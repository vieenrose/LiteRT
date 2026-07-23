"""Windowed transcription + cross-window speaker linking.

Extracted from scripts/85_window_sweep.py (distil-vibevoice-asr repo) where it
was developed and validated. Copied verbatim, no logic changes, so the numbers
measured there (WER, speaker accuracy, turn-crossing rate) describe exactly
what this module does.

This is a WRAPPER around two C APIs, not an engine change:
  - moss_transcribe_capi_transcribe_pcm() (moss_td/, vendored MIT, unmodified)
    is called once per window; results are stitched in Python.
  - rs_speaker_* (rapidspeech-core's CAM++, unmodified) embeds each window-
    local speaker's pooled audio; a constrained-agglomerative clustering pass
    links identity across windows.

Window length is fixed at 90s (WINDOW_S below), chosen after a length sweep on
both languages: it is the best point found for zh (turn-crossing 97.3% clean,
highest of 60/90/180/300s tested) and is not meaningfully worse than any other
tested length for en (91-94% clean band, no length stands out). See
project memory "project-staged-delivery" for the full sweep.
"""
from __future__ import annotations

import re

import numpy as np
import soundfile as sf

SR = 16000
WINDOW_S = 90.0
TOKENS_PER_AUDIO_SECOND = 12.0
LINK_THRESHOLD = 0.50

SEG_RE = re.compile(r"\[(\d+(?:\.\d+)?)(?:-(\d+(?:\.\d+)?))?\](?:\[(S\d+)\])?([^\[]*)")


# ------------------------------------------------------------- engine seam ---
# This build's engine is litert_engine.MossLiteRT (LiteRT CompiledModel,
# CPU/XNNPACK) instead of the C++ .so. The seam keeps the exact call shapes
# the rest of this module was validated with: engine = (engine_obj, None),
# transcribe(lib, ctx, pcm, max_new, on_event) where on_event(kind, text,
# cur, total) receives kind-1 phase events ("encode"/"prefill"/"decode") and
# kind-0 token events carrying the FULL partial transcript.

def transcribe(lib, ctx, pcm: np.ndarray, max_new: int, on_event=None) -> str:
    return lib.transcribe(pcm, max_new, on_event=on_event)


def transcribe_batch(lib, ctx, pcm_list, max_new: int, on_event=None):
    raise NotImplementedError(
        "batched multi-window decode is an rs.cpp-engine feature; "
        "the LiteRT engine decodes windows sequentially (batch=1)")


def load_speaker_model(so_path: str, gguf: str):
    raise NotImplementedError(
        "CAM++ speaker linking requires librapidspeech-core (C++ build); "
        "this LiteRT build ships without cross-window linking")


def embed(lib, sp, dim, pcm: np.ndarray):
    raise NotImplementedError("no speaker model in the LiteRT build")


def pause_cut(piece: np.ndarray, window_s: float, snap_s: float) -> float:
    """Seconds into `piece` of the quietest 0.4s frame within the last snap_s of
    the window -- so a cut lands in silence, not mid-utterance."""
    n = len(piece)
    if n < window_s * SR:
        return n / SR
    frm = max(0, n - int(snap_s * SR))
    win, hop = int(0.4 * SR), int(0.1 * SR)
    best, best_e = n - win, float("inf")
    o = frm
    while o + win <= n:
        e = float(np.sum(piece[o:o + win] ** 2))
        if e < best_e:
            best_e, best = e, o
        o += hop
    return (best + win / 2) / SR


def parse_window(text: str, win_start_s: float):
    segs, prev_spk = [], "S01"
    for m in SEG_RE.finditer(text):
        t = float(m.group(1))
        if m.group(3):
            prev_spk = m.group(3)
        body = (m.group(4) or "").strip()
        if body:
            segs.append({"start": win_start_s + t,
                        "rawEnd": win_start_s + float(m.group(2)) if m.group(2) else None,
                        "spk": prev_spk, "text": body})
    return segs


def run_windowed(lib, ctx, pcm: np.ndarray, window_s: float, tokens_per_second: float):
    dur_s = len(pcm) / SR
    snap_s = 12 if window_s >= 90 else 5
    cursor, all_segs = 0.0, []
    win_idx = 0
    while cursor < dur_s - 0.5:
        frm = int(cursor * SR)
        piece = pcm[frm: frm + int(window_s * SR)]
        if len(piece) < SR:
            break
        is_last = frm + len(piece) >= len(pcm)
        cut = pause_cut(piece, window_s, snap_s) if not is_last else len(piece) / SR
        win_start = cursor
        max_new = max(5120, int(tokens_per_second * window_s))
        text = transcribe(lib, ctx, piece[:int(cut * SR)], max_new)
        segs = parse_window(text, win_start)
        cut_abs = win_start + cut
        kept = [s for s in segs if s["start"] < cut_abs - 0.01]
        for s in kept:
            s["win"] = win_idx
            s["local_spk"] = s["spk"]
        all_segs.extend(kept)
        cursor = cut_abs
        win_idx += 1
        if is_last:
            break
    return all_segs


def render(segs):
    out = []
    for i, s in enumerate(segs):
        out.append(f"[{s['start']:.2f}][{s['spk']}]{s['text']}")
    return "".join(out) + (f"[{segs[-1].get('rawEnd') or segs[-1]['start']:.2f}]" if segs else "")


# --------------------------------------------------------- speaker linking ---
def load_speaker_model(so_path: str, gguf: str):
    lib = ctypes.CDLL(so_path)
    lib.rs_speaker_init_from_file.restype = ctypes.c_void_p
    lib.rs_speaker_init_from_file.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.rs_speaker_dim.argtypes = [ctypes.c_void_p]
    lib.rs_speaker_dim.restype = ctypes.c_int32
    lib.rs_speaker_embed.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
    sp = lib.rs_speaker_init_from_file(gguf.encode(), 0)
    assert sp, "campplus load failed"
    dim = lib.rs_speaker_dim(ctypes.c_void_p(sp))
    return lib, sp, dim


def embed(lib, sp, dim, pcm: np.ndarray):
    if len(pcm) < 1600:
        return None
    buf = (ctypes.c_float * dim)()
    pcm = np.ascontiguousarray(pcm, dtype=np.float32)
    err = lib.rs_speaker_embed(ctypes.c_void_p(sp),
                               pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                               len(pcm), buf, dim)
    if err != 0:
        return None
    v = np.frombuffer(buf, dtype=np.float32).copy()
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def units_for_window(segs_in_window: list, win_start: float, win_cut_abs: float,
                     end_cap: float = 12.0):
    """(window, local-tag) -> [(start, end)] relative time ranges, all inside
    ONE window. A unit's segments are, by construction (the key includes
    win_idx), always drawn from the SAME window -- so its embedding audio is
    always a subset of that window's own samples and never needs any other
    window's audio. This is what makes streaming (discard each window's audio
    once its units are embedded) exact, not an approximation."""
    units: dict = {}
    for i, s in enumerate(segs_in_window):
        key = (s["win"], s["local_spk"])
        nxt = segs_in_window[i + 1]["start"] if i + 1 < len(segs_in_window) else win_cut_abs
        end = min(nxt, s["start"] + end_cap, win_cut_abs)
        units.setdefault(key, []).append((s["start"], end))
    return units


def embed_units(lib, sp, dim, pcm: np.ndarray, units: dict, t0: float = 0.0):
    """Embed each unit's pooled audio (up to 30s). `pcm` covers [t0, t0+len(pcm)/SR);
    unit time ranges are absolute and must fall inside that span (true both for
    a single window's `piece`, t0=window start, and for a whole-file `pcm`,
    t0=0 -- so this one function serves both the streaming and array paths)."""
    out = {}
    for key, ranges in units.items():
        total = sum(e - a for a, e in ranges)
        budget, chunks = 30.0, []
        for a, e in sorted(ranges):
            take = min(e - a, max(0.0, budget))
            if take <= 0:
                break
            rel_a = a - t0
            chunks.append(pcm[int(rel_a * SR):int((rel_a + take) * SR)])
            budget -= take
        pooled = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        out[key] = (embed(lib, sp, dim, pooled), total)
    return out


def cluster_units(segs: list, unit_emb: dict, threshold: float = LINK_THRESHOLD):
    """Constrained agglomerative clustering over PRE-COMPUTED unit embeddings
    -- shared by both link_speakers (array path) and the streaming path, so
    they produce identical clustering given identical embeddings. Not
    per-utterance (39% of AMI utterances are <2s backchannels, too little
    audio for a stable CAM++ embedding -- unit pooling fixed that). Not greedy
    time-order streaming (one bad merge cascaded via the running-mean centroid
    -- measured 68%->99%+ speaker accuracy fixing this). Cannot-link: two
    units in the SAME window with DIFFERENT local tags are provably different
    speakers (raw per-window purity 0.94-1.00), enforced as a soft penalty
    rather than a hard rule since the engine occasionally over-splits one real
    speaker within a window."""
    keys = [k for k in unit_emb if unit_emb[k][0] is not None]
    orphans = [k for k in unit_emb if unit_emb[k][0] is None]
    n = len(keys)
    clusters = [{k} for k in keys]
    cluster_emb = {i: unit_emb[keys[i]][0].copy() for i in range(n)}
    cluster_n = {i: 1 for i in range(n)}
    cannot_link = set()
    by_win: dict = {}
    for k in keys:
        by_win.setdefault(k[0], []).append(k)
    for win, ks in by_win.items():
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                cannot_link.add(frozenset((ks[a], ks[b])))

    CONSTRAINT_PENALTY = 0.35
    alive = set(range(n))
    while True:
        best_pair, best_eff = None, threshold
        alive_l = list(alive)
        for ai in range(len(alive_l)):
            for bi in range(ai + 1, len(alive_l)):
                i, j = alive_l[ai], alive_l[bi]
                constrained = any(frozenset((u, v)) in cannot_link
                                  for u in clusters[i] for v in clusters[j])
                sim = float(np.dot(cluster_emb[i], cluster_emb[j]))
                eff = sim - (CONSTRAINT_PENALTY if constrained else 0.0)
                if eff > best_eff:
                    best_eff, best_pair = eff, (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        clusters[i] |= clusters[j]
        cluster_emb[i] = (cluster_emb[i] * cluster_n[i] + cluster_emb[j] * cluster_n[j])
        cluster_emb[i] /= np.linalg.norm(cluster_emb[i])
        cluster_n[i] += cluster_n[j]
        alive.discard(j)

    MIN_UNIT_SECONDS = 8.0
    cluster_dur = {i: sum(unit_emb[k][1] for k in clusters[i]) for i in alive}
    big = {i for i in alive if cluster_dur[i] >= MIN_UNIT_SECONDS}
    tiny = sorted(alive - big, key=lambda i: cluster_dur[i])
    for i in tiny:
        if not big:
            break
        cands = []
        for j in big:
            if any(frozenset((u, v)) in cannot_link
                   for u in clusters[i] for v in clusters[j]):
                continue
            cands.append((float(np.dot(cluster_emb[i], cluster_emb[j])), j))
        if not cands:
            continue
        _, j = max(cands)
        clusters[j] |= clusters[i]
        alive.discard(i)

    unit_gid = {}
    for gid, i in enumerate(sorted(alive)):
        for k in clusters[i]:
            unit_gid[k] = gid
    for k in orphans:
        unit_gid[k] = None

    for s in segs:
        gid = unit_gid.get((s["win"], s["local_spk"]))
        s["spk"] = f"S{gid + 1:02d}" if gid is not None else s["local_spk"]
    return segs


def link_speakers(lib, sp, dim, pcm: np.ndarray, segs: list, threshold: float = LINK_THRESHOLD):
    """Array-path linking: whole-file `pcm` already resident (used by the
    offline research sweep, scripts/85_window_sweep.py). See
    transcribe_windowed_streaming for the memory-bounded path the Space uses,
    which never holds the whole file."""
    dur_s = len(pcm) / SR
    units = units_for_window(segs, 0.0, dur_s)
    unit_emb = embed_units(lib, sp, dim, pcm, units, t0=0.0)
    return cluster_units(segs, unit_emb, threshold)


def transcribe_windowed(engine, speaker_model, pcm: np.ndarray):
    """Array path (whole file resident) -- kept for the offline research
    sweep. See transcribe_windowed_streaming for what the Space runs.
    engine = (lib, ctx); speaker_model = (lib, sp, dim) or None to skip linking."""
    lib, ctx = engine
    segs = run_windowed(lib, ctx, pcm, WINDOW_S, TOKENS_PER_AUDIO_SECOND)
    if speaker_model is not None:
        splib, sp, dim = speaker_model
        segs = link_speakers(splib, sp, dim, pcm, segs, LINK_THRESHOLD)
    return segs, render(segs)


def transcribe_windowed_streaming(engine, speaker_model, audio_path: str,
                                  window_s: float = WINDOW_S,
                                  tokens_per_second: float = TOKENS_PER_AUDIO_SECOND,
                                  threshold: float = LINK_THRESHOLD,
                                  on_event=None, batch_n: int = 1):
    """Memory-bounded path: never holds more than one window's audio at once.

    Measured need for this on real long meetings (2026-07-21 validation): the
    array path's peak RSS grew ~3.8 MB per minute of audio (a 16 kHz mono
    float32 buffer costs exactly that), predicted-vs-observed within 3% across
    16/87.5/123-minute clips -- confirming the growth was the whole-file numpy
    array, not the engine (whose per-window KV cache is genuinely flat
    regardless of total duration; windowing already solved THAT half).

    Fix: read each window's samples directly from the file via
    soundfile.SoundFile (seek + read, no upfront full-file load), and embed
    that window's speaker units immediately while its samples are still
    resident, before moving on. This is exact, not approximate: by
    construction a unit's key includes its window index, so every unit's
    audio always lies entirely within the one window it came from (see
    units_for_window) -- there is no case where linking needs audio outside
    the window currently in memory. Only two things survive across windows:
    segment metadata (text/timestamps/tags -- negligible) and per-unit
    embeddings (192 floats each -- negligible). Peak memory is now O(one
    window + embeddings + model weights), flat regardless of total duration.

    batch_n > 1 decodes windows in GROUPS of batch_n through the engine's
    batched API (weight reads amortized across streams; identity-gated
    byte-identical to sequential). batch_n == 1 takes exactly the original
    single-window path below — nothing regresses.
    """
    if batch_n > 1:
        return _transcribe_windowed_streaming_batched(
            engine, speaker_model, audio_path, window_s, tokens_per_second,
            threshold, on_event, batch_n)
    lib, ctx = engine
    splib = sp = dim = None
    if speaker_model is not None:
        splib, sp, dim = speaker_model

    f = sf.SoundFile(audio_path)
    assert f.samplerate == SR, f"expected {SR} Hz, got {f.samplerate}"
    dur_s = len(f) / SR
    snap_s = 12 if window_s >= 90 else 5
    cursor, all_segs, unit_emb = 0.0, [], {}
    win_idx = 0
    try:
        while cursor < dur_s - 0.5:
            f.seek(int(cursor * SR))
            piece = f.read(frames=int(window_s * SR), dtype="float32", always_2d=False)
            if piece.ndim > 1:
                piece = piece.mean(axis=1)
            if len(piece) < SR:
                break
            is_last = int(cursor * SR) + len(piece) >= len(f)
            cut = pause_cut(piece, window_s, snap_s) if not is_last else len(piece) / SR
            win_start = cursor
            max_new = max(5120, int(tokens_per_second * window_s))
            engine_cb = None
            if on_event is not None:
                _wi, _ws = win_idx, win_start  # bind per-window values
                def engine_cb(kind, text, cur, total):  # noqa: E306
                    on_event({"type": "tail", "window": _wi, "winStart": _ws,
                              "text": text, "tokens": cur} if kind == 0 else
                             {"type": "phase", "window": _wi, "winStart": _ws,
                              "phase": text, "cur": cur, "total": total,
                              "durationS": dur_s})
            text = transcribe(lib, ctx, piece[:int(cut * SR)], max_new,
                              on_event=engine_cb)
            segs = parse_window(text, win_start)
            cut_abs = win_start + cut
            kept = [s for s in segs if s["start"] < cut_abs - 0.01]
            for s in kept:
                s["win"] = win_idx
                s["local_spk"] = s["spk"]
            all_segs.extend(kept)

            if splib is not None and kept:
                win_units = units_for_window(kept, win_start, cut_abs)
                unit_emb.update(embed_units(splib, sp, dim, piece, win_units, t0=win_start))

            if on_event is not None:
                # Commit everything so far, RE-CLUSTERED over all units seen:
                # speaker labels of already-committed segments may legitimately
                # change as later windows disambiguate -- consumers re-render
                # the whole committed list, they never append blindly.
                committed = (cluster_units([dict(s) for s in all_segs], unit_emb,
                                           threshold)
                             if splib is not None else all_segs)
                on_event({"type": "window", "window": win_idx,
                          "processedS": round(cut_abs, 2),
                          "durationS": round(dur_s, 2),
                          "segments": committed})

            del piece  # explicit: don't wait for the next loop iteration's rebind
            cursor = cut_abs
            win_idx += 1
            if is_last:
                break
    finally:
        f.close()

    if splib is None:
        return all_segs, render(all_segs)
    segs = cluster_units(all_segs, unit_emb, threshold)
    return segs, render(segs)


def _transcribe_windowed_streaming_batched(engine, speaker_model, audio_path: str,
                                           window_s: float, tokens_per_second: float,
                                           threshold: float, on_event, batch_n: int):
    """Batched variant of transcribe_windowed_streaming: windows are cut
    EXACTLY as in the single path (same streaming reader, same pause_cut in
    the same order -- cuts depend only on audio, never on transcripts, so
    buffering a group ahead of decoding changes nothing), buffered in groups
    of batch_n (one group of window PCMs resident at a time), then decoded
    simultaneously via transcribe_batch. Per-window parsing / unit embedding
    is unchanged; commit + re-cluster happens per GROUP: the "window" event
    keeps its schema with "window" = LAST absolute window index of the group
    and processedS = the group's end cut."""
    lib, ctx = engine
    splib = sp = dim = None
    if speaker_model is not None:
        splib, sp, dim = speaker_model

    f = sf.SoundFile(audio_path)
    assert f.samplerate == SR, f"expected {SR} Hz, got {f.samplerate}"
    dur_s = len(f) / SR
    snap_s = 12 if window_s >= 90 else 5
    cursor, all_segs, unit_emb = 0.0, [], {}
    win_idx = 0
    max_new = max(5120, int(tokens_per_second * window_s))

    def flush_group(group):
        gn = len(group)
        engine_cb = None
        if on_event is not None:
            tok_counts = [0] * gn
            def engine_cb(kind, text, cur, total):  # noqa: E306
                g = group[cur] if 0 <= cur < gn else group[0]
                if kind == 0:
                    tok_counts[cur] += 1
                    on_event({"type": "tail", "window": g["wi"],
                              "winStart": g["ws"], "text": text,
                              "tokens": tok_counts[cur], "group": gn})
                else:
                    on_event({"type": "phase", "window": g["wi"],
                              "winStart": g["ws"], "phase": text,
                              "cur": cur, "total": total, "durationS": dur_s})
        texts = transcribe_batch(lib, ctx, [g["pcm"] for g in group], max_new,
                                 on_event=engine_cb)
        for g, text in zip(group, texts):
            segs = parse_window(text, g["ws"])
            kept = [s for s in segs if s["start"] < g["cut_abs"] - 0.01]
            for s in kept:
                s["win"] = g["wi"]
                s["local_spk"] = s["spk"]
            all_segs.extend(kept)
            if splib is not None and kept:
                win_units = units_for_window(kept, g["ws"], g["cut_abs"])
                unit_emb.update(embed_units(splib, sp, dim, g["pcm"], win_units,
                                            t0=g["ws"]))
        if on_event is not None:
            committed = (cluster_units([dict(s) for s in all_segs], unit_emb,
                                       threshold)
                         if splib is not None else all_segs)
            on_event({"type": "window", "window": group[-1]["wi"],
                      "processedS": round(group[-1]["cut_abs"], 2),
                      "durationS": round(dur_s, 2), "segments": committed})

    try:
        group = []
        while cursor < dur_s - 0.5:
            f.seek(int(cursor * SR))
            piece = f.read(frames=int(window_s * SR), dtype="float32",
                           always_2d=False)
            if piece.ndim > 1:
                piece = piece.mean(axis=1)
            if len(piece) < SR:
                break
            is_last = int(cursor * SR) + len(piece) >= len(f)
            cut = pause_cut(piece, window_s, snap_s) if not is_last else len(piece) / SR
            cut_abs = cursor + cut
            group.append({"wi": win_idx, "ws": cursor, "cut_abs": cut_abs,
                          "pcm": np.ascontiguousarray(piece[:int(cut * SR)],
                                                      dtype=np.float32)})
            del piece  # only the cut portion survives in the group buffer
            cursor = cut_abs
            win_idx += 1
            if len(group) >= batch_n or is_last:
                flush_group(group)
                group = []
            if is_last:
                break
        if group:
            flush_group(group)
    finally:
        f.close()

    if splib is None:
        return all_segs, render(all_segs)
    segs = cluster_units(all_segs, unit_emb, threshold)
    return segs, render(segs)
