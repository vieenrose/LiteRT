# Copyright 2026. Apache-2.0.
"""RSS budget breakdown + shared-KV-buffer decode prototype.

Part 1 (--mode breakdown): stage-by-stage RSS/anon deltas of the classic
Interpreter pipeline (q8, zh90s prefix), attributing the footprint.

Part 2 (--mode compiled): decode loop on ai_edge_litert CompiledModel with the
SAME TensorBuffer bound as KV input and KV output (LiteRT Next buffer
binding) — no per-step host KV round-trips. Verifies tokens match the
Interpreter path, reports decode tok/s + RSS.
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np


def rss_mb():
    out = {}
    with open("/proc/self/smaps_rollup") as f:
        for line in f:
            for k in ("Rss:", "Anonymous:", "Private_Clean:", "Private_Dirty:",
                      "Shared_Clean:"):
                if line.startswith(k):
                    out[k[:-1]] = int(line.split()[1]) / 1024.0
    return out


_last = {"Rss": 0.0}


def stage(name):
    global _last
    gc.collect()
    cur = rss_mb()
    d = cur["Rss"] - _last["Rss"]
    print(f"[mem] {name:44s} RSS={cur['Rss']:8.0f} MB (+{d:7.0f})  "
          f"anon={cur['Anonymous']:8.0f}  file={cur['Rss']-cur['Anonymous']:7.0f}")
    _last = cur
    return cur


def load_prefix(args):
    """audio -> fused embeds + prompt ids (uses the normal runner)."""
    import soundfile as sf
    from moss_td.runner import MossTdLiteRT
    from moss_td import common

    audio, sr = sf.read(args.wav, dtype="float32")
    rt = MossTdLiteRT(args.encoder, args.embedder, args.decoder,
                      args.checkpoint, args.threads)
    return rt, audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["breakdown", "compiled", "compiled_full"], required=True)
    ap.add_argument("--wav", default="/tmp/claude-1001/zh90s.wav")
    ap.add_argument("--encoder", default="models/moss_td_encoder_q8.tflite")
    ap.add_argument("--embedder", default="models/moss_td_embedder_q8.tflite")
    ap.add_argument("--decoder", default="models/moss_td_decoder_q8_ekv2048.tflite")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    from moss_td import common

    stage("baseline (python+numpy)")
    import soundfile as sf
    from transformers import AutoTokenizer, WhisperFeatureExtractor
    snap = common.resolve_snapshot(args.checkpoint)
    tok = AutoTokenizer.from_pretrained(snap)
    fe = WhisperFeatureExtractor.from_pretrained(snap)
    stage("transformers tokenizer+fe")

    if args.mode == "breakdown":
        from ai_edge_litert.interpreter import Interpreter
        enc = Interpreter(model_path=args.encoder, num_threads=args.threads)
        stage("encoder Interpreter created (mmap)")
        emb = Interpreter(model_path=args.embedder, num_threads=args.threads)
        stage("embedder Interpreter created")
        dec = Interpreter(model_path=args.decoder, num_threads=args.threads)
        stage("decoder Interpreter created")

        audio, _ = sf.read(args.wav, dtype="float32")
        from moss_td.runner import MossTdLiteRT, KvIo, NEG_INF
        rt = MossTdLiteRT(args.encoder, args.embedder, args.decoder,
                          args.checkpoint, args.threads)
        stage("runner init (all 3 interpreters again)")
        embeds = rt.encode_audio(audio)
        stage("after encoder invoke (arena+repack)")
        ids = common.build_input_ids(rt.tok, embeds.shape[0])
        fused = rt.embed_tokens(ids)
        stage("after embedder invoke")
        apos = [i for i, t in enumerate(ids) if t == common.AUDIO_TOKEN_ID]
        fused[apos] = embeds
        S = len(ids)

        kv = rt.kv_io.zeros(rt.decode_sig)
        stage(f"numpy KV staging alloc (28x2x(1,{rt.kv_len},8,128) f32)")
        # first prefill_1024 invoke
        sig = rt.prefills[1024]
        kvio = KvIo(sig)
        emb1 = np.zeros((1, 1024, common.HIDDEN), dtype=np.float32)
        emb1[0, :min(1024, S)] = fused[:1024]
        mask = np.full((1, 1, 1024, rt.kv_len), NEG_INF, dtype=np.float32)
        for r in range(min(1024, S)):
            mask[0, 0, r, :r + 1] = 0.0
        call = {kvio.in_names[k]: kv[k] for k in kvio.in_names}
        for want, arr in (("input_embeds", emb1),
                          ("input_pos", np.arange(1024, dtype=np.int32)),
                          ("mask", mask)):
            call[[nm for nm in sig.inputs if want in nm][0]] = arr
        out = sig(**call)
        stage("after first prefill_1024 invoke (arena+KV I/O)")
        for k, nm in kvio.out_names.items():
            kv[k] = out[nm]
        stage("after prefill output copy-out (fresh numpy KV)")
        # decode
        dsig = rt.decode_sig
        dkv = KvIo(dsig)
        for step in range(4):
            e = np.zeros((1, 1, 1024), dtype=np.float32)
            m = np.full((1, 1, 1, rt.kv_len), NEG_INF, dtype=np.float32)
            m[0, 0, 0, :1025 + step] = 0.0
            c = {dkv.in_names[k]: kv[k] for k in dkv.in_names}
            c[[nm for nm in dsig.inputs if "input_embeds" in nm][0]] = e
            c[[nm for nm in dsig.inputs if "input_pos" in nm][0]] = (
                np.array([1024 + step], dtype=np.int32))
            c[[nm for nm in dsig.inputs if "mask" in nm][0]] = m
            o = dsig(**c)
            for k, nm in dkv.out_names.items():
                kv[k] = o[nm]
        stage("decode steady state (4 steps)")
        weights = {"encoder": 321, "embedder": 161, "decoder": 456}
        print(f"[mem] flatbuffer file sizes: {weights} MB total "
              f"{sum(weights.values())} MB")
        kv_mb = 28 * 2 * rt.kv_len * 8 * 128 * 4 / 1e6
        print(f"[mem] analytic: one full KV set = {kv_mb:.0f} MB "
              f"(numpy staging holds 1x; runtime holds input+output per "
              f"allocated signature)")
        return

    if args.mode == "compiled_full":
        return compiled_full(args, stage)

    # ---- compiled mode -------------------------------------------------------
    from ai_edge_litert.interpreter import Interpreter
    from ai_edge_litert import compiled_model as cm_lib

    audio, _ = sf.read(args.wav, dtype="float32")
    from moss_td.runner import MossTdLiteRT, KvIo, NEG_INF
    rt = MossTdLiteRT(args.encoder, args.embedder, args.decoder,
                      args.checkpoint, args.threads)
    embeds = rt.encode_audio(audio)
    ids = common.build_input_ids(rt.tok, embeds.shape[0])
    fused = rt.embed_tokens(ids)
    apos = [i for i, t in enumerate(ids) if t == common.AUDIO_TOKEN_ID]
    fused[apos] = embeds
    S = len(ids)

    # reference: interpreter path decode of N tokens (also produces the
    # prefilled KV that we hand to the compiled model)
    kv = rt.kv_io.zeros(rt.decode_sig)
    plens = sorted(rt.prefills.keys(), reverse=True)
    pos = 0
    last_hidden = None
    while pos < S:
        n = next((l for l in plens if l <= S - pos), plens[-1])
        sig = rt.prefills[n]
        kvio = KvIo(sig)
        real = min(n, S - pos)
        embn = np.zeros((1, n, common.HIDDEN), dtype=np.float32)
        embn[0, :real] = fused[pos:pos + real]
        mask = np.full((1, 1, n, rt.kv_len), NEG_INF, dtype=np.float32)
        for r in range(n):
            mask[0, 0, r, :pos + r + 1] = 0.0
        call = {kvio.in_names[k]: kv[k] for k in kvio.in_names}
        for want, arr in (("input_embeds", embn),
                          ("input_pos", np.arange(pos, pos + n, dtype=np.int32)),
                          ("mask", mask)):
            call[[nm for nm in sig.inputs if want in nm][0]] = arr
        out = sig(**call)
        for k, nm in kvio.out_names.items():
            kv[k] = out[nm]
        hid = [v for nm, v in out.items() if "hidden" in nm][0]
        last_hidden = hid[0, real - 1]
        pos += real
    logits = rt.logits(last_hidden)
    stage("prefill done (interpreter path)")

    # interpreter-path greedy reference tokens
    import copy as _copy
    kv_ref = {k: v.copy() for k, v in kv.items()}
    ref_tokens = []
    lg = logits.copy()
    p = S
    dsig = rt.decode_sig
    dkv = KvIo(dsig)
    t0 = time.perf_counter()
    for _ in range(args.max_new):
        t = int(np.argmax(lg))
        ref_tokens.append(t)
        if t == common.EOS_TOKEN_ID:
            break
        e = rt.embed_tokens([t]).reshape(1, 1, -1)
        m = np.full((1, 1, 1, rt.kv_len), NEG_INF, dtype=np.float32)
        m[0, 0, 0, :p + 1] = 0.0
        c = {dkv.in_names[k]: kv_ref[k] for k in dkv.in_names}
        c[[nm for nm in dsig.inputs if "input_embeds" in nm][0]] = e
        c[[nm for nm in dsig.inputs if "input_pos" in nm][0]] = np.array([p], dtype=np.int32)
        c[[nm for nm in dsig.inputs if "mask" in nm][0]] = m
        o = dsig(**c)
        for k, nm in dkv.out_names.items():
            kv_ref[k] = o[nm]
        lg = rt.logits([v for nm, v in o.items() if "hidden" in nm][0][0, 0])
        p += 1
    interp_s = time.perf_counter() - t0
    n_ref = len(ref_tokens)
    print(f"[interp ] {n_ref} tokens in {interp_s:.1f}s "
          f"({n_ref/interp_s:.2f} tok/s)")
    del kv_ref
    stage("after interpreter decode ref")

    # compiled model with shared KV buffers
    cm = cm_lib.CompiledModel.from_file(args.decoder)
    sigs = cm.get_signature_list()
    in_names = sigs["decode"]["inputs"]
    out_names = sigs["decode"]["outputs"]
    di = cm.get_signature_index("decode")
    in_bufs = cm.create_input_buffers(di)
    out_bufs = cm.create_output_buffers(di)
    stage("CompiledModel created + buffers")

    import re
    pat = re.compile(r"kv_(k|v)_(\d+)$|kv_cache_(k|v)_(\d+)$")
    def kvkey(n):
        m = pat.search(n)
        if not m:
            return None
        return (m.group(1) or m.group(3), int(m.group(2) or m.group(4)))

    # Alias: for every kv output, reuse the corresponding INPUT buffer object.
    shared = 0
    for oi, on in enumerate(out_names):
        k = kvkey(on)
        if k is None:
            continue
        ii = [j for j, n in enumerate(in_names) if kvkey(n) == k][0]
        out_bufs[oi] = in_bufs[ii]
        shared += 1
    print(f"[compiled] aliased {shared} KV buffers (in==out)")

    # load prefilled KV into the input buffers once
    for j, n in enumerate(in_names):
        k = kvkey(n)
        if k is not None:
            in_bufs[j].write(np.ascontiguousarray(kv[k]))
    stage("KV loaded into runtime buffers (one-time)")

    e_idx = [j for j, n in enumerate(in_names) if "input_embeds" in n][0]
    p_idx = [j for j, n in enumerate(in_names) if "input_pos" in n][0]
    m_idx = [j for j, n in enumerate(in_names) if "mask" in n][0]
    h_oidx = [j for j, n in enumerate(out_names) if "hidden" in n][0]

    cm_tokens = []
    lg = logits.copy()
    p = S
    t0 = time.perf_counter()
    for _ in range(args.max_new):
        t = int(np.argmax(lg))
        cm_tokens.append(t)
        if t == common.EOS_TOKEN_ID:
            break
        e = rt.embed_tokens([t]).reshape(1, 1, -1)
        m = np.full((1, 1, 1, rt.kv_len), NEG_INF, dtype=np.float32)
        m[0, 0, 0, :p + 1] = 0.0
        in_bufs[e_idx].write(np.ascontiguousarray(e))
        in_bufs[p_idx].write(np.array([p], dtype=np.int32))
        in_bufs[m_idx].write(np.ascontiguousarray(m))
        cm.run_by_index(di, in_bufs, out_bufs)
        hid = out_bufs[h_oidx].read(1024, np.float32)
        lg = rt.logits(np.asarray(hid).reshape(-1))
        p += 1
    cm_s = time.perf_counter() - t0
    n_cm = len(cm_tokens)
    print(f"[compiled] {n_cm} tokens in {cm_s:.1f}s ({n_cm/cm_s:.2f} tok/s)")
    stage("after compiled decode")
    match = cm_tokens == ref_tokens
    print(f"[verify] tokens match interpreter path: {match}")
    if not match:
        for i, (a, b) in enumerate(zip(cm_tokens, ref_tokens)):
            if a != b:
                print(f"  first diff at step {i}: {a} vs {b}")
                break




def compiled_full(args, stage):
    """Clean-measurement pipeline: decoder runs ONLY on CompiledModel with a
    single shared KV TensorBuffer set reused across prefill and decode
    signatures (aliased as both input and output). No host KV staging."""
    import re
    import soundfile as sf
    import numpy as np
    import time
    from moss_td import common
    from moss_td.runner import MossTdLiteRT, NEG_INF
    from ai_edge_litert import compiled_model as cm_lib

    audio, _ = sf.read(args.wav, dtype="float32")
    rt = MossTdLiteRT(args.encoder, args.embedder, args.decoder,
                      args.checkpoint, args.threads)
    stage("runner init (enc/emb interpreters; dec interpreter unused)")
    embeds = rt.encode_audio(audio)
    stage("after encoder invoke")
    ids = common.build_input_ids(rt.tok, embeds.shape[0])
    fused = rt.embed_tokens(ids)
    apos = [i for i, t in enumerate(ids) if t == common.AUDIO_TOKEN_ID]
    fused[apos] = embeds
    S = len(ids)
    stage("after embedder invoke + fuse")

    cm = cm_lib.CompiledModel.from_file(args.decoder)
    sigs = cm.get_signature_list()
    pat = re.compile(r"kv_(k|v)_(\d+)$|kv_cache_(k|v)_(\d+)$")
    def kvkey(n):
        m = pat.search(n)
        return None if not m else (m.group(1) or m.group(3),
                                   int(m.group(2) or m.group(4)))

    dec_idx = cm.get_signature_index("decode")
    dec_in_names = sigs["decode"]["inputs"]
    dec_out_names = sigs["decode"]["outputs"]
    dec_in = cm.create_input_buffers(dec_idx)
    dec_out = cm.create_output_buffers(dec_idx)
    # shared KV set: the decode-signature input buffers ARE the canonical KV
    kvbuf = {}
    for j, n in enumerate(dec_in_names):
        k = kvkey(n)
        if k is not None:
            kvbuf[k] = dec_in[j]
    for j, n in enumerate(dec_out_names):
        k = kvkey(n)
        if k is not None:
            dec_out[j] = kvbuf[k]      # alias decode outputs onto same bufs
    stage("CompiledModel + single shared KV buffer set")

    # prefill signatures reuse the same KV buffers for input AND output
    pf = {}
    for name in sigs:
        if not name.startswith("prefill_"):
            continue
        n = int(name.split("_")[1])
        idx = cm.get_signature_index(name)
        ins = cm.create_input_buffers(idx)
        outs = cm.create_output_buffers(idx)
        innames = sigs[name]["inputs"]
        outnames = sigs[name]["outputs"]
        for j, nm in enumerate(innames):
            k = kvkey(nm)
            if k is not None:
                ins[j] = kvbuf[k]
        for j, nm in enumerate(outnames):
            k = kvkey(nm)
            if k is not None:
                outs[j] = kvbuf[k]
        pf[n] = (idx, ins, outs, innames, outnames)
    stage("prefill signatures wired to shared KV buffers")

    def named(names, needle):
        return [j for j, n in enumerate(names) if needle in n][0]

    t0 = time.perf_counter()
    plens = sorted(pf.keys(), reverse=True)
    pos = 0
    last_hidden = None
    while pos < S:
        n = next((l for l in plens if l <= S - pos), plens[-1])
        idx, ins, outs, innames, outnames = pf[n]
        real = min(n, S - pos)
        embn = np.zeros((1, n, common.HIDDEN), dtype=np.float32)
        embn[0, :real] = fused[pos:pos + real]
        mask = np.full((1, 1, n, rt.kv_len), NEG_INF, dtype=np.float32)
        for r in range(n):
            mask[0, 0, r, :pos + r + 1] = 0.0
        ins[named(innames, "input_embeds")].write(np.ascontiguousarray(embn))
        ins[named(innames, "input_pos")].write(
            np.arange(pos, pos + n, dtype=np.int32))
        ins[named(innames, "mask")].write(np.ascontiguousarray(mask))
        cm.run_by_index(idx, ins, outs)
        hid = np.asarray(outs[named(outnames, "hidden")].read(
            n * common.HIDDEN, np.float32)).reshape(n, common.HIDDEN)
        last_hidden = hid[real - 1]
        pos += real
    prefill_s = time.perf_counter() - t0
    stage(f"prefill done ({S} tokens, {prefill_s:.1f}s)")

    lg = rt.logits(last_hidden)
    e_i = named(dec_in_names, "input_embeds")
    p_i = named(dec_in_names, "input_pos")
    m_i = named(dec_in_names, "mask")
    h_o = named(dec_out_names, "hidden")
    tokens = []
    p = S
    t0 = time.perf_counter()
    while len(tokens) < args.max_new:
        t = int(np.argmax(lg))
        tokens.append(t)
        if t == common.EOS_TOKEN_ID:
            break
        e = rt.embed_tokens([t]).reshape(1, 1, -1)
        m = np.full((1, 1, 1, rt.kv_len), NEG_INF, dtype=np.float32)
        m[0, 0, 0, :p + 1] = 0.0
        dec_in[e_i].write(np.ascontiguousarray(e))
        dec_in[p_i].write(np.array([p], dtype=np.int32))
        dec_in[m_i].write(np.ascontiguousarray(m))
        cm.run_by_index(dec_idx, dec_in, dec_out)
        hid = np.asarray(dec_out[h_o].read(common.HIDDEN, np.float32))
        lg = rt.logits(hid.reshape(-1))
        p += 1
    dec_s = time.perf_counter() - t0
    stage("decode done")
    ntok = len(tokens)
    print(f"[compiled_full] prefill {S} tok in {prefill_s:.1f}s; "
          f"decode {ntok} tok in {dec_s:.1f}s ({ntok/dec_s:.2f} tok/s)")
    text = rt.tok.decode([t for t in tokens if t != common.EOS_TOKEN_ID],
                         skip_special_tokens=True).strip()
    print("[compiled_full] text:", text[:300])


if __name__ == "__main__":
    main()
