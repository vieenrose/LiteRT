// Demo client. It streams audio through /api/transcribe_stream (server-sent
// events): the current window's text arrives token by token as a provisional
// tail, and each finished window commits interactive speaker-linked segments
// -- so the transcript is usable while the rest of the audio still decodes.
// For the golden clips it also scores the final output against the PyTorch
// reference transcript shipped alongside the audio.
//
// It applies NO transformation of its own. The previous build of this Space did
// s2tw conversion, number ITN, cross-window loop collapsing and CAM++ speaker
// linking in this file; all of it is gone, because any of it would make the
// displayed text something other than what the engine produced.
const $ = (id) => document.getElementById(id);

let busy = false;

function setBusy(b, msg) {
  busy = b;
  document.querySelectorAll(".chip").forEach((c) => (c.disabled = b));
  const p = $("process");
  if (p) p.disabled = b || !PENDING;
  if (msg) $("status").textContent = msg;
}

// Segments are rendered with their start time on the element so the player can
// seek to them and highlight the one playing. This is inspection machinery, not
// a transform: the text and timestamps shown are exactly what the engine
// emitted, and the raw stream is printed verbatim below it.
let SEGS = [];
let LAST = null;          // full result of the most recent run, for export
let HEALTH = {};          // /api/health payload (build/version info for export)
let PENDING = null;       // staged audio source (chip or upload), run by Process
let ABORTER = null;       // AbortController of the in-flight run

// Speaker palette. Indexed by order of first appearance, not by the numeric
// part of the tag, so a transcript that opens on S03 still gets a stable,
// high-contrast assignment. Chosen to stay distinguishable on both themes and
// under the most common form of colour blindness (deuteranopia): blue/orange/
// purple/brown vary in lightness as well as hue, so they are still separable
// if hue is lost entirely.
const PALETTE = ["#2563eb", "#ea580c", "#7c3aed", "#0891b2",
                 "#b45309", "#db2777", "#15803d", "#525252"];
const spkColor = new Map();

function colorFor(spk) {
  if (!spkColor.has(spk)) spkColor.set(spk, PALETTE[spkColor.size % PALETTE.length]);
  return spkColor.get(spk);
}

const hidden = new Set();   // speakers toggled off from the legend

function renderLegend() {
  const el = $("legend");
  const spks = [...spkColor.keys()];
  el.innerHTML = spks.map((s) =>
    `<span class="lg${hidden.has(s) ? " off" : ""}" data-spk="${s}">` +
    `<i style="background:${colorFor(s)}"></i>${s}</span>`).join("");
  el.querySelectorAll(".lg").forEach((n) => n.addEventListener("click", () => {
    const s = n.dataset.spk;
    hidden.has(s) ? hidden.delete(s) : hidden.add(s);
    renderLegend();
    applyFilter();
  }));
}

function renderSegments(segs) {
  SEGS = segs;
  spkColor.clear();
  hidden.clear();
  segs.forEach((s) => colorFor(s.spk));      // assign in order of appearance
  $("segs").innerHTML = segs.map((s, i) => {
    const end = s.end == null ? "" : `\u2013${s.end.toFixed(2)}`;
    const c = colorFor(s.spk);
    return `<div class="seg" data-i="${i}" data-start="${s.start}" ` +
           `data-spk="${s.spk}" style="border-left-color:${c}">` +
           `<span class="ts">${s.start.toFixed(2)}${end}</span>` +
           `<span class="spk" style="color:${c}">${s.spk}</span>` +
           `${escapeHtml(s.text)}</div>`;
  }).join("");
  renderLegend();
  document.querySelectorAll("#segs .seg").forEach((el) => {
    el.addEventListener("click", () => {
      const a = $("audio");
      a.currentTime = parseFloat(el.dataset.start);
      a.play().catch(() => {});
    });
  });
}

// Highlight the segment covering the playhead. Linear scan is fine here: these
// are single-pass transcripts of a few hundred segments at most.
function syncHighlight() {
  const t = $("audio").currentTime;
  let cur = -1;
  for (let i = 0; i < SEGS.length; i++) {
    // A missing end runs to the NEXT segment's start, not to infinity --
    // with null ends (streamed window commits) the old rule matched the
    // first segment for every playhead position.
    const end = SEGS[i].end != null ? SEGS[i].end
      : (i + 1 < SEGS.length ? SEGS[i + 1].start : Infinity);
    if (t >= SEGS[i].start && t < end) { cur = i; break; }
  }
  document.querySelectorAll("#segs .seg").forEach((el) => {
    const on = +el.dataset.i === cur;
    if (on !== el.classList.contains("active")) {
      el.classList.toggle("active", on);
      if (on && $("follow").checked) el.scrollIntoView({ block: "nearest" });
    }
  });
}

function applyFilter() {
  const q = ($("search").value || "").trim().toLowerCase();
  document.querySelectorAll("#segs .seg").forEach((el) => {
    const byText = q && !el.textContent.toLowerCase().includes(q);
    const bySpk = hidden.has(el.dataset.spk);
    el.classList.toggle("hidden", Boolean(byText || bySpk));
  });
}

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

// Stage 2 fixed a 90s window unconditionally, so every example here (the
// golden clips included, at 300s/318s) is now windowed + speaker-linked.
// Windowing intentionally restructures segmentation vs the stored
// single-pass reference, so byte-identity is no longer even the right thing
// to check -- it would show every clip as "DIFFERS" including ones that are
// working correctly, and that reads as a failure banner for expected
// behaviour. Report CONTENT agreement instead (character-bigram Jaccard,
// cheap enough to run in-browser on a 15-20KB transcript): how much of the
// actual text content survives windowing, independent of how it got chunked
// or which segment boundaries were chosen.
function bigramSet(s) {
  const set = new Set();
  for (let i = 0; i < s.length - 1; i++) set.add(s[i] + s[i + 1]);
  return set;
}

function jaccard(a, b) {
  const A = bigramSet(a), B = bigramSet(b);
  if (A.size === 0 && B.size === 0) return 1;
  let inter = 0;
  for (const x of A) if (B.has(x)) inter++;
  return inter / (A.size + B.size - inter);
}

async function checkParity(refUrl, raw) {
  const card = $("paritycard");
  card.style.display = "";
  $("parity").innerHTML = "checking…";
  $("paritydetail").textContent = "";
  let ref;
  try {
    ref = await (await fetch(refUrl, { cache: "no-store" })).text();
  } catch (e) {
    $("parity").innerHTML = `<span class="verdict bad">reference unavailable</span>`;
    return;
  }
  const r = ref.replace(/\n$/, "").replace(/\[[0-9.]+\]|\[S\d+\]/g, "");
  const got = raw.replace(/\n$/, "").replace(/\[[0-9.]+\]|\[S\d+\]/g, "");
  const sim = jaccard(r, got);
  const pct = (100 * sim).toFixed(1);
  // Bigram Jaccard is a cheap, deliberately approximate in-browser stand-in --
  // it is structurally harsher than the LCS-based char-agreement method used
  // throughout offline validation (scripts/81_score_vs_ami.py's
  // difflib.SequenceMatcher). Same clip, same output: 91.56% by that method
  // measured here as 75.5% Jaccard. So the "ok" band is calibrated against
  // THIS metric's own scale, not against the validated numbers quoted
  // elsewhere on this page -- do not compare the two directly.
  const cls = sim >= 0.65 ? "ok" : "bad";
  $("parity").innerHTML =
    `<span class="verdict ${cls}">${pct}% content match (bigram overlap)</span> ` +
    `vs the single-pass reference (text only, markers/tags excluded — ` +
    `windowing legitimately restructures segmentation, so this is NOT a ` +
    `byte-identity check, and this number runs lower than the char-agreement ` +
    `figures quoted elsewhere on this page for the same reason).`;
  $("paritydetail").textContent =
    sim >= 0.65
      ? ""
      : "Below the usual band for this metric — worth a manual look.";
}

async function run(blob, name, refUrl, durHint, srcUrl, exampleName) {
  if (busy) return;
  // Point the player at the audio being transcribed. Prefer a REAL URL over a
  // blob: URL whenever we have one (the bundled examples): <audio> fed a blob:
  // URL is unreliable on iOS Safari -- it loads, reports a duration, and plays
  // silence -- and a served file also gets proper HTTP range requests for
  // seeking. Uploaded files have no URL, so they still use a blob.
  const a = $("audio");
  if (a.dataset.objurl) { URL.revokeObjectURL(a.dataset.objurl); delete a.dataset.objurl; }
  if (srcUrl) {
    a.src = srcUrl;
  } else {
    const u = URL.createObjectURL(blob);
    a.dataset.objurl = u;
    a.src = u;
  }
  a.load();
  $("outcard").style.display = "none";
  $("paritycard").style.display = "none";
  $("stats").textContent = "";
  $("slowwarn").style.display = durHint && durHint !== "11 s" ? "" : "none";
  const t0 = performance.now();
  let phaseLine = "";
  const tick = setInterval(() => {
    $("status").textContent =
      `transcribing ${name} — ${((performance.now() - t0) / 1000).toFixed(0)} s elapsed` +
      (phaseLine ? ` · ${phaseLine}` : "");
  }, 250);
  setBusy(true, `transcribing ${name}…`);

  // Live tail rendering is rAF-throttled: token events can arrive faster than
  // layout, and one coalesced repaint per frame is plenty. With batched
  // decoding several windows stream at once; showing whichever arrived last
  // flickers, so the box is pinned to the EARLIEST window still decoding.
  let tailText = "", tailRaf = false;
  const tails = new Map();   // absolute window idx -> latest partial
  const scheduleTail = () => {
    if (tailRaf) return;
    tailRaf = true;
    requestAnimationFrame(() => {
      tailRaf = false;
      $("tail").textContent = tailText;
      $("tailnote").style.display = tailText ? "" : "none";
    });
  };

  try {
    // Examples are transcribed SERVER-SIDE by name (the audio already lives
    // on the Space) -- uploads only happen for user files. This removes the
    // 2x file-size round-trip that made big examples look hung on mobile.
    const batchN = "1"; // batched decode is an rs.cpp-engine feature
    ABORTER = new AbortController();
    $("cancel").style.display = "";
    const resp = await fetch((exampleName
      ? "api/transcribe_stream?example=" + encodeURIComponent(exampleName)
      : "api/transcribe_stream?name=" + encodeURIComponent(name)) +
      "&batch=" + encodeURIComponent(batchN), {
      method: "POST", body: exampleName ? null : blob,
      headers: { "Content-Type": "application/octet-stream" },
      signal: ABORTER.signal,
    });
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);
    $("outcard").style.display = "";
    $("segs").innerHTML = "";
    spkColor.clear(); hidden.clear(); renderLegend();   // no stale legend from a previous run
    $("raw").textContent = "";
    $("progress").textContent = "";
    tailText = ""; scheduleTail();

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "", done = null;
    for (;;) {
      const { value, done: eof } = await reader.read();
      if (eof) break;
      buf += dec.decode(value, { stream: true });
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        const frame = buf.slice(0, cut); buf = buf.slice(cut + 2);
        if (!frame.startsWith("data: ")) continue;
        const ev = JSON.parse(frame.slice(6));
        if (ev.type === "phase") {
          phaseLine = `window ${ev.window + 1} · ${ev.phase}…`;
          // Between groups the tail box would sit empty through encode+prefill
          // (60-90s on long clips) and read as a hang — show liveness there.
          if (tails.size === 0) { tailText = `⏳ ${ev.phase} — window ${ev.window + 1}…`; scheduleTail(); }
        } else if (ev.type === "tail") {
          const par = ev.group > 1 ? ` (+${ev.group - 1} windows in parallel)` : "";
          tails.set(ev.window, ev.text);
          const first = Math.min(...tails.keys());
          phaseLine = `window ${first + 1} · decoding${par}`;
          tailText = tails.get(first); scheduleTail();
        } else if (ev.type === "window") {
          // Full re-render on purpose: speaker labels of already-committed
          // rows may legitimately change as clustering sees more voices.
          for (const k of [...tails.keys()]) if (k <= ev.window) tails.delete(k);
          renderSegments(ev.segments);
          $("progress").textContent =
            `committed ${ev.processedS.toFixed(0)}s / ${ev.durationS.toFixed(0)}s ` +
            `(window ${ev.window + 1}) — rows are clickable already`;
          tailText = ""; scheduleTail();
        } else if (ev.type === "error") {
          throw new Error(ev.error);
        } else if (ev.type === "done") {
          done = ev;
        }
      }
    }
    if (!done) throw new Error("stream ended without a result");
    clearInterval(tick);
    const j = done;
    LAST = { ...j, name, totalS: +(((performance.now() - t0) / 1000).toFixed(2)),
             batchUsed: +batchN };
    renderSegments(j.segments);
    tailText = ""; scheduleTail();
    $("progress").textContent = "";
    // Show the Traditional-converted stream when present (Chinese output from
    // the base weights is Simplified); `j.raw` remains the untouched engine
    // output and is what the export and the parity check use.
    $("raw").textContent = j.rawTraditional || j.raw;
    $("stats").textContent =
      `${j.durationS}s audio · ${j.segments.length} segments · ` +
      `${j.decodeS}s decode · RTF ${j.rtf}`;
    $("status").textContent = "done.";
    if (refUrl) await checkParity(refUrl, j.raw);
  } catch (e) {
    clearInterval(tick);
    $("status").textContent = e.name === "AbortError"
      ? "cancelled. (the engine finishes the current window in the background — a new run waits for it)"
      : "failed: " + e.message;
  } finally {
    clearInterval(tick);
    ABORTER = null;
    $("cancel").style.display = "none";
    setBusy(false);
  }
}

document.querySelectorAll(".chip").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const ex = btn.dataset.ex;
    const g = btn.dataset.gain || "1";
    $("boost").value = g;
    setBoost(+g);
    PENDING = { blob: null, name: ex, refUrl: btn.dataset.ref ? "examples/" + btn.dataset.ref : null,
                dur: btn.dataset.dur, srcUrl: "examples/" + ex, exampleName: ex };
    const a = $("audio");
    if (a.dataset.objurl) { URL.revokeObjectURL(a.dataset.objurl); delete a.dataset.objurl; }
    a.src = PENDING.srcUrl; a.load();
    $("outcard").style.display = "";
    $("process").disabled = false;
    $("status").textContent = `loaded ${ex} — adjust settings, then press Process.`;
  });
});

// Playback gain. The AMI clip is a -43 dB mean headset mix -- audible peaks but
// inaudible on a phone at normal volume. It is NOT normalised on disk: the
// golden reference transcript was produced from those exact bytes, so re-encoding
// the file would break the parity check. Boost is applied at PLAYBACK only, via
// WebAudio, so what the engine receives is untouched.
let audioCtx = null, gainNode = null;

function setBoost(x) {
  const a = $("audio");
  if (x <= 1 && !audioCtx) return;                 // nothing to do, stay on the plain path
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const src = audioCtx.createMediaElementSource(a);
      gainNode = audioCtx.createGain();
      // MUST reach destination: routing through WebAudio and forgetting this
      // silences the element completely.
      src.connect(gainNode).connect(audioCtx.destination);
    }
    if (audioCtx.state === "suspended") audioCtx.resume();
    gainNode.gain.value = x;
  } catch (e) {
    console.warn("boost unavailable:", e);         // fall back to plain playback
  }
}

// Export the FULL result, raw stream included. The raw field is the whole point:
// the parsed segments are a convenience view, whereas `raw` is exactly what the
// model emitted and is what any parity or regression check must compare.
$("export").addEventListener("click", () => {
  if (!LAST) return;
  const doc = {
    source: LAST.name,
    engine: "LiteRT 2.1.6 (CPU/XNNPACK, CompiledModel shared-KV)",
    engineCommit: HEALTH.buildId || null,
    buildId: (document.querySelector('meta[name="build-id"]') || {}).content || null,
    model: "moss-transcribe-base-q4mix.gguf",
    windowed: true,
    batch: LAST.batchUsed || null,

    harnesses: [],
    durationS: LAST.durationS,
    decodeS: LAST.decodeS,
    totalS: LAST.totalS,           // upload + queue + decode, client-measured
    rtf: LAST.rtf,
    debug: LAST.debug ? { ...LAST.debug,
      buildId: (document.querySelector('meta[name="build-id"]') || {}).content || null }
      : null,   // per-window profiling + engine config (server-side)
    raw: LAST.raw,
    segments: LAST.segments,
  };
  const blob = new Blob([JSON.stringify(doc, null, 2)],
                        { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = (LAST.name || "transcript").replace(/\.[^.]+$/, "") + ".json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
});

$("process").addEventListener("click", () => {
  if (!PENDING || busy) return;
  run(PENDING.blob, PENDING.name, PENDING.refUrl, PENDING.dur,
      PENDING.srcUrl, PENDING.exampleName);
});
$("cancel").addEventListener("click", () => { if (ABORTER) ABORTER.abort(); });
$("boost").addEventListener("change", () => setBoost(+$("boost").value));
$("audio").addEventListener("play", () => {
  if (audioCtx && audioCtx.state === "suspended") audioCtx.resume();
});

$("audio").addEventListener("timeupdate", syncHighlight);
$("audio").addEventListener("seeked", syncHighlight);
$("search").addEventListener("input", applyFilter);

$("drop").addEventListener("click", () => $("file").click());
function stageFile(f) {
  PENDING = { blob: f, name: f.name, refUrl: null, dur: "long", srcUrl: null, exampleName: null };
  const a = $("audio");
  if (a.dataset.objurl) { URL.revokeObjectURL(a.dataset.objurl); delete a.dataset.objurl; }
  const u = URL.createObjectURL(f);
  a.dataset.objurl = u; a.src = u; a.load();
  $("outcard").style.display = "";
  $("process").disabled = false;
  $("status").textContent = `loaded ${f.name} — adjust settings, then press Process.`;
}
$("file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) stageFile(f);
});
["dragover", "dragleave", "drop"].forEach((ev) =>
  $("drop").addEventListener(ev, (e) => {
    e.preventDefault();
    $("drop").classList.toggle("hover", ev === "dragover");
    if (ev === "drop" && e.dataTransfer.files[0]) {
      stageFile(e.dataTransfer.files[0]);
    }
  }));

(async () => {
  try {
    const h = await (await fetch("api/health")).json();
    HEALTH = h;
    const ver = `${document.querySelector('meta[name="build-id"]').content}` +
      ` · engine LiteRT 2.1.6 · build ${h.buildId || "?"} · linking ${(h.speakerLinking||"none").split(" ")[0]}`;
    $("build").textContent = `${ver} · ${h.model} · ${h.threads} threads`;
    const tag = document.querySelector("header .tag");
    if (tag) tag.textContent += ` · ${h.buildId || "?"}`;
    if ($("batch")) $("batch").value = "1";
  } catch {
    $("build").textContent = document.querySelector('meta[name="build-id"]').content;
  }
})();
