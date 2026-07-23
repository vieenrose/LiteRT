"""MOSS-Transcribe-Diarize demo — windowed LiteRT pipeline (CPU/XNNPACK).

Pipeline:
  * ENGINE: LiteRT (ai-edge-litert 2.1.6) port of MOSS-TD — three flatbuffers
    (Whisper-medium encoder q8, tied embedder q8, Qwen3-0.6B decoder int4
    blockwise-32) run through the CompiledModel API with the KV cache held in
    TensorBuffers aliased as BOTH input and output of every call, so the cache
    never crosses the host boundary. Conversion + parity harness:
    github.com/vieenrose/LiteRT branch `moss-td-port`. The f32 build of this
    exact pipeline is byte-identical to the project's pinned PyTorch f32
    reference on all three golden clips; the int4 decoder deployed here scores
    98.99% text fidelity (95.45% with timestamps) vs that reference on the zh
    90 s window and 100% text on jfk.
  * WINDOWING: 90 s windows cut at the quietest point in the last 12 s (a real
    pause, not a fixed boundary) — identical logic to the C++ Space.
  * SPEAKER LINKING: NOT PRESENT in this build. The C++ Space links speakers
    across windows with CAM++ (rapidspeech-core); that library requires the
    full C++ build tree this Space intentionally drops. [Sxx] tags here are
    WINDOW-LOCAL: the numbering restarts at every window boundary. This is an
    honest limitation, not a silent regression.
  * WEIGHTS: 0.73 GB total (encoder q8 0.32 + embedder q8 0.16 + decoder
    int4-b32 0.25), downloaded from Luigi/moss-transcribe-diarize-litert.

Deliberately absent (same policy as the C++ Space): repetition penalty,
EOS-coverage suppression, loop guards, number ITN. Batched multi-window decode
and audio-KV eviction are rs.cpp-engine features and do not exist here.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import soundfile as sf

# This Space targets ZeroGPU hardware (downgrading to cpu-basic needs PRO), and
# the ZeroGPU runtime expects the `spaces` package to be imported and at least
# one @spaces.GPU entry point to exist. A build without them started cleanly and
# was then torn down into RUNTIME_ERROR. The ASR engine itself is a CPU
# LiteRT/XNNPACK build and never touches the GPU; this is purely to satisfy the
# runtime. (Preserved verbatim from the C++ Space, where this failure mode was
# observed live.)
try:
    import spaces  # noqa: F401
except ImportError:
    spaces = None

# XNNPACK threads. The ZeroGPU-class host advertises ~192 cores but the
# container gets ~16 usable vCPUs; the C++ Space measured 16T beating 4T by 3x
# on the same host class. On smaller hosts fall back to what exists.
THREADS = int(os.environ.get("MTD_THREADS", "0") or min(16, os.cpu_count() or 4))

os.environ["GRADIO_SSR_MODE"] = "false"
import gradio as gr
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).parent.resolve()
MODEL_REPO = "Luigi/moss-transcribe-diarize-litert"
ENCODER_FILE = "moss_td_encoder_q8.tflite"
EMBEDDER_FILE = "moss_td_embedder_q8.tflite"
# int4 blockwise-32 decoder, KV sized for 90 s windows: prompt <= ~1250 tokens
# + generation headroom (the engine additionally caps max_new at
# kv_len - prompt - 1, so the budget can never run past the static cache).
DECODER_FILE = "moss_td_decoder_q4b32_ekv2560.tflite"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                   "merges.txt", "added_tokens.json", "special_tokens_map.json",
                   "preprocessor_config.json"]
BUILD_ID = "litert-1"
SR = 16000

print("[startup] fetching LiteRT models…", flush=True)
ENC = hf_hub_download(MODEL_REPO, ENCODER_FILE)
EMB = hf_hub_download(MODEL_REPO, EMBEDDER_FILE)
DEC = hf_hub_download(MODEL_REPO, DECODER_FILE)
TOK_DIR = None
for f in TOKENIZER_FILES:
    p = hf_hub_download(MODEL_REPO, f"tokenizer/{f}")
    TOK_DIR = str(Path(p).parent)

# One decode at a time -- enforced at TWO levels (semaphore for fair queueing,
# mutex held by the engine thread for the full decode so a disconnected
# streaming client can never let a second request overlap the running one).
# Same rationale and incident history as the C++ Space.
_engine_lock = asyncio.Semaphore(1)
_engine_mutex = threading.Lock()

import litert_engine  # noqa: E402
import windowing  # noqa: E402
import zhtw  # noqa: E402

print(f"[startup] loading LiteRT engine (threads={THREADS}, once, resident)…",
      flush=True)
_t0 = time.time()
_ENGINE = litert_engine.MossLiteRT(ENC, EMB, DEC, TOK_DIR, threads=THREADS)
print(f"[startup] engine resident in {time.time() - _t0:.1f}s "
      f"(kv_len={_ENGINE.kv_len})", flush=True)

# Cross-window speaker linking is a C++-Space feature (CAM++ via
# rapidspeech-core). This build ships without it: window-local [Sxx] tags.
_SPEAKER_MODEL = None


def transcribe_file(wav_path: str) -> str:
    with _engine_mutex:
        _, raw = windowing.transcribe_windowed_streaming(
            (_ENGINE, None), _SPEAKER_MODEL, wav_path)
        return raw


def ensure_wav16k(path: str) -> tuple[str, bool]:
    """Return (path-to-16kHz-mono-WAV, is_temp_file)."""
    try:
        info = sf.info(path)
        if info.samplerate == SR and info.channels == 1:
            return path, False
    except Exception:
        pass
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                    "-vn", "-ac", "1", "-ar", str(SR), tmp], check=True)
    return tmp, True


SEG_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]\s*(?:\[(S\d+)\])?\s*([^\[]*)")


def parse_segments(text: str):
    segs, spk = [], "S01"
    for m in SEG_RE.finditer(text):
        if m.group(2):
            spk = m.group(2)
        body = (m.group(3) or "").strip()
        if body:
            segs.append({"start": float(m.group(1)), "spk": spk, "text": body})
    for i, s in enumerate(segs):
        s["end"] = segs[i + 1]["start"] if i + 1 < len(segs) else None
    return segs


async def health():
    return {"ok": True,
            "model": f"{ENCODER_FILE} + {EMBEDDER_FILE} + {DECODER_FILE}",
            "engine": "LiteRT 2.1.6 (CPU/XNNPACK, CompiledModel shared-KV)",
            "threads": THREADS, "hostCpus": os.cpu_count(), "windowed": True,
            "windowS": windowing.WINDOW_S,
            "speakerLinking": "none (window-local [Sxx] tags; CAM++ linking is a C++-Space feature)",
            "zhTraditional": zhtw.available(), "liveStreaming": True,
            "buildId": BUILD_ID, "batchDefault": 1,
            "kvLen": _ENGINE.kv_len,
            "parity": "f32 pipeline byte-identical to the pinned PyTorch f32 "
                      "reference on the 3 golden clips; deployed int4 decoder "
                      "98.99% text fidelity on the zh 90s window",
            "tokensPerAudioSecond": windowing.TOKENS_PER_AUDIO_SECOND}


async def transcribe_route(request: Request):
    body = await request.body()
    q = request.query_params
    name = q.get("name", "upload")
    tmp = tempfile.mktemp(suffix="_" + os.path.basename(name)[-40:])
    Path(tmp).write_bytes(body)
    wav, is_temp_wav = tmp, False
    try:
        t0 = time.time()
        wav, is_temp_wav = ensure_wav16k(tmp)
        dur = sf.info(wav).frames / float(SR)
        t1 = time.time()
        async with _engine_lock:
            raw = await asyncio.to_thread(transcribe_file, wav)
        wall = time.time() - t1
        return JSONResponse({
            "raw": raw, "rawTraditional": zhtw.to_traditional(raw),
            "segments": parse_segments(zhtw.to_traditional(raw)),
            "durationS": round(dur, 2), "decodeS": round(wall, 2),
            "decodeMs": round(wall * 1000), "rtf": round(wall / max(dur, 1e-6), 3),
            "loadS": round(t1 - t0, 2),
        })
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        if is_temp_wav:
            try:
                os.unlink(wav)
            except OSError:
                pass


async def transcribe_stream_route(request: Request):
    """Server-sent events: live transcript while the pipeline runs.
    Same event schema as the C++ Space (phase / tail / window / done / error).
    batch is forced to 1: batched multi-window decode is an rs.cpp-engine
    feature this LiteRT engine does not implement."""
    q = request.query_params
    batch_n = 1  # forced; see docstring
    example = q.get("example")
    if example:
        src = ROOT / "examples" / os.path.basename(example)
        if not src.is_file():
            return JSONResponse({"error": f"unknown example {example!r}"},
                                status_code=404)
        name = os.path.basename(example)
        tmp = tempfile.mktemp(suffix="_" + name[-40:])
        shutil.copyfile(src, tmp)
    else:
        body = await request.body()
        name = q.get("name", "upload")
        tmp = tempfile.mktemp(suffix="_" + os.path.basename(name)[-40:])
        Path(tmp).write_bytes(body)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    last_tail = [0.0]
    prof: dict = {}

    def _prof(ev: dict):
        now = time.time()
        w = ev.get("window")
        if w is None:
            return
        p = prof.setdefault(w, {})
        if ev["type"] == "phase":
            p.setdefault("phases", {})[ev["phase"]] = now
        elif ev["type"] == "tail":
            p.setdefault("firstTok", now)
            p["lastTok"] = now
            p["tokens"] = ev.get("tokens", 0)

    def on_event(ev: dict):
        _prof(ev)
        if ev["type"] == "tail":
            now = time.time()
            if now - last_tail[0] < 0.25:
                return
            last_tail[0] = now
            ev = dict(ev, text=zhtw.to_traditional(ev["text"]))
        elif ev["type"] == "window":
            segs = ev["segments"]
            ev = dict(ev, segments=[
                {"start": round(s["start"], 2),
                 "end": (round(segs[i + 1]["start"], 2) if i + 1 < len(segs)
                         else s.get("rawEnd")),
                 "spk": s["spk"],
                 "text": zhtw.to_traditional(s["text"]), "win": s.get("win")}
                for i, s in enumerate(segs)])
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    def run_pipeline(wav_path: str):
        with _engine_mutex:
            segs, raw = windowing.transcribe_windowed_streaming(
                (_ENGINE, None), _SPEAKER_MODEL, wav_path,
                on_event=on_event, batch_n=batch_n)
            return raw

    async def gen():
        wav, is_temp_wav = tmp, False
        try:
            t0 = time.time()
            wav, is_temp_wav = ensure_wav16k(tmp)
            dur = sf.info(wav).frames / float(SR)
            yield "data: " + json.dumps(
                {"type": "start", "durationS": round(dur, 2)}) + "\n\n"
            async with _engine_lock:
                task = asyncio.get_running_loop().run_in_executor(
                    None, run_pipeline, wav)
                t1 = time.time()
                while True:
                    get = asyncio.create_task(queue.get())
                    await asyncio.wait({get, task},
                                       return_when=asyncio.FIRST_COMPLETED)
                    if get.done():
                        yield "data: " + json.dumps(get.result(),
                                                    ensure_ascii=False) + "\n\n"
                    else:
                        get.cancel()
                    if task.done() and queue.empty():
                        break
                raw = await task
            wall = time.time() - t1
            windows_dbg = []
            for w in sorted(prof):
                p = prof[w]
                ph = p.get("phases", {})
                enc_t = ph.get("encode"); pre_t = ph.get("prefill")
                first, last = p.get("firstTok"), p.get("lastTok")
                dec_s = round(last - first, 2) if first and last else None
                toks = p.get("tokens", 0)
                windows_dbg.append({
                    "window": w, "tokens": toks,
                    "encodeToPrefillS": round(pre_t - enc_t, 2) if enc_t and pre_t else None,
                    "prefillToFirstTokS": round(first - pre_t, 2) if pre_t and first else None,
                    "decodeS": dec_s,
                    "tokPerS": round(toks / dec_s, 2) if dec_s and toks else None,
                })
            debug = {
                "buildId": BUILD_ID,
                "engine": "LiteRT 2.1.6 (CPU/XNNPACK, CompiledModel shared-KV)",
                "threads": THREADS,
                "hostCpus": os.cpu_count(),
                "kvLen": _ENGINE.kv_len,
                "windowS": windowing.WINDOW_S,
                "model": DECODER_FILE,
                "windows": windows_dbg,
            }
            yield "data: " + json.dumps({
                "type": "done", "debug": debug, "raw": raw,
                "rawTraditional": zhtw.to_traditional(raw),
                "segments": parse_segments(zhtw.to_traditional(raw)),
                "durationS": round(dur, 2), "decodeS": round(wall, 2),
                "rtf": round(wall / max(dur, 1e-6), 3),
            }, ensure_ascii=False) + "\n\n"
        except Exception as e:  # noqa: BLE001
            yield "data: " + json.dumps(
                {"type": "error", "error": f"{type(e).__name__}: {e}"}) + "\n\n"
        finally:
            for pth, cond in ((tmp, True), (wav, is_temp_wav)):
                if cond:
                    try:
                        os.unlink(pth)
                    except OSError:
                        pass

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def transcribe_api(audio_path: str) -> str:
    """Transcribe meeting audio with window-local speaker tags and timestamps
    (zh/en).

    Args:
        audio_path: path to an audio file (wav/mp3/m4a…).
    """
    wav, is_temp_wav = ensure_wav16k(audio_path)
    try:
        return transcribe_file(wav)
    finally:
        if is_temp_wav:
            try:
                os.unlink(wav)
            except OSError:
                pass


if spaces is not None:
    @spaces.GPU(duration=10)
    def gpu_ping() -> str:
        """Report the allocated ZeroGPU device (diagnostic only — the ASR engine
        runs natively on this Space's CPU)."""
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        return r.stdout.strip() or r.stderr.strip() or "no gpu"
else:
    def gpu_ping() -> str:
        """Report the allocated GPU (local/no-ZeroGPU fallback)."""
        return "spaces package unavailable"


with gr.Blocks(title="MOSS-TD LiteRT — API") as demo:
    gr.Interface(
        fn=transcribe_api,
        inputs=gr.Audio(type="filepath", label="Audio"),
        outputs=gr.Textbox(label="Raw transcript ([start][Sxx]text[end])"),
    )
    with gr.Accordion("diagnostics", open=False):
        _b = gr.Button("GPU ping")
        _o = gr.Textbox(label="ZeroGPU device")
        _b.click(gpu_ping, [], _o)

fast_app, _, _ = demo.launch(server_name="0.0.0.0", server_port=7860,
                             mcp_server=True, prevent_thread_lock=True,
                             ssr_mode=False)

from starlette.responses import FileResponse  # noqa: E402
from starlette.routing import Mount, Route  # noqa: E402

STATIC = ROOT / "static"

async def _index_route(request):
    return FileResponse(STATIC / "index.html")


async def _health_route(request):
    return JSONResponse(await health())


for _r in (Route("/", _index_route, methods=["GET"]),
           Route("/api/health", _health_route, methods=["GET"]),
           Route("/api/transcribe", transcribe_route, methods=["POST"]),
           Route("/api/transcribe_stream", transcribe_stream_route, methods=["POST"])):
    fast_app.router.routes.insert(0, _r)

fast_app.router.routes.append(
    Mount("/examples", app=StaticFiles(directory=str(ROOT / "examples")),
          name="examples"))
fast_app.router.routes.append(
    Mount("/", app=StaticFiles(directory=str(STATIC)), name="static"))

demo.block_thread()
