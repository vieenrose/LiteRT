# VoxSum LiteRT ASR engines — MOSS-TD · X-ASR · SenseVoice

Reference implementations of the three ASR backends ported to **LiteRT 2.1.6**
for [VoxSumDroid](https://github.com/vieenrose/VoxSumDroid), upstreamed here so
the ports live next to the Nemotron sample (`../nemotron/`). All engines run on
the LiteRT-Next `CompiledModel` C++ API (`litert::Model` / `litert::Environment`
/ `litert::CompiledModel`) — no TFLite classic `Interpreter`, no ONNX Runtime,
no ggml.

Snapshot of VoxSumDroid `v0.30.0` (versionCode 103, 2026-07-24).
SenseVoice was removed from the shipping app in that release (quality verdict:
zh audio partially recognized as ja) — its engine is preserved here from git
history (`2202064`, deleted in `00d04e0`).

## Layout

```
cpp/                              — JNI + engine sources (built into one .so by the app)
  moss_lite_engine.{h,cc}         — MOSS-TD encoder/decoder loop on CompiledModel
  moss_lite_jni.cpp               — JNI surface for MossLiteEngine.kt (gpu flag param)
  xasr_lite_jni.cpp               — full greedy zipformer-transducer decode in C++
  sv_lite_jni.cpp                 — SenseVoice CTC decode (recovered from history)
  lite_pod_jni.cpp                — "pods": Silero-VAD, pyannote segmentation, CAM++ embedder
  whisper_mel.{h,cc}              — 80/128-bin log-mel front end (Whisper-style, for MOSS)
  bench_lite.cc                   — standalone on-device benchmark CLI (no JNI)
kotlin/                           — Kotlin drivers (front ends, tokenizers, orchestration)
  MossLiteEngine.kt  XasrLiteEngine.kt  XasrLiteAsr.kt
  SenseVoiceLiteEngine.kt  SenseVoiceLiteAsr.kt   (historical)
  LitePod.kt  VadSegmenter.kt
```

The app builds `cpp/` (minus `sv_lite_jni.cpp` and `bench_lite.cc`) into a
single `libvoxsum-mosslite.so` linked against the **prebuilt** `libLiteRt.so`
extracted from the official `com.google.ai.edge.litert:litert:2.1.6` AAR
(plus `libLiteRtClGlAccelerator.so` for the GPU path). Vendored C headers under
the app's `cpp/mosslite/litert/`.

## Engines at a glance

| | MOSS-TD | X-ASR (zipformer) | SenseVoice |
|---|---|---|---|
| model | Luigi/moss-transcribe-diarize-litert (q4 0.9B enc-dec, +fp16 trio) | Luigi/xasr-litert (295 MB OCTAV dyn-int8) | Luigi/sensevoice-litert (q8) |
| task | one-pass ASR **+ diarization** + ts | zh-en transducer, ts | zh/en/ja/ko/yue CTC |
| decode | AR token loop, KvStore KV cache | greedy RNN-T, ≤1 sym/frame, blank 0, unk 4015 suppressed | CTC argmax + collapse |
| front end | whisper 80-mel (`whisper_mel.cc`) | povey fbank 80, **normalized** samples (no ×32768) | 80-mel + LFR(7,6) + CMVN |
| ts step | model-emitted `[ss.ss]` markers | 0.04 s / frame | — (CTC frame-derived) |
| RTF (Samsung A53, classic) | ≈3.8 | 0.13–0.14 | 0.17 |
| GPU | encoder GPU-first w/ sticky CPU fallback | CPU (GPU no win) | CPU |

Key implementation notes:

- **X-ASR export**: 979 params transplanted from the sherpa ONNX zipformer2
  (icefall zh-en punct transducer) into an ai-edge-torch re-implementation;
  masked bucketed signatures `enc_375/750/1500/3000` + `decoder` + `joiner`;
  encoder parity gate 3.1e-06 vs ONNX; OCTAV dynamic-int8 adds **zero** CER
  over the fp32 tflite. Decoder context is `y[1,2]` i32 with `-1` pad init.
- **MOSS-TD**: `Component` wrapper owns env+model; GPU compile attempted per
  graph with `kLiteRtHwAcceleratorGpu|Cpu`, engine falls back (and rebuilds a
  fresh KvStore) if the GPU compile fails — sticky per process.
- **Pods** (`lite_pod_jni.cpp`): Silero-VAD tflite, pyannote-segmentation-3.0
  tflite, CAM++/wespeaker speaker embedder — used for diarization of the
  non-diarizing backends.
- **Thread count matters**: `CompiledModel`'s default is far too low; set the
  big-core count explicitly (measured ~8× on big.LITTLE).

## bench_lite.cc

```sh
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang++ \
  -O2 bench_lite.cc moss_lite_engine.cc whisper_mel.cc -I. \
  -L<jniLibs>/arm64-v8a -lLiteRt -llog -landroid -o bench_lite
adb push bench_lite libLiteRt.so /data/local/tmp/
adb shell 'cd /data/local/tmp && LD_LIBRARY_PATH=. ./bench_lite …'
```

## Provenance / license

Sources are Apache-2.0 (same as VoxSumDroid). Model artifacts carry their own
licenses (see the HF repos). The X-ASR export & quantization scripts live in
the VoxSumDroid history and `Luigi/xasr-litert`'s model card.
