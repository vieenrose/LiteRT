---
title: MOSS Transcribe Diarize — LiteRT port
emoji: 🎙️
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: LiteRT (CPU/XNNPACK) port of MOSS-Transcribe-Diarize
---

The same windowed MOSS-Transcribe-Diarize demo as
[moss-transcribe-diarize-cpp](https://huggingface.co/spaces/Luigi/moss-transcribe-diarize-cpp),
with the backend swapped from the ggml/C++ engine to the
[LiteRT port](https://github.com/vieenrose/LiteRT/tree/moss-td-port/litert/samples/asr/moss_td)
(ai-edge-litert 2.1.6, CPU/XNNPACK, CompiledModel with a buffer-bound shared
KV cache).

Weights: [Luigi/moss-transcribe-diarize-litert](https://huggingface.co/Luigi/moss-transcribe-diarize-litert)
— encoder q8 + tied embedder q8 + decoder int4-b32 (0.73 GB total, base
model, no fine-tuning).

Parity: the f32 build of this exact pipeline is byte-identical to the pinned
PyTorch f32 reference on the three golden clips; the deployed int4 decoder
scores 98.99% text fidelity vs that reference on the zh 90 s window.

Differences vs the C++ Space, stated honestly: no cross-window CAM++ speaker
linking ([Sxx] tags are window-local), no batched multi-window decode, no
audio-KV eviction — those are rs.cpp-engine features.
