# Deploying this Space

This directory is the complete HF Space for the LiteRT port of
MOSS-Transcribe-Diarize (CPU/XNNPACK, int4-b32 decoder). It was built and
tested locally end-to-end (health + SSE streaming on jfk + the zh 5-min
golden clip) on 2026-07-23.

## Blocker at time of writing

The `Luigi` HF account cannot create the Space yet:

* ZeroGPU: free accounts are limited to 2 ZeroGPU Spaces (both slots in use:
  `moss-transcribe-diarize-cpp` + `edge-fall-vlm-demo`).
* cpu-basic: since 2026, creating NEW Gradio/Docker Spaces on free cpu-basic
  requires a PRO subscription (HTTP 402 on both duplicate_space and
  create_repo).

Either subscribe to PRO, or free one ZeroGPU slot, then:

```bash
hf repo create Luigi/moss-transcribe-diarize-litert-demo --repo-type space --space-sdk gradio
hf upload Luigi/moss-transcribe-diarize-litert-demo . . --repo-type space
# then request zero-a10g hardware in Space settings (app.py has the
# @spaces.GPU shim the ZeroGPU runtime requires)
```

Run the `examples/` populate step first (they are NOT committed to git —
60 MB of wavs). They are byte-identical to the C++ Space's examples:

```bash
hf download Luigi/moss-transcribe-diarize-cpp --repo-type space \
  --include "examples/*" --local-dir .
```

## Weights

Pulled at startup from `Luigi/moss-transcribe-diarize-litert` (public):
encoder q8 + embedder q8 + decoder `moss_td_decoder_q4b32_ekv2560.tflite`
+ `tokenizer/`. Total download ~0.73 GB.

## What differs from the C++ Space (honest deltas, also stated in the UI)

* NO cross-window CAM++ speaker linking — [Sxx] tags are window-local.
* NO batched multi-window decode (batch is forced to 1).
* NO audio-KV eviction.
* Parity: f32 pipeline byte-identical to the pinned PyTorch f32 reference on
  the 3 golden clips; deployed int4-b32 decoder 98.99% text fidelity
  (95.45% incl. timestamps) on the zh 90 s window; 100% on jfk.
