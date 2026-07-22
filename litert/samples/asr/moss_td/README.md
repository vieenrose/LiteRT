# MOSS-Transcribe-Diarize on LiteRT

LiteRT (TFLite) port of [OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
(0.9B, Apache-2.0): joint zh/en transcription + speaker diarization + utterance
timestamps, output format `[ss.ss][Sxx]text[ss.ss]…`.

Architecture: Whisper-medium encoder (24L, d=1024) -> 4x time merge
(12.5 audio tokens/s) -> VQAdaptor MLP (4096->1024) -> Qwen3-0.6B decoder
(28L, GQA 16/8, head_dim 128, RoPE 1e6, tied lm_head, vocab 151936). Audio
embeddings are masked_scatter'd into the token sequence; the prompt interleaves
time-marker digit tokens (every 2 s) within the audio span.

## Model files

The network is split into three flatbuffers (keeps every file < 2 GB at f32;
mirrors the reference C++ implementation, which also keeps embedding lookup +
lm_head outside the decoder graph):

| file | signatures | I/O |
| --- | --- | --- |
| `moss_td_encoder_*.tflite` | (default) | mel (1,80,3000) f32 -> audio embeds (1,375,1024) |
| `moss_td_embedder_*.tflite` | `embed_1`, `embed_128`, `logits` | token ids -> embeds; hidden (1,1,1024) -> logits (1,1,151936) |
| `moss_td_decoder_*_ekv{N}.tflite` | `prefill_128`, `prefill_1024`, `decode` | input_embeds + input_pos + mask (1,1,S,N) + external KV cache (BTNH, 28x2 tensors (1,N,8,128)) -> hidden + updated KV |

Variants: `f32` (parity reference, ekv6144), `fp16`, `q8` (dynamic-range int8,
ekv2048 for on-device use). Converted with `litert-torch` 0.9.1 (the renamed
`ai-edge-torch` generative API): the decoder is re-authored on
`litert_torch.generative` layers (the stock Qwen3-0.6B example config) with an
embeddings-in / hidden-out forward, mask-as-input, and externalized KV cache —
the same conventions used by litert-community LLM releases, so the decoder also
composes with the standard LiteRT LLM tooling.

Prebuilt models: https://huggingface.co/Luigi/moss-transcribe-diarize-litert

## Usage

```bash
# one-off: export from the HF checkpoint
python -m moss_td.export --component encoder  --quantize none --out models
python -m moss_td.export --component embedder --quantize none --out models
python -m moss_td.export --component decoder  --quantize none \
    --kv-cache-max-len 6144 --prefill-lens 128,1024 --out models

# transcribe (host, XNNPACK)
python -m moss_td.runner --wav audio_16k_mono.wav \
    --encoder models/moss_td_encoder_f32.tflite \
    --embedder models/moss_td_embedder_f32.tflite \
    --decoder models/moss_td_decoder_f32_ekv6144.tflite
```

The host runner reproduces the reference pipeline exactly: 30 s-chunked
Whisper mels -> encoder -> per-chunk length slice -> chat-template prompt with
time-marker-interleaved audio span -> embed + masked scatter -> chunked prefill
-> greedy decode until `<|im_end|>`.

Parity gates: `moss_td/verify_torch.py` (re-authored torch vs official
PyTorch), `moss_td/verify_tflite.py` (tflite vs torch), `moss_td/parity.py`
(full-transcript agreement, `difflib.SequenceMatcher(autojunk=False)`).

## Parity status (f32, greedy)

Stage gates:

* encoder (torch re-author vs official): max abs diff 2.4e-07
* encoder.tflite vs torch: max abs diff 6.7e-05
* embedder.tflite embed: exact; logits: max abs diff 3.2e-06
* decoder teacher-forced logits vs official: max abs diff 1.2e-04, argmax
  agreement 100% over a real 230-token prompt; greedy continuations identical

End-to-end transcripts (LiteRT f32 pipeline vs official PyTorch f32 greedy and
vs the reference C++ (rapidspeech/moss-transcribe.cpp) f32 GGUF output):

<!--PARITY_TABLE-->

## Samsung SM-A5360 (Exynos 1280) benchmarks

<!--BENCH_TABLE-->

## Provenance

* Weights: OpenMOSS-Team/MOSS-Transcribe-Diarize (Apache-2.0), converted from
  the bf16 safetensors checkpoint to f32.
* Reference implementations used for parity: the official PyTorch package and
  the vendored moss-transcribe.cpp (rapidspeech) f32 GGUF build.
