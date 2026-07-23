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

| clip | vs official PyTorch f32 | vs C++ f32 (full) | vs C++ f32 (text w/o timestamps) |
| --- | --- | --- | --- |
| jfk 11 s | **100.000% (byte-identical)** | 96.27% | 100.00% |
| golden_en_5min 300 s | **100.000% (byte-identical)** | 91.55% | 98.54% |
| golden_zh_5min 318 s | **100.000% (byte-identical)** | 56.88% | 79.62% |

The C++ reference itself agrees with the official PyTorch model only to the
same degree (en 91.52%/98.54%, zh 57.17%/79.62%): its hand-rolled f32 mel/FFT
flips near-tie timestamp-digit tokens, and on the 318 s zh clip that flip
cascades into a coarser segmentation. Both implementations are individually
deterministic (the C++ rerun reproduces its stored goldens byte-for-byte).
This port tracks the original PyTorch model exactly (the stronger parity), so
the residual gap to the C++ goldens is a documented C++-side deviation, not a
port defect.

Quantized: q8 text-identical to f32 on jfk/zh90s (one timestamp digit differs
on jfk); fp16 byte-identical to f32 on jfk (host XNNPACK).

Decoder context: the zh 5-min clip needs prompt 4486 + 1909 generated tokens;
use the ekv8192 decoder for 5-min single-pass decodes (ekv6144 truncates the
tail; every token generated before the cap still matched PyTorch).

## Samsung SM-A5360 (Exynos 1280) benchmarks

All CPU, 8 threads, XNNPACK. Per-signature latencies via the LiteRT
`benchmark_model` tool (q8 variant):

| signature | avg latency | peak RSS |
| --- | --- | --- |
| encoder (one 30 s mel chunk) | 7.60 s | 0.80 GB |
| decoder `decode` (1 token, ekv2048) | 859 ms | 3.15 GB |
| decoder `prefill_128` | 2.52 s | 3.17 GB |
| decoder `prefill_1024` | 13.13 s | 3.33 GB |
| embedder `logits` | 10.4 ms | 0.31 GB |
| embedder `embed_1` | 5 us | 0.32 GB |

Composed pipeline (q8, measured signature latencies x per-clip counts from the
host runner) vs rs.cpp (moss-transcribe.cpp) q4mix measured end-to-end on the
same device:

| clip | LiteRT q8 (composed) | rs.cpp q4mix (measured) |
| --- | --- | --- |
| jfk 11 s | ~74 s, RTF 6.8 (1.15 tok/s decode) | 142.9 s, RTF 13.0, RSS 1.03 GB |
| zh 90 s | ~367 s, RTF 4.1 | 762.6 s, RTF 8.5, RSS 1.34 GB |

LiteRT q8 decode is ~1.15 tok/s vs ~0.5 tok/s for rs.cpp q4mix on this device
— about 1.9-2.1x faster end-to-end, at the cost of ~3.2 GB peak RSS vs
~1.3 GB (externalized f32 KV cache + benchmark-tool double-buffering; an
integrated runner with shared KV I/O buffers would sit lower).

The fp16 decoder variant OOM'd this 8 GB device hard enough to reboot it
(XNNPACK upconverts fp16 weights to f32 at init): use q8 on-device; fp16 is a
host/GPU-delegate variant.

### Host x86 (32-core workstation, CPU-only, 16 threads)

Same clips, single window, greedy. rs.cpp = moss-transcribe.cpp CPU backend;
LiteRT = Python host runner (XNNPACK; per-step KV round-trips through numpy —
a native integrated runner would cut most of the LiteRT decode overhead).

| config | jfk 11 s wall | zh90s wall | peak RSS | decode rate (zh90s) |
| --- | --- | --- | --- | --- |
| rs.cpp q4mix | 3.6 s | 18.4 s | 1.5 GB | 26.5 tok/s |
| rs.cpp f32 | 7.3 s | 35.1 s | 4.3 GB | 12.6 tok/s |
| LiteRT q8 (ekv2048) | 17.2 s | 72.0 s | 7.9 GB | ~5.6 tok/s |
| LiteRT fp16 (ekv2048) | 36.0 s | 125.2 s | 15.8 GB | ~4.6 tok/s |
| LiteRT f32 (ekv6144) | 43.5 s | 195.6 s | 24.4 GB | ~1.9 tok/s |
| LiteRT int4-b32 dec + fp16 enc (ekv2048) | 31.2 s | 103.5 s | 8.5 GB | ~5.9 tok/s |

rs.cpp per-stage on zh90s (profile build): q4mix encoder 4.85 s + generate
12.5 s; f32 encoder 4.51 s + generate 29.4 s. LiteRT q8 split on zh90s:
encoder 1.6 s, prefill 3.0 s, decode 64.7 s (the decode loop is dominated by
externalized-KV copies through the Python signature API).

int4 (ai-edge-quantizer blockwise-32 weight-only via the litert-torch
dynamic_int4_block32 recipe, same family litert-community uses for
Gemma/Qwen int4): decoder file 251 MB (vs 456 MB q8; rs.cpp q4mix full-model
GGUF is 759 MB incl. encoder+embeddings). int4 decode speed matches q8 in the
Python harness (KV-copy bound); transcript agreement vs rs.cpp f32:
jfk 96.3% full / 100.0% text-only, zh90s 90.4% / 97.7% text-only. Caveat: the
fp16 encoder is ~25x slower than the q8 encoder on x86 XNNPACK — pair the
int4 decoder with the q8 encoder for speed.

## Provenance

* Weights: OpenMOSS-Team/MOSS-Transcribe-Diarize (Apache-2.0), converted from
  the bf16 safetensors checkpoint to f32.
* Reference implementations used for parity: the official PyTorch package and
  the vendored moss-transcribe.cpp (rapidspeech) f32 GGUF build.
