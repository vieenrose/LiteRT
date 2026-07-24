# Integration note — Nemotron-3.5-ASR 3.5 (q4-mix, LiteRT) as a VoxSum ASR backend

*Status: integration guide · 2026-07-24 · first draft for the q4-mix port*

> Companion to the port at `litert/samples/asr/nemotron/` in
> [`vieenrose/LiteRT`](https://github.com/vieenrose/LiteRT) (branch
> `nemotron-port`). This describes wiring that port into
> [VoxSumDroid](https://github.com/vieenrose/VoxSumDroid) as a **third LiteRT ASR
> backend**, alongside X-ASR (zipformer) and MOSS-TD.

## What the model does — and why add it

Base [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
(0.6B FastConformer-RNNT, OpenMDW-1.1): **multilingual** streaming ASR, 25
languages via a 128-slot language prompt. It is the only current VoxSum ASR
option that natively spans en/zh/**ja**/ko/es/fr/de/… in one model.

- **Plain transcription only** — no timestamps-with-speakers, no diarization
  (unlike MOSS-TD). It emits punctuated, cased text. VoxSum's pyannote-seg +
  CAM++ stages are still needed if you want speaker labels for this backend.
- **zh-TW**: the base model's `zh-TW` slot is **untrained** (100% CER). Produce
  Traditional output the way VoxSum already does it — decode with the `zh-CN`
  slot, then OpenCC `s2t` (the `app/src/main/assets/opencc/` tables are already
  shipped). A future zh-TW fine-tune will fix the native slot.
- **Accuracy / size (this q4-mix build)**: ASCEND zh-CN **CER 16.32%** vs the
  fp32 reference 15.61% (n=100) — INT4 is near-lossless here. Bundle **693 MB**
  (encoder 596 INT4 + decoder 61 + joint 36, fp32). Bigger than X-ASR (295 MB)
  but multilingual.

## Model files — manifest.json entries

Three flatbuffers + the HF tokenizer/processor. Host them in an HF repo (e.g.
`Luigi/nemotron-asr-litert`) pinned by revision, mirroring the existing
`models/manifest.json` `asr[]` shape (id / url / sha256 / license / default):

| id | file | size | precision |
|---|---|---|---|
| `nemotron-encoder-q4` | `nemotron_encoder_q4.tflite` | 596 MB | INT4 FC (blockwise-128) + fp32 convs/norms |
| `nemotron-decoder-fp32` | `nemotron_decoder_fp32.tflite` | 61 MB | fp32 |
| `nemotron-joint-fp32` | `nemotron_joint_fp32.tflite` | 36 MB | fp32 |
| `nemotron-tokenizer` | `tokenizer.json` (HF ParakeetTokenizer) | 0.8 MB | — |

Regenerate any file from the port: `python -m nemotron.export --component <c>
--checkpoint qat_q4mix_enc.pt --prebake`. All Apache-2.0-compatible export of an
OpenMDW-1.1 base; the QAT checkpoint is the only trained artifact.

## Runtime — the graph flow

```
 pcm 16k mono
   │  128-bin log-mel  (n_fft=512, hop=160, win=400, preemph=0.97, RAW log-mel — NO per-feature norm)
   ▼
 encoder_q4(mel[1,T,128])              → hidden[1,T',1024]          (T' = ceil(T/8))
   │  prompt fusion (fp32, host/4th-graph):
   │    fused = prompt_projector( concat(hidden, one_hot(slot)[128]) )   # [1,T',1024], NO residual
   ▼
 RNN-T greedy over T' frames:
   init  token=BLANK(13087), h=c=0[2,1,640];  dec_out,h,c = decoder(token,h,c)
   for t in T':
     loop (≤10 symbols/frame):
       logits = joint(enc=fused[:,t], dec=dec_out)        # joint folds encoder_projector 1024→640
       k = argmax(logits[13088]);  if k==BLANK: break
       emit k; dec_out,h,c = decoder(token=k, h, c)
   → detok(ParakeetTokenizer, strip <lang-tag> tokens, ▁→space)
   → [if zh-TW] OpenCC s2t
```

**Graph I/O contract** (all batch-1):

| graph | inputs | outputs |
|---|---|---|
| encoder | `input_features` (1,T,128) f32 | `hidden` (1,T',1024) f32 |
| decoder | `token` (1,1) **i32**, `h` (2,1,640) f32, `c` (2,1,640) f32 | `dec_out` (1,1,640), `h`, `c` |
| joint | `enc` (1,1,1024) f32, `dec` (1,1,640) f32 | `logits` (1,1,13088) f32 |

- **Language slots** (from `processor_config.json` `prompt_dictionary`):
  `en-US`=0, `zh-CN`=4, **`zh-TW`=5 (dead)**, `ja-JP`=10, `ko`=14, `es-ES`=2,
  `fr`=8, `de`=9, … Pass the slot as a one-hot[128] into the fusion step.
- **Timestamps**: encoder subsamples 8×, hop 160 @16k ⇒ **0.08 s per output
  frame**. `ts(frame) = frame_index × 0.08 s`.
- **Prompt fusion** is currently host-side fp32 (INT4 collapses it). Two options
  on-device: (a) recommended — export it as a 4th tiny fp32 graph
  `prompt_fuse(hidden[1,T',1024], one_hot[1,128]) → fused[1,T',1024]`; or (b)
  do the two matmuls in Kotlin (`linear_1` 1152→I, ReLU, `linear_2` I→1024).

## Delegate — this is the important bit

The INT4 encoder uses **dynamic-range INT4 `FULLY_CONNECTED`**. The classic
`Interpreter` reference CPU kernel **cannot allocate it** (fails in
`fully_connected.cc`). Run it through the **LiteRT-Next `CompiledModel`** path —
the same `com.google.ai.edge.litert:litert:2.1.6` AAR already vendored for the
MOSS-LiteRT backend (`app/src/main/jniLibs/**/libLiteRt.so`, see
`mosslite/PROVENANCE.md`) — with an Android NNAPI / XNNPACK-QD8 delegate.
decoder/joint are fp32 and run on either path.

**Set the CPU thread count explicitly** — `CompiledModel` otherwise picks a very
low default (measured ~8× slower). Use the big-core count on big.LITTLE.

## Engine: what to build

Reuse the existing `XasrLiteEngine` / `xasr_lite_jni.cpp` pattern — both are RNN-T
transducers driven the same way. Fork it to `NemotronLiteEngine` and change:

| aspect | X-ASR (zipformer) | Nemotron |
|---|---|---|
| graphs | enc(multi-sig 375/750/1500/3000) + decoder + joiner | encoder(offline fixed-T) + decoder + joint + prompt-fuse |
| blank id | 0 | **13087** |
| vocab | 5000 | **13088** |
| tokenizer | `tokens.txt` | HF ParakeetTokenizer (SentencePiece BPE, `▁`, `<lang>` tags) |
| front end | 80-bin povey fbank | **128-bin log-mel, raw (no norm)**, preemph 0.97 |
| decoder ctx | `y[1,2]` i32, `-1` pad | LSTM state `h,c[2,1,640]`, token`[1,1]` |
| language | fixed zh-en | 128-slot prompt (one-hot + `prompt_projector`) |
| max sym/frame | 1 | ≤10 |
| ts step | 0.04 s | 0.08 s |

Everything else — `AsrBackend` registration, the `CompiledModel`/`TensorBuffer`
shared-buffer plumbing, model download + sha256 pinning, OpenCC — is unchanged
from the MOSS-LiteRT / X-ASR backends.

## Caveats & open items

- **Offline, fixed length** (`T`≈1101 mel frames ≈ 11 s). For longer audio, chunk
  on VoxSum's Silero-VAD boundaries and concatenate, or re-export the encoder
  with multiple length signatures (as X-ASR does). Padding is masked-safe:
  pad to `T`, trim output to `ceil(T_valid/8)` (padding-vs-exact cos 0.9996).
- **Prompt fusion 4th graph** not yet exported (host-side today) — see above.
- **decoder/joint are fp32** — convert to fp16 to shave ~50 MB if size matters.
- **No diarization.** Speaker labels still need pyannote-seg + CAM++.
- Model files **not yet uploaded** to a public HF repo / no sha256 pins yet.
- On-device latency/RTF **not yet measured** (desktop XNNPACK INT4 encoder is
  ~0.3 s per short clip; expect the phone to be the real gate — benchmark before
  making it a default backend).
