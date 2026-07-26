# Integration note — Nemotron-3.5-ASR 3.5 (q4-mix, LiteRT) as a VoxSum ASR backend

*Status: integration guide · updated 2026-07-26 for the **zh-TW fine-tuned v2** build*

> **v2 (2026-07-26)** replaces the v1.1 warm-start weights with a **zh-TW
> fine-tuned** model: on-device zh-TW CER **13.90 → from ~38**, ~2.7× better, same
> 663 MB. Repo and pins changed — see "Model files" below.

> Companion to the port at `litert/samples/asr/nemotron/` in
> [`vieenrose/LiteRT`](https://github.com/vieenrose/LiteRT) (branch
> `nemotron`). This describes wiring that port into
> [VoxSumDroid](https://github.com/vieenrose/VoxSumDroid) as a **third LiteRT ASR
> backend**, alongside X-ASR (zipformer) and MOSS-TD.

## What the model does — and why add it

Base [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
(0.6B FastConformer-RNNT, OpenMDW-1.1): **multilingual** streaming ASR, 25
languages via a 128-slot language prompt. It is the only current VoxSum ASR
option that natively spans en/zh/**ja**/ko/es/fr/de/… in one model.

- **zh-TW is fine-tuned in v2.** The base model's `zh-TW` slot was untrained
  (~100% CER). v1.1 revived it with a warm-start (prompt-column copy); **v2 goes
  further with a real fine-tune** on Common Voice zh-TW + IVOD 立法院 long-form,
  taking Common Voice zh-TW CER from **38.43 → 12.03 (fp32)** and **13.90 on this
  q4-mix build**. Source model:
  [`Luigi/nemotron-3.5-asr-streaming-0.6b-zhtw`](https://huggingface.co/Luigi/nemotron-3.5-asr-streaming-0.6b-zhtw).
- **Output is Simplified Chinese — apply OpenCC `s2t` for Traditional.** This is
  not a preference, it's a hard constraint: the 13,087-token tokenizer has no
  tokens for many common Traditional characters (點 兒 區 說 麼 嗎 **灣** 黨 體 產;
  6.3% of Traditional chars fail a round-trip vs 0.1% Simplified). VoxSum already
  ships the tables in `app/src/main/assets/opencc/`. `runner.py --s2t` does it;
  `--itn` adds inverse text normalization (百分之五十 → 50%).
- **Which prompt slot for Taiwan Mandarin? Use `zh-CN` (slot 4), not `zh-TW`.**
  Measured on the shipped v2 q4-mix build, two disjoint Common Voice zh-TW slices:
  | slot | clips 0-120 | clips 200-440 |
  |---|---|---|
  | `zh-TW` (5) | 13.90 | 16.33 |
  | `auto` (101) | 13.79 | 16.02 |
  | **`zh-CN` (4)** | **13.20** | **15.97** |
  The ranking replicates on both slices. `zh-CN` wins because it has far more
  pretraining behind it than the fine-tuned zh-TW slot; `auto` is ~equivalent.
  All three land within ~0.7 CER, so this is a small optimisation — but free.
  Output is Simplified either way, so the OpenCC `s2t` step is unchanged.
- **The fine-tune did not cost other languages.** vs base (FLEURS/LibriSpeech,
  fp32): ko −0.80, de −0.75, ja −0.41, hi −0.17, en −0.13 (all *improved*),
  ar +0.21 (flat), fr +1.55 and es +1.27 (the only regressions; fr still beats
  NVIDIA's published 15.93). Full table in the source model card.
- **Plain transcription only** — no timestamps-with-speakers, no diarization
  (unlike MOSS-TD). VoxSum's pyannote-seg + CAM++ stages are still needed for
  speaker labels on this backend.
- **Size**: 663 MB (encoder 596 INT4 + prompt-fuse 18 + decoder 31 + joint 18).
  Bigger than X-ASR (295 MB) but multilingual.
- **Not a zh-en replacement for X-ASR.** Measured head-to-head on identical
  clips, X-ASR wins both of its languages (zh-TW 6.66 CER / en 2.18 WER vs this
  model's 12.03 / 2.58 at fp32). Nemotron's value here is **breadth** — 25
  languages including ja/ko/es/fr/de that X-ASR cannot do at all. Keep both.

## Model files — manifest.json entries

Published, pinned by revision, at
[`Luigi/nemotron-asr-litert-zhtw`](https://huggingface.co/Luigi/nemotron-asr-litert-zhtw)
(commit `bbc906fe`, **v2 fine-tuned**). Four flatbuffers + the HF
tokenizer/processor, mirroring the existing `models/manifest.json` `asr[]` shape
(id / url / sha256 / license).

> The base-faithful (non-zh-TW) build stays at
> [`Luigi/nemotron-asr-litert`](https://huggingface.co/Luigi/nemotron-asr-litert)
> if you ever want the unmodified model.

| id | file | size | precision |
|---|---|---|---|
| `nemotron-encoder-q4` | `nemotron_encoder_q4.tflite` | 596 MB | INT4 FC (blockwise-128) + fp32 convs/norms |
| `nemotron-prompt-fuse` | `nemotron_prompt_fuse_fp32.tflite` | 18 MB | fp32 |
| `nemotron-decoder-fp16` | `nemotron_decoder_fp16.tflite` | 31 MB | fp16 |
| `nemotron-joint-fp16` | `nemotron_joint_fp16.tflite` | 18 MB | fp16 |
| `nemotron-tokenizer` | `tokenizer.json` (HF ParakeetTokenizer) | 0.8 MB | — |

Ready-to-paste `asr[]` entries (URLs pinned to commit `bbc906fe`, sha256 verified):

```json
{ "id": "nemotron-encoder-q4", "kind": "ASR",
  "url": "https://huggingface.co/Luigi/nemotron-asr-litert-zhtw/resolve/bbc906fe254b8c1b84d53fc64b9204efd3d08b57/nemotron_encoder_q4.tflite",
  "sha256": "b1b3c93add91ee2253c8d6d24172614a83f6572720dea0150fb34285be53a0c2",
  "license": "nvidia-open-model-license" },
{ "id": "nemotron-prompt-fuse", "kind": "ASR",
  "url": "https://huggingface.co/Luigi/nemotron-asr-litert-zhtw/resolve/bbc906fe254b8c1b84d53fc64b9204efd3d08b57/nemotron_prompt_fuse_fp32.tflite",
  "sha256": "21c59326f8633c3824f9e92dcaded6148978dcd53591846c85c9b1ac982a1bba",
  "license": "nvidia-open-model-license" },
{ "id": "nemotron-decoder-fp16", "kind": "ASR",
  "url": "https://huggingface.co/Luigi/nemotron-asr-litert-zhtw/resolve/bbc906fe254b8c1b84d53fc64b9204efd3d08b57/nemotron_decoder_fp16.tflite",
  "sha256": "e92dfa900ebd9d7cd87429c9bb7c304b7e3fa61dc233c74f2e074fbb4342222b",
  "license": "nvidia-open-model-license" },
{ "id": "nemotron-joint-fp16", "kind": "ASR",
  "url": "https://huggingface.co/Luigi/nemotron-asr-litert-zhtw/resolve/bbc906fe254b8c1b84d53fc64b9204efd3d08b57/nemotron_joint_fp16.tflite",
  "sha256": "d728fb09aa034b85b1549772fef6cfc4f85d7df0faf59c6db4ad2e7fbbfdc848",
  "license": "nvidia-open-model-license" },
{ "id": "nemotron-tokenizer", "kind": "ASR",
  "url": "https://huggingface.co/Luigi/nemotron-asr-litert-zhtw/resolve/bbc906fe254b8c1b84d53fc64b9204efd3d08b57/tokenizer.json",
  "sha256": "3f3d481deb073b64c2082e8c7860d487a3a62774bf4e9e4faac83007e181f246",
  "license": "nvidia-open-model-license" },
{ "id": "nemotron-processor-config", "kind": "ASR",
  "url": "https://huggingface.co/Luigi/nemotron-asr-litert-zhtw/resolve/bbc906fe254b8c1b84d53fc64b9204efd3d08b57/processor_config.json",
  "sha256": "ec47870f1091ea4f25539208387b45b902c92d0e3f997a30061ef88f73437ab0",
  "license": "nvidia-open-model-license" },
{ "id": "nemotron-config", "kind": "ASR",
  "url": "https://huggingface.co/Luigi/nemotron-asr-litert-zhtw/resolve/bbc906fe254b8c1b84d53fc64b9204efd3d08b57/config.json",
  "sha256": "3fcc4f88c746b9f4b3f0b174d7b5db1bb6eb3997c209f6199186f560d21b85ca",
  "license": "nvidia-open-model-license" }
```

Regenerate any file from the port (the exporter takes any HF model dir/id):

```bash
python -m nemotron.export --component all --model Luigi/nemotron-3.5-asr-streaming-0.6b-zhtw --prebake --out models/
```

`--prebake` bakes the INT4 grid into the weights before quantization (worth
~0.8 CER). The fine-tuned source model is the only trained artifact; the export
itself is numerics-preserving.

## Runtime — the graph flow

```
 pcm 16k mono
   │  128-bin log-mel  (n_fft=512, hop=160, win=400, preemph=0.97, RAW log-mel — NO per-feature norm)
   ▼
 encoder_q4(mel[1,T,128])              → hidden[1,T',1024]          (T' = ceil(T/8))
   │  prompt fusion (nemotron_prompt_fuse_fp32.tflite):
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
   → [if zh-TW] OpenCC s2t  → [optional] ITN (wetext, spoken→written: 百分之五十→50%)
```

**Graph I/O contract** (all batch-1):

| graph | inputs | outputs |
|---|---|---|
| encoder | `input_features` (1,T,128) f32 | `hidden` (1,T',1024) f32 |
| decoder | `token` (1,1) **i32**, `h` (2,1,640) f32, `c` (2,1,640) f32 | `dec_out` (1,1,640), `h`, `c` |
| joint | `enc` (1,1,1024) f32, `dec` (1,1,640) f32 | `logits` (1,1,13088) f32 |

- **Language slots** (from `processor_config.json` `prompt_dictionary`):
  `en-US`=0, `zh-CN`=4, **`zh-TW`=5 (fine-tuned in v2)**, `ja-JP`=10, `ko`=14, `es-ES`=2,
  `fr`=8, `de`=9, … Pass the slot as a one-hot[128] into the fusion step.
- **Timestamps**: encoder subsamples 8×, hop 160 @16k ⇒ **0.08 s per output
  frame**. `ts(frame) = frame_index × 0.08 s`.
- **Prompt fusion** is a dedicated fp32 graph `nemotron_prompt_fuse_fp32.tflite`
  (`hidden[1,T',1024]` + `one_hot[1,128]` → `fused[1,T',1024]`; INT4 collapses
  it). Called once per utterance, numerically identical to the in-model fusion
  (numerically identical to the in-model fusion). Fixed `T'` = encoder output length (139
  for T=1101) — feed the full untrimmed encoder output, fuse, then trim to
  `ceil(T_valid/8)`.

## Delegate — this is the important bit

The INT4 encoder uses **dynamic-range INT4 `FULLY_CONNECTED`**. The classic
`Interpreter` reference CPU kernel **cannot allocate it** (fails in
`fully_connected.cc`). Run it through the **LiteRT-Next `CompiledModel`** path —
the same `com.google.ai.edge.litert:litert:2.1.6` AAR already vendored for the
MOSS-LiteRT backend (`app/src/main/jniLibs/**/libLiteRt.so`, see
`mosslite/PROVENANCE.md`) — with an Android NNAPI / XNNPACK-QD8 delegate.
prompt-fuse / decoder / joint are fp16 and run on either path.

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
- **ITN** (inverse text normalization, spoken→written numerals) is deterministic
  post-processing (`wetext` / WeTextProcessing WFST), not in the weights — a
  fine-tune to bake it in was tried and does not beat post-proc (see below).
- **No diarization.** Speaker labels still need pyannote-seg + CAM++.
- **zh-TW is Simplified out + OpenCC `s2t`** — native Traditional is impossible
  without extending the tokenizer (see above). Not a bug; don't "fix" it.
- On-device latency/RTF **not yet measured** (desktop XNNPACK INT4 encoder is
  ~0.3 s per short clip; expect the phone to be the real gate — benchmark before
  making it a default backend).
