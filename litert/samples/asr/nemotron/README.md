# Nemotron-3.5-ASR-Streaming on LiteRT (q4-mix)

LiteRT (TFLite) port of [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
(0.6B FastConformer-RNNT, OpenMDW-1.1), multilingual streaming ASR. This sample
builds a **q4-mix** on-device build — INT4 encoder + fp32 decoder/joint — with an
optional QAT step that makes the INT4 near-lossless.

Architecture: cache-aware FastConformer encoder (24L, d=1024, 8x subsampling,
217 FC + 77 conv) + a 128-slot language prompt (`prompt_projector`) + RNN-T
prediction net (2-layer LSTM, d=640) + joint (→ vocab 13088). ASR only — no
diarization.

## Model files

Four flatbuffers, mirroring the reference split (embedding lookup + lm_head stay
inside the decoder/joint graphs; every file < 2 GB):

| file | I/O | precision |
| --- | --- | --- |
| `nemotron_encoder_q4.tflite` | `input_features` (1,T,128) f32 → `hidden` (1,T',1024) f32 | **INT4** FC (blockwise-128), convs/norms fp32 |
| `nemotron_prompt_fuse_fp32.tflite` | `hidden` (1,T',1024) + `one_hot` (1,128) → `fused` (1,T',1024) | fp32 |
| `nemotron_decoder_fp32.tflite` | `token` (1,1) i32 + `h`,`c` (2,1,640) → `dec_out` (1,1,640) + `h`,`c` | fp32 |
| `nemotron_joint_fp32.tflite` | `enc` (1,1,1024) + `dec` (1,1,640) → `logits` (1,1,13088) | fp32 (folds `encoder_projector` 1024→640) |

The `prompt_projector` language fusion (between encoder and greedy) is its own
tiny fp32 graph — INT4 there collapses the model. It runs once per utterance;
the exported `nemotron_prompt_fuse_fp32.tflite` is numerically identical to the
in-model fusion (end-to-end CER unchanged).

## Why q4-mix (not uniform INT4)

Quantizing the RNN-T decoder / joint / embedding / prompt to INT4 is catastrophic
(measured **435% WER** — the discrete token decisions flip). Only the encoder's
`FULLY_CONNECTED` weights (83% of params) tolerate INT4. Converted with
`litert_torch` 0.9.1 (the renamed `ai-edge-torch`): the transformers encoder
**traces directly** — no torch rebuild needed — and `ai_edge_quantizer`'s
`dynamic_wi4_afp32` (min_max, blockwise-128) matches the QAT grid.

`min_max` beats `OCTAV` here: OCTAV's optimal-clipping discards the weight
outliers the QAT trained the model to keep.

## Accuracy (ASCEND zh, CER, n=100)

| build | zh-CN CER |
| --- | --- |
| FP32 `model.generate` (reference) | 15.61% |
| **q4-mix LiteRT** (INT4 enc + QAT + pre-bake) | **16.32%** (+0.7, near-lossless) |
| q4-mix, no QAT pre-bake | 17.10% |

Naive INT4 PTQ (no QAT) costs +0.85 CER (zh) / +3.65 WER (en) vs FP; the QAT
(label-based RNN-T self-distillation on the FP model's own greedy output)
recovers it. zh-TW: the base model's zh-TW slot is untrained (100% CER) — use the
zh-CN slot + OpenCC `s2t` (`runner.py --s2t`).

## Export

Requires `transformers>=5.13`, `litert_torch==0.9.1`, `ai_edge_quantizer==0.7.0`,
`ai-edge-litert==2.1.5` (the quantizer pins 2.1.5; the runtime tolerates 2.1.6).

```bash
# base model (no QAT) — INT4 PTQ:
python -m nemotron.export --component all --out models/
# with a QAT checkpoint (near-lossless):
python -m nemotron.export --component encoder --checkpoint qat_q4mix_enc.pt --prebake --out models/
python -m nemotron.export --component decoder --out models/
python -m nemotron.export --component joint   --out models/
```

## Run

```bash
python -m nemotron.runner --models models/ --wav clip.wav --lang zh-CN --s2t
```

The INT4 encoder needs a delegate that supports dynamic-range INT4
`FULLY_CONNECTED` — the desktop reference CPU kernel cannot allocate it (fails at
`fully_connected.cc`); it runs under the LiteRT-Next `CompiledModel` runtime and
on Android (NNAPI / XNNPACK-QD8). Encoder is offline / fixed-length (pad to `T`,
trim output to `ceil(T_valid/8)`); for streaming, export multiple length
signatures (cf. the Xasr sample).

## Provenance

Base weights: nvidia/nemotron-3.5-asr-streaming-0.6b (OpenMDW-1.1). The QAT
checkpoint is the only trained artifact: label-based RNN-T QAT (INT4 STE on
encoder FC weights, FP self-distillation pseudo-labels) — no language change,
"as original as possible". Export = numerics-preserving conversion + INT4 quant.
