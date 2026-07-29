# TurboQuant TQ3 KV cache for Gemma 4 E2B on LiteRT

A 3-bit KV-cache port of **TurboQuant** (Zandieh, Dickens, Kacham, Karbasi —
"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate",
ICLR 2026) to LiteRT, running Gemma 4 E2B (16k context) summarization
end-to-end on CPU with the KV cache stored **only** as packed TQ3 — no
full-length fp32 KV tensor anywhere in the process.

The port runs on the **stock prebuilt `libLiteRt.so`** — no runtime fork or
rebuild. The fused attention kernel is injected via
`LiteRtAddCustomOpKernelOption`, the same mechanism as the ternary-GEMM custom
op on the `vibevoice-ternary` branch of this fork.

## What it does

- **TQ3 quantization** (`cpp/tq3.c`): per-vector L2 norm + fixed random
  rotation (seed-42 QR, torch-bit-identical) + per-coordinate 3-bit Lloyd-Max
  scalar quantization over the Beta coordinate pdf. Block = 4-byte fp32 norm +
  ceil(3d/8) codes: d=256 → 100 B, d=512 → 196 B. MSE 0.0343 at d=256/b=3,
  matching the paper's 0.034.
- **Graph rewrite** (`python/rewrite_tq3.py`): flatbuffer surgery that replaces
  every externalized-KV attention block (14 in prefill_128, 35 in decode) with
  one fused custom op `voxsum.tq3_attention(q, kv_slice_k, kv_slice_v, mask,
  packed_k, packed_v) → ctx`, and swaps the 30 fp32 `kv_cache_*` signature
  inputs for 30 uint8 `packed_{k,v}_i` inputs. Handles the extended-format
  .tflite (weight buffers at absolute u64 offsets after the flatbuffer) with a
  two-pass offset fixup; the 2.28 GB data section is carried over verbatim.
  Runs in ~3 s. KV-shared blocks (Gemma 4 E2B shares 20 of 35 layers'
  KV) receive a second custom code `voxsum.tq3_attention_t` for the transposed
  new-token slice layout.
- **Fused kernel** (`cpp/tq3_attn.cc`): dequant-on-the-fly attention over the
  packed side-cache. Live rows are derived from the mask (sliding layers ≤512
  rows, global layers `pos` rows — validated bit-exact by the Phase 2b
  window-equivalence check). A generation-keyed memo dequantizes each distinct
  packed pair once per interpreter Run even when 17 decode blocks share one
  layer's cache.
- **Engines**: `cpp/engine2.cc` is the driver — it auto-detects staging vs
  fused mode from the signature inputs (`kv_cache_*` → staging, `packed_*` →
  fused) and reproduces the staging engine bit-for-bit in staging mode.
  `cpp/engine.cc` is the earlier Phase 2b staging-only engine, kept as an A/B
  reference (packed TQ3 side-cache is the source of truth; fp32 staging is its
  dequantized image, verified bit-exact every run).

## Acceptance numbers (9950X3D, XNNPACK 32 threads, attn 8 OMP threads)

Teacher-forced top-1 agreement vs the fp32-KV baseline over 64 steps
(gate ≥ 0.94 — PASS):

| prompt | fused top-1 | staging top-1 |
|---|---|---|
| p0 (en) | **0.9455** | 0.9455 |
| p1 (zh) | **0.9688** | 0.9531 |

Free-running summaries are faithful in en and zh-TW (key figures intact); a
1244-token meeting transcript yields a coherent structured zh summary. Full
outputs in `results/acceptance3.json`.

Memory (16k context):

| component | staging engine | fused engine |
|---|---|---|
| fp32 KV staging | 576.0 MiB | **0** |
| packed TQ3 side-cache (15 pairs × 16384) | 55.9 MiB | 55.9 MiB |
| kernel dequant memo (live rows) | — | 9–37 MiB |
| **KV-attributable total** | 632 MiB (vs ~1152 MiB Python harness) | **65–93 MiB** |
| peak RSS (no weight cache) | 6.13 GB | **5.12–5.18 GB** |

Speed:

| config | prefill tok/s | decode tok/s |
|---|---|---|
| staging engine | 153–159 | 5.6–5.7 |
| **fused** | **402–410** | **14.3** |

The fused op is ~2.5× *faster*, not slower: decode no longer BMMs over all
16384 cache columns per layer (only ≤512 live sliding / `pos` global rows) and
prefill drops the 16512-wide score/mask-broadcast arenas entirely.

## Two traps worth knowing

1. **Double accumulation is mandatory.** fp32 accumulation inside the fused op
   leaves a ~2e-5 wobble that the surrounding dynamic-activation-quant FCs
   amplify to ~1e-2 relative on the next KV slice (scale = max/127
   hypersensitivity) and costs 5 points of top-1. With double accumulation the
   op matches exact fp64 attention to 4.8e-7 and top-1 recovers fully. A
   corollary: end-to-end "near-bit-identical" logits are unachievable across
   *any* attention reimplementation boundary on a dynamic-activation-quant
   int8 graph — top-1 vs the fp32 baseline is the meaningful gate.
2. **Cap the op's OpenMP threads (≤8).** Letting OMP take all 32 threads
   oversubscribes against XNNPACK's own 32-thread pool and collapses decode to
   2 tok/s. 8 threads is the sweet spot (4/8/12/16 ≈ 12.4/13.3/13.5/13.1
   tok/s); `--attn-threads` controls it.

## Build & run

Prerequisites: a Gemma 4 E2B litert_torch export at 16k
(`final/{model_quantized,auxiliary,embedder_quantized}.tflite` — not committed,
2.28 GB), the stock prebuilt `libLiteRt.so`, LiteRT C headers, and the local
Gemma 4 E2B HF snapshot (the per-layer-embedding table is mmap'd from
`model.safetensors`; `python/prep_assets.py` records the byte offset).

```bash
# one-time asset prep (rotations, codebooks, PLE offset, prompts) — needs torch
python python/prep_assets.py            # -> assets/ (small .bin/.json committed here)

# graph rewrite: staging model -> fused model (custom ops + packed signatures)
python python/rewrite_tq3.py <export>/final/model_quantized.tflite model_tq3.tflite

# build (OpenMP required)
mkdir -p build && cd build
cmake .. -DLITERT_INCLUDE=/path/to/litert/c/headers \
         -DLITERT_LIB=/path/to/libLiteRt.so
make    # -> engine2 (staging+fused auto-detect), engine (Phase 2b reference)

# fused run                                  # staging run: omit --model
./build/engine2 --model model_tq3.tflite \
  --final <export>/final --assets assets \
  --prompt-file assets/prompt_p0.json --teacher \
  --threads 32 --attn-threads 8 --out out/x.json

# acceptance suites
python python/accept3.py                # fused + staging A/B, free runs
python python/run_accept.py 32          # Phase 2b staging-only suite
```

## Layout

- `cpp/tq3.{c,h}` — TQ3 quantize/dequant kernel (torch-matched; rotation and
  codebooks loaded from `assets/`, not the reference repo's non-torch RNG).
- `cpp/tq3_attn.{cc,h}` — fused `voxsum.tq3_attention` kernel (memo, double
  accumulation, OMP; `TQ3_DUMP_OP=dir` dumps first-invocation IO).
- `cpp/engine2.cc` — driver: mode auto-detect, zero-copy packed binding
  (`LiteRtCreateTensorBufferFromHostMemory`), custom-op registration,
  quantize-on-write scatter, `--dump-logits`.
- `cpp/engine.cc` — Phase 2b staging engine (A/B reference).
- `python/rewrite_tq3.py` — flatbuffer lowering; asserts on every matched op so
  a different export fails loudly rather than silently.
- `python/prep_assets.py` — codebook/rotation/PLE/prompt asset generation
  (the codebook-generation reference: Lloyd-Max via the upstream
  `turboquant` Python package's `get_codebook(d, bits)`).
- `python/accept3.py`, `python/run_accept.py` — acceptance drivers.
- `assets/` — committed small artifacts: rotation matrices (fp32, seed-42 QR),
  d=256/512 b=3 codebooks, PLE offset metadata, prompt/teacher token files.
- `results/acceptance.json` (Phase 2b), `results/acceptance3.json` (fused).

## Known limitations

- ~~Global-layer memo growth~~ **mitigated** via `--global-memo full|fp16|stream`
  (sliding pairs are always window-capped, ~14 MiB total; the flag governs the
  3 global pairs whose live rows grow with position):

  | mode | p0 top-1 | p1 top-1 | decode tok/s (p0 / p2@1244) | memo resident @1244 | projected resident @16k |
  |---|---|---|---|---|---|
  | `full` (x86 default) | 0.9455 | 0.9688 | 14.2 / 10.5 | 37 MiB | ~215 MiB |
  | `fp16` | 0.9636 | 0.9844 | 13.6 / 7.8 | 24.5 MiB | ~114 MiB |
  | `stream` (device default) | 0.9455 | 0.9688 | 13.3 / 9.9 | 12 MiB | **~14 MiB + O(1) scratch** |

  `stream` keeps NO persistent global-layer memo: decode runs two tile-streamed
  passes (2 MiB tile + 1 MiB score rows, O(1) in context) with per-(row,column)
  operation order identical to `full` — logits are **bit-identical**
  (verified max|Δ| = 0.0 over 64 teacher-forced steps) at a measured −6%
  decode cost. Prefill in `stream` mode uses a transient dequant buffer freed
  per op (5 MiB at ctx 1244; up to 64 MiB for one op at a full-16k chunk,
  resident zero). `fp16` (memo stored as `_Float16`) is dominated: more memory
  than `stream` AND slower (the fp16→fp32 conversion sits in the hot dot loop),
  though its top-1 stays above the 0.94 gate. Pick `stream` on device.
- **`per_layer_embedder.tflite` re-export pending** — the exported one is
  corrupt; the engines mmap the PLE table from the HF safetensors instead
  (desktop convenience; on-device you'd precompute the table).
- **x86-only so far.** Nothing needs beyond ARMv8.0 (`-march=native` →
  `-march=armv8-a`), OpenMP → NDK libomp; cross-compile is straightforward but
  not yet done.
- Engine-side integration for LiteRT-LM lives on the `turboquant-tq3` branch of
  [vieenrose/litert-lm](https://github.com/vieenrose/litert-lm) (LiteRT-LM
  itself is not buildable from public sources — google-ai-edge/LiteRT-LM
  issues #3002/#2932/#2945 — so everything here builds standalone via CMake
  against the prebuilt `libLiteRt.so`).
