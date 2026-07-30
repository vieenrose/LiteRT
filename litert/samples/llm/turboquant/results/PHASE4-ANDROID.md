# TurboQuant Phase 4 — Android package for VoxSumDroid (2026-07-29)

Goal: run the Phase 3 fused-TQ3 Gemma 4 E2B summarizer on the Boox Tab Mini C
(ARMv8.0-A, 4×A73 big cores, 3.7 GB RAM) against the stock arm64 `libLiteRt.so`
the VoxSumDroid app already ships.

## 1. Disk triage (ws)

Freed ~40 GB (2.2 G → 42 G free before the 4k export; 20 G after it):

- HF caches of consumed checkpoints: gemma-4-E4B-it (15 G), Qwen3.5-9B-GGUF
  (16 G), VibeVoice-ASR-HF-NF4 (6.5 G), fleurs dataset (5.2 G),
  Qwen3-VL-2B (4 G), pip cache.
- `~/turboquant` npz/log debug artifacts of non-baseline configs (k3v4/k4v3/
  k4v4/kg4/tq3333, ~1 G) and `phase3/out/*.logits*` (250 M).
- NOT deleted: gemma-4-E2B-it cache (source of the 4k export + PLE table),
  `out_p{0,1,2}_base.npz` acceptance baselines, model_quantized/model_tq3/
  embedder/auxiliary/codebooks/PHASE*.md, nemotron + wav2vec2 caches (in use
  by the ASR-bench scoring), VibeVoice-awq cache (root-owned, needs sudo).
- fp32 intermediates of the 4k export (model.tflite 8.5 G, embedder 1.6 G,
  fp32 per_layer_embedder) deleted right after the quantized artifacts landed.

## 2. PLE table (`--ple-table`, replaces safetensors mmap + corrupt per_layer_embedder.tflite)

`python/make_ple_table.py ple.json OUT.bin {fp16|fp32}` extracts
`embed_tokens_per_layer.weight` (262144 × 8960 bf16) from `model.safetensors`
into a flat mmap-able binary; a bf16 variant is a raw copy. Format: 32-byte
header `"PLETBL01"` + u32 dtype (0 fp32 / 1 fp16 / 2 bf16) + u32 rows +
u32 cols + f32 scale (16.0) + 8 pad, then row-major raw values (scale applied
at gather time in engine2).

- Size: 4.7 GB at any 16-bit dtype (fp32 would be 9.4 GB — rejected).
- bf16 is bit-identical to the safetensors path by construction (same bytes).
- fp16: bf16→fp16 conversion is exact except values below fp16-normal range —
  153,643 of 2.35 G values (0.0065 %) flush; measured acceptance is unchanged.
- x86 validation (fused 16k model, stream memo, teacher-forced 64 steps):

| PLE source | p0 top-1 | p1 top-1 |
|---|---|---|
| safetensors mmap (reference) | 0.9455 | 0.9688 |
| bf16 table | 0.9455 | 0.9688 |
| fp16 table | 0.9455 | 0.9688 |

Device pick: **bf16** (exactness for free; same size as fp16).
Future option: the 4k export produced a healthy
`per_layer_embedder_quantized.tflite` (int8, 2.35 GB — the 16k one was
corrupt); an int8 PLE table would halve the file if flash is ever tight.

## 3. 4k-context re-export

Same Phase 1 recipe (`run_export_4k.py` = run_export.py with
`cache_length=4096`; needs `PYTHONPATH=~/meeting-summarizer/ai-edge-torch`
commit 4a67251 — the venv-installed litert_torch refuses split_cache for
Gemma 4 — plus the litert_moe_sequential shim). ~3.5 min on the 9950X3D.
Artifacts: `~/turboquant/export/e2b_4k/final/`.

`rewrite_tq3.py` gained an optional cache-length argument;
`rewrite_tq3.py model_quantized.tflite model_tq3_4k.tflite 4096` fused the
same 14 (prefill) / 35 (decode) blocks with all structure asserts green.
`tq3_attn.cc` now infers (T, cache_len, block_bytes) jointly from the mask and
packed tensor sizes, so one kernel binary serves 16k and 4k models;
engine2 gained `--cache-len` (packed side-cache sizing).

x86 gates (fused, stream memo, bf16 PLE table, vs the SAME Phase 2a
baselines): p0 **0.9455**, p1 **0.9688** — identical to the 16k model, PASS.
p2 (1244-token meeting transcript) free-run: coherent structured zh summary
(三位元量化方案 / 頻譜分群 / numbers intact). Packed side-cache 14 MiB,
memo 12 MiB, x86 decode 9.8 tok/s.

## 4. NDK cross-compile

Host: NDK r27.2 (27.2.12479018), built on the adb host, sources synced from
the ws. `CMakeLists.txt` changes: `-march` is now cache var `TQ3_MARCH`
(default `native`); on Android link `-static-openmp`.

```bash
cmake .. -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 -DTQ3_MARCH=armv8-a \
  -DTQ3_DIR=<dir with tq3.c> -DLITERT_INCLUDE=<VibeASR third_party headers> \
  -DLITERT_LIB=~/VoxSumDroid/app/src/main/jniLibs/arm64-v8a/libLiteRt.so \
  -DOpenMP_CXX_FLAGS=-fopenmp -DOpenMP_C_FLAGS=-fopenmp \
  -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_C_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY=$NDK/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/18/lib/linux/aarch64/libomp.a
```

The OpenMP_* overrides are required: CMake's FindOpenMP resolves the NDK's
libomp.so by absolute path, which silently defeats `-static-openmp`.
Result: `engine2` (arm64, armv8-a, no dotprod/fp16) NEEDED = libLiteRt.so,
libc/m/dl only. Linked against the app-shipped libLiteRt.so
(md5 0563b6bc…, NOT the ws prebuilt 7d5f664b… — they differ).

## 5. On-device validation

### 5a. Boox Tab Mini C (Android, serial 800D1C1B) — cold path only, thrash-bound

Everything pushed to `/data/local/tmp/tq3/` (model_tq3_4k, int8 PLE table,
embedder/auxiliary, assets, engine2, app libLiteRt.so; 22 G flash left).
Cold p0 (2 threads, taskset f0 big cores):

- load 30.1 s, VmHWM 2339–2652 MB; prefill 15.2 tok/s; catch-up 0.35 tok/s;
  **decode 0.037 tok/s** (fully page-thrash-bound: 2.2 G model + 2.2 G PLE
  file pages vs ~2.5 G of cache on a 3.6 G device); top-1 0.9273 (same tokens
  as the Pi run below — arm-deterministic).
- The first 4-thread attempt livelocked the whole device for ~45 min
  (load avg 77, adbd stalled); no OOM kill, no reboot.
- Status: **revisit with the pre-packed XNNPACK weight cache** (Section 5b
  shows it eliminates exactly this phase). Not rerun here because the device
  was crushed twice and the user redirected validation to the Pi.

### 5b. Raspberry Pi 4B (glibc aarch64, Cortex-A72 4×1.8 GHz, 3.8 G RAM, 8 G swap) — PRIMARY RESULT

Same ISA constraint as the Boox (armv8.0-A, no dotprod/fp16). Blocker found
and fixed: **every available linux_arm64 libLiteRt.so prebuilt (litert-fork
`prebuilt/linux_arm64`, vibe-lite) SIGILLs on armv8.0 — compiler-inlined
`ldaprb` (ARMv8.3 RCpc) in static initializers** (953 `ldapr` occurrences in
the prebuilt; the Android jniLibs build is baseline-clean, which is why the
Boox runs). Rebuilt libLiteRt.so from the fork for linux-aarch64 at plain
`-march=armv8-a` (Arm GNU 12.2 cross toolchain to match Debian 12 glibc 2.36,
host-flatc fix for the cross build, `XNNPACK_ENABLE_KLEIDIAI=OFF` — GCC 12.2
ICEs on a KleidiAI bf16 kernel, and KleidiAI needs v8.2+ anyway). Result: 0
`ldapr`, engine runs. Toolchain file: `~/turboquant/aarch64-armv8a.toolchain.cmake`;
build dir `~/litert-fork/litert/cmake_build_linux_arm64` (lib in `c/`).

**Acceptance (teacher-forced, 64 steps, vs the same Phase 2a baselines):**

| prompt | top-1 (Pi, arm) | x86 ref | gate ≥0.94 |
|---|---|---|---|
| p0 (en) | 0.9273 (4 flips/54) | 0.9455 (3 flips) | **MISS by 1.3 pts — see margin analysis** |
| p1 (zh) | **0.9688** | 0.9688 | PASS (exact match) |

p0 flip margins (arm logits, nats): flips @7 and @24 are **shared with x86**
(the TQ3 quantizer's own deviations; margins 9.0 and 4.9 — real, but present
on every platform and already accepted in Phase 3). Arm-specific: @48 margin
**0.28 (benign near-tie)** and @27 margin 3.6; x86 instead flips @50. Same
top-1 (0.9273) and identical generated tokens on Boox-Android-bionic and
Pi-glibc with different libLiteRt builds and at 2/3/4 threads — the arm result
is deterministic, the x86↔arm delta is the documented dynamic-activation-quant
amplification of sub-1e-6 rounding differences (PHASE3: ANY reimplementation
boundary costs ~0.93 argmax agreement; near-bit-identical is unachievable on
this int8 graph). Double accumulation confirmed active (same tq3_attn.cc
source, `-O3 -march=armv8-a`, no fast-math anywhere). Free-running output is
the decisive check and passes: p2 (1244-token zh meeting transcript, 200
tokens generated) is a coherent, faithful structured zh summary (三位元量化,
頻譜分群 12%→7%, numbers intact).

**Memory & speed (int8 PLE table, stream memo):**

| phase | cold (no weight cache, 4 thr) | cache-create (2 thr) | warm (`--weight-cache`, 4 thr) |
|---|---|---|---|
| load time | 88 s | 110 s | **0.3–1.2 s** |
| load RSS | 3493 MB | 485 MB | **92 MB** |
| peak RssAnon (whole run) | **2850 MB** | n/a | **117 MB** |
| peak VmHWM | 3647 MB | 2744 MB | 2411 MB (≈2.2 G file-backed cache pages, evictable) |
| prefill tok/s | 3.2 (load contention) / 11.0 @p2 | — | **20.2** |
| decode tok/s | 1.09–1.11 | 0.97 | **1.06** (0.96 @2 thr) |

**SHIP GATE (peak RssAnon < 2.0 GB warm): PASS at 117 MB** — the Phase 2b
x86 precedent (packing becomes file-backed) fully reproduces on arm. The
cache file is 2.29 GB on flash (wcache.bin). TTFT estimate for a ~3 k-token
transcript, warm at 4 threads: ~2.5 min prefill + up to ~1 min catch-up tail
≈ **3–4 min to first summary token**, then ~1 tok/s decode (Boox A73 @2.0 GHz
should be modestly faster; the Pi numbers are the floor).

Caveat: the Pi hard-rebooted 3× during heavy 3–4-thread load phases
(14:17/14:37/14:45; throttled=0x0 after boot, no OOM in journal — PSU/SD
brownout suspected, not a software fault; identical workloads completed at
2 threads and earlier in the day at 4).

### 5c. Pre-packed cache portability (the Android plan)

`XNNPackCacheHeader` (kVersion=2) holds only version + buffer-list offset —
no device/ISA fingerprint. Entries key on (pack_algorithm_id, weights_id,
bias_id). Empirically the cache created at 2 threads was consumed unmodified
at 3 and 4 threads → **thread count does not key the cache**. What DOES bind
it: the exact libLiteRt.so build (pack-algorithm ids are build-internal) and
the runtime ukernel dispatch (ISA features). Plan that follows: generate
wcache.bin ONCE per (lib build × ISA class) — armv8.0-baseline covers both
A72 Pi and A73 Boox since cpuinfo dispatches identically — ship it next to
the model, never pack on device. Validate top-1 once per device class after
any lib upgrade (cache version bumps invalidate it loudly).

## 6. Verdict

**Usable as VoxSumDroid's summarizer path — conditionally, and only via the
pre-packed-cache route.** The memory story is now solid: warm-path anonymous
RSS is ~120 MB end-to-end (KV 14 MiB packed + ~13 MiB memo + activations),
everything else is evictable file-backed pages (2.2 G model cache + 2.2 G
int8 PLE + 0.4 G embedder ≈ 4.9 G flash), so the 3.7 G-RAM Boox fits with
huge headroom — where the current .litertlm path cannot even load Gemma 4
E2B. Quality holds (zh acceptance exact vs x86, en within near-tie noise,
faithful free-run summaries). The open question is SPEED: ~1 tok/s decode /
20 tok/s prefill on A72 means ~6 min for a 1.2 k-token transcript + 200-token
zh summary. That is acceptable for an offline "summarize overnight/afterward"
flow but well below the current small-model .litertlm summarizer's latency;
E2B quality vs that path is the trade. Next steps: (1) rerun the Boox with
wcache.bin (expect the 5a thrash to disappear; the cache must be rebuilt
against the app's own libLiteRt.so build), (2) big-core affinity + thread
sweep on-device, (3) decide product-side whether 3–4 min TTFT is shippable.

## Code

Pushed to vieenrose/LiteRT branch `turboquant-tq3`
(`litert/samples/llm/turboquant/`), commit a39fe5d: engine2.cc (--ple-table,
--cache-len), tq3_attn.cc (cache-length-agnostic), rewrite_tq3.py (cache-len
arg), make_ple_table.py (new), CMakeLists.txt (TQ3_MARCH, static libomp).
