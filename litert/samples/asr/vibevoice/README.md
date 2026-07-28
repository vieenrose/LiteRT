# VibeVoice-ASR-BitNet on LiteRT

Runs [microsoft/VibeVoice-ASR-BitNet](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet)
— a conv audio front end plus a **ternary (BitNet I2_S) Qwen2.5-1.5B** decoder —
entirely on LiteRT, with no ggml.

LiteRT/XNNPACK has no ternary kernel, and exporting BitNet the normal way turns it
into an int8 matmul that keeps the *values* and throws away the *packing*. Packing
is the performance: batch-1 decode is memory-bandwidth-bound, so a 1.31 B-parameter
decoder moves ~328 MB per token at 2 bits versus ~1310 MB at int8. This sample adds
a **custom op** backed by a NEON ternary GEMM, so the packing survives to the
runtime.

**No LiteRT fork is required to run it.** `LiteRtAddCustomOpKernelOption` is
exported by the stock prebuilt `libLiteRt.so` on both arm64 and linux-x64
(verified with `nm`). The sample lives here because this is where it belongs, not
because the runtime needed changing.

## Measured

Boox Tab Mini C — Snapdragon 662, Cortex-A73, **ARMv8.0 with no dotprod**, which is
close to worst case for int8 SIMD. Back to back against the ggml build of the same
model, minutes apart:

| | this sample | ggml (`asr_infer`) |
|---|---|---|
| decode | **123.5 ms/token** | 123.1 |
| prefill (batched, T=16) | **68 ms/token** | — |
| peak RSS | **786 MB** | 1297 MB |
| peak RssAnon | **241 MB** | 507 MB |

Parity on speed at **half the unevictable memory**. Accuracy: cosine 0.994938
against a dense f32 reference over the full 28-layer stack; the ternary kernel
itself is bit-exact against its scalar reference at every shape and thread count.

## Layout

    cpp/
      ternary_gemm.{h,cc}     the NEON ternary GEMM (4-row blocked, spin-pool)
      q6k.{h,c}               Q6_K row dequant, for host-side embedding lookup
      vibe_lite_engine.{h,cc} four graphs + KV aliasing + greedy decode
      vibe_lite_jni.cpp       Android JNI surface
      test_ternary_gemm.cc    exactness + throughput
      test_i2s_layout.cc      proves the GGUF weight layout, using ggml as oracle
      test_ternary_custom_op.cc  the op end to end on the stock runtime
      run_mlp_layer.cc        any exported block, against a reference
      generate.cc             CLI greedy generation
    python/
      gguf_read.py            minimal GGUF reader + I2_S/Q6_K dequant
      ternary_op.py           torch custom op + stablehlo lowering
      retarget_custom_op.py   STABLEHLO_CUSTOM_CALL -> tfl.custom
      export_*.py             encoder / decoder / prefill / head exports
    VibeLiteEngine.kt         Kotlin wrapper + byte-level BPE detokenizer

## Pipeline

    audio ──► encoder.tflite ──► features [frames, 1536] ──┐
                                                           ├─► prefill/decode ──► head ──► argmax
    token id ──► Q6_K embedding lookup (host) ─────────────┘         (ternary custom op)

Four graphs, separate for concrete reasons:

* **The head bakes its weights in as constants** (233 MB int8) because it has no
  custom op. The decoder's weights must be runtime **inputs** — the dispatcher
  refuses to hand constant tensors to a custom kernel.
* **Prefill and decode share weight and cache buffers exactly**; only the graph
  differs. Each KV pair is aliased input↔output so the cache updates in place.

## Things that cost real time to discover

* **The I2_S weight layout is bit-plane major over 128-element groups**, not the
  sequential 4-per-byte packing it looks like: element `j` (with `g = j/32`,
  `i = j%32`) lives in byte `i` at bit-pair `3-g`. The converter source is
  *misleading* — `transform_to_i2` does `x + 2`, implying codes {1,2,3}, while the
  bytes say {0,1,2}. Getting it wrong fails **silently**. `test_i2s_layout.cc`
  derives it by feeding one-hot vectors through ggml (cosine 1.0000 per probe).
* **A condition-variable thread pool is catastrophic here.** A decoder dispatches
  the GEMM ~84 times per token for a few ms each; with a futex handoff, 4 threads
  were 1.8x *slower* than 1 (341 vs 188 ms). Spinning makes 4 threads 2.4x faster
  than 1. ggml's threadpool spins for the same reason.
* **The chat template is not optional.** A minimal
  `<|im_start|><|speech_start|>feats<|speech_end|>` wrapper loaded, ran and
  detokenized cleanly, and transcribed a 7 s English clip as *"Hmm. The guy."*
* **`LiteRtAddExternalTensorBinding` does not override an explicit run input** — it
  is silently ignored. Use `LiteRtCreateTensorBufferFromHostMemory`, and grow the
  buffer to the runtime's *requirement*, not the logical size.
* **`dynamic_update_slice` needs 0-D scalar indices**, and it is essential: the
  obvious one-hot cache scatter is O(ctx) and cost 7.57 vs 5.30 ms/layer at ctx=64
  alone, growing linearly with context.
* **`uint8` graph tensors are rejected**; carry packed bytes as int8.

## Benchmarking caveat

Absolute timings on a passively-cooled device are **not comparable across time**.
This tablet's `schedutil` governor parks the big cores at ~1050 of 2016 MHz when a
memory-bound workload stalls, and without root the governor cannot be pinned.
Chasing a 2.5x "regression" between two binaries here, eight hypotheses were tested
and rejected before re-running the *old* binary revealed it had also slowed by the
same factor. **Only trust A/B measurements taken back to back.**
