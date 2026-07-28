// End-to-end proof that a ternary custom op works on the STOCK LiteRT runtime.
//
// The chain being validated:
//   torch custom op -> stablehlo.custom_call -> (flatbuffer retarget) -> tfl.custom
//   -> LiteRtAddCustomOpKernelOption -> ternary_gemm
//
// The point is that none of this needs a LiteRT fork or rebuild:
// LiteRtAddCustomOpKernelOption is exported by the shipped libLiteRt.so, so a
// packed-ternary kernel can be attached to Google's binary as-is.
//
// Also reports whether ops AROUND the custom op still get delegated — custom ops
// are opaque to XNNPACK, and if the surrounding graph falls out of delegation the
// ternary win could be given straight back.
//
//   ./test_ternary_custom_op model.tflite

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "ternary_gemm.h"

#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_custom_op_kernel.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"

namespace {

#define CHECK_OK(expr, what)                                             \
    do {                                                                 \
        if ((expr) != kLiteRtStatusOk) {                                 \
            fprintf(stderr, "FAIL: %s\n", what);                         \
            return 1;                                                    \
        }                                                                \
    } while (0)

struct TernaryOpState {
    int calls = 0;
};

// inputs: x [m,k] f32, packed_w [n_rows, k/4] int8, w_scale [n_rows] f32
// output: y [m, n_rows] f32
LiteRtStatus TernaryInit(void*, const void*, size_t n) {
    fprintf(stderr, "[op] Init(init_data_size=%zu)\n", n);
    return kLiteRtStatusOk;
}

LiteRtStatus TernaryGetOutputLayouts(void*, size_t num_inputs,
                                     const LiteRtLayout* in, size_t num_outputs,
                                     LiteRtLayout* out) {
    fprintf(stderr, "[op] GetOutputLayouts(in=%zu out=%zu)", num_inputs, num_outputs);
    for (size_t i = 0; i < num_inputs; i++) {
        fprintf(stderr, "  in%zu rank=%u [", i, in[i].rank);
        for (unsigned d = 0; d < in[i].rank; d++) fprintf(stderr, "%d,", in[i].dimensions[d]);
        fprintf(stderr, "]");
    }
    fprintf(stderr, "\n");
    if (num_inputs != 3 || num_outputs != 1) return kLiteRtStatusErrorInvalidArgument;
    const int32_t m = in[0].dimensions[0];
    const int32_t n_rows = in[1].dimensions[0];
    // Zero the whole struct: everything past `rank` is documented as undefined,
    // and the dispatcher reads more of it than just the dimensions we set.
    memset(&out[0], 0, sizeof(out[0]));
    out[0].rank = 2;
    out[0].has_strides = false;
    out[0].dimensions[0] = m;
    out[0].dimensions[1] = n_rows;
    fprintf(stderr, "[op]   -> out [%d,%d]\n", m, n_rows);
    return kLiteRtStatusOk;
}

LiteRtStatus TernaryRun(void* user_data, size_t num_inputs,
                        const LiteRtTensorBuffer* inputs, size_t num_outputs,
                        LiteRtTensorBuffer* outputs) {
    fprintf(stderr, "[op] Run(in=%zu out=%zu)\n", num_inputs, num_outputs);
    if (num_inputs != 3 || num_outputs != 1) return kLiteRtStatusErrorInvalidArgument;
    auto* st = static_cast<TernaryOpState*>(user_data);
    st->calls++;

    void *px = nullptr, *pw = nullptr, *ps = nullptr, *py = nullptr;
    size_t bx = 0, bw = 0, bs = 0;
    LiteRtGetTensorBufferSize(inputs[0], &bx);
    LiteRtGetTensorBufferSize(inputs[1], &bw);
    LiteRtGetTensorBufferSize(inputs[2], &bs);
    if (LiteRtLockTensorBuffer(inputs[0], &px, kLiteRtTensorBufferLockModeRead) != kLiteRtStatusOk ||
        LiteRtLockTensorBuffer(inputs[1], &pw, kLiteRtTensorBufferLockModeRead) != kLiteRtStatusOk ||
        LiteRtLockTensorBuffer(inputs[2], &ps, kLiteRtTensorBufferLockModeRead) != kLiteRtStatusOk ||
        LiteRtLockTensorBuffer(outputs[0], &py, kLiteRtTensorBufferLockModeWrite) != kLiteRtStatusOk) {
        fprintf(stderr, "[op] lock failed\n");
        return kLiteRtStatusErrorRuntimeFailure;
    }
    fprintf(stderr, "[op] sizes bx=%zu bw=%zu bs=%zu\n", bx, bw, bs);

    const int n_rows = (int)(bs / sizeof(float));
    const int kq = (int)(bw / n_rows);
    const int k = kq * 4;
    const int m = (int)(bx / sizeof(float) / k);

    std::vector<int8_t> q((size_t)m * k);
    std::vector<float> xs(m);
    ternary_quantize_activations((const float*)px, m, k, q.data(), xs.data());
    ternary_gemm((const uint8_t*)pw, n_rows, k, q.data(), xs.data(), m,
                 (const float*)ps, /*per_row=*/1, /*bias=*/nullptr, (float*)py);

    LiteRtUnlockTensorBuffer(inputs[0]);
    LiteRtUnlockTensorBuffer(inputs[1]);
    LiteRtUnlockTensorBuffer(inputs[2]);
    LiteRtUnlockTensorBuffer(outputs[0]);
    return kLiteRtStatusOk;
}

LiteRtStatus TernaryDestroy(void*) { return kLiteRtStatusOk; }

}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "ternary_probe_custom.tflite";
    printf("model: %s\nkernel: %s\n\n", path, ternary_gemm_impl_name());

    LiteRtEnvironment env = nullptr;
    CHECK_OK(LiteRtCreateEnvironment(0, nullptr, &env), "CreateEnvironment");
    LiteRtModel model = nullptr;
    CHECK_OK(LiteRtCreateModelFromFile(env, path, &model), "CreateModelFromFile");

    LiteRtOptions opts = nullptr;
    CHECK_OK(LiteRtCreateOptions(&opts), "CreateOptions");
    CHECK_OK(LiteRtSetOptionsHardwareAccelerators(opts, kLiteRtHwAcceleratorCpu), "SetAccelerators");

    // VIBEASR_EXTERNAL_W: supply the packed weights ZERO-COPY from caller memory.
    //
    // Needed because the dispatcher will not hand CONSTANT tensors to a custom
    // kernel, so weights must enter as a graph input — and a graph input would
    // otherwise mean allocating a managed buffer and copying a decoder's worth of
    // packed bytes into anonymous memory, which is exactly what we are trying to
    // avoid. LiteRtAddExternalTensorBinding looked like the answer but does NOT
    // override an explicit run input (tested: the managed buffer won, silently),
    // so the zero-copy has to come from the buffer itself.
    std::vector<uint8_t> ext_w;
    const bool use_external = getenv("VIBEASR_EXTERNAL_W") != nullptr;
    if (use_external) {
        ext_w.resize(32 * 16);
        for (size_t j = 0; j < ext_w.size(); j++) ext_w[j] = (uint8_t)(0x1B + (j % 7));
    }

    TernaryOpState state;
    LiteRtCustomOpKernel kernel = {TernaryInit, TernaryGetOutputLayouts, TernaryRun,
                                   TernaryDestroy};
    CHECK_OK(LiteRtAddCustomOpKernelOption(opts, "voxsum.ternary_matmul", 1, &kernel, &state),
             "AddCustomOpKernelOption");

    LiteRtCompiledModel cm = nullptr;
    CHECK_OK(LiteRtCreateCompiledModel(env, model, opts, &cm), "CreateCompiledModel");
    printf("compiled with the custom op registered\n");

    LiteRtSignature sig = nullptr;
    CHECK_OK(LiteRtGetModelSignature(model, 0, &sig), "GetModelSignature");

    const auto make = [&](bool is_input, LiteRtParamIndex idx, LiteRtTensorBuffer* buf) {
        LiteRtTensor t = nullptr;
        if ((is_input ? LiteRtGetSignatureInputTensorByIndex(sig, idx, &t)
                      : LiteRtGetSignatureOutputTensorByIndex(sig, idx, &t)) != kLiteRtStatusOk)
            return false;
        LiteRtRankedTensorType tt;
        if (LiteRtGetRankedTensorType(t, &tt) != kLiteRtStatusOk) return false;
        LiteRtTensorBufferRequirements reqs = nullptr;
        if ((is_input ? LiteRtGetCompiledModelInputBufferRequirements(cm, 0, idx, &reqs)
                      : LiteRtGetCompiledModelOutputBufferRequirements(cm, 0, idx, &reqs))
            != kLiteRtStatusOk) return false;
        size_t bytes = 0;
        if (LiteRtGetTensorBufferRequirementsBufferSize(reqs, &bytes) != kLiteRtStatusOk) return false;
        return LiteRtCreateManagedTensorBuffer(env, kLiteRtTensorBufferTypeHostMemory, &tt, bytes,
                                               buf) == kLiteRtStatusOk;
    };

    // Create a buffer for EVERY signature input: when the packed weights are
    // graph arguments rather than constants there are three, not one.
    LiteRtParamIndex n_in = 0;
    CHECK_OK(LiteRtGetNumSignatureInputs(sig, &n_in), "GetNumSignatureInputs");
    std::vector<LiteRtTensorBuffer> ins((size_t)n_in, nullptr);
    for (LiteRtParamIndex i = 0; i < n_in; i++) {
        // Weights: wrap the caller's memory instead of allocating a managed buffer.
        // LiteRtAddExternalTensorBinding does NOT override an explicit run input
        // (verified: the managed buffer won), so zero-copy has to come from the
        // buffer itself. With real weights this is what keeps a decoder's worth of
        // packed bytes mmap'd and file-backed rather than copied into anon memory.
        if (use_external && i == 1) {
            LiteRtTensor t = nullptr;
            LiteRtRankedTensorType tt;
            if (LiteRtGetSignatureInputTensorByIndex(sig, i, &t) != kLiteRtStatusOk ||
                LiteRtGetRankedTensorType(t, &tt) != kLiteRtStatusOk ||
                LiteRtCreateTensorBufferFromHostMemory(&tt, ext_w.data(), ext_w.size(),
                                                       nullptr, &ins[i]) != kLiteRtStatusOk) {
                fprintf(stderr, "FAIL: CreateTensorBufferFromHostMemory\n");
                return 1;
            }
            printf("wrapped args_1 zero-copy from host memory (%zu bytes)\n", ext_w.size());
            continue;
        }
        if (!make(true, i, &ins[i])) { fprintf(stderr, "FAIL: input buffer %llu\n",
                                               (unsigned long long)i); return 1; }
        size_t nb = 0;
        LiteRtGetTensorBufferSize(ins[i], &nb);
        void* q = nullptr;
        LiteRtLockTensorBuffer(ins[i], &q, kLiteRtTensorBufferLockModeWrite);
        memset(q, 0, nb);
        // 0: activations, 1: packed ternary weights (raw bytes), 2: per-row scale.
        // Real values, not zeros — zeroed weights make every output 0.0, which
        // "passes" a liveness check while proving nothing about the arithmetic.
        if (i == 0) for (size_t j = 0; j < nb / sizeof(float); j++)
            ((float*)q)[j] = sinf((float)j * 0.37f);
        else if (i == 1) for (size_t j = 0; j < nb; j++)
            // When testing the external binding, leave the MANAGED buffer holding a
            // different pattern (0x55) than the external one (0x1B+): if the output
            // still matches the external data, the binding is demonstrably in use
            // rather than the managed copy silently winning.
            ((uint8_t*)q)[j] = use_external ? 0x55 : (uint8_t)(0x1B + (j % 7));
        else for (size_t j = 0; j < nb / sizeof(float); j++)
            ((float*)q)[j] = 0.02f;
        LiteRtUnlockTensorBuffer(ins[i]);
    }
    LiteRtTensorBuffer out = nullptr;
    if (!make(false, 0, &out)) { fprintf(stderr, "FAIL: output buffer\n"); return 1; }

    size_t in_bytes = 0, out_bytes = 0;
    LiteRtGetTensorBufferSize(ins[0], &in_bytes);
    LiteRtGetTensorBufferSize(out, &out_bytes);
    const int k = (int)(in_bytes / sizeof(float));
    const int n_rows = (int)(out_bytes / sizeof(float));
    void* p = nullptr;

    CHECK_OK(LiteRtRunCompiledModel(cm, 0, (LiteRtParamIndex)ins.size(), ins.data(), 1, &out),
             "RunCompiledModel");

    LiteRtLockTensorBuffer(out, &p, kLiteRtTensorBufferLockModeRead);
    double sum = 0, amax = 0;
    for (int i = 0; i < n_rows; i++) {
        const double v = ((float*)p)[i];
        sum += v;
        amax = fmax(amax, fabs(v));
    }
    LiteRtUnlockTensorBuffer(out);

    printf("ran: input %d floats -> output %d floats\n", k, n_rows);
    printf("custom kernel invocations: %d\n", state.calls);
    printf("output: sum=%.4f max_abs=%.4f\n", sum, amax);

    // Cross-check against calling the kernel directly on the same bytes: the
    // graph result must equal what ternary_gemm produces standalone.
    bool matches = true;
    if (ins.size() == 3) {
        void *qx = nullptr, *qw = nullptr, *qs = nullptr;
        size_t bw = 0, bs = 0;
        LiteRtGetTensorBufferSize(ins[1], &bw);
        LiteRtGetTensorBufferSize(ins[2], &bs);
        LiteRtLockTensorBuffer(ins[0], &qx, kLiteRtTensorBufferLockModeRead);
        LiteRtLockTensorBuffer(ins[1], &qw, kLiteRtTensorBufferLockModeRead);
        const uint8_t* wsrc = use_external ? ext_w.data() : (const uint8_t*)qw;
        LiteRtLockTensorBuffer(ins[2], &qs, kLiteRtTensorBufferLockModeRead);
        const int nr = (int)(bs / sizeof(float));
        const int kk = (int)(bw / nr) * 4;
        std::vector<int8_t> q((size_t)kk);
        std::vector<float> xs(1), expect((size_t)nr);
        ternary_quantize_activations((const float*)qx, 1, kk, q.data(), xs.data());
        ternary_gemm(wsrc, nr, kk, q.data(), xs.data(), 1,
                     (const float*)qs, 1, nullptr, expect.data());
        LiteRtLockTensorBuffer(out, &p, kLiteRtTensorBufferLockModeRead);
        for (int i = 0; i < nr; i++)
            if (fabs(expect[i] - ((float*)p)[i]) > 1e-4) matches = false;
        LiteRtUnlockTensorBuffer(out);
        LiteRtUnlockTensorBuffer(ins[0]);
        LiteRtUnlockTensorBuffer(ins[1]);
        LiteRtUnlockTensorBuffer(ins[2]);
        printf("graph output vs direct ternary_gemm: %s\n", matches ? "identical" : "DIFFERS");
    }

    const bool pass = state.calls == 1 && amax > 0.0 && matches;
    printf("\n%s\n", pass ? "PASS - stock LiteRT dispatched to the ternary kernel"
                          : "FAIL - custom kernel was not invoked");
    return pass ? 0 : 1;
}
