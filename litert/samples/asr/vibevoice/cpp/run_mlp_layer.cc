// Runs one REAL Qwen2.5 MLP block on LiteRT with the ternary custom op, using
// weights lifted straight from the shipped I2_S GGUF, and checks the result
// against a PyTorch reference computed from the same weights.
//
// This is the first time the whole chain carries real model data end to end:
//   GGUF I2_S -> unpack_i2s -> ternary_pack -> tfl.custom -> ternary_gemm
//
// The MLP is 88% of a decoder layer's weight bytes (41.3 M of 46.8 M), and decode
// is bandwidth-bound, so getting this right is most of the decoder.
//
//   ./run_mlp_layer <model.tflite> <dir> [iters] [manifest.txt] [expected.bin]
//
// manifest.txt lists one input filename per line, in signature order; without it
// the MLP block's own file set is assumed.

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

std::vector<uint8_t> read_file(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);
    const size_t n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> b(n);
    if (fread(b.data(), 1, n, f) != n) { fprintf(stderr, "short read %s\n", p.c_str()); exit(1); }
    fclose(f);
    return b;
}

struct State { int calls = 0; };

LiteRtStatus Init(void*, const void*, size_t) { return kLiteRtStatusOk; }
LiteRtStatus Destroy(void*) { return kLiteRtStatusOk; }

LiteRtStatus GetOutputLayouts(void*, size_t n_in, const LiteRtLayout* in,
                              size_t n_out, LiteRtLayout* out) {
    if (n_in != 3 || n_out != 1) return kLiteRtStatusErrorInvalidArgument;
    memset(&out[0], 0, sizeof(out[0]));
    out[0].rank = 2;
    out[0].dimensions[0] = in[0].dimensions[0];
    out[0].dimensions[1] = in[1].dimensions[0];
    return kLiteRtStatusOk;
}

LiteRtStatus Run(void* ud, size_t n_in, const LiteRtTensorBuffer* inputs,
                 size_t n_out, LiteRtTensorBuffer* outputs) {
    if (n_in != 3 || n_out != 1) return kLiteRtStatusErrorInvalidArgument;
    static_cast<State*>(ud)->calls++;
    void *px = nullptr, *pw = nullptr, *ps = nullptr, *py = nullptr;
    size_t bx = 0, bw = 0, bs = 0;
    LiteRtGetTensorBufferSize(inputs[0], &bx);
    LiteRtGetTensorBufferSize(inputs[1], &bw);
    LiteRtGetTensorBufferSize(inputs[2], &bs);
    LiteRtLockTensorBuffer(inputs[0], &px, kLiteRtTensorBufferLockModeRead);
    LiteRtLockTensorBuffer(inputs[1], &pw, kLiteRtTensorBufferLockModeRead);
    LiteRtLockTensorBuffer(inputs[2], &ps, kLiteRtTensorBufferLockModeRead);
    LiteRtLockTensorBuffer(outputs[0], &py, kLiteRtTensorBufferLockModeWrite);
    const int n_rows = (int)(bs / sizeof(float));
    const int k = (int)(bw / n_rows) * 4;
    const int m = (int)(bx / sizeof(float) / k);
    static std::vector<int8_t> q;
    static std::vector<float> xs;
    q.resize((size_t)m * k);
    xs.resize(m);
    ternary_quantize_activations((const float*)px, m, k, q.data(), xs.data());
    ternary_gemm((const uint8_t*)pw, n_rows, k, q.data(), xs.data(), m,
                 (const float*)ps, 1, nullptr, (float*)py);
    for (int i = 0; i < 3; i++) LiteRtUnlockTensorBuffer(inputs[i]);
    LiteRtUnlockTensorBuffer(outputs[0]);
    return kLiteRtStatusOk;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model.tflite> <dir> [iters]\n", argv[0]); return 1; }
    const std::string model_path = argv[1], dir = argv[2];
    const int iters = argc > 3 ? atoi(argv[3]) : 10;
    const std::string manifest = argc > 4 ? argv[4] : "";
    const std::string expected_name = argc > 5 ? argv[5] : "mlp_expected.bin";

    // Signature order matches export_mlp_layer.py when no manifest is given:
    //   args_0 x, 1 gate_w, 2 gate_s, 3 up_w, 4 up_s, 5 down_w, 6 down_s
    std::vector<std::string> files = {"mlp_input.bin", "mlp_gate_w.bin", "mlp_gate_s.bin",
                                      "mlp_up_w.bin", "mlp_up_s.bin",
                                      "mlp_down_w.bin", "mlp_down_s.bin"};
    if (!manifest.empty()) {
        files.clear();
        FILE* mf = fopen((dir + "/" + manifest).c_str(), "r");
        if (!mf) { fprintf(stderr, "no manifest %s\n", manifest.c_str()); return 1; }
        char line[512];
        while (fgets(line, sizeof(line), mf)) {
            std::string t(line);
            while (!t.empty() && (t.back() == '\n' || t.back() == '\r' || t.back() == ' ')) t.pop_back();
            if (!t.empty()) files.push_back(t);
        }
        fclose(mf);
        printf("manifest: %zu inputs\n", files.size());
    }

    LiteRtEnvironment env = nullptr;
    LiteRtModel model = nullptr;
    if (LiteRtCreateEnvironment(0, nullptr, &env) != kLiteRtStatusOk ||
        LiteRtCreateModelFromFile(env, model_path.c_str(), &model) != kLiteRtStatusOk) {
        fprintf(stderr, "load failed\n"); return 1;
    }
    LiteRtOptions opts = nullptr;
    LiteRtCreateOptions(&opts);
    LiteRtSetOptionsHardwareAccelerators(opts, kLiteRtHwAcceleratorCpu);
    State st;
    LiteRtCustomOpKernel kern = {Init, GetOutputLayouts, Run, Destroy};
    LiteRtAddCustomOpKernelOption(opts, "voxsum.ternary_matmul", 1, &kern, &st);

    LiteRtCompiledModel cm = nullptr;
    if (LiteRtCreateCompiledModel(env, model, opts, &cm) != kLiteRtStatusOk) {
        fprintf(stderr, "compile failed\n"); return 1;
    }
    LiteRtSignature sig = nullptr;
    LiteRtGetModelSignature(model, 0, &sig);
    LiteRtParamIndex n_in = 0;
    LiteRtGetNumSignatureInputs(sig, &n_in);
    printf("model has %llu inputs\n", (unsigned long long)n_in);

    // Weights are wrapped ZERO-COPY: a real decoder holds 328 MB of them and
    // copying into managed buffers would defeat the packing.
    std::vector<std::vector<uint8_t>> blobs;
    std::vector<LiteRtTensorBuffer> ins((size_t)n_in, nullptr);
    blobs.reserve(n_in);
    for (LiteRtParamIndex i = 0; i < n_in; i++) {
        blobs.push_back(read_file(dir + "/" + files[i]));
        LiteRtTensor t = nullptr;
        LiteRtRankedTensorType tt;
        LiteRtGetSignatureInputTensorByIndex(sig, i, &t);
        LiteRtGetRankedTensorType(t, &tt);
        // The runtime can demand more than the tensor's natural size (padding and
        // alignment), and CreateTensorBufferFromHostMemory rejects a buffer that is
        // merely the right logical size. Grow the blob to whatever it asks for —
        // the extra bytes are never read.
        LiteRtTensorBufferRequirements reqs = nullptr;
        size_t need = blobs.back().size();
        if (LiteRtGetCompiledModelInputBufferRequirements(cm, 0, i, &reqs) == kLiteRtStatusOk) {
            size_t want = 0;
            if (LiteRtGetTensorBufferRequirementsBufferSize(reqs, &want) == kLiteRtStatusOk &&
                want > need) {
                blobs.back().resize(want);
                need = want;
            }
        }
        if (LiteRtCreateTensorBufferFromHostMemory(&tt, blobs.back().data(), need, nullptr,
                                                   &ins[i]) != kLiteRtStatusOk) {
            fprintf(stderr, "zero-copy wrap refused for %s (%zu bytes); "
                            "falling back to a managed copy\n", files[i].c_str(), need);
            if (LiteRtCreateManagedTensorBuffer(env, kLiteRtTensorBufferTypeHostMemory, &tt,
                                                need, &ins[i]) != kLiteRtStatusOk) {
                fprintf(stderr, "managed buffer also failed for %s\n", files[i].c_str());
                return 1;
            }
            void* dst = nullptr;
            LiteRtLockTensorBuffer(ins[i], &dst, kLiteRtTensorBufferLockModeWrite);
            memcpy(dst, blobs.back().data(), blobs.back().size());
            LiteRtUnlockTensorBuffer(ins[i]);
        }
    }
    // A decoder step returns the hidden state AND every updated KV cache, so
    // allocate all of them; only output 0 is compared.
    LiteRtParamIndex n_out_sig = 0;
    LiteRtGetNumSignatureOutputs(sig, &n_out_sig);
    std::vector<LiteRtTensorBuffer> outs((size_t)n_out_sig, nullptr);
    for (LiteRtParamIndex i = 0; i < n_out_sig; i++) {
        LiteRtTensor t = nullptr;
        LiteRtRankedTensorType tt;
        LiteRtTensorBufferRequirements reqs = nullptr;
        size_t bytes = 0;
        LiteRtGetSignatureOutputTensorByIndex(sig, i, &t);
        LiteRtGetRankedTensorType(t, &tt);
        LiteRtGetCompiledModelOutputBufferRequirements(cm, 0, i, &reqs);
        LiteRtGetTensorBufferRequirementsBufferSize(reqs, &bytes);
        LiteRtCreateManagedTensorBuffer(env, kLiteRtTensorBufferTypeHostMemory, &tt, bytes, &outs[i]);
    }
    printf("model has %llu outputs\n", (unsigned long long)n_out_sig);

    // ALIAS each KV cache in/out pair to ONE buffer. Without this the runtime
    // treats cache_out as a separate tensor and copies the whole cache every
    // layer every token — 28 MB per token at ctx=512 — instead of updating one
    // row in place. This is the trick MossLiteEngine's KvStore uses.
    //
    // Layout comes from export_decoder.py: outputs are (hidden, k0..k[L-1],
    // v0..v[L-1]) and the cache inputs are the trailing 2L, interleaved k,v per
    // layer. A graph shaped that way is recognised; anything else is left alone.
    int aliased = 0;
    if (n_out_sig >= 3 && (n_out_sig - 1) % 2 == 0) {
        const LiteRtParamIndex L = (n_out_sig - 1) / 2;
        if (n_in >= 2 * L) {
            const LiteRtParamIndex base = n_in - 2 * L;
            for (LiteRtParamIndex i = 0; i < L; i++) {
                LiteRtDestroyTensorBuffer(outs[1 + i]);
                outs[1 + i] = ins[base + 2 * i];            // k_out[i] <- k_in[i]
                LiteRtDestroyTensorBuffer(outs[1 + L + i]);
                outs[1 + L + i] = ins[base + 2 * i + 1];    // v_out[i] <- v_in[i]
                aliased += 2;
            }
        }
    }
    if (aliased) printf("aliased %d KV buffers in place\n", aliased);
    LiteRtTensorBuffer out = outs[0];

    if (LiteRtRunCompiledModel(cm, 0, (LiteRtParamIndex)ins.size(), ins.data(),
                               (LiteRtParamIndex)outs.size(), outs.data()) != kLiteRtStatusOk) {
        fprintf(stderr, "invoke failed\n"); return 1;
    }
    // Latch before the timing loop below adds to it.
    const int calls_first_run = st.calls;
    printf("custom op invocations: %d\n", calls_first_run);

    const auto expect_raw = read_file(dir + "/" + expected_name);
    const float* expect = (const float*)expect_raw.data();
    const size_t n_out = expect_raw.size() / sizeof(float);
    void* p = nullptr;
    LiteRtLockTensorBuffer(out, &p, kLiteRtTensorBufferLockModeRead);
    const float* got = (const float*)p;
    double dot = 0, na = 0, nb = 0, worst = 0;
    for (size_t i = 0; i < n_out; i++) {
        dot += (double)expect[i] * got[i];
        na += (double)expect[i] * expect[i];
        nb += (double)got[i] * got[i];
        worst = fmax(worst, fabs((double)expect[i] - got[i]));
    }
    LiteRtUnlockTensorBuffer(out);
    const double cos = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    printf("reference rms=%.6g  litert rms=%.6g\ncosine=%.6f  max_abs_diff=%.4g\n",
           sqrt(na / n_out), sqrt(nb / n_out), cos, worst);

    const double t0 = now_s();
    for (int i = 0; i < iters; i++)
        LiteRtRunCompiledModel(cm, 0, (LiteRtParamIndex)ins.size(), ins.data(),
                               (LiteRtParamIndex)outs.size(), outs.data());
    const double each = (now_s() - t0) / iters;
    // Packed weight bytes actually streamed per run — every int8 input that is
    // large enough to be a weight matrix rather than a scale vector.
    double wbytes = 0;
    for (size_t i = 0; i < blobs.size(); i++)
        if (blobs[i].size() > 65536) wbytes += (double)blobs[i].size();
    printf("\n%.3f ms per block, %.2f GB/s over packed weights (%.1f MB)\n",
           each * 1e3, wbytes / each / 1e9, wbytes / 1e6);
    // Layers cannot be inferred from the custom op count: an unfused decoder has 7
    // projections per layer, a qkv/gate-up fused one has 4. Guessing 7 reported a
    // fused 28-layer graph as 16 layers. VIBEASR_LAYERS states it.
    int layers = 1;
    if (const char* e = getenv("VIBEASR_LAYERS")) layers = std::max(1, atoi(e));
    printf("whole graph %.1f ms/step; %d layer(s) => %.2f ms/layer, "
           "%.0f ms/token at 28 layers (%s)\n",
           each * 1e3, layers, each * 1e3 / layers, each * 1e3 / layers * 28,
           ternary_gemm_impl_name());

    // Activation quantization is the only expected difference from the dense
    // reference; the weights themselves are bit-identical.
    const bool pass = calls_first_run > 0 && cos > 0.99;
    printf("\n%s\n", pass ? "PASS - real ternary block runs on LiteRT"
                          : "FAIL");
    return pass ? 0 : 1;
}
