// End-to-end greedy generation on LiteRT: token ids in, token ids out.
//
// Ties together everything the port has built:
//   host Q6_K embedding lookup -> 28-layer graph (ternary custom op, KV cache)
//   -> int8 head graph -> argmax -> next token
//
// The two graphs are separate on purpose. Layer weights must be runtime INPUTS
// (the dispatcher will not hand constants to a custom kernel) while the head has
// no custom op and bakes its weights in, so they have different loading needs and
// different re-export cadences.
//
//   ./generate <layers.tflite> <head.tflite> <weights-dir> <embd.bin> [n_tokens] [start_id]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "q6k.h"
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

constexpr int DIM = 1536;

// Flush denormals to zero.
//
// The layer graph measured 125-140 ms/step in isolation but 312 ms inside the
// generation loop — same graph, same weights. The isolated bench ran with an
// all-ZERO KV cache; generation fills it with real values, and the softmax over a
// mask of -1e30 produces denormals. Denormal arithmetic traps to microcode on
// Cortex-A cores and costs orders of magnitude more than normal FP.
//
// FZ (bit 24) flushes denormal results to zero; FZ16 is not needed here. This is
// what ggml and XNNPACK both do at init, which is part of why ggml never showed
// this cliff.
void flush_denormals_to_zero() {
#if defined(__aarch64__)
    uint64_t fpcr;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1ull << 24);
    __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));
#endif
}

double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// mmap a file read-only. Weight tensors are the bulk of resident memory (344 MB
// of layer weights, 191 MB of embedding table), and reading them into heap buffers
// makes every byte ANONYMOUS: unevictable, and counted in full against an Android
// app's budget. Mapped, they are clean file-backed pages the kernel can drop and
// re-read, and they are shared between processes opening the same model.
//
// The same fix was worth 1020 -> 347 MB of RssAnon on the VAE side of this repo.
struct Mapped {
    void* addr = nullptr;
    size_t size = 0;
    Mapped() = default;
    Mapped(const Mapped&) = delete;
    Mapped(Mapped&& o) noexcept : addr(o.addr), size(o.size) { o.addr = nullptr; o.size = 0; }
    Mapped& operator=(Mapped&& o) noexcept {
        if (this != &o) {
            if (addr) munmap(addr, size);
            addr = o.addr; size = o.size;
            o.addr = nullptr; o.size = 0;
        }
        return *this;
    }
    ~Mapped() { if (addr) munmap(addr, size); }
    const uint8_t* data() const { return (const uint8_t*)addr; }
};

static Mapped map_file(const std::string& p) {
    Mapped m;
    const int fd = open(p.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "missing %s\n", p.c_str()); exit(1); }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); fprintf(stderr, "stat %s\n", p.c_str()); exit(1); }
    m.size = (size_t)st.st_size;
    m.addr = mmap(nullptr, m.size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m.addr == MAP_FAILED) { m.addr = nullptr; fprintf(stderr, "mmap %s\n", p.c_str()); exit(1); }
    // Ask the kernel to read the pages in now rather than faulting them one by one
    // during inference. Mapping cut RssAnon by 3x but moved 330 MB of page faults
    // into the first decode step: a cold run measured 207 ms/token against 124 warm.
    // MADV_WILLNEED is advisory and cheap; MADV_SEQUENTIAL tells readahead that the
    // access pattern is a straight sweep, which is exactly how weights are read.
    madvise(m.addr, m.size, MADV_WILLNEED);
    madvise(m.addr, m.size, MADV_SEQUENTIAL);
    return m;
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

LiteRtStatus Run(void*, size_t n_in, const LiteRtTensorBuffer* inputs,
                 size_t n_out, LiteRtTensorBuffer* outputs) {
    if (n_in != 3 || n_out != 1) return kLiteRtStatusErrorInvalidArgument;
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

// A compiled graph plus its bound buffers.
struct Graph {
    LiteRtModel model = nullptr;
    LiteRtCompiledModel cm = nullptr;
    LiteRtSignature sig = nullptr;
    std::vector<LiteRtTensorBuffer> ins, outs;
    std::vector<std::vector<uint8_t>> blobs;

    bool make_buffer(LiteRtEnvironment env, bool is_input, LiteRtParamIndex i,
                     size_t* bytes, LiteRtTensorBuffer* buf) {
        LiteRtTensor t = nullptr;
        LiteRtRankedTensorType tt;
        LiteRtTensorBufferRequirements reqs = nullptr;
        if ((is_input ? LiteRtGetSignatureInputTensorByIndex(sig, i, &t)
                      : LiteRtGetSignatureOutputTensorByIndex(sig, i, &t)) != kLiteRtStatusOk)
            return false;
        if (LiteRtGetRankedTensorType(t, &tt) != kLiteRtStatusOk) return false;
        if ((is_input ? LiteRtGetCompiledModelInputBufferRequirements(cm, 0, i, &reqs)
                      : LiteRtGetCompiledModelOutputBufferRequirements(cm, 0, i, &reqs))
            != kLiteRtStatusOk) return false;
        if (LiteRtGetTensorBufferRequirementsBufferSize(reqs, bytes) != kLiteRtStatusOk) return false;
        return LiteRtCreateManagedTensorBuffer(env, kLiteRtTensorBufferTypeHostMemory, &tt,
                                               *bytes, buf) == kLiteRtStatusOk;
    }

    void run() {
        LiteRtRunCompiledModel(cm, 0, (LiteRtParamIndex)ins.size(), ins.data(),
                               (LiteRtParamIndex)outs.size(), outs.data());
    }
};

void write_buf(LiteRtTensorBuffer b, const void* src, size_t bytes) {
    void* p = nullptr;
    LiteRtLockTensorBuffer(b, &p, kLiteRtTensorBufferLockModeWrite);
    memcpy(p, src, bytes);
    LiteRtUnlockTensorBuffer(b);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <layers.tflite> <head.tflite> <dir> <embd.bin> "
                        "[n_tokens] [start_id]\n", argv[0]);
        return 1;
    }
    const std::string layers_path = argv[1], head_path = argv[2], dir = argv[3];
    const std::string embd_path = argv[4];
    const int n_tokens = argc > 5 ? atoi(argv[5]) : 8;
    const int start_id = argc > 6 ? atoi(argv[6]) : 9707;

    flush_denormals_to_zero();

    LiteRtEnvironment env = nullptr;
    LiteRtCreateEnvironment(0, nullptr, &env);

    LiteRtCustomOpKernel kern = {Init, GetOutputLayouts, Run, Destroy};
    const auto make_opts = [&](bool with_custom_op) {
        LiteRtOptions o = nullptr;
        LiteRtCreateOptions(&o);
        LiteRtSetOptionsHardwareAccelerators(o, kLiteRtHwAcceleratorCpu);
        if (with_custom_op)
            LiteRtAddCustomOpKernelOption(o, "voxsum.ternary_matmul", 1, &kern, nullptr);
        return o;
    };

    // --- layer graph -----------------------------------------------------------
    Graph L;
    if (LiteRtCreateModelFromFile(env, layers_path.c_str(), &L.model) != kLiteRtStatusOk ||
        LiteRtCreateCompiledModel(env, L.model, make_opts(true), &L.cm) != kLiteRtStatusOk) {
        fprintf(stderr, "layers: load/compile failed\n"); return 1;
    }
    LiteRtGetModelSignature(L.model, 0, &L.sig);
    LiteRtParamIndex n_in = 0, n_out = 0;
    LiteRtGetNumSignatureInputs(L.sig, &n_in);
    LiteRtGetNumSignatureOutputs(L.sig, &n_out);

    // Manifest order matches export_decoder.py: x, pos, weights..., caches...
    std::vector<std::string> files;
    {
        FILE* mf = fopen((dir + "/dec_28L_manifest.txt").c_str(), "r");
        if (!mf) { fprintf(stderr, "no manifest\n"); return 1; }
        char line[512];
        while (fgets(line, sizeof(line), mf)) {
            std::string t(line);
            while (!t.empty() && (t.back() == '\n' || t.back() == '\r')) t.pop_back();
            if (!t.empty()) files.push_back(t);
        }
        fclose(mf);
    }
    if (files.size() != (size_t)n_in) {
        fprintf(stderr, "manifest %zu vs %llu inputs\n", files.size(), (unsigned long long)n_in);
        return 1;
    }
    L.ins.resize(n_in);
    std::vector<Mapped> maps(n_in);
    size_t n_mapped = 0, n_copied = 0, mapped_bytes = 0;
    for (LiteRtParamIndex i = 0; i < n_in; i++) {
        maps[i] = map_file(dir + "/" + files[i]);

        // Wrap the MAPPING directly where the runtime allows it, so 344 MB of layer
        // weights stay clean file-backed pages instead of becoming anonymous heap.
        // Inputs 0 and 1 are the embedding and pos, rewritten every step, so they
        // need writable managed buffers.
        LiteRtTensor t = nullptr;
        LiteRtRankedTensorType tt;
        LiteRtTensorBufferRequirements reqs = nullptr;
        size_t want = maps[i].size;
        LiteRtGetSignatureInputTensorByIndex(L.sig, i, &t);
        LiteRtGetRankedTensorType(t, &tt);
        if (LiteRtGetCompiledModelInputBufferRequirements(L.cm, 0, i, &reqs) == kLiteRtStatusOk) {
            size_t req = 0;
            if (LiteRtGetTensorBufferRequirementsBufferSize(reqs, &req) == kLiteRtStatusOk)
                want = req;
        }
        // Writable, so NOT mappable read-only: inputs 0/1 are the embedding and pos
        // (rewritten every step), and the trailing 2*L are the KV caches, which are
        // aliased as OUTPUTS so the graph writes into them. Mapping those PROT_READ
        // segfaults on the first decode step.
        const LiteRtParamIndex n_layers_guess = (n_out - 1) / 2;
        const LiteRtParamIndex cache_first = n_in - 2 * n_layers_guess;
        const bool writable = (i <= 1) || (i >= cache_first);
        if (!writable && want <= maps[i].size &&
            LiteRtCreateTensorBufferFromHostMemory(&tt, (void*)maps[i].data(), want, nullptr,
                                                   &L.ins[i]) == kLiteRtStatusOk) {
            n_mapped++;
            mapped_bytes += want;
        } else {
            size_t need = 0;
            if (!L.make_buffer(env, true, i, &need, &L.ins[i])) {
                fprintf(stderr, "layer input %llu failed\n", (unsigned long long)i); return 1;
            }
            write_buf(L.ins[i], maps[i].data(), std::min(need, maps[i].size));
            n_copied++;
        }
    }
    L.outs.resize(n_out);
    for (LiteRtParamIndex i = 0; i < n_out; i++) {
        size_t need = 0;
        if (!L.make_buffer(env, false, i, &need, &L.outs[i])) {
            fprintf(stderr, "layer output %llu failed\n", (unsigned long long)i); return 1;
        }
    }
    // Alias each KV pair so the cache is updated in place across steps rather than
    // copied out and re-fed. This is what makes generation stateful at all.
    const LiteRtParamIndex n_layers = (n_out - 1) / 2;
    const LiteRtParamIndex cache_base = n_in - 2 * n_layers;
    for (LiteRtParamIndex i = 0; i < n_layers; i++) {
        LiteRtDestroyTensorBuffer(L.outs[1 + i]);
        L.outs[1 + i] = L.ins[cache_base + 2 * i];
        LiteRtDestroyTensorBuffer(L.outs[1 + n_layers + i]);
        L.outs[1 + n_layers + i] = L.ins[cache_base + 2 * i + 1];
    }
    printf("layer inputs: %zu mapped (%.0f MB file-backed), %zu copied\n",
           n_mapped, mapped_bytes / 1e6, n_copied);
    printf("layers: %llu inputs, %llu outputs, %llu layers, KV aliased\n",
           (unsigned long long)n_in, (unsigned long long)n_out,
           (unsigned long long)n_layers);

    // --- head graph ------------------------------------------------------------
    Graph H;
    if (LiteRtCreateModelFromFile(env, head_path.c_str(), &H.model) != kLiteRtStatusOk ||
        LiteRtCreateCompiledModel(env, H.model, make_opts(false), &H.cm) != kLiteRtStatusOk) {
        fprintf(stderr, "head: load/compile failed\n"); return 1;
    }
    LiteRtGetModelSignature(H.model, 0, &H.sig);
    H.ins.resize(1);
    H.outs.resize(1);
    size_t head_in_bytes = 0, logits_bytes = 0;
    if (!H.make_buffer(env, true, 0, &head_in_bytes, &H.ins[0]) ||
        !H.make_buffer(env, false, 0, &logits_bytes, &H.outs[0])) {
        fprintf(stderr, "head buffers failed\n"); return 1;
    }
    const int vocab = (int)(logits_bytes / sizeof(float));
    printf("head: vocab %d\n\n", vocab);

    // --- embedding table -------------------------------------------------------
    const Mapped embd = map_file(embd_path);
    std::vector<float> scratch(DIM + 2 * Q6K_BLOCK), emb(DIM);

    // --- generate --------------------------------------------------------------
    int token = start_id;
    std::vector<int> produced;
    // Per-phase, because the whole-loop number came out 2.5x the sum of the same
    // graphs benchmarked separately and guessing at the reason was not working.
    double t_embed = 0, t_layers = 0, t_head = 0, t_argmax = 0;
    const double t0 = now_s();
    for (int step = 0; step < n_tokens; step++) {
        double a = now_s();
        // VIBEASR_NO_INPUT_WRITE: run with stale inputs, to test whether locking an
        // input buffer for write makes the runtime re-process every bound buffer
        // (there are 282, holding 344 MB) rather than just the one written.
        static const bool no_write = getenv("VIBEASR_NO_INPUT_WRITE") != nullptr;
        q6k_embedding_row(embd.data(), token, DIM, scratch.data(), emb.data());
        if (!no_write) write_buf(L.ins[0], emb.data(), (size_t)DIM * sizeof(float));
        // VIBEASR_FIXED_POS pins the position, to separate "work that depends on
        // pos" from "work that depends on the cache having real values in it".
        static const char* fixed = getenv("VIBEASR_FIXED_POS");
        const int64_t pos = fixed ? atoll(fixed) : step;
        if (!no_write) write_buf(L.ins[1], &pos, sizeof(pos));
        t_embed += now_s() - a;

        a = now_s();
        L.run();
        t_layers += now_s() - a;

        // VIBEASR_NO_HEAD isolates the layer graph inside this same process, to
        // separate "two graphs alternating" from "something about this loop".
        static const bool no_head = getenv("VIBEASR_NO_HEAD") != nullptr;
        a = now_s();
        void* p = nullptr;
        if (!no_head) {
            LiteRtLockTensorBuffer(L.outs[0], &p, kLiteRtTensorBufferLockModeRead);
            write_buf(H.ins[0], p, (size_t)DIM * sizeof(float));
            LiteRtUnlockTensorBuffer(L.outs[0]);
            H.run();
        }
        t_head += now_s() - a;

        a = now_s();
        int best = 0;
        float bv = 0;
        if (!no_head) {
            LiteRtLockTensorBuffer(H.outs[0], &p, kLiteRtTensorBufferLockModeRead);
            const float* logits = (const float*)p;
            bv = logits[0];
            for (int i = 1; i < vocab; i++)
                if (logits[i] > bv) { bv = logits[i]; best = i; }
            LiteRtUnlockTensorBuffer(H.outs[0]);
        }

        t_argmax += now_s() - a;
        produced.push_back(best);
        printf("step %2d  pos %2d  token %6d -> %6d  (logit %.3f)\n", step, step, token, best, bv);
        token = best;
    }
    const double dt = now_s() - t0;

    printf("\n%d tokens in %.2f s  =  %.1f ms/token, %.2f tok/s\n",
           n_tokens, dt, dt * 1e3 / n_tokens, n_tokens / dt);
    printf("  embed %.1f  layers %.1f  head %.1f  argmax %.1f  (ms/token)\n",
           t_embed * 1e3 / n_tokens, t_layers * 1e3 / n_tokens,
           t_head * 1e3 / n_tokens, t_argmax * 1e3 / n_tokens);
    printf("ids:");
    for (int t : produced) printf(" %d", t);
    printf("\n");
    return 0;
}
