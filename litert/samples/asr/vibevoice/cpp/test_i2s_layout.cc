// Is the shipped I2_S weight layout the same sequential 4-per-byte packing that
// ternary_gemm uses?
//
// The code MAPPING is settled (u = w+1 over {0,1,2}, established from the byte
// histogram — see utils/gguf_read.py). The element ORDER is not: a histogram cannot
// distinguish orderings, and ggml's NEON kernel permutes ACTIVATIONS to match its
// weights (I2S_Y_BASE), which is the same strategy ternary_gemm uses but not
// necessarily the same permutation.
//
// Getting this wrong would not crash anything — it would silently produce a decoder
// that emits plausible-looking nonsense. So ggml is used as the oracle: run the same
// matvec through ggml's own I2_S path and through ternary_gemm over the same bytes,
// and compare.
//
// Takes a RAW dump of one I2_S tensor rather than the gguf: ggml's own
// gguf_init_from_file cannot size the custom I2_S type and fails with "failed to
// read tensor data". utils/gguf_read.py writes the dump.
//
//   ./test_i2s_layout <ggml.bin> <K> <N> [--probe | <transcoded.bin>]
//
// One file cannot serve both sides: ggml reads its own bit-plane layout, so handing
// it a re-packed file just makes it misread. Pass the ORIGINAL for ggml and the
// TRANSCODED one for ternary_gemm to check a transcoder.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cstdlib>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ternary_gemm.h"

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <i2s_tensor.bin> <K> <N>\n", argv[0]); return 1; }
    const char* path = argv[1];
    const int64_t K = atoll(argv[2]), N = atoll(argv[3]);

    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "could not open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END);
    const size_t file_bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> blob(file_bytes);
    if (fread(blob.data(), 1, file_bytes, f) != file_bytes) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);
    printf("%s: %zu bytes, K=%lld N=%lld (expect %lld + 32)\n", path, file_bytes,
           (long long)K, (long long)N, (long long)(K * N / 4));

    // Wrap the bytes as an I2_S ggml tensor by hand.
    struct ggml_init_params wip = { ggml_tensor_overhead() * 4, nullptr, /*no_alloc=*/true };
    struct ggml_context* ctx = ggml_init(wip);
    struct ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_I2_S, K, N);
    w->data = blob.data();

    // Deterministic activations, small enough that int8 quantization is faithful.
    std::vector<float> x((size_t)K);
    for (int64_t i = 0; i < K; i++) x[i] = sinf((float)i * 0.017f) * 0.5f;

    // --probe: feed one-hot vectors. With x = e_j, ggml returns column j of the
    // matrix AS IT SEES IT. Comparing that against every column of our unpacking
    // reveals the index permutation directly, instead of guessing at bit orders.
    const bool probe = (argc > 4 && std::string(argv[4]) == "--probe");
    // Optional second blob, already in ternary_gemm's sequential layout.
    std::vector<uint8_t> mine_blob;
    if (argc > 4 && !probe) {
        FILE* mf = fopen(argv[4], "rb");
        if (!mf) { fprintf(stderr, "could not open %s\n", argv[4]); return 1; }
        fseek(mf, 0, SEEK_END); const size_t mb = ftell(mf); fseek(mf, 0, SEEK_SET);
        mine_blob.resize(mb);
        if (fread(mine_blob.data(), 1, mb, mf) != mb) { fprintf(stderr, "short read\n"); return 1; }
        fclose(mf);
        printf("transcoded weights: %s (%zu bytes)\n", argv[4], mb);
    }

    // --- ggml's own I2_S matvec -------------------------------------------------
    // (built once, re-run per input below)
    ggml_backend_t backend = ggml_backend_cpu_init();
    const size_t meta = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    struct ggml_init_params ip = { meta, nullptr, /*no_alloc=*/true };
    struct ggml_context* gctx = ggml_init(ip);
    struct ggml_tensor* xin = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, K, 1);
    ggml_set_input(xin);
    struct ggml_tensor* y = ggml_mul_mat(gctx, w, xin);
    ggml_set_output(y);
    struct ggml_cgraph* gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) { fprintf(stderr, "alloc failed\n"); return 1; }
    memcpy(xin->data, x.data(), x.size() * sizeof(float));
    ggml_backend_cpu_set_n_threads(backend, 4);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml compute failed\n"); return 1;
    }
    std::vector<float> y_ggml((size_t)N);
    memcpy(y_ggml.data(), y->data, y_ggml.size() * sizeof(float));

    if (probe) {
        // Our unpacking, as a dense [N,K] matrix of {-1,0,+1}.
        std::vector<int8_t> mine((size_t)N * K);
        ternary_unpack(blob.data(), (int)N, (int)K, mine.data());
        printf("\nprobe: for x = e_j, which of OUR columns matches ggml's answer?\n");
        for (int64_t j : {0, 1, 2, 3, 4, 8, 16, 64}) {
            if (j >= K) break;
            std::vector<float> e((size_t)K, 0.0f);
            e[j] = 1.0f;
            memcpy(xin->data, e.data(), e.size() * sizeof(float));
            ggml_backend_graph_compute(backend, gf);
            const float* got = (const float*)y->data;
            // Sign pattern of ggml's column, as ternary.
            int best = -1; double best_score = -1;
            for (int64_t c = 0; c < K; c++) {
                double d = 0, na2 = 0, nb2 = 0;
                for (int64_t r = 0; r < N; r++) {
                    const double a = got[r], b = mine[(size_t)r * K + c];
                    d += a * b; na2 += a * a; nb2 += b * b;
                }
                const double sc = d / (sqrt(na2) * sqrt(nb2) + 1e-30);
                if (sc > best_score) { best_score = sc; best = (int)c; }
            }
            printf("  ggml e_%-4lld  <-> our column %-5d  (cos %.4f)\n",
                   (long long)j, best, best_score);
        }
        return 0;
    }

    // --- ternary_gemm over the SAME bytes, assuming sequential 4-per-byte -------
    // ggml stores nelements/4 packed bytes then 32 bytes whose first float is the
    // per-tensor scale.
    const uint8_t* packed = mine_blob.empty() ? (const uint8_t*)w->data : mine_blob.data();
    const size_t packed_bytes = (size_t)(K * N) / 4;
    const float scale = *(const float*)(packed + packed_bytes);
    printf("scale from tail: %.6g\n", scale);

    std::vector<int8_t> q((size_t)K);
    std::vector<float> xs(1), y_mine((size_t)N);
    ternary_quantize_activations(x.data(), 1, (int)K, q.data(), xs.data());
    ternary_gemm(packed, (int)N, (int)K, q.data(), xs.data(), 1,
                 &scale, /*per_row=*/0, nullptr, y_mine.data());

    // --- compare ----------------------------------------------------------------
    double dot = 0, na = 0, nb = 0, worst = 0;
    for (int64_t i = 0; i < N; i++) {
        dot += (double)y_ggml[i] * y_mine[i];
        na += (double)y_ggml[i] * y_ggml[i];
        nb += (double)y_mine[i] * y_mine[i];
        worst = fmax(worst, fabs((double)y_ggml[i] - y_mine[i]));
    }
    const double cos = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    printf("ggml  rms=%.6g\nmine  rms=%.6g\ncosine=%.6f  max_abs_diff=%.6g\n",
           sqrt(na / N), sqrt(nb / N), cos, worst);

    // Activation quantization differs between the two, so exact equality is not
    // expected; a matching ORDER shows up as cosine ~1. A wrong order gives ~0,
    // because the products land on unrelated weights.
    const bool pass = cos > 0.99;
    printf("\n%s\n", pass
        ? (mine_blob.empty() ? "PASS - sequential layout matches ggml's I2_S order"
                             : "PASS - transcoded weights match ggml")
        : (mine_blob.empty() ? "FAIL - layouts differ, weights need transcoding"
                             : "FAIL - transcoding is wrong"));
    return pass ? 0 : 1;
}
