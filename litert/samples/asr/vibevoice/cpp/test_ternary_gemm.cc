// Correctness and speed for the ternary GEMM.
//
// Correctness is checked against a plain-C reference AND against a float matmul
// over the unpacked {-1,0,+1} weights, so a packing bug and a kernel bug cannot
// cancel out. The only expected difference from the float result is activation
// quantization, which is bounded.
//
//   ./test_ternary_gemm [k] [n_rows] [m]

#include "ternary_gemm.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static uint32_t rng_state = 12345;
static uint32_t rnd() {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return rng_state;
}

int main(int argc, char** argv) {
    const int k      = argc > 1 ? atoi(argv[1]) : 1536;
    const int n_rows = argc > 2 ? atoi(argv[2]) : 1536;
    const int m      = argc > 3 ? atoi(argv[3]) : 1;

    printf("ternary_gemm impl: %s\n", ternary_gemm_impl_name());
    printf("shape: [%d x %d] weights, %d activation row(s)\n\n", n_rows, k, m);

    // Ternary weights, roughly a third each of -1/0/+1.
    std::vector<int8_t> w((size_t)n_rows * k);
    for (auto& v : w) v = (int8_t)((int)(rnd() % 3) - 1);

    std::vector<uint8_t> packed(ternary_packed_bytes(n_rows, k));
    ternary_pack(w.data(), n_rows, k, packed.data());

    // Packing must round-trip exactly, or nothing below means anything.
    std::vector<int8_t> back((size_t)n_rows * k);
    ternary_unpack(packed.data(), n_rows, k, back.data());
    if (memcmp(w.data(), back.data(), w.size()) != 0) {
        printf("FAIL: pack/unpack does not round-trip\n");
        return 1;
    }
    printf("pack/unpack round-trip: OK (%zu bytes for %zu weights, %.2f bits each)\n",
           packed.size(), w.size(), 8.0 * packed.size() / w.size());

    std::vector<float> x((size_t)m * k);
    for (auto& v : x) v = ((float)(rnd() % 2001) - 1000.0f) / 1000.0f;

    std::vector<int8_t> q((size_t)m * k);
    std::vector<float> xs(m);
    ternary_quantize_activations(x.data(), m, k, q.data(), xs.data());

    std::vector<float> wscale(n_rows);
    for (int r = 0; r < n_rows; r++) wscale[r] = 0.01f + (float)(r % 7) * 0.001f;

    std::vector<float> y_ref((size_t)m * n_rows), y_simd((size_t)m * n_rows);
    ternary_gemm_reference(packed.data(), n_rows, k, q.data(), xs.data(), m,
                           wscale.data(), 1, nullptr, y_ref.data());
    ternary_gemm(packed.data(), n_rows, k, q.data(), xs.data(), m,
                 wscale.data(), 1, nullptr, y_simd.data());

    // 1) SIMD must match the reference EXACTLY — both are integer accumulations
    //    of the same products, so any difference is a real bug, not rounding.
    double worst_simd = 0;
    for (size_t i = 0; i < y_ref.size(); i++)
        worst_simd = fmax(worst_simd, fabs((double)y_ref[i] - y_simd[i]));
    printf("simd vs reference: max abs diff %.3e %s\n", worst_simd,
           worst_simd == 0.0 ? "(exact)" : "<-- MISMATCH");

    // 2) Both must approximate a float matmul over the unpacked weights. This is
    //    what catches a packing/order bug that the reference would share.
    // Normalise by the RMS of the exact result, not per element: outputs of a
    // ternary matmul are a mean-zero random walk, so individual entries land
    // arbitrarily close to zero and a per-element relative error is meaningless
    // there. Error RMS over signal RMS is the honest quantization measure.
    double sum_sq_err = 0, sum_sq_ref = 0, worst_abs = 0;
    for (int i = 0; i < m; i++) {
        for (int r = 0; r < n_rows; r++) {
            double acc = 0;
            for (int j = 0; j < k; j++) acc += (double)back[(size_t)r * k + j] * x[(size_t)i * k + j];
            acc *= wscale[r];
            const double d = acc - y_ref[(size_t)i * n_rows + r];
            sum_sq_err += d * d; sum_sq_ref += acc * acc;
            worst_abs = fmax(worst_abs, fabs(d));
        }
    }
    const double worst_rel = sqrt(sum_sq_err / (sum_sq_ref > 0 ? sum_sq_ref : 1));
    printf("vs float matmul:   err_rms/sig_rms %.5f, max abs %.4f "
           "(activation quantization only)\n", worst_rel, worst_abs);

    const bool pass = worst_simd == 0.0 && worst_rel < 0.05;

    // Speed, reported as decoder-equivalent bandwidth: batch-1 decode reads the
    // whole weight matrix per token, so MB/s here is the number that matters.
    const int iters = m == 1 ? 50 : 5;
    ternary_gemm(packed.data(), n_rows, k, q.data(), xs.data(), m,
                 wscale.data(), 1, nullptr, y_simd.data());   // warm
    const double t0 = now_s();
    for (int it = 0; it < iters; it++)
        ternary_gemm(packed.data(), n_rows, k, q.data(), xs.data(), m,
                     wscale.data(), 1, nullptr, y_simd.data());
    const double each = (now_s() - t0) / iters;
    printf("\nspeed: %.3f ms/call, %.2f GB/s over packed weights (%.1f MB)\n",
           each * 1e3, packed.size() / each / 1e9, packed.size() / 1e6);
    printf("       int8 equivalent would move %.1f MB -> %.3f ms at the same GB/s\n",
           w.size() / 1e6, w.size() / (packed.size() / each) * 1e3);

    printf("\n%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
