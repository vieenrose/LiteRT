// Minimal LiteRT model bench: compile with the app's own runtime (2.1.6),
// zero-fill inputs, time run() on the first signature.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "moss_lite_engine.h"
#include "litert/c/litert_common.h"

using mosslite::Component;

int main(int argc, char** argv) {
  if (argc < 3) { printf("usage: bench_lite <model> <threads> [runs]\n"); return 1; }
  const int threads = atoi(argv[2]);
  const int runs = argc > 3 ? atoi(argv[3]) : 12;
  LiteRtEnvironment env = nullptr;
  if (LiteRtCreateEnvironment(0, nullptr, &env) != kLiteRtStatusOk) return 2;
  auto t0 = std::chrono::steady_clock::now();
  Component c(env, argv[1], nullptr, threads, "");
  if (!c.ok() || c.sigs().empty()) { printf("compile FAILED\n"); return 3; }
  auto t1 = std::chrono::steady_clock::now();
  auto& io = const_cast<mosslite::SigIO&>(c.sigs().begin()->second);
  for (size_t i = 0; i < io.in.size(); ++i) {
    size_t b = Component::buf_bytes(io.in[i]);
    std::vector<char> z(b, 0);
    Component::write_buf(io.in[i], z.data(), b);
  }
  c.run(io);  // warmup
  auto t2 = std::chrono::steady_clock::now();
  for (int i = 0; i < runs; ++i) c.run(io);
  auto t3 = std::chrono::steady_clock::now();
  printf("compile=%.1fs warmup=%.0fms avg_invoke=%.1fms (%d runs, %d threads)\n",
         std::chrono::duration<double>(t1 - t0).count(),
         std::chrono::duration<double, std::milli>(t2 - t1).count(),
         std::chrono::duration<double, std::milli>(t3 - t2).count() / runs,
         runs, threads);
  return 0;
}
