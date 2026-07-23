// Copyright 2026. Apache-2.0.
// Thin C++ MOSS-TD engine on the LiteRT C API (CompiledModel + TensorBuffer).
//
// The decoder KV cache lives in TensorBuffers that are passed as BOTH inputs
// and outputs of every prefill/decode invocation (buffer aliasing), so the
// cache never crosses the host boundary and exists exactly once.
//
// Front end (mel + tokenization) is precomputed host-side; this binary
// measures the engine: encoder -> fuse -> prefill -> greedy decode.
//
// Usage:
//   moss_td_engine --encoder E.tflite --embedder M.tflite --decoder D.tflite \
//     --mel mel.bin --toklens lens.bin --ids ids.bin --out tokens.bin \
//     [--max-new 5120]
//
// mel.bin: n_chunks * 80 * 3000 f32; lens.bin: n_chunks i32 (audio tokens per
// chunk); ids.bin: prompt token ids i32 (audio placeholders = 151671).

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_options.h"

#define CHECK_OK(expr)                                                    \
  do {                                                                    \
    LiteRtStatus s_ = (expr);                                             \
    if (s_ != kLiteRtStatusOk) {                                          \
      fprintf(stderr, "FATAL %s:%d: %s -> %d\n", __FILE__, __LINE__, #expr, \
              (int)s_);                                                   \
      exit(1);                                                            \
    }                                                                     \
  } while (0)

static const int32_t kAudioTokenId = 151671;
static const int32_t kEosTokenId = 151645;
static const int kHidden = 1024;

static double now_s() {
  using namespace std::chrono;
  return duration_cast<duration<double>>(
             steady_clock::now().time_since_epoch())
      .count();
}

static long read_status_kb(const char* key) {
  FILE* f = fopen("/proc/self/status", "r");
  if (!f) return -1;
  char line[256];
  long val = -1;
  size_t klen = strlen(key);
  while (fgets(line, sizeof(line), f)) {
    if (!strncmp(line, key, klen)) {
      val = atol(line + klen + 1);
      break;
    }
  }
  fclose(f);
  return val;
}

static void stage(const char* name) {
  fprintf(stderr, "[mem] %-40s VmRSS=%6.0f MB  VmHWM=%6.0f MB\n", name,
          read_status_kb("VmRSS:") / 1024.0, read_status_kb("VmHWM:") / 1024.0);
}

static std::vector<uint8_t> read_file(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", p.c_str());
    exit(1);
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(n);
  if (fread(buf.data(), 1, n, f) != (size_t)n) exit(1);
  fclose(f);
  return buf;
}

struct SigIO {
  LiteRtParamIndex index = 0;
  std::vector<std::string> in_names, out_names;
  std::vector<LiteRtTensorBuffer> in, out;   // not owned if aliased
};

struct KvStore {
  std::map<std::string, LiteRtTensorBuffer> bufs;  // canonical KV buffers
  size_t elem_bytes = 4;
  size_t kv_bytes_total = 0;
};

static bool is_kv_name(const std::string& n, std::string* key) {
  // match ...kv_k_<i> / ...kv_v_<i> / ...kv_cache_k_<i> variants
  size_t p = n.rfind("kv_");
  if (p == std::string::npos) return false;
  *key = n.substr(p);
  return true;
}

class Component {
 public:
  Component(LiteRtEnvironment env, const std::string& path, KvStore* kv)
      : env_(env), kv_(kv) {
    CHECK_OK(LiteRtCreateModelFromFile(env, path.c_str(), &model_));
    LiteRtOptions opts;
    CHECK_OK(LiteRtCreateOptions(&opts));
    CHECK_OK(LiteRtSetOptionsHardwareAccelerators(
        opts, kLiteRtHwAcceleratorCpu));
    CHECK_OK(LiteRtCreateCompiledModel(env, model_, opts, &cm_));
    LiteRtParamIndex nsigs = 0;
    CHECK_OK(LiteRtGetNumModelSignatures(model_, &nsigs));
    for (LiteRtParamIndex si = 0; si < nsigs; ++si) {
      LiteRtSignature sig;
      CHECK_OK(LiteRtGetModelSignature(model_, si, &sig));
      const char* key_c = nullptr;
      CHECK_OK(LiteRtGetSignatureKey(sig, &key_c));
      SigIO io;
      io.index = si;
      LiteRtParamIndex nin = 0, nout = 0;
      CHECK_OK(LiteRtGetNumSignatureInputs(sig, &nin));
      CHECK_OK(LiteRtGetNumSignatureOutputs(sig, &nout));
      for (LiteRtParamIndex i = 0; i < nin; ++i) {
        const char* nm = nullptr;
        CHECK_OK(LiteRtGetSignatureInputName(sig, i, &nm));
        io.in_names.push_back(nm);
        io.in.push_back(make_buffer(sig, si, i, /*is_input=*/true, nm));
      }
      for (LiteRtParamIndex i = 0; i < nout; ++i) {
        const char* nm = nullptr;
        CHECK_OK(LiteRtGetSignatureOutputName(sig, i, &nm));
        io.out_names.push_back(nm);
        io.out.push_back(make_buffer(sig, si, i, /*is_input=*/false, nm));
      }
      sigs_[key_c] = io;
    }
  }

  ~Component() {
    // Destroy owned (non-aliased) buffers.
    for (auto b : owned_) LiteRtDestroyTensorBuffer(b);
    if (cm_) LiteRtDestroyCompiledModel(cm_);
    if (model_) LiteRtDestroyModel(model_);
  }

  SigIO& sig(const std::string& name) {
    auto it = sigs_.find(name);
    if (it == sigs_.end()) {
      fprintf(stderr, "signature %s not found\n", name.c_str());
      exit(1);
    }
    return it->second;
  }
  const std::map<std::string, SigIO>& sigs() const { return sigs_; }

  void run(SigIO& io) {
    CHECK_OK(LiteRtRunCompiledModel(cm_, io.index, io.in.size(),
                                    io.in.data(), io.out.size(),
                                    io.out.data()));
  }

  static void write_buf(LiteRtTensorBuffer b, const void* src, size_t bytes) {
    void* host = nullptr;
    CHECK_OK(LiteRtLockTensorBuffer(b, &host, kLiteRtTensorBufferLockModeWrite));
    memcpy(host, src, bytes);
    CHECK_OK(LiteRtUnlockTensorBuffer(b));
  }
  static void read_buf(LiteRtTensorBuffer b, void* dst, size_t bytes) {
    void* host = nullptr;
    CHECK_OK(LiteRtLockTensorBuffer(b, &host, kLiteRtTensorBufferLockModeRead));
    memcpy(dst, host, bytes);
    CHECK_OK(LiteRtUnlockTensorBuffer(b));
  }
  static void zero_buf(LiteRtTensorBuffer b, size_t bytes) {
    void* host = nullptr;
    CHECK_OK(LiteRtLockTensorBuffer(b, &host, kLiteRtTensorBufferLockModeWrite));
    memset(host, 0, bytes);
    CHECK_OK(LiteRtUnlockTensorBuffer(b));
  }

 private:
  LiteRtTensorBuffer make_buffer(LiteRtSignature sig, LiteRtParamIndex si,
                                 LiteRtParamIndex ti, bool is_input,
                                 const std::string& name) {
    std::string kvkey;
    const bool kv = kv_ && is_kv_name(name, &kvkey);
    if (kv) {
      auto it = kv_->bufs.find(kvkey);
      if (it != kv_->bufs.end()) return it->second;  // alias existing
    }
    LiteRtTensor tensor;
    if (is_input) {
      CHECK_OK(LiteRtGetSignatureInputTensorByIndex(sig, ti, &tensor));
    } else {
      CHECK_OK(LiteRtGetSignatureOutputTensorByIndex(sig, ti, &tensor));
    }
    LiteRtRankedTensorType tt;
    CHECK_OK(LiteRtGetRankedTensorType(tensor, &tt));
    LiteRtTensorBufferRequirements reqs;
    if (is_input) {
      CHECK_OK(LiteRtGetCompiledModelInputBufferRequirements(cm_, si, ti,
                                                             &reqs));
    } else {
      CHECK_OK(LiteRtGetCompiledModelOutputBufferRequirements(cm_, si, ti,
                                                              &reqs));
    }
    size_t bytes = 0;
    CHECK_OK(LiteRtGetTensorBufferRequirementsBufferSize(reqs, &bytes));
    LiteRtTensorBuffer buf;
    CHECK_OK(LiteRtCreateManagedTensorBuffer(
        env_, kLiteRtTensorBufferTypeHostMemory, &tt, bytes, &buf));
    owned_.push_back(buf);
    if (kv) {
      kv_->bufs[kvkey] = buf;
      kv_->kv_bytes_total += bytes;
      zero_buf(buf, bytes);
    }
    return buf;
  }

  LiteRtEnvironment env_;
  KvStore* kv_;
  LiteRtModel model_ = nullptr;
  LiteRtCompiledModel cm_ = nullptr;
  std::map<std::string, SigIO> sigs_;
  std::vector<LiteRtTensorBuffer> owned_;
};

static int find_name(const std::vector<std::string>& names,
                     const char* needle) {
  for (size_t i = 0; i < names.size(); ++i)
    if (names[i].find(needle) != std::string::npos) return (int)i;
  fprintf(stderr, "input %s not found\n", needle);
  exit(1);
}

int main(int argc, char** argv) {
  std::string enc_p, emb_p, dec_p, mel_p, lens_p, ids_p, out_p;
  int max_new = 5120;
  for (int i = 1; i < argc - 1; ++i) {
    std::string a = argv[i];
    if (a == "--encoder") enc_p = argv[++i];
    else if (a == "--embedder") emb_p = argv[++i];
    else if (a == "--decoder") dec_p = argv[++i];
    else if (a == "--mel") mel_p = argv[++i];
    else if (a == "--toklens") lens_p = argv[++i];
    else if (a == "--ids") ids_p = argv[++i];
    else if (a == "--out") out_p = argv[++i];
    else if (a == "--max-new") max_new = atoi(argv[++i]);
  }
  stage("baseline");

  LiteRtEnvironment env;
  CHECK_OK(LiteRtCreateEnvironment(0, nullptr, &env));

  // ---- inputs ----
  auto mel_raw = read_file(mel_p);
  auto lens_raw = read_file(lens_p);
  auto ids_raw = read_file(ids_p);
  const int n_chunks = lens_raw.size() / 4;
  const int32_t* tok_lens = (const int32_t*)lens_raw.data();
  const int S = ids_raw.size() / 4;
  const int32_t* ids = (const int32_t*)ids_raw.data();
  int n_audio = 0;
  for (int i = 0; i < n_chunks; ++i) n_audio += tok_lens[i];

  // ---- encoder (scoped: freed before decode) ----
  std::vector<float> audio_embeds((size_t)n_audio * kHidden);
  double t0 = now_s();
  {
    Component enc(env, enc_p, nullptr);
    auto& io = enc.sigs().begin()->first == "" ? enc.sig("")
               : const_cast<SigIO&>(enc.sigs().begin()->second);
    stage("encoder compiled");
    size_t mel_bytes = (size_t)80 * 3000 * 4;
    std::vector<float> chunk_out((size_t)375 * kHidden);
    int off = 0;
    for (int c = 0; c < n_chunks; ++c) {
      Component::write_buf(io.in[0], mel_raw.data() + (size_t)c * mel_bytes,
                           mel_bytes);
      enc.run(io);
      Component::read_buf(io.out[0], chunk_out.data(),
                          chunk_out.size() * 4);
      memcpy(audio_embeds.data() + (size_t)off * kHidden, chunk_out.data(),
             (size_t)tok_lens[c] * kHidden * 4);
      off += tok_lens[c];
    }
    stage("encoder done (pre-free)");
  }
  double enc_s = now_s() - t0;
  stage("encoder freed");

  // ---- embedder + decoder ----
  Component emb(env, emb_p, nullptr);
  KvStore kv;
  Component dec(env, dec_p, &kv);
  stage("embedder+decoder compiled (KV zeroed)");
  fprintf(stderr, "[kv] canonical KV bytes: %.0f MB (%zu tensors)\n",
          kv.kv_bytes_total / 1e6, kv.bufs.size());

  // embed signatures inventory
  std::map<int, std::string> embed_sizes;  // n -> signature name
  for (auto& kvp : emb.sigs()) {
    if (kvp.first.rfind("embed_", 0) == 0)
      embed_sizes[atoi(kvp.first.c_str() + 6)] = kvp.first;
  }
  auto embed_tokens = [&](const int32_t* toks, int n, float* dst) {
    int i = 0;
    while (i < n) {
      int pick = -1;
      for (auto it = embed_sizes.rbegin(); it != embed_sizes.rend(); ++it)
        if (it->first <= n - i) { pick = it->first; break; }
      if (pick < 0) pick = embed_sizes.begin()->first;
      auto& sio = emb.sig(embed_sizes[pick]);
      std::vector<int32_t> padbuf(pick, 0);
      int real = n - i < pick ? n - i : pick;
      memcpy(padbuf.data(), toks + i, (size_t)real * 4);
      Component::write_buf(sio.in[0], padbuf.data(), (size_t)pick * 4);
      emb.run(sio);
      std::vector<float> outv((size_t)pick * kHidden);
      Component::read_buf(sio.out[0], outv.data(), outv.size() * 4);
      memcpy(dst + (size_t)i * kHidden, outv.data(),
             (size_t)real * kHidden * 4);
      i += real;
    }
  };

  // fused prompt embeddings
  std::vector<float> fused((size_t)S * kHidden);
  embed_tokens(ids, S, fused.data());
  {
    int k = 0;
    for (int p = 0; p < S; ++p)
      if (ids[p] == kAudioTokenId)
        memcpy(fused.data() + (size_t)p * kHidden,
               audio_embeds.data() + (size_t)k++ * kHidden, kHidden * 4);
    if (k != n_audio) { fprintf(stderr, "audio scatter mismatch\n"); exit(1); }
  }
  std::vector<float>().swap(audio_embeds);
  stage("fused embeds ready");

  // decoder signatures
  std::map<int, std::string> prefills;
  for (auto& kvp : dec.sigs())
    if (kvp.first.rfind("prefill_", 0) == 0)
      prefills[atoi(kvp.first.c_str() + 8)] = kvp.first;
  auto& dsig = dec.sig("decode");
  const int d_e = find_name(dsig.in_names, "input_embeds");
  const int d_p = find_name(dsig.in_names, "input_pos");
  const int d_m = find_name(dsig.in_names, "mask");
  const int d_h = find_name(dsig.out_names, "hidden");
  // kv_len from decode mask buffer size
  size_t mask_bytes1 = 0;
  {
    LiteRtRankedTensorType tt;
    CHECK_OK(LiteRtGetTensorBufferTensorType(dsig.in[d_m], &tt));
    mask_bytes1 = 4;
    int kvlen = tt.layout.dimensions[tt.layout.rank - 1];
    fprintf(stderr, "[dec] kv_len=%d prefill sizes:", kvlen);
    for (auto& p : prefills) fprintf(stderr, " %d", p.first);
    fprintf(stderr, "\n");
    mask_bytes1 = (size_t)kvlen * 4;
  }
  const int kv_len = mask_bytes1 / 4;

  // ---- prefill ----
  t0 = now_s();
  std::vector<float> last_hidden(kHidden);
  {
    int pos = 0;
    std::vector<float> mask;
    while (pos < S) {
      int pick = -1;
      for (auto it = prefills.rbegin(); it != prefills.rend(); ++it)
        if (it->first <= S - pos) { pick = it->first; break; }
      if (pick < 0) pick = prefills.begin()->first;
      auto& sio = dec.sig(prefills[pick]);
      const int e_i = find_name(sio.in_names, "input_embeds");
      const int p_i = find_name(sio.in_names, "input_pos");
      const int m_i = find_name(sio.in_names, "mask");
      const int h_o = find_name(sio.out_names, "hidden");
      int real = S - pos < pick ? S - pos : pick;
      std::vector<float> embn((size_t)pick * kHidden, 0.f);
      memcpy(embn.data(), fused.data() + (size_t)pos * kHidden,
             (size_t)real * kHidden * 4);
      Component::write_buf(sio.in[e_i], embn.data(), embn.size() * 4);
      std::vector<int32_t> ipos(pick);
      for (int r = 0; r < pick; ++r) ipos[r] = pos + r;
      Component::write_buf(sio.in[p_i], ipos.data(), ipos.size() * 4);
      mask.assign((size_t)pick * kv_len, -INFINITY);
      for (int r = 0; r < pick; ++r)
        for (int c2 = 0; c2 <= pos + r && c2 < kv_len; ++c2)
          mask[(size_t)r * kv_len + c2] = 0.f;
      Component::write_buf(sio.in[m_i], mask.data(), mask.size() * 4);
      dec.run(sio);
      std::vector<float> hid((size_t)pick * kHidden);
      Component::read_buf(sio.out[h_o], hid.data(), hid.size() * 4);
      memcpy(last_hidden.data(), hid.data() + (size_t)(real - 1) * kHidden,
             kHidden * 4);
      pos += real;
    }
  }
  double prefill_s = now_s() - t0;
  stage("prefill done");

  // ---- greedy decode ----
  auto& lsig = emb.sig("logits");
  size_t vocab_bytes = 0;
  {
    LiteRtRankedTensorType tt;
    CHECK_OK(LiteRtGetTensorBufferTensorType(lsig.out[0], &tt));
    vocab_bytes = 4;
    for (unsigned r = 0; r < tt.layout.rank; ++r)
      vocab_bytes *= tt.layout.dimensions[r];
  }
  const int vocab = vocab_bytes / 4;
  std::vector<float> logits(vocab);
  Component::write_buf(lsig.in[0], last_hidden.data(), kHidden * 4);
  emb.run(lsig);
  Component::read_buf(lsig.out[0], logits.data(), vocab_bytes);

  std::vector<int32_t> new_ids;
  std::vector<float> mask1((size_t)kv_len, -INFINITY);
  std::vector<float> e1(kHidden), h1(kHidden);
  t0 = now_s();
  int p = S;
  while ((int)new_ids.size() < max_new) {
    int best = 0;
    for (int i = 1; i < vocab; ++i)
      if (logits[i] > logits[best]) best = i;
    new_ids.push_back(best);
    if (best == kEosTokenId) break;
    if (p + 1 > kv_len) break;
    int32_t tk = best;
    embed_tokens(&tk, 1, e1.data());
    Component::write_buf(dsig.in[d_e], e1.data(), kHidden * 4);
    int32_t pp = p;
    Component::write_buf(dsig.in[d_p], &pp, 4);
    for (int c2 = 0; c2 < kv_len; ++c2)
      mask1[c2] = c2 <= p ? 0.f : -INFINITY;
    Component::write_buf(dsig.in[d_m], mask1.data(), mask1.size() * 4);
    dec.run(dsig);
    Component::read_buf(dsig.out[d_h], h1.data(), kHidden * 4);
    Component::write_buf(lsig.in[0], h1.data(), kHidden * 4);
    emb.run(lsig);
    Component::read_buf(lsig.out[0], logits.data(), vocab_bytes);
    ++p;
  }
  double dec_s = now_s() - t0;
  stage("decode done");

  FILE* f = fopen(out_p.c_str(), "wb");
  fwrite(new_ids.data(), 4, new_ids.size(), f);
  fclose(f);
  fprintf(stderr,
          "STATS encoder_s=%.2f prefill_s=%.2f prompt_tokens=%d "
          "decode_s=%.2f new_tokens=%zu tok_per_s=%.2f peak_rss_mb=%.0f\n",
          enc_s, prefill_s, S, dec_s, new_ids.size(),
          new_ids.size() / dec_s, read_status_kb("VmHWM:") / 1024.0);
  return 0;
}
