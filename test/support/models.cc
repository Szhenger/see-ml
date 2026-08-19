#include <cstring>
#include <random>

#include "test/support/builders.h"

// =============================================================================
// models/ discipline of the shared fixtures: deterministic SMF model
// builders and the suites' default compiler configuration. Everything is
// seeded — two calls with the same arguments produce identical bytes, which
// the determinism tests rely on.
// =============================================================================

namespace seeml::testing {

using update::SmfModel;
using update::SmfOpKind;
using update::UpdateConfig;

std::vector<uint8_t> AsBytes(const std::vector<float>& v) {
  std::vector<uint8_t> b(v.size() * sizeof(float));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

std::vector<float> RandnVector(size_t n, uint64_t seed, float stddev) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, stddev);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

SmfModel MakeMlp(int64_t in_dim, int64_t hidden, int64_t out_dim,
                 uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.5f);
  auto randv = [&](size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
  };

  SmfModel m;
  m.input_name = "x";
  m.output_name = "logits";
  m.tensors.push_back({.name = "x", .dims = {-1, in_dim}, .is_const = false});
  m.tensors.push_back({.name = "w1",
                       .dims = {in_dim, hidden},
                       .is_const = true,
                       .data = AsBytes(randv(in_dim * hidden))});
  m.tensors.push_back({.name = "b1",
                       .dims = {hidden},
                       .is_const = true,
                       .data = AsBytes(randv(hidden))});
  m.tensors.push_back({.name = "w2",
                       .dims = {hidden, out_dim},
                       .is_const = true,
                       .data = AsBytes(randv(hidden * out_dim))});
  m.tensors.push_back({.name = "b2",
                       .dims = {out_dim},
                       .is_const = true,
                       .data = AsBytes(randv(out_dim))});
  for (auto& t : m.tensors)
    if (t.is_const) t.byte_size = t.data.size();

  m.ops.push_back({SmfOpKind::kMatMul, "mm1", {"x", "w1"}, "z1"});
  m.ops.push_back({SmfOpKind::kAddBias, "ab1", {"z1", "b1"}, "z1b"});
  m.ops.push_back({SmfOpKind::kRelu, "relu1", {"z1b"}, "h1"});
  m.ops.push_back({SmfOpKind::kMatMul, "mm2", {"h1", "w2"}, "z2"});
  m.ops.push_back({SmfOpKind::kAddBias, "ab2", {"z2", "b2"}, "logits"});
  return m;
}

SmfModel MakeTiedMlp(int64_t dim, uint64_t seed) {
  SmfModel m;
  m.input_name = "x";
  m.output_name = "y";
  m.tensors.push_back({.name = "x", .dims = {-1, dim}, .is_const = false});
  m.tensors.push_back(
      {.name = "w",
       .dims = {dim, dim},
       .is_const = true,
       .data = AsBytes(RandnVector(static_cast<size_t>(dim * dim), seed))});
  m.tensors.back().byte_size = m.tensors.back().data.size();

  m.ops.push_back({SmfOpKind::kMatMul, "mm1", {"x", "w"}, "z"});
  m.ops.push_back({SmfOpKind::kRelu, "relu1", {"z"}, "h"});
  m.ops.push_back({SmfOpKind::kMatMul, "mm2", {"h", "w"}, "y"});
  return m;
}

SmfModel MakeGatedNet(int64_t in_dim, int64_t hidden, int64_t out_dim,
                      uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.5f);
  auto randv = [&](size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
  };

  SmfModel m;
  m.input_name = "x";
  m.output_name = "logits";
  m.tensors.push_back({.name = "x", .dims = {-1, in_dim}, .is_const = false});
  auto add_const = [&](const char* name, std::vector<int64_t> dims,
                       std::vector<float> data) {
    m.tensors.push_back({.name = name,
                         .dims = std::move(dims),
                         .is_const = true,
                         .data = AsBytes(data)});
    m.tensors.back().byte_size = m.tensors.back().data.size();
  };
  add_const("w1", {in_dim, hidden}, randv(in_dim * hidden));
  add_const("b1", {hidden}, randv(hidden));
  add_const("w2", {in_dim, hidden}, randv(in_dim * hidden));
  add_const("b2", {hidden}, randv(hidden));
  std::vector<float> gamma(hidden, 1.0f), beta(hidden, 0.0f);
  for (auto& g : gamma) g += 0.1f * dist(rng);  // non-trivial affine
  add_const("gamma", {hidden}, gamma);
  add_const("beta", {hidden}, beta);
  add_const("w3", {hidden, out_dim}, randv(hidden * out_dim));
  add_const("b3", {out_dim}, randv(out_dim));

  m.ops.push_back({SmfOpKind::kMatMul, "mm1", {"x", "w1"}, "z1"});
  m.ops.push_back({SmfOpKind::kAddBias, "ab1", {"z1", "b1"}, "z1b"});
  m.ops.push_back({SmfOpKind::kGelu, "gelu1", {"z1b"}, "h1"});
  m.ops.push_back({SmfOpKind::kMatMul, "mm2", {"x", "w2"}, "z2"});
  m.ops.push_back({SmfOpKind::kAddBias, "ab2", {"z2", "b2"}, "z2b"});
  m.ops.push_back({SmfOpKind::kSilu, "silu1", {"z2b"}, "h2"});
  m.ops.push_back({SmfOpKind::kMul, "gate", {"h1", "h2"}, "g"});
  m.ops.push_back({SmfOpKind::kLayerNorm, "ln", {"g", "gamma", "beta"}, "n"});
  m.ops.push_back({SmfOpKind::kMatMul, "mm3", {"n", "w3"}, "z3"});
  m.ops.push_back({SmfOpKind::kAddBias, "ab3", {"z3", "b3"}, "logits"});
  return m;
}

SmfModel MakeMlpStack(int64_t in_dim, int64_t hidden, int64_t layers,
                      int64_t out_dim, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.5f);
  auto randv = [&](size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
  };

  SmfModel m;
  m.input_name = "x";
  m.output_name = "logits";
  m.tensors.push_back({.name = "x", .dims = {-1, in_dim}, .is_const = false});
  std::string prev = "x";
  int64_t prev_dim = in_dim;
  auto linear = [&](int idx, int64_t width, const std::string& out) {
    const std::string w = "w" + std::to_string(idx);
    const std::string b = "b" + std::to_string(idx);
    m.tensors.push_back({.name = w,
                         .dims = {prev_dim, width},
                         .is_const = true,
                         .data = AsBytes(randv(prev_dim * width))});
    m.tensors.push_back({.name = b,
                         .dims = {width},
                         .is_const = true,
                         .data = AsBytes(randv(width))});
    m.ops.push_back({SmfOpKind::kMatMul, "mm" + std::to_string(idx),
                     {prev, w}, "z" + std::to_string(idx)});
    m.ops.push_back({SmfOpKind::kAddBias, "ab" + std::to_string(idx),
                     {"z" + std::to_string(idx), b}, out});
    prev = out;
    prev_dim = width;
  };
  for (int i = 1; i <= layers; ++i) {
    linear(i, hidden, "zb" + std::to_string(i));
    m.ops.push_back({SmfOpKind::kRelu, "relu" + std::to_string(i),
                     {"zb" + std::to_string(i)}, "h" + std::to_string(i)});
    prev = "h" + std::to_string(i);
  }
  linear(static_cast<int>(layers) + 1, out_dim, "logits");
  for (auto& t : m.tensors)
    if (t.is_const) t.byte_size = t.data.size();
  return m;
}

SmfModel MakeTinyDecoder(int64_t dim, int64_t heads, int64_t seq, int64_t ffn,
                         int64_t vocab, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.5f);
  auto randv = [&](size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
  };

  SmfModel m;
  m.input_name = "x";
  m.output_name = "logits";
  m.seq_len = static_cast<uint64_t>(seq);
  m.tensors.push_back({.name = "x", .dims = {-1, dim}, .is_const = false});
  auto add_const = [&](const std::string& name, std::vector<int64_t> dims,
                       std::vector<float> data) {
    m.tensors.push_back({.name = name,
                         .dims = std::move(dims),
                         .is_const = true,
                         .data = AsBytes(data)});
    m.tensors.back().byte_size = m.tensors.back().data.size();
  };
  std::vector<float> g1(dim, 1.0f), g2(dim, 1.0f), gf(dim, 1.0f);
  for (auto* g : {&g1, &g2, &gf})
    for (auto& v : *g) v += 0.1f * dist(rng);  // non-trivial gains
  add_const("g1", {dim}, g1);
  add_const("wq", {dim, dim}, randv(dim * dim));
  add_const("wk", {dim, dim}, randv(dim * dim));
  add_const("wv", {dim, dim}, randv(dim * dim));
  add_const("wo", {dim, dim}, randv(dim * dim));
  add_const("g2", {dim}, g2);
  add_const("wg", {dim, ffn}, randv(dim * ffn));
  add_const("wu", {dim, ffn}, randv(dim * ffn));
  add_const("wd", {ffn, dim}, randv(ffn * dim));
  add_const("gf", {dim}, gf);
  add_const("wh", {dim, vocab}, randv(dim * vocab));

  const auto h = static_cast<uint32_t>(heads);
  m.ops.push_back({SmfOpKind::kRmsNorm, "ln1", {"x", "g1"}, "n1"});
  m.ops.push_back({SmfOpKind::kMatMul, "mmq", {"n1", "wq"}, "q"});
  m.ops.push_back({SmfOpKind::kMatMul, "mmk", {"n1", "wk"}, "k"});
  m.ops.push_back({SmfOpKind::kMatMul, "mmv", {"n1", "wv"}, "v"});
  m.ops.push_back({SmfOpKind::kRope, "ropeq", {"q"}, "qr", h});
  m.ops.push_back({SmfOpKind::kRope, "ropek", {"k"}, "kr", h});
  m.ops.push_back({SmfOpKind::kAttention, "attn", {"qr", "kr", "v"}, "a", h});
  m.ops.push_back({SmfOpKind::kMatMul, "mmo", {"a", "wo"}, "o"});
  m.ops.push_back({SmfOpKind::kAdd, "res1", {"x", "o"}, "x1"});
  m.ops.push_back({SmfOpKind::kRmsNorm, "ln2", {"x1", "g2"}, "n2"});
  m.ops.push_back({SmfOpKind::kMatMul, "mmg", {"n2", "wg"}, "gt"});
  m.ops.push_back({SmfOpKind::kMatMul, "mmu", {"n2", "wu"}, "u"});
  m.ops.push_back({SmfOpKind::kSilu, "silu", {"gt"}, "s"});
  m.ops.push_back({SmfOpKind::kMul, "swiglu", {"s", "u"}, "mg"});
  m.ops.push_back({SmfOpKind::kMatMul, "mmd", {"mg", "wd"}, "dn"});
  m.ops.push_back({SmfOpKind::kAdd, "res2", {"x1", "dn"}, "x2"});
  m.ops.push_back({SmfOpKind::kRmsNorm, "lnf", {"x2", "gf"}, "nf"});
  m.ops.push_back({SmfOpKind::kMatMul, "mmh", {"nf", "wh"}, "logits"});
  return m;
}

SmfModel MakeDecoderStack(int64_t dim, int64_t heads, int64_t seq,
                          int64_t ffn, int64_t vocab, int64_t blocks,
                          uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.5f);
  auto randv = [&](size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
  };

  SmfModel m;
  m.input_name = "x";
  m.output_name = "logits";
  m.seq_len = static_cast<uint64_t>(seq);
  m.tensors.push_back({.name = "x", .dims = {-1, dim}, .is_const = false});
  auto add_const = [&](const std::string& name, std::vector<int64_t> dims,
                       std::vector<float> data) {
    m.tensors.push_back({.name = name,
                         .dims = std::move(dims),
                         .is_const = true,
                         .data = AsBytes(data)});
    m.tensors.back().byte_size = m.tensors.back().data.size();
  };
  auto gain = [&](int64_t n) {
    std::vector<float> g(n, 1.0f);
    for (auto& v : g) v += 0.1f * dist(rng);  // non-trivial gains
    return g;
  };

  const auto h = static_cast<uint32_t>(heads);
  std::string prev = "x";
  for (int64_t i = 0; i < blocks; ++i) {
    const std::string p = "l" + std::to_string(i) + ".";
    add_const(p + "g1", {dim}, gain(dim));
    add_const(p + "wq", {dim, dim}, randv(dim * dim));
    add_const(p + "wk", {dim, dim}, randv(dim * dim));
    add_const(p + "wv", {dim, dim}, randv(dim * dim));
    add_const(p + "wo", {dim, dim}, randv(dim * dim));
    add_const(p + "g2", {dim}, gain(dim));
    add_const(p + "wg", {dim, ffn}, randv(dim * ffn));
    add_const(p + "wu", {dim, ffn}, randv(dim * ffn));
    add_const(p + "wd", {ffn, dim}, randv(ffn * dim));
    m.ops.push_back({SmfOpKind::kRmsNorm, p + "ln1", {prev, p + "g1"},
                     p + "n1"});
    for (const char* w : {"q", "k", "v"})
      m.ops.push_back({SmfOpKind::kMatMul, p + "mm" + w,
                       {p + "n1", p + "w" + std::string(w)}, p + w});
    m.ops.push_back({SmfOpKind::kRope, p + "ropeq", {p + "q"}, p + "qr", h});
    m.ops.push_back({SmfOpKind::kRope, p + "ropek", {p + "k"}, p + "kr", h});
    m.ops.push_back({SmfOpKind::kAttention, p + "attn",
                     {p + "qr", p + "kr", p + "v"}, p + "a", h});
    m.ops.push_back({SmfOpKind::kMatMul, p + "mmo", {p + "a", p + "wo"},
                     p + "o"});
    m.ops.push_back({SmfOpKind::kAdd, p + "res1", {prev, p + "o"}, p + "x1"});
    m.ops.push_back({SmfOpKind::kRmsNorm, p + "ln2", {p + "x1", p + "g2"},
                     p + "n2"});
    m.ops.push_back({SmfOpKind::kMatMul, p + "mmg", {p + "n2", p + "wg"},
                     p + "gt"});
    m.ops.push_back({SmfOpKind::kMatMul, p + "mmu", {p + "n2", p + "wu"},
                     p + "u"});
    m.ops.push_back({SmfOpKind::kSilu, p + "silu", {p + "gt"}, p + "s"});
    m.ops.push_back({SmfOpKind::kMul, p + "swiglu", {p + "s", p + "u"},
                     p + "mg"});
    m.ops.push_back({SmfOpKind::kMatMul, p + "mmd", {p + "mg", p + "wd"},
                     p + "dn"});
    m.ops.push_back({SmfOpKind::kAdd, p + "res2", {p + "x1", p + "dn"},
                     p + "x2"});
    prev = p + "x2";
  }
  add_const("gf", {dim}, gain(dim));
  add_const("wh", {dim, vocab}, randv(dim * vocab));
  m.ops.push_back({SmfOpKind::kRmsNorm, "lnf", {prev, "gf"}, "nf"});
  m.ops.push_back({SmfOpKind::kMatMul, "mmh", {"nf", "wh"}, "logits"});
  return m;
}

SmfModel MakeTinyTokenDecoder(int64_t vocab, int64_t dim, int64_t heads,
                              int64_t seq, int64_t ffn, uint64_t seed) {
  // The decoder block of MakeTinyDecoder, fed by an embedding gather over
  // a rank-1 dynamic i32 input instead of pre-embedded rows.
  SmfModel m = MakeTinyDecoder(dim, heads, seq, ffn, vocab, seed);
  m.tensors[0].dims = {-1};  // x: token ids, one per row
  std::mt19937_64 rng(seed ^ 0x9E3779B97F4A7C15ULL);
  std::normal_distribution<float> dist(0.0f, 0.5f);
  std::vector<float> table(static_cast<size_t>(vocab * dim));
  for (auto& v : table) v = dist(rng);
  m.tensors.push_back({.name = "emb",
                       .dims = {vocab, dim},
                       .is_const = true,
                       .data = AsBytes(table)});
  m.tensors.back().byte_size = m.tensors.back().data.size();
  // The embedding replaces x at the front of the op list; every op that
  // read "x" as features now reads the gathered rows "e".
  m.ops.insert(m.ops.begin(),
               {SmfOpKind::kEmbedding, "embed", {"x", "emb"}, "e"});
  for (size_t i = 1; i < m.ops.size(); ++i)
    for (auto& in : m.ops[i].inputs)
      if (in == "x") in = "e";
  return m;
}

UpdateConfig BaseConfig(int64_t batch) {
  UpdateConfig config;
  config.batch = batch;
  config.loss = update::LossKind::kSoftmaxXEnt;
  config.lora.rank = 4;
  config.lora.alpha = 8.0f;
  config.lora.seed = 7;
  return config;
}

}  // namespace seeml::testing
