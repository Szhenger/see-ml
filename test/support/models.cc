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
