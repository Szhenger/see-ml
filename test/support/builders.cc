#include "test/support/builders.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

namespace seeml::testing {

using update::SmfModel;
using update::SmfOpKind;
using update::UpdateConfig;
using update_rt::Dataset;
using update_rt::UpdateEngine;

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

std::string RepoRoot() {
#ifdef SEEML_SOURCE_DIR
  return SEEML_SOURCE_DIR;
#else
  return ".";
#endif
}

std::expected<Dataset, std::string> MakeClassificationData(uint64_t n,
                                                           int64_t in_dim,
                                                           uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> w_true(in_dim);
  for (auto& w : w_true) w = dist(rng);

  std::vector<float> inputs(n * in_dim);
  std::vector<uint8_t> labels(n * sizeof(int32_t));
  auto* lab = reinterpret_cast<int32_t*>(labels.data());
  for (uint64_t i = 0; i < n; ++i) {
    float dot = 0.0f;
    for (int64_t d = 0; d < in_dim; ++d) {
      inputs[i * in_dim + d] = dist(rng);
      dot += inputs[i * in_dim + d] * w_true[d];
    }
    lab[i] = dot > 0.0f ? 1 : 0;
  }
  return Dataset::FromMemory(std::move(inputs), std::move(labels), n, in_dim,
                             /*label_kind=*/1, /*label_dim=*/0);
}

std::expected<Dataset, std::string> MakeRegressionData(uint64_t n,
                                                       int64_t in_dim,
                                                       int64_t out_dim,
                                                       uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> w_true(in_dim * out_dim);
  for (auto& w : w_true) w = dist(rng);

  std::vector<float> inputs(n * in_dim);
  std::vector<float> targets(n * out_dim, 0.0f);
  for (uint64_t i = 0; i < n; ++i) {
    for (int64_t d = 0; d < in_dim; ++d) inputs[i * in_dim + d] = dist(rng);
    for (int64_t o = 0; o < out_dim; ++o)
      for (int64_t d = 0; d < in_dim; ++d)
        targets[i * out_dim + o] +=
            inputs[i * in_dim + d] * w_true[d * out_dim + o];
  }
  return Dataset::FromMemory(std::move(inputs), AsBytes(targets), n, in_dim,
                             /*label_kind=*/2,
                             /*label_dim=*/static_cast<uint64_t>(out_dim));
}

std::expected<Dataset, std::string> MakeUnlabeledData(uint64_t n,
                                                      int64_t in_dim,
                                                      uint64_t seed) {
  std::vector<float> inputs =
      RandnVector(static_cast<size_t>(n * in_dim), seed);
  return Dataset::FromMemory(std::move(inputs), {}, n, in_dim,
                             /*label_kind=*/0, /*label_dim=*/0);
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

float ReadArenaF32(UpdateEngine& e, uint64_t ref, uint64_t index) {
  return reinterpret_cast<const float*>(e.arena() +
                                        update::RefOffset(ref))[index];
}

void WriteArenaF32(UpdateEngine& e, uint64_t ref, uint64_t index, float v) {
  reinterpret_cast<float*>(e.arena() + update::RefOffset(ref))[index] = v;
}

void FillSlots(UpdateEngine& e, const std::vector<float>& x,
               const std::vector<int32_t>& labels) {
  if (x.size() != e.header().input_floats ||
      (!labels.empty() &&
       labels.size() * sizeof(int32_t) != e.header().label_bytes)) {
    std::fprintf(stderr, "FillSlots: batch does not match the plan header\n");
    std::abort();
  }
  std::memcpy(e.arena() + update::RefOffset(e.header().input_ref), x.data(),
              x.size() * sizeof(float));
  if (!labels.empty())
    std::memcpy(e.arena() + update::RefOffset(e.header().label_ref),
                labels.data(), labels.size() * sizeof(int32_t));
}

}  // namespace seeml::testing
