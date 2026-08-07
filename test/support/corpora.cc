#include <random>

#include "test/support/builders.h"

// =============================================================================
// corpora/ discipline of the shared fixtures: deterministic synthetic
// datasets for the runtime's training loop — a separable classification
// problem, a dense-target regression, and an unlabeled distillation corpus.
// =============================================================================

namespace seeml::testing {

using update_rt::Dataset;

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

std::expected<Dataset, std::string> MakeTokenCorpus(uint64_t records,
                                                    uint64_t seq,
                                                    int64_t vocab,
                                                    uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<int32_t> tokens;
  tokens.reserve(records * (seq + 1));
  for (uint64_t r = 0; r < records; ++r) {
    int32_t t = static_cast<int32_t>(rng() % static_cast<uint64_t>(vocab));
    for (uint64_t i = 0; i <= seq; ++i) {
      tokens.push_back(t);
      t = static_cast<int32_t>((3 * t + 1) % vocab);
    }
  }
  return Dataset::FromTokens(std::move(tokens), records, seq);
}

}  // namespace seeml::testing
