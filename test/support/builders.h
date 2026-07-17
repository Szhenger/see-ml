#ifndef SEEML_TEST_SUPPORT_BUILDERS_H_
#define SEEML_TEST_SUPPORT_BUILDERS_H_

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "compiler/backend/update_types.h"
#include "runtime/dataset.h"
#include "runtime/update_engine.h"
#include "source/smf.h"

// =============================================================================
// Shared fixtures for the SeeML test suites: deterministic model builders,
// synthetic datasets, and arena introspection helpers for the UpdateEngine.
// Everything is seeded — two calls with the same arguments produce identical
// bytes, which the determinism tests rely on.
// =============================================================================

namespace seeml::testing {

/// Reinterprets a float vector as its raw little-endian bytes (SMF blobs).
std::vector<uint8_t> AsBytes(const std::vector<float>& v);

/// `n` samples from N(0, std), deterministic in `seed`.
std::vector<float> RandnVector(size_t n, uint64_t seed, float stddev = 1.0f);

/// A deterministic 2-layer MLP:  in -> hidden (relu) -> out.
/// Tensors: x (dynamic batch), w1, b1, w2, b2; ops: mm1, ab1, relu1, mm2, ab2.
update::SmfModel MakeMlp(int64_t in_dim, int64_t hidden, int64_t out_dim,
                         uint64_t seed);

/// A two-layer net whose MatMuls share one square weight tensor "w"
/// (weight tying):  x[?,dim] @ w -> relu -> @ w -> y.
update::SmfModel MakeTiedMlp(int64_t dim, uint64_t seed);

/// A linearly separable 2-class problem: label = 1[x . w_true > 0].
std::expected<update_rt::Dataset, std::string> MakeClassificationData(
    uint64_t n, int64_t in_dim, uint64_t seed);

/// A linear regression problem with dense f32 targets: y = x @ W_true.
std::expected<update_rt::Dataset, std::string> MakeRegressionData(
    uint64_t n, int64_t in_dim, int64_t out_dim, uint64_t seed);

/// Unlabeled inputs (distillation corpora).
std::expected<update_rt::Dataset, std::string> MakeUnlabeledData(
    uint64_t n, int64_t in_dim, uint64_t seed);

/// The suites' default update configuration: softmax cross-entropy, LoRA
/// rank 4 / alpha 8 / seed 7.
update::UpdateConfig BaseConfig(int64_t batch);

// --- UpdateEngine arena helpers -----------------------------------------------

float ReadArenaF32(update_rt::UpdateEngine& e, uint64_t ref,
                   uint64_t index = 0);
void WriteArenaF32(update_rt::UpdateEngine& e, uint64_t ref, uint64_t index,
                   float v);

/// Copies a batch directly into the plan's I/O slots (bypassing Dataset).
/// Sizes must match the plan header exactly; aborts otherwise.
void FillSlots(update_rt::UpdateEngine& e, const std::vector<float>& x,
               const std::vector<int32_t>& labels);

}  // namespace seeml::testing

#endif  // SEEML_TEST_SUPPORT_BUILDERS_H_
