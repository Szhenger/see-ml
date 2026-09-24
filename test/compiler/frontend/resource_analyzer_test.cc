// =============================================================================
// Resource analyzer tests: the static training-footprint lower bound, local
// memory detection, and the fail-fast gate in UpdateCompiler::Compile.
// =============================================================================

#include <cstdint>

#include "compiler/driver/update_compiler.h"
#include "compiler/frontend/accountant/resource_analyzer.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"

namespace {

using namespace seeml::update;
using seeml::testing::MakeMlp;

constexpr int64_t kBatch = 16;

TEST(ResourceAnalyzer, FootprintCountsWeightsAndActivations) {
  // MakeMlp(4, 8, 2): w1[4x8] b1[8] w2[8x2] b2[2], ops mm1/ab1/relu1/mm2/ab2.
  SmfModel model = MakeMlp(4, 8, 2, /*seed=*/1);
  TrainingFootprint fp = EstimateTrainingFootprint(model, kBatch);

  EXPECT_EQ(fp.weight_bytes, (4 * 8 + 8 + 8 * 2 + 2) * sizeof(float));
  // Activations: mm1/ab1/relu1 are [16x8], mm2/ab2 are [16x2].
  EXPECT_EQ(fp.activation_bytes,
            (3 * kBatch * 8 + 2 * kBatch * 2) * sizeof(float));
  EXPECT_EQ(fp.total_bytes(), fp.weight_bytes + fp.activation_bytes);
}

TEST(ResourceAnalyzer, LayerNormCachesRowStatistics) {
  SmfModel model;
  model.input_name = "x";
  model.output_name = "y";
  model.tensors.push_back({.name = "x", .dims = {-1, 8}, .is_const = false});
  model.tensors.push_back(
      {.name = "gamma", .dims = {8}, .is_const = true, .byte_size = 32});
  model.tensors.push_back(
      {.name = "beta", .dims = {8}, .is_const = true, .byte_size = 32});
  model.ops.push_back(
      {SmfOpKind::kLayerNorm, "ln", {"x", "gamma", "beta"}, "y"});

  TrainingFootprint fp = EstimateTrainingFootprint(model, /*batch=*/4);
  EXPECT_EQ(fp.weight_bytes, 64u);
  // Output [4x8] plus per-row mean/rstd: 2 f32 per batch row.
  EXPECT_EQ(fp.activation_bytes, (4 * 8 + 2 * 4) * sizeof(float));
}

TEST(ResourceAnalyzer, UnresolvableWidthStaysALowerBound) {
  SmfModel model;
  model.input_name = "x";
  model.output_name = "y";
  model.tensors.push_back({.name = "x", .dims = {-1, 4}, .is_const = false});
  // "w" is missing entirely: the estimate must not invent bytes for it.
  model.ops.push_back({SmfOpKind::kMatMul, "mm", {"x", "w"}, "y"});

  TrainingFootprint fp = EstimateTrainingFootprint(model, kBatch);
  EXPECT_EQ(fp.weight_bytes, 0u);
  EXPECT_EQ(fp.activation_bytes, 0u);
}

TEST(ResourceAnalyzer, DetectsLocalMemory) {
  EXPECT_TRUE(DetectLocalMemoryBytes() > 0);
}

TEST(ResourceAnalyzer, CheckPassesWithinBudget) {
  TrainingFootprint fp{.weight_bytes = 1024, .activation_bytes = 1024};
  EXPECT_OK(CheckTrainableLocally(fp, 4096));
  // Budget 0 = detect physical memory; a 2 KiB footprint always fits.
  EXPECT_OK(CheckTrainableLocally(fp, 0));
}

TEST(ResourceAnalyzer, CheckRejectsBeyondBudget) {
  TrainingFootprint fp{.weight_bytes = 4096, .activation_bytes = 4096};
  EXPECT_ERROR_CONTAINS(CheckTrainableLocally(fp, 1024),
                        "too big to train locally");
}

TEST(ResourceAnalyzer, CompilerFailsFastOnOversizedModel) {
  SmfModel model = MakeMlp(4, 8, 2, /*seed=*/1);
  UpdateConfig config;
  config.batch = kBatch;
  config.memory_budget_bytes = 256;  // below the model's proven lower bound
  EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(model),
                        "too big to train locally");
}

TEST(ResourceAnalyzer, CompilerAcceptsModelWithinBudget) {
  SmfModel model = MakeMlp(4, 8, 2, /*seed=*/1);
  UpdateConfig config;
  config.batch = kBatch;
  config.memory_budget_bytes = 64ull << 20;  // 64 MiB
  ASSERT_OK(UpdateCompiler(config).Compile(model));
}

TEST(ResourceAnalyzer, FrozenForwardEstimateChargesPeakNotSum) {
  // A teacher under distillation has no backward: its transients die at
  // their single reader and the binder reuses their slots, so the honest
  // lower bound is the single widest live activation — summing them (the
  // training model) over-counts and can refuse a feasible compile.
  SmfModel model = MakeMlp(4, 8, 2, /*seed=*/1);
  const TrainingFootprint train = EstimateTrainingFootprint(model, kBatch);
  const TrainingFootprint frozen =
      EstimateFrozenForwardFootprint(model, kBatch);
  EXPECT_EQ(frozen.weight_bytes, train.weight_bytes);
  // Peak term: the widest activation is [16 x 8] f32.
  EXPECT_EQ(frozen.activation_bytes, kBatch * 8 * sizeof(float));
  EXPECT_TRUE(frozen.activation_bytes < train.activation_bytes);
}

// --- The merged-delta term of the step-0 gate (E2, #81 — finding #8) ---------

TEST(ResourceAnalyzer, DeltaSegmentCountsExactlyWhatLoraWillAdapt) {
  // MakeMlp(4, 8, 2): w1[4x8] and w2[8x2] are MatMul weights; the biases
  // are consumed by AddBias and can never be adapted.
  SmfModel model = MakeMlp(4, 8, 2, /*seed=*/1);
  EXPECT_EQ(EstimateLoraDeltaBytes(model, {}),
            (4 * 8 + 8 * 2) * sizeof(float));
  const std::vector<std::string> only_w2 = {"w2"};
  EXPECT_EQ(EstimateLoraDeltaBytes(model, only_w2), 8 * 2 * sizeof(float));
  const std::vector<std::string> nothing = {"no-such-tensor"};
  EXPECT_EQ(EstimateLoraDeltaBytes(model, nothing), 0u);

  // The grafter's rule: a weight read any other way — here w2 doubling as
  // a MatMul's LEFT operand — is not adapted, so it must not be charged.
  SmfModel tied = model;
  tied.ops.push_back({.kind = SmfOpKind::kMatMul,
                      .name = "odd",
                      .inputs = {"w2", "w2"},
                      .output = "odd_out"});
  EXPECT_EQ(EstimateLoraDeltaBytes(tied, {}), 4 * 8 * sizeof(float));
}

TEST(ResourceAnalyzer, TheEarlyGateSeesTheDeltasAQuantizedBaseHides) {
  // Under --quantize-base the weights count at 1/4, but the merged deltas
  // are full-size f32 — 4x the weights they patch. A budget between the
  // two estimates used to pass step 0 and fail only at the final gate,
  // after all the work; now it is refused up front, and names the term.
  SmfModel model = MakeMlp(64, 256, 32, /*seed=*/2);
  UpdateConfig config;
  config.batch = kBatch;
  config.quantize_base = true;
  TrainingFootprint fp = EstimateTrainingFootprint(model, kBatch);
  fp.weight_bytes /= 4;
  const uint64_t without_deltas = fp.total_bytes();
  fp.delta_bytes = EstimateLoraDeltaBytes(model, {});
  ASSERT_GT(fp.delta_bytes, fp.weight_bytes);
  EXPECT_EQ(fp.total_bytes(), without_deltas + fp.delta_bytes);

  config.memory_budget_bytes = without_deltas + fp.delta_bytes / 2;
  auto refused = UpdateCompiler(config).Compile(model);
  EXPECT_ERROR_CONTAINS(refused, "too big to train locally");
  EXPECT_ERROR_CONTAINS(refused, "merged LoRA deltas");
  // A lower bound still: what it admits, the exact final gate may refuse,
  // but it never refuses what fits.
  ASSERT_OK_AND_ASSIGN(CompiledUpdate roomy, [&] {
    config.memory_budget_bytes = 64ull << 20;
    return UpdateCompiler(config).Compile(model);
  }());
  EXPECT_LE(fp.total_bytes(), roomy.arena_size + roomy.plan.size());
}

}  // namespace
