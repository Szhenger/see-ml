// =============================================================================
// Resource analyzer tests: the static training-footprint lower bound, local
// memory detection, and the fail-fast gate in UpdateCompiler::Compile.
// =============================================================================

#include <cstdint>

#include "compiler/backend/update_compiler.h"
#include "compiler/frontend/ingressor/resource_analyzer.h"
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

}  // namespace
