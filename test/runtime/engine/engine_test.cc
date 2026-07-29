// =============================================================================
// Engine tests: the subsystem usage contracts the engine enforces at each
// boundary (plan / executor / feeder), the diagnostics contract at the
// engine's Train boundary, and the runtime unit registry.
// =============================================================================

#include <cstring>
#include <span>
#include <string>

#include "compiler/driver/update_compiler.h"
#include "runtime/engine/contract.h"
#include "runtime/engine/update_engine.h"
#include "source/plan/update_types.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"

namespace {

using namespace seeml::update_rt;
namespace up = seeml::update;
using seeml::testing::BaseConfig;
using seeml::testing::MakeClassificationData;
using seeml::testing::MakeMlp;

constexpr int64_t kInDim = 6;
constexpr int64_t kHidden = 10;
constexpr int64_t kOutDim = 3;
constexpr int64_t kBatch = 4;

up::PlanHeader HeaderOf(const up::CompiledUpdate& compiled) {
  up::PlanHeader h;
  std::memcpy(&h, compiled.plan.data(), sizeof(h));
  return h;
}

// =============================================================================
// The diagnostics contract
// =============================================================================

TEST(EngineContract, RecognizesEveryRegisteredUnit) {
  EXPECT_TRUE(WellFormedDiagnostic("Dataset: bad magic in 'c.sds'"));
  EXPECT_TRUE(WellFormedDiagnostic("PlanValidator: unknown opcode 9"));
  EXPECT_TRUE(WellFormedDiagnostic("UpdateEngine: no plan loaded"));
  EXPECT_TRUE(WellFormedDiagnostic("DurableIO: cannot write '/x'"));
  EXPECT_TRUE(WellFormedDiagnostic("Checkpoint: payload is corrupt"));
}

TEST(EngineContract, RejectsUnattributableDiagnostics) {
  EXPECT_FALSE(WellFormedDiagnostic("something went wrong"));
  EXPECT_FALSE(WellFormedDiagnostic("Frobnicator: unknown unit"));
  EXPECT_FALSE(WellFormedDiagnostic("Dataset"));
  EXPECT_FALSE(WellFormedDiagnostic("Dataset: "));
  EXPECT_FALSE(WellFormedDiagnostic(""));
}

// =============================================================================
// The plan boundary
// =============================================================================

TEST(EngineContract, PlanContractAcceptsACompiledPlanAndCatchesCorruption) {
  up::SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 1);
  ASSERT_OK_AND_ASSIGN(
      up::CompiledUpdate compiled,
      up::UpdateCompiler(BaseConfig(kBatch)).Compile(model));
  const up::PlanHeader good = HeaderOf(compiled);
  EXPECT_TRUE(VerifyPlanContract(good, compiled.plan.size()).has_value());

  up::PlanHeader oversized = good;
  oversized.persistent_size = oversized.arena_size + 1;
  const auto r_seg = VerifyPlanContract(oversized, compiled.plan.size());
  ASSERT_FALSE(r_seg.has_value());
  EXPECT_TRUE(WellFormedDiagnostic(r_seg.error()));
  EXPECT_NE(r_seg.error().find("persistent segment exceeds arena"),
            std::string::npos);

  up::PlanHeader escaped = good;
  escaped.train_instr_offset = compiled.plan.size();
  EXPECT_FALSE(
      VerifyPlanContract(escaped, compiled.plan.size()).has_value());
}

// =============================================================================
// The executor boundary
// =============================================================================

TEST(EngineContract, ExecutorContractSpeaksAsThePlanValidator) {
  up::UpdateInstruction bogus{};
  bogus.opcode = 0xFFFF;
  up::PlanHeader header{};
  header.arena_size = 1024;

  const auto r = VerifyExecutorContract(std::span(&bogus, 1), {}, {}, {},
                                        header);
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(WellFormedDiagnostic(r.error()));
  EXPECT_EQ(r.error().find("PlanValidator: "), 0u);
}

// =============================================================================
// The feeder boundary
// =============================================================================

TEST(EngineContract, FeederContractMatchesCorpusToPlanGeometry) {
  up::SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 2);
  ASSERT_OK_AND_ASSIGN(
      up::CompiledUpdate compiled,
      up::UpdateCompiler(BaseConfig(kBatch)).Compile(model));
  const up::PlanHeader header = HeaderOf(compiled);

  ASSERT_OK_AND_ASSIGN(Dataset fits,
                       MakeClassificationData(32, kInDim, 3));
  EXPECT_TRUE(VerifyFeederContract(header, fits, kOutDim).has_value());

  ASSERT_OK_AND_ASSIGN(Dataset wrong_width,
                       MakeClassificationData(32, kInDim + 1, 3));
  const auto r = VerifyFeederContract(header, wrong_width, kOutDim);
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(WellFormedDiagnostic(r.error()));
  EXPECT_NE(r.error().find("input width does not match"), std::string::npos);
}

// =============================================================================
// The engine end to end
// =============================================================================

TEST(Engine, ErrorsCrossingTheTrainBoundaryAreWellFormed) {
  ASSERT_OK_AND_ASSIGN(Dataset data, MakeClassificationData(16, kInDim, 5));
  UpdateEngine engine;  // no plan loaded
  const auto r = engine.Train(data, 1);
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(WellFormedDiagnostic(r.error()));
  EXPECT_EQ(r.error(), "UpdateEngine: no plan loaded");
}

}  // namespace
