// =============================================================================
// UpdateEngine tests: plan ingestion hardening (corrupt headers, out-of-bounds
// instructions), dataset/plan contract validation at Train() time, checkpoint
// round-trips, and the merge-before-commit protocol.
// =============================================================================

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "compiler/backend/update_compiler.h"
#include "compiler/backend/update_types.h"
#include "runtime/dataset.h"
#include "runtime/update_engine.h"
#include "source/smf.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"
#include "test/support/scoped_temp_dir.h"

namespace {

using namespace seeml::update;
using seeml::update_rt::Dataset;
using seeml::update_rt::TrainOptions;
using seeml::update_rt::UpdateEngine;
using seeml::testing::BaseConfig;
using seeml::testing::MakeClassificationData;
using seeml::testing::MakeMlp;
using seeml::testing::ScopedTempDir;

constexpr int64_t kInDim = 6;
constexpr int64_t kBatch = 4;

/// Compiles the standard test MLP into a plan blob.
std::vector<uint8_t> CompilePlan(UpdateConfig config, uint64_t seed = 1) {
  SmfModel model = MakeMlp(kInDim, 10, 3, seed);
  auto compiled = UpdateCompiler(config).Compile(model);
  if (!compiled) return {};
  return compiled->plan;
}

PlanHeader HeaderOf(const std::vector<uint8_t>& plan) {
  PlanHeader h;
  std::memcpy(&h, plan.data(), sizeof(h));
  return h;
}

void PutHeader(std::vector<uint8_t>& plan, const PlanHeader& h) {
  std::memcpy(plan.data(), &h, sizeof(h));
}

TrainOptions Quiet() {
  TrainOptions options;
  options.log_every = 0;
  return options;
}

/// Byte offset of the first train instruction with `oc`, or 0 if absent.
size_t FindTrainInstr(const std::vector<uint8_t>& plan, OpCode oc) {
  const PlanHeader h = HeaderOf(plan);
  for (uint64_t i = 0; i < h.train_instr_count; ++i) {
    const size_t off = h.train_instr_offset + i * sizeof(UpdateInstruction);
    UpdateInstruction ins;
    std::memcpy(&ins, plan.data() + off, sizeof(ins));
    if (ins.opcode == static_cast<uint16_t>(oc)) return off;
  }
  return 0;
}

TEST(UpdateEngineLoad, AcceptsCompiledPlan) {
  const std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  EXPECT_OK(engine.LoadFromMemory(plan.data(), plan.size()));
  EXPECT_EQ(engine.step(), 0u);
  EXPECT_EQ(engine.header().magic, kSeeuMagic);
  EXPECT_EQ(engine.header().batch, static_cast<uint64_t>(kBatch));
}

TEST(UpdateEngineLoad, RejectsTruncatedHeader) {
  const std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  EXPECT_ERROR_CONTAINS(engine.LoadFromMemory(plan.data(), 16),
                        "smaller than its header");
}

TEST(UpdateEngineLoad, RejectsBadMagicAndVersion) {
  std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());

  {
    std::vector<uint8_t> bad = plan;
    PlanHeader h = HeaderOf(bad);
    h.magic = 0xDEADBEEF;
    PutHeader(bad, h);
    UpdateEngine engine;
    EXPECT_ERROR_CONTAINS(engine.LoadFromMemory(bad.data(), bad.size()),
                          "bad plan magic");
  }
  {
    std::vector<uint8_t> bad = plan;
    PlanHeader h = HeaderOf(bad);
    h.version = 42;
    PutHeader(bad, h);
    UpdateEngine engine;
    EXPECT_ERROR_CONTAINS(engine.LoadFromMemory(bad.data(), bad.size()),
                          "unsupported plan version");
  }
}

TEST(UpdateEngineLoad, RejectsSectionOutOfBounds) {
  std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  PlanHeader h = HeaderOf(plan);
  h.train_instr_offset = plan.size();  // count > 0 pushes past the end
  PutHeader(plan, h);
  UpdateEngine engine;
  EXPECT_ERROR_CONTAINS(engine.LoadFromMemory(plan.data(), plan.size()),
                        "section out of bounds");
}

TEST(UpdateEngineLoad, RejectsPersistentSegmentExceedingArena) {
  std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  PlanHeader h = HeaderOf(plan);
  h.persistent_size = h.arena_size + 1;
  PutHeader(plan, h);
  UpdateEngine engine;
  EXPECT_ERROR_CONTAINS(engine.LoadFromMemory(plan.data(), plan.size()),
                        "persistent segment exceeds arena");
}

TEST(UpdateEngineLoad, RejectsUnknownOpcode) {
  std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  const PlanHeader h = HeaderOf(plan);
  UpdateInstruction ins;
  std::memcpy(&ins, plan.data() + h.train_instr_offset, sizeof(ins));
  ins.opcode = 999;
  std::memcpy(plan.data() + h.train_instr_offset, &ins, sizeof(ins));

  UpdateEngine engine;
  EXPECT_ERROR_CONTAINS(engine.LoadFromMemory(plan.data(), plan.size()),
                        "unknown opcode");
}

TEST(UpdateEngineLoad, RejectsOperandOutsideItsAddressSpace) {
  std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  const PlanHeader h = HeaderOf(plan);
  UpdateInstruction ins;
  std::memcpy(&ins, plan.data() + h.train_instr_offset, sizeof(ins));
  ins.in[0] = MakeArenaRef(h.arena_size + (1ULL << 32));  // far outside
  std::memcpy(plan.data() + h.train_instr_offset, &ins, sizeof(ins));

  UpdateEngine engine;
  EXPECT_ERROR_CONTAINS(engine.LoadFromMemory(plan.data(), plan.size()),
                        "out of bounds");
}

TEST(UpdateEngineLoad, FailedReloadLeavesEngineUnloaded) {
  const std::vector<uint8_t> good = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(good.empty());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(good.data(), good.size()));

  // A plan that fails validation *late* in the load — after its header and
  // programs have been decoded.
  std::vector<uint8_t> bad = good;
  const PlanHeader h = HeaderOf(bad);
  UpdateInstruction ins;
  std::memcpy(&ins, bad.data() + h.train_instr_offset, sizeof(ins));
  ins.in[0] = MakeArenaRef(h.arena_size + (1ULL << 32));
  std::memcpy(bad.data() + h.train_instr_offset, &ins, sizeof(ins));
  EXPECT_ERROR(engine.LoadFromMemory(bad.data(), bad.size()));

  // The failed load must not leave the bad plan's programs paired with the
  // good plan's arena; the engine must be cleanly unloaded.
  ASSERT_OK_AND_ASSIGN(Dataset data, MakeClassificationData(32, kInDim, 1));
  EXPECT_ERROR_CONTAINS(engine.Train(data, 1, Quiet()), "no plan loaded");
  EXPECT_ERROR_CONTAINS(engine.RunMerge(), "no plan loaded");
}

TEST(UpdateEngineLoad, RejectsSoftmaxLabelsUnboundFromLabelSlot) {
  std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  const PlanHeader h = HeaderOf(plan);
  const size_t off = FindTrainInstr(plan, OpCode::kSoftmaxXEntFwd);
  ASSERT_NE(off, 0u);

  // Point the labels operand at a different (individually valid) arena
  // region: the input slot. The kernel would then index probs rows with
  // whatever bit patterns live there — an unbounded read/write.
  UpdateInstruction ins;
  std::memcpy(&ins, plan.data() + off, sizeof(ins));
  ins.in[1] = MakeArenaRef(RefOffset(h.input_ref));
  std::memcpy(plan.data() + off, &ins, sizeof(ins));

  UpdateEngine engine;
  EXPECT_ERROR_CONTAINS(engine.LoadFromMemory(plan.data(), plan.size()),
                        "not bound to the plan's label slot");
}

TEST(UpdateEngineLoad, RejectsInconsistentSoftmaxClassCounts) {
  std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  const size_t off = FindTrainInstr(plan, OpCode::kSoftmaxXEntBwd);
  ASSERT_NE(off, 0u);

  // Shrink the backward instruction's class count: labels validated against
  // the forward's C would still index out of the backward's narrower rows.
  UpdateInstruction ins;
  std::memcpy(&ins, plan.data() + off, sizeof(ins));
  ASSERT_GT(ins.out[1], 1u);
  ins.out[1] -= 1;
  std::memcpy(plan.data() + off, &ins, sizeof(ins));

  UpdateEngine engine;
  EXPECT_ERROR_CONTAINS(engine.LoadFromMemory(plan.data(), plan.size()),
                        "inconsistent class count");
}

TEST(UpdateEngineLoad, RejectsSoftmaxWithoutClassLabels) {
  std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  PlanHeader h = HeaderOf(plan);
  h.label_kind = 2;  // dense targets: ValidateClassLabels() would never run
  PutHeader(plan, h);

  UpdateEngine engine;
  EXPECT_ERROR_CONTAINS(engine.LoadFromMemory(plan.data(), plan.size()),
                        "without class-index labels");
}

TEST(UpdateEngineLoad, LoadFromFileMatchesLoadFromMemory) {
  ScopedTempDir dir;
  const std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  const std::string path = dir.File("plan.seeu");
  {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(plan.data()),
            static_cast<std::streamsize>(plan.size()));
  }

  UpdateEngine from_file, from_memory;
  ASSERT_OK(from_file.LoadFromFile(path));
  ASSERT_OK(from_memory.LoadFromMemory(plan.data(), plan.size()));
  EXPECT_EQ(from_file.header().arena_size, from_memory.header().arena_size);

  // Both engines start from the same persistent image.
  EXPECT_EQ(std::memcmp(from_file.arena(), from_memory.arena(),
                        from_file.header().persistent_size),
            0);

  EXPECT_ERROR_CONTAINS(from_file.LoadFromFile(dir.File("missing.seeu")),
                        "cannot open");
}

TEST(UpdateEngineTrain, ValidatesDatasetAgainstPlan) {
  const std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));

  // Wrong input width.
  ASSERT_OK_AND_ASSIGN(Dataset wrong_width,
                       MakeClassificationData(32, kInDim + 1, 1));
  EXPECT_ERROR_CONTAINS(engine.Train(wrong_width, 5, Quiet()),
                        "input width");

  // Wrong label kind (unlabeled data on a cross-entropy plan).
  ASSERT_OK_AND_ASSIGN(Dataset unlabeled,
                       seeml::testing::MakeUnlabeledData(32, kInDim, 2));
  EXPECT_ERROR_CONTAINS(engine.Train(unlabeled, 5, Quiet()), "label kind");

  // Class label out of range for the plan's softmax width (3 classes).
  std::vector<uint8_t> labels(kBatch * sizeof(int32_t), 0);
  reinterpret_cast<int32_t*>(labels.data())[1] = 7;
  ASSERT_OK_AND_ASSIGN(
      Dataset bad_labels,
      Dataset::FromMemory(std::vector<float>(kBatch * kInDim, 0.5f),
                          std::move(labels), kBatch, kInDim, 1, 0));
  EXPECT_ERROR_CONTAINS(engine.Train(bad_labels, 5, Quiet()),
                        "outside [0, 3)");
}

TEST(UpdateEngineTrain, UsesDefaultStepsWhenZeroRequested) {
  UpdateConfig config = BaseConfig(kBatch);
  config.default_steps = 5;
  const std::vector<uint8_t> plan = CompilePlan(config);
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));

  ASSERT_OK_AND_ASSIGN(Dataset data, MakeClassificationData(64, kInDim, 3));
  ASSERT_OK_AND_ASSIGN(auto report, engine.Train(data, 0, Quiet()));
  EXPECT_EQ(report.steps, 5u);
  EXPECT_EQ(engine.step(), 5u);
}

TEST(UpdateEngineTrain, RejectsZeroStepsWithoutDefault) {
  UpdateConfig config = BaseConfig(kBatch);
  config.default_steps = 0;
  const std::vector<uint8_t> plan = CompilePlan(config);
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));

  ASSERT_OK_AND_ASSIGN(Dataset data, MakeClassificationData(64, kInDim, 4));
  EXPECT_ERROR_CONTAINS(engine.Train(data, 0, Quiet()), "no steps requested");
}

TEST(UpdateEngineTrain, ExecuteTrainOnceBumpsStepFromZero) {
  const std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));
  EXPECT_EQ(engine.step(), 0u);

  seeml::testing::FillSlots(
      engine, std::vector<float>(kBatch * kInDim, 0.1f), {0, 1, 2, 1});
  engine.ExecuteTrainOnce();  // AdamW bias correction is 1-indexed
  EXPECT_EQ(engine.step(), 1u);
  EXPECT_GT(engine.LossValue(), 0.0f);
}

TEST(UpdateEngineCheckpoint, RoundTripRestoresPersistentState) {
  ScopedTempDir dir;
  const std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));
  ASSERT_OK_AND_ASSIGN(Dataset data, MakeClassificationData(64, kInDim, 5));

  ASSERT_OK(engine.Train(data, 3, Quiet()));
  EXPECT_EQ(engine.step(), 3u);
  const std::string path = dir.File("ckpt.bin");
  ASSERT_OK(engine.SaveCheckpoint(path));
  std::vector<uint8_t> saved(engine.header().persistent_size);
  std::memcpy(saved.data(), engine.arena(), saved.size());

  // Diverge, then restore.
  ASSERT_OK(engine.Train(data, 4, Quiet()));
  EXPECT_EQ(engine.step(), 7u);
  ASSERT_OK(engine.LoadCheckpoint(path));
  EXPECT_EQ(engine.step(), 3u);
  EXPECT_EQ(std::memcmp(engine.arena(), saved.data(), saved.size()), 0);
}

TEST(UpdateEngineCheckpoint, RejectsIncompatibleCheckpoint) {
  ScopedTempDir dir;
  const std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));

  EXPECT_ERROR_CONTAINS(engine.LoadCheckpoint(dir.File("missing.bin")),
                        "no checkpoint");

  // A checkpoint from a plan with a different persistent layout (rank 2
  // instead of 4) must be refused.
  UpdateConfig other_config = BaseConfig(kBatch);
  other_config.lora.rank = 2;
  const std::vector<uint8_t> other_plan = CompilePlan(other_config);
  ASSERT_FALSE(other_plan.empty());
  UpdateEngine other;
  ASSERT_OK(other.LoadFromMemory(other_plan.data(), other_plan.size()));
  const std::string path = dir.File("other.bin");
  ASSERT_OK(other.SaveCheckpoint(path));

  EXPECT_ERROR_CONTAINS(engine.LoadCheckpoint(path),
                        "incompatible with plan");
}

TEST(UpdateEngineCommit, RequiresMergeFirst) {
  ScopedTempDir dir;
  const std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));
  EXPECT_ERROR_CONTAINS(
      engine.CommitToModel(dir.File("src.smf"), dir.File("out.smf")),
      "RunMerge() must precede");
}

TEST(UpdateEngineCommit, RejectsOutOfSyncModelFile) {
  ScopedTempDir dir;
  const std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));
  ASSERT_OK(engine.RunMerge());

  // The emit table's byte ranges cannot fit inside this 3-byte impostor.
  const std::string tiny = dir.File("tiny.smf");
  std::ofstream(tiny, std::ios::binary) << "smf";
  EXPECT_ERROR_CONTAINS(engine.CommitToModel(tiny, dir.File("out.smf")),
                        "out of sync");
}

}  // namespace
