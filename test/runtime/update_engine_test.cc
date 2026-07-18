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
#include "source/update_types.h"
#include "runtime/dataset.h"
#include "runtime/update_engine.h"
#include "source/hash.h"
#include "source/parallel_for.h"
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

/// Recomputes the plan's integrity hash after a deliberate mutation, so a
/// test reaches the specific validator it targets instead of tripping the
/// corruption gate. (The gate itself is covered by RejectsCorruptedBlob.)
void ResealPlan(std::vector<uint8_t>& plan) {
  constexpr size_t kHashAt = offsetof(PlanHeader, plan_hash);
  std::memset(plan.data() + kHashAt, 0, sizeof(uint64_t));
  const uint64_t h = Fnv1a64(plan.data(), plan.size());
  std::memcpy(plan.data() + kHashAt, &h, sizeof(h));
}

void PutHeader(std::vector<uint8_t>& plan, const PlanHeader& h) {
  std::memcpy(plan.data(), &h, sizeof(h));
  ResealPlan(plan);
}

TrainOptions Quiet() {
  TrainOptions options;
  options.log_every = 0;
  return options;
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

TEST(UpdateEngineLoad, RejectsCorruptedBlob) {
  std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  // One flipped bit anywhere in the blob — deliberately NOT resealed.
  plan[plan.size() / 2] ^= 0x40;
  UpdateEngine engine;
  EXPECT_ERROR_CONTAINS(engine.LoadFromMemory(plan.data(), plan.size()),
                        "hash mismatch");
}

TEST(UpdateEngineLoad, RejectsUnknownOpcode) {
  std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  const PlanHeader h = HeaderOf(plan);
  UpdateInstruction ins;
  std::memcpy(&ins, plan.data() + h.train_instr_offset, sizeof(ins));
  ins.opcode = 999;
  std::memcpy(plan.data() + h.train_instr_offset, &ins, sizeof(ins));
  ResealPlan(plan);

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
  ResealPlan(plan);

  UpdateEngine engine;
  EXPECT_ERROR_CONTAINS(engine.LoadFromMemory(plan.data(), plan.size()),
                        "out of bounds");
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

  // A checkpoint from any other plan must be refused — here one with a
  // different persistent layout (rank 2 instead of 4). The plan-hash
  // binding catches it before layout is even considered.
  UpdateConfig other_config = BaseConfig(kBatch);
  other_config.lora.rank = 2;
  const std::vector<uint8_t> other_plan = CompilePlan(other_config);
  ASSERT_FALSE(other_plan.empty());
  UpdateEngine other;
  ASSERT_OK(other.LoadFromMemory(other_plan.data(), other_plan.size()));
  const std::string path = dir.File("other.bin");
  ASSERT_OK(other.SaveCheckpoint(path));

  EXPECT_ERROR_CONTAINS(engine.LoadCheckpoint(path), "different plan");

  // Same persistent layout, different plan bytes (another LoRA seed): still
  // a foreign checkpoint, still refused.
  UpdateConfig same_layout = BaseConfig(kBatch);
  same_layout.lora.seed = 1234;
  const std::vector<uint8_t> twin_plan = CompilePlan(same_layout);
  ASSERT_FALSE(twin_plan.empty());
  UpdateEngine twin;
  ASSERT_OK(twin.LoadFromMemory(twin_plan.data(), twin_plan.size()));
  const std::string twin_path = dir.File("twin.bin");
  ASSERT_OK(twin.SaveCheckpoint(twin_path));
  EXPECT_ERROR_CONTAINS(engine.LoadCheckpoint(twin_path), "different plan");
}

TEST(UpdateEngineCheckpoint, RejectsCorruptedPayload) {
  ScopedTempDir dir;
  const std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));
  const std::string path = dir.File("ckpt.bin");
  engine.SetStep(41);
  ASSERT_OK(engine.SaveCheckpoint(path));
  ASSERT_OK(engine.LoadCheckpoint(path));
  EXPECT_EQ(engine.step(), 41u);

  // Flip the last payload byte on disk: the payload hash must catch it.
  {
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    f.seekg(-1, std::ios::end);
    char c;
    f.get(c);
    f.seekp(-1, std::ios::end);
    f.put(static_cast<char>(c ^ 0x11));
  }
  EXPECT_ERROR_CONTAINS(engine.LoadCheckpoint(path), "corrupt");
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
  // (An in-memory model carries no content hash, so the range check is the
  // active defense here; the hash binding is covered below.)
  const std::string tiny = dir.File("tiny.smf");
  std::ofstream(tiny, std::ios::binary) << "smf";
  EXPECT_ERROR_CONTAINS(engine.CommitToModel(tiny, dir.File("out.smf")),
                        "out of sync");
}

TEST(UpdateEngineCommit, RejectsModelFileThePlanWasNotCompiledFrom) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(kInDim, 10, 3, 1);
  SmfModel other = MakeMlp(kInDim, 10, 3, 99);  // same shapes, other bytes
  const std::string right = dir.File("right.smf");
  const std::string wrong = dir.File("wrong.smf");
  ASSERT_OK(SaveSmf(right, model));
  ASSERT_OK(SaveSmf(wrong, other));

  // Compile from the loaded file so the plan is hash-bound to it.
  ASSERT_OK_AND_ASSIGN(SmfModel saved, LoadSmf(right));
  auto compiled = UpdateCompiler(BaseConfig(kBatch)).Compile(saved);
  ASSERT_OK(compiled);
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled->plan.data(),
                                  compiled->plan.size()));
  ASSERT_OK(engine.RunMerge());

  // Every offset stays in range for the impostor — only the hash binding
  // stands between the plan and silent corruption of the wrong model.
  EXPECT_ERROR_CONTAINS(engine.CommitToModel(wrong, dir.File("never.smf")),
                        "source_model_hash");
  EXPECT_OK(engine.CommitToModel(right, dir.File("ok.smf")));
}

TEST(UpdateEngineValidate, EvaluateRunsWithoutMutatingState) {
  const std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));

  ASSERT_OK_AND_ASSIGN(Dataset data, MakeClassificationData(64, kInDim, 6));
  ASSERT_OK_AND_ASSIGN(float before, engine.Evaluate(data));
  EXPECT_GT(before, 0.0f);
  // Deterministic and side-effect free: the persistent state is untouched,
  // so a repeat evaluation returns the identical loss.
  ASSERT_OK_AND_ASSIGN(float again, engine.Evaluate(data));
  EXPECT_EQ(before, again);
  EXPECT_EQ(engine.step(), 0u);  // no training step consumed
}

TEST(UpdateEngineValidate, ValidationDrivesTheRegressionGate) {
  UpdateConfig config = BaseConfig(kBatch);
  config.optimizer.lr = 5e-3f;
  const std::vector<uint8_t> plan = CompilePlan(config);
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));

  ASSERT_OK_AND_ASSIGN(Dataset data, MakeClassificationData(320, kInDim, 7));
  ASSERT_OK_AND_ASSIGN(Dataset val, data.SplitValidation(0.2));
  data.EnableShuffle(3);

  TrainOptions options = Quiet();
  options.validation = &val;
  ASSERT_OK_AND_ASSIGN(auto report, engine.Train(data, 300, options));
  EXPECT_TRUE(report.has_validation);
  EXPECT_LT(report.val_final_loss, report.val_initial_loss);
  EXPECT_TRUE(report.improved());
}

TEST(UpdateEngineTrain, ShouldStopInterruptsAndLossCurveRecords) {
  const std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));
  ASSERT_OK_AND_ASSIGN(Dataset data, MakeClassificationData(64, kInDim, 8));

  TrainOptions options = Quiet();
  options.record_loss_curve = true;
  uint64_t polled = 0;
  options.should_stop = [&polled]() { return ++polled > 25; };
  ASSERT_OK_AND_ASSIGN(auto report, engine.Train(data, 400, options));
  EXPECT_TRUE(report.stopped_early);
  EXPECT_EQ(report.steps, 25u);
  EXPECT_EQ(report.loss_curve.size(), 25u);
}

TEST(UpdateEngineTrain, TrainingIsBitwiseInvariantAcrossThreadCounts) {
  // The whole update — batch pipeline, parallel kernels, ordered loss
  // reductions — must compute identical BITS at any pool width: with one
  // thread the feeder and every kernel run inline, with four the feeder
  // overlaps compute and the kernels fan out, and nothing may change.
  const std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());

  auto run = [&](size_t threads, std::vector<float>* curve,
                 std::vector<uint8_t>* persistent) {
    seeml::update::SetParallelThreadCount(threads);
    UpdateEngine engine;
    EXPECT_OK(engine.LoadFromMemory(plan.data(), plan.size()));
    auto data = MakeClassificationData(30, kInDim, 5);
    EXPECT_OK(data);
    data->EnableShuffle(3);  // batches cross epoch boundaries mid-run
    TrainOptions options = Quiet();
    options.record_loss_curve = true;
    auto report = engine.Train(*data, 25, options);
    EXPECT_OK(report);
    if (!report) return;
    *curve = report->loss_curve;
    persistent->assign(engine.arena(),
                       engine.arena() + engine.header().persistent_size);
  };

  std::vector<float> curve_serial, curve_wide;
  std::vector<uint8_t> persist_serial, persist_wide;
  run(1, &curve_serial, &persist_serial);
  run(4, &curve_wide, &persist_wide);
  seeml::update::SetParallelThreadCount(0);

  ASSERT_EQ(curve_serial.size(), 25u);
  ASSERT_EQ(curve_wide.size(), 25u);
  EXPECT_EQ(std::memcmp(curve_serial.data(), curve_wide.data(),
                        curve_serial.size() * sizeof(float)),
            0);
  ASSERT_FALSE(persist_serial.empty());
  EXPECT_TRUE(persist_serial == persist_wide);
}

}  // namespace
