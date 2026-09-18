// =============================================================================
// UpdateEngine tests: plan ingestion hardening (corrupt headers, out-of-bounds
// instructions), dataset/plan contract validation at Train() time, checkpoint
// round-trips, and the merge-before-commit protocol.
// =============================================================================

#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "compiler/driver/update_compiler.h"
#include "source/plan/update_types.h"
#include "runtime/feeder/dataset.h"
#include "runtime/custodian/checkpoint.h"
#include "runtime/engine/update_engine.h"
#include "source/identity/hash.h"
#include "source/parallel/parallel_for.h"
#include "compiler/frontend/ingressor/model_reader.h"
#include "compiler/frontend/ingressor/model_writer.h"
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
  const uint64_t h = PlanSelfHash(plan.data(), plan.size(), kHashAt);
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

TEST(UpdateEngineLoad, HeaderGemmTilesReachTheBackendAndNeverChangeBits) {
  // The compiler's kernel policy (v11) rides in the header; the engine
  // hands it to the backend after Bind, and a trained arena is bit for bit
  // the arena the defaults produce — the property that lets a measured
  // table (tool/autotune.py) pick any geometry the kernels accept.
  UpdateConfig tuned = BaseConfig(kBatch);
  tuned.gemm_tile_k = 8;
  tuned.gemm_tile_n = 4;  // tiny tiles: every boundary the 10-wide MLP has
  const std::vector<uint8_t> plan_default = CompilePlan(BaseConfig(kBatch));
  const std::vector<uint8_t> plan_tuned = CompilePlan(tuned);
  ASSERT_FALSE(plan_default.empty());
  ASSERT_FALSE(plan_tuned.empty());
  EXPECT_EQ(HeaderOf(plan_default).gemm_tile_k, 0u);
  EXPECT_EQ(HeaderOf(plan_tuned).gemm_tile_k, 8u);
  EXPECT_EQ(HeaderOf(plan_tuned).gemm_tile_n, 4u);
  EXPECT_EQ(HeaderOf(plan_tuned).version, kSeeuVersion);

  // Two datasets: a Dataset carries its shuffle position across Train
  // calls, so sharing one would feed the second engine different batches.
  auto data_default = MakeClassificationData(64, kInDim, 5);
  auto data_tuned = MakeClassificationData(64, kInDim, 5);
  ASSERT_TRUE(data_default.has_value() && data_tuned.has_value());
  UpdateEngine by_default, by_tuned;
  EXPECT_OK(by_default.LoadFromMemory(plan_default.data(), plan_default.size()));
  EXPECT_OK(by_tuned.LoadFromMemory(plan_tuned.data(), plan_tuned.size()));
  EXPECT_EQ(by_default.gemm_tiles().k, seeml::update_rt::kernels::kDefaultGemmTiles.k);
  EXPECT_EQ(by_tuned.gemm_tiles().k, 8u);
  EXPECT_EQ(by_tuned.gemm_tiles().n, 4u);
  EXPECT_OK(by_default.Train(*data_default, 12, Quiet()));
  EXPECT_OK(by_tuned.Train(*data_tuned, 12, Quiet()));
  ASSERT_EQ(by_default.header().arena_size, by_tuned.header().arena_size);
  EXPECT_EQ(std::memcmp(by_default.arena(), by_tuned.arena(),
                        by_default.header().arena_size),
            0);

  // The compiler refuses a K tile off the unroll before any plan exists.
  UpdateConfig bad = BaseConfig(kBatch);
  bad.gemm_tile_k = 6;
  EXPECT_TRUE(CompilePlan(bad).empty());
  // And the runtime refuses one smuggled into a sealed plan.
  std::vector<uint8_t> forged = plan_tuned;
  PlanHeader h = HeaderOf(forged);
  h.gemm_tile_k = 6;
  PutHeader(forged, h);
  UpdateEngine engine;
  EXPECT_ERROR_CONTAINS(engine.LoadFromMemory(forged.data(), forged.size()),
                        "4-wide unroll");
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
    // A plan from a future runtime: readable-range rejection, and the
    // message names the negotiated range so the operator knows which side
    // to upgrade.
    std::vector<uint8_t> bad = plan;
    PlanHeader h = HeaderOf(bad);
    h.version = kSeeuVersion + 1;
    PutHeader(bad, h);
    UpdateEngine engine;
    EXPECT_ERROR_CONTAINS(engine.LoadFromMemory(bad.data(), bad.size()),
                          "unsupported plan version");
    const auto r = engine.LoadFromMemory(bad.data(), bad.size());
    ASSERT_FALSE(r.has_value());
    EXPECT_STR_CONTAINS(r.error(),
                        "v" + std::to_string(kSeeuOldestReadable) + "..v" +
                            std::to_string(kSeeuVersion));
  }
  {
    // A plan below the semantic-compatibility floor (v2's source_model_hash
    // would mis-verify under the current hash): rejected, never misread.
    std::vector<uint8_t> bad = plan;
    PlanHeader h = HeaderOf(bad);
    h.version = kSeeuOldestReadable - 1;
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

TEST(UpdateEngineLoad, RejectsClassLabelPlanWithoutSoftmax) {
  // label_kind == 1 promises the raw dataset labels are validated against a
  // softmax class width; a plan claiming class labels while carrying no
  // softmax would leave them indexing kernels unvalidated. Regression: the
  // check must run against the candidate plan being loaded — a previous
  // implementation scanned the engine's (still empty) member programs and
  // the previous plan's header, so on a fresh engine it never fired.
  UpdateConfig config = BaseConfig(kBatch);
  config.loss = LossKind::kMse;
  std::vector<uint8_t> plan = CompilePlan(config);
  ASSERT_FALSE(plan.empty());
  PlanHeader h = HeaderOf(plan);
  ASSERT_EQ(h.label_kind, 2u);
  h.label_kind = 1;  // lie: class labels, but no softmax in any program
  h.label_bytes = kBatch * sizeof(int32_t);
  PutHeader(plan, h);

  UpdateEngine engine;
  EXPECT_ERROR_CONTAINS(engine.LoadFromMemory(plan.data(), plan.size()),
                        "carries no softmax");
}

TEST(UpdateEngineLoad, BindsSoftmaxLabelsToTheStagedSlot) {
  // The softmax pair indexes probability rows with raw i32 labels — safe
  // only because the feeder contract validates exactly the label slot's
  // `batch` staged entries. A plan pointing a softmax's label operand
  // anywhere else, or claiming more rows than the staged batch, or
  // claiming no class labels at all, would turn unvalidated bytes into
  // indices (an OOB write through the backward kernel) — it must never
  // reach dispatch.
  const std::vector<uint8_t> pristine = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(pristine.empty());
  const PlanHeader h = HeaderOf(pristine);

  // Locate the train program's softmax forward.
  uint64_t at = 0;
  UpdateInstruction ins;
  for (uint64_t i = 0; i < h.train_instr_count; ++i) {
    std::memcpy(&ins, pristine.data() + h.train_instr_offset + i * sizeof(ins),
                sizeof(ins));
    if (ins.opcode == static_cast<uint16_t>(OpCode::kSoftmaxXEntFwd)) {
      at = h.train_instr_offset + i * sizeof(ins);
      break;
    }
  }
  ASSERT_NE(at, 0u);

  auto patched = [&](auto&& mutate) {
    std::vector<uint8_t> plan = pristine;
    UpdateInstruction m = ins;
    mutate(m);
    std::memcpy(plan.data() + at, &m, sizeof(m));
    ResealPlan(plan);
    UpdateEngine engine;
    return engine.LoadFromMemory(plan.data(), plan.size());
  };

  EXPECT_ERROR_CONTAINS(
      patched([&](UpdateInstruction& m) { m.in[1] = h.input_ref; }),
      "not the plan's staged label slot");
  EXPECT_ERROR_CONTAINS(
      patched([&](UpdateInstruction& m) { m.out[0] = h.batch / 2; }),
      "row count disagrees");

  std::vector<uint8_t> no_labels = pristine;
  PlanHeader lied = h;
  lied.label_kind = 0;
  PutHeader(no_labels, lied);
  UpdateEngine engine;
  EXPECT_ERROR_CONTAINS(engine.LoadFromMemory(no_labels.data(),
                                              no_labels.size()),
                        "without class labels");
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
  // The compiler refuses a zero budget (E8), so no current plan has one;
  // the runtime guard is for the plans that predate that, reproduced here
  // by patching the header.
  std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  PlanHeader h = HeaderOf(plan);
  h.default_steps = 0;
  PutHeader(plan, h);
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));

  ASSERT_OK_AND_ASSIGN(Dataset data, MakeClassificationData(64, kInDim, 4));
  EXPECT_ERROR_CONTAINS(engine.Train(data, 0, Quiet()), "no steps requested");
}

TEST(UpdateEngineTrain, ExecuteTrainOnceInvalidatesAnEarlierMerge) {
  // Every parameter-mutating path must stale out previously materialized
  // deltas — including this probe hook, which was the one gap: commit
  // after RunMerge -> ExecuteTrainOnce would patch the model with deltas
  // that no longer match the adapters.
  const std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));
  ASSERT_OK(engine.RunMerge());
  engine.ExecuteTrainOnce();
  EXPECT_ERROR_CONTAINS(engine.CommitToModel("unused.smf", "unused.out"),
                        "RunMerge() must precede");
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

TEST(UpdateEngineCheckpoint, ResumedRunMatchesUninterruptedRunBitwise) {
  ScopedTempDir dir;
  const std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());

  // The uninterrupted twin: 6 steps over a shuffled feeder.
  UpdateEngine straight;
  ASSERT_OK(straight.LoadFromMemory(plan.data(), plan.size()));
  ASSERT_OK_AND_ASSIGN(Dataset d1, MakeClassificationData(10, kInDim, 5));
  d1.EnableShuffle(42);
  ASSERT_OK(straight.Train(d1, 6, Quiet()));

  // The interrupted twin: 3 steps, checkpoint, process death — then a cold
  // resume (fresh engine, freshly loaded feeder) for 3 more. With 10
  // samples the feeder crosses an epoch boundary mid-way, so the resumed
  // serving position must replay a reshuffle, not just a cursor.
  const std::string ck = dir.File("ck.bin");
  {
    UpdateEngine first;
    ASSERT_OK(first.LoadFromMemory(plan.data(), plan.size()));
    ASSERT_OK_AND_ASSIGN(Dataset d2, MakeClassificationData(10, kInDim, 5));
    d2.EnableShuffle(42);
    ASSERT_OK(first.Train(d2, 3, Quiet()));
    ASSERT_OK(first.SaveCheckpoint(ck));
  }
  UpdateEngine resumed;
  ASSERT_OK(resumed.LoadFromMemory(plan.data(), plan.size()));
  ASSERT_OK_AND_ASSIGN(Dataset d3, MakeClassificationData(10, kInDim, 5));
  d3.EnableShuffle(42);
  TrainOptions opt = Quiet();
  opt.checkpoint_path = ck;
  opt.resume = true;
  ASSERT_OK(resumed.Train(d3, 3, opt));
  // Both engines must actually have trained to step 6 — otherwise the byte
  // comparison below could pass vacuously on two equally-early states.
  EXPECT_EQ(straight.step(), 6u);
  EXPECT_EQ(resumed.step(), 6u);

  // "Power cuts are ordinary": the resumed run and the uninterrupted run
  // must be the same run, bit for bit.
  EXPECT_EQ(std::memcmp(straight.arena(), resumed.arena(),
                        straight.header().persistent_size), 0);
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

TEST(UpdateEngineCommit, VerifySourceModelFailsFastOnTheWrongFile) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(kInDim, 10, 3, 1);
  SmfModel other = MakeMlp(kInDim, 10, 3, 99);  // same shapes, other bytes
  const std::string right = dir.File("right.smf");
  const std::string wrong = dir.File("wrong.smf");
  ASSERT_OK(SaveSmf(right, model));
  ASSERT_OK(SaveSmf(wrong, other));

  ASSERT_OK_AND_ASSIGN(SmfModel saved, LoadSmf(right));
  auto compiled = UpdateCompiler(BaseConfig(kBatch)).Compile(saved);
  ASSERT_OK(compiled);
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled->plan.data(),
                                  compiled->plan.size()));

  // No train, no merge: the pre-flight check must work straight after load —
  // its whole purpose is refusing before any training cycles are spent.
  EXPECT_ERROR_CONTAINS(engine.VerifySourceModel(wrong), "source_model_hash");
  EXPECT_OK(engine.VerifySourceModel(right));
}

TEST(UpdateEngineCommit, EvaluateBetweenMergeAndCommitDoesNotCorruptDeltas) {
  // Train -> RunMerge -> Evaluate -> CommitToModel is the natural
  // "validate after merging" sequence. The merge's delta buffers must live
  // above the train/eval transients — if they shared offsets, the eval
  // forward pass here would overwrite every delta and commit would patch
  // activation bytes into the model as weight deltas.
  ScopedTempDir dir;
  SmfModel model = MakeMlp(kInDim, 10, 3, 1);
  const std::string src = dir.File("src.smf");
  ASSERT_OK(SaveSmf(src, model));
  ASSERT_OK_AND_ASSIGN(SmfModel saved, LoadSmf(src));
  auto compiled = UpdateCompiler(BaseConfig(kBatch)).Compile(saved);
  ASSERT_OK(compiled);

  auto commit = [&](bool eval_between,
                    const std::string& out) -> std::vector<uint8_t> {
    UpdateEngine engine;
    if (!engine.LoadFromMemory(compiled->plan.data(), compiled->plan.size()))
      return {};
    // Fresh-but-identical corpus per run: training is deterministic, so
    // both engines reach the same adapters.
    auto data = MakeClassificationData(64, kInDim, 6);
    if (!data) return {};
    if (!engine.Train(*data, 20, Quiet())) return {};
    if (!engine.RunMerge()) return {};
    if (eval_between && !engine.Evaluate(*data)) return {};
    if (!engine.CommitToModel(src, out)) return {};
    std::ifstream f(out, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f),
                                std::istreambuf_iterator<char>());
  };

  const std::vector<uint8_t> plain = commit(false, dir.File("plain.smf"));
  const std::vector<uint8_t> evald = commit(true, dir.File("evald.smf"));
  ASSERT_FALSE(plain.empty());
  ASSERT_FALSE(evald.empty());
  EXPECT_TRUE(plain == evald);
}

TEST(UpdateEngineCommit, TrainingAfterMergeStalesTheDeltas) {
  // A step executed after RunMerge changes the adapters; the materialized
  // deltas no longer describe them, so commit must demand a re-merge.
  ScopedTempDir dir;
  const std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));
  ASSERT_OK_AND_ASSIGN(Dataset data, MakeClassificationData(64, kInDim, 6));
  ASSERT_OK(engine.RunMerge());
  ASSERT_OK(engine.Train(data, 5, Quiet()));
  EXPECT_ERROR_CONTAINS(
      engine.CommitToModel(dir.File("src.smf"), dir.File("out.smf")),
      "RunMerge() must precede");
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

TEST(UpdateEngineValidate, EvaluateMetricsReportsExactAccuracy) {
  const std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));

  // 10 samples at batch 4: three eval batches, the last of which wraps two
  // duplicate rows. All inputs identical, so every probability row is the
  // same and exactly one class wins every argmax. With labels all set to
  // class k the exact accuracy is 1 for the winning k and 0 otherwise —
  // so the accuracies over k must sum to exactly 1. A wrapped duplicate
  // leaking into the count would break that identity.
  constexpr uint64_t kN = 10, kClasses = 3;
  const std::vector<float> one_input = {0.3f, -0.1f, 0.7f, 0.2f, -0.5f, 0.4f};
  std::vector<float> inputs;
  for (uint64_t i = 0; i < kN; ++i)
    inputs.insert(inputs.end(), one_input.begin(), one_input.end());

  float sum = 0.0f;
  for (uint64_t k = 0; k < kClasses; ++k) {
    std::vector<int32_t> labels(kN, static_cast<int32_t>(k));
    std::vector<uint8_t> label_bytes(kN * sizeof(int32_t));
    std::memcpy(label_bytes.data(), labels.data(), label_bytes.size());
    ASSERT_OK_AND_ASSIGN(
        Dataset data,
        Dataset::FromMemory(inputs, std::move(label_bytes), kN, kInDim,
                            /*label_kind=*/1, /*label_dim=*/0));
    ASSERT_OK_AND_ASSIGN(auto m, engine.EvaluateMetrics(data));
    EXPECT_TRUE(m.has_accuracy);
    EXPECT_TRUE(m.accuracy == 0.0f || m.accuracy == 1.0f);
    sum += m.accuracy;
  }
  EXPECT_EQ(sum, 1.0f);
}

TEST(UpdateEngineValidate, TrainReportCarriesValidationAccuracy) {
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
  ASSERT_OK_AND_ASSIGN(auto report, engine.Train(data, 100, options));
  EXPECT_TRUE(report.has_val_accuracy);
  EXPECT_GE(report.val_initial_accuracy, 0.0f);
  EXPECT_LE(report.val_initial_accuracy, 1.0f);
  EXPECT_GE(report.val_final_accuracy, 0.0f);
  EXPECT_LE(report.val_final_accuracy, 1.0f);
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

TEST(UpdateEngineAccumulation, StepsCountOptimizerStepsAndConsumeGBatches) {
  // A G = 3 plan: Train(5) is five optimizer steps over fifteen
  // micro-batches; the loss curve has five entries (each the mean of three
  // micro-batch losses); the persistent moments moved; and a resume after
  // three optimizer steps replays the feeder past 3 x 3 x batch rows so
  // the resumed run is bit-identical to the uninterrupted one.
  UpdateConfig config = BaseConfig(kBatch);
  config.grad_accum_steps = 3;
  const std::vector<uint8_t> plan = CompilePlan(config);
  ASSERT_FALSE(plan.empty());

  auto data_at = [&](uint64_t seed) -> Dataset {
    auto d = MakeClassificationData(26, kInDim, seed);
    if (!d) std::abort();  // the fixture is seeded and cannot fail
    d->EnableShuffle(5);
    return std::move(*d);
  };
  UpdateEngine straight;
  ASSERT_OK(straight.LoadFromMemory(plan.data(), plan.size()));
  EXPECT_EQ(straight.grad_accum_steps(), 3u);
  Dataset d1 = data_at(8);
  TrainOptions opts = Quiet();
  opts.record_loss_curve = true;
  ASSERT_OK_AND_ASSIGN(auto report, straight.Train(d1, 6, opts));
  EXPECT_EQ(report.steps, 6u);
  EXPECT_EQ(straight.step(), 6u);
  EXPECT_EQ(report.loss_curve.size(), 6u);
  const std::vector<uint8_t> straight_bytes(
      straight.arena(), straight.arena() + straight.header().persistent_size);

  ScopedTempDir tmp;
  const std::string ckpt = tmp.File("accum.ckpt");
  UpdateEngine first;
  ASSERT_OK(first.LoadFromMemory(plan.data(), plan.size()));
  Dataset d2 = data_at(8);
  ASSERT_OK(first.Train(d2, 3, Quiet()));
  ASSERT_OK(first.SaveCheckpoint(ckpt));
  UpdateEngine resumed;
  ASSERT_OK(resumed.LoadFromMemory(plan.data(), plan.size()));
  Dataset d3 = data_at(8);
  TrainOptions ropt = Quiet();
  ropt.checkpoint_path = ckpt;
  ropt.resume = true;
  ASSERT_OK(resumed.Train(d3, 3, ropt));
  EXPECT_EQ(resumed.step(), 6u);
  const std::vector<uint8_t> resumed_bytes(
      resumed.arena(), resumed.arena() + resumed.header().persistent_size);
  EXPECT_TRUE(straight_bytes == resumed_bytes);
}

TEST(UpdateEngineAccumulation, SgdWithClipTrainsAndImproves) {
  UpdateConfig config = BaseConfig(kBatch);
  config.optimizer.kind = OptimizerKind::kSgd;
  config.optimizer.clip_norm = 0.5f;
  config.optimizer.lr = 5e-2f;
  config.grad_accum_steps = 2;
  const std::vector<uint8_t> plan = CompilePlan(config);
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));
  auto data = MakeClassificationData(40, kInDim, 6);
  ASSERT_OK(data);
  data->EnableShuffle(2);
  TrainOptions options = Quiet();
  options.record_loss_curve = true;
  ASSERT_OK_AND_ASSIGN(auto report, engine.Train(*data, 30, options));
  EXPECT_EQ(report.steps, 30u);
  EXPECT_EQ(report.loss_curve.size(), 30u);
  EXPECT_LT(report.final_avg_loss, report.initial_avg_loss);
}

TEST(UpdateEngineAccumulation, IsBitwiseInvariantAcrossThreadCounts) {
  UpdateConfig config = BaseConfig(kBatch);
  config.grad_accum_steps = 2;
  const std::vector<uint8_t> plan = CompilePlan(config);
  ASSERT_FALSE(plan.empty());
  auto run = [&](size_t threads, std::vector<float>* curve,
                 std::vector<uint8_t>* persistent) {
    seeml::update::SetParallelThreadCount(threads);
    UpdateEngine engine;
    EXPECT_OK(engine.LoadFromMemory(plan.data(), plan.size()));
    auto data = MakeClassificationData(30, kInDim, 5);
    EXPECT_OK(data);
    data->EnableShuffle(3);
    TrainOptions options = Quiet();
    options.record_loss_curve = true;
    auto report = engine.Train(*data, 10, options);
    EXPECT_OK(report);
    if (!report) return;
    *curve = report->loss_curve;
    persistent->assign(engine.arena(),
                       engine.arena() + engine.header().persistent_size);
  };
  std::vector<float> c1, c4;
  std::vector<uint8_t> p1, p4;
  run(1, &c1, &p1);
  run(4, &c4, &p4);
  seeml::update::SetParallelThreadCount(0);
  ASSERT_EQ(c1.size(), 10u);
  EXPECT_TRUE(c1 == c4);
  EXPECT_TRUE(p1 == p4);
}

TEST(UpdateEngineGate, ImprovedByAndAccuracyHeld) {
  seeml::update_rt::TrainReport r;
  r.has_validation = true;
  r.val_initial_loss = 2.0f;
  r.val_final_loss = 1.9f;  // a 5% fall
  EXPECT_TRUE(r.improved());
  EXPECT_TRUE(r.ImprovedBy(0.0f));   // zero margin == improved()
  EXPECT_TRUE(r.ImprovedBy(0.05f));  // exactly at the margin
  EXPECT_FALSE(r.ImprovedBy(0.06f));
  r.val_final_loss = 1.9999999f;     // a calibration-only drift
  EXPECT_TRUE(r.ImprovedBy(0.0f));
  EXPECT_FALSE(r.ImprovedBy(0.01f));
  r.val_final_loss = 2.0f;           // no strict fall: fails at every margin
  EXPECT_FALSE(r.ImprovedBy(0.0f));
  EXPECT_FALSE(r.ImprovedBy(0.5f));
  // Without a split the training windows carry the gate.
  seeml::update_rt::TrainReport t;
  t.initial_avg_loss = 1.0f;
  t.final_avg_loss = 0.5f;
  EXPECT_TRUE(t.ImprovedBy(0.5f));
  EXPECT_FALSE(t.ImprovedBy(0.51f));
  // Accuracy: vacuous without a measurement, strict "did not drop" with one.
  EXPECT_TRUE(r.AccuracyHeld());
  r.has_val_accuracy = true;
  r.val_initial_accuracy = 0.40f;
  r.val_final_accuracy = 0.40f;
  EXPECT_TRUE(r.AccuracyHeld());
  r.val_final_accuracy = 0.39f;
  EXPECT_FALSE(r.AccuracyHeld());
}

TEST(UpdateEngineBackend, KindsParseAndNameRoundTrip) {
  using seeml::update_rt::BackendKind;
  using seeml::update_rt::BackendKindName;
  using seeml::update_rt::ParseBackendKind;
  for (BackendKind k : {BackendKind::kCpu, BackendKind::kMetal,
                        BackendKind::kAuto}) {
    auto parsed = ParseBackendKind(BackendKindName(k));
    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(static_cast<int>(*parsed), static_cast<int>(k));
  }
  EXPECT_FALSE(ParseBackendKind("gpu").has_value());
  EXPECT_FALSE(ParseBackendKind("CPU").has_value());
  EXPECT_FALSE(ParseBackendKind("").has_value());
}

TEST(UpdateEngineBackend, ExplicitCpuSelectionIsTheDefaultBitForBit) {
  // The seam is zero-cost on the reference path: an engine that never
  // called SelectBackend and one that selected cpu explicitly (before and
  // after the load) train to identical bits.
  const std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  auto run = [&](int mode, std::vector<float>* curve,
                 std::vector<uint8_t>* persistent) {
    UpdateEngine engine;
    if (mode == 1)
      EXPECT_OK(engine.SelectBackend(seeml::update_rt::BackendKind::kCpu));
    EXPECT_OK(engine.LoadFromMemory(plan.data(), plan.size()));
    if (mode == 2)
      EXPECT_OK(engine.SelectBackend(seeml::update_rt::BackendKind::kCpu));
    EXPECT_EQ(std::string(engine.backend_name()), "cpu");
    EXPECT_TRUE(engine.backend_note().empty());
    auto data = MakeClassificationData(30, kInDim, 5);
    EXPECT_OK(data);
    data->EnableShuffle(3);
    TrainOptions options = Quiet();
    options.record_loss_curve = true;
    auto report = engine.Train(*data, 12, options);
    EXPECT_OK(report);
    if (!report) return;
    *curve = report->loss_curve;
    persistent->assign(engine.arena(),
                       engine.arena() + engine.header().persistent_size);
  };
  std::vector<float> c0, c1, c2;
  std::vector<uint8_t> p0, p1, p2;
  run(0, &c0, &p0);
  run(1, &c1, &p1);
  run(2, &c2, &p2);
  ASSERT_EQ(c0.size(), 12u);
  EXPECT_TRUE(c0 == c1);
  EXPECT_TRUE(c0 == c2);
  EXPECT_TRUE(p0 == p1);
  EXPECT_TRUE(p0 == p2);
}

TEST(UpdateEngineBackend, AutoNeverFailsAndMetalFailsLoudlyWhenAbsent) {
  using seeml::update_rt::BackendKind;
  UpdateEngine engine;
  EXPECT_OK(engine.SelectBackend(BackendKind::kAuto));
  const bool on_metal = std::string(engine.backend_name()) == "metal";
  // auto resolves to a concrete backend and explains a fallback.
  EXPECT_TRUE(on_metal || !engine.backend_note().empty());
  UpdateEngine strict;
  auto r = strict.SelectBackend(BackendKind::kMetal);
  if (on_metal) {
    EXPECT_OK(r);
  } else {
    EXPECT_ERROR(r);
    EXPECT_EQ(std::string(strict.backend_name()), "cpu");  // unchanged
  }
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

TEST(UpdateEngineMerge, RejectsNonFiniteMergeDeltas) {
  // The training loop's loss guard reads the loss written BEFORE each
  // step's backward + optimizer, so a gradient that overflows on the final
  // executed step can poison the adapters unseen; RunMerge is the last
  // line between a NaN delta and the committed model file.
  UpdateConfig config = BaseConfig(kBatch);
  SmfModel model = MakeMlp(kInDim, 10, 3, 1);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));

  // Sanity: clean adapters merge fine.
  ASSERT_OK(engine.RunMerge());

  // Poison one element of the first adapter's A. B is zeros at step 0, and
  // NaN * 0 is NaN, so the materialized delta carries the poison.
  ASSERT_FALSE(compiled.adapters.empty());
  seeml::testing::WriteArenaF32(engine, compiled.adapters[0].a_ref, 0,
                                std::numeric_limits<float>::quiet_NaN());
  EXPECT_ERROR_CONTAINS(engine.RunMerge(), "non-finite");
}

TEST(UpdateEngineLr, CosineWithWarmupBoundaryBehavior) {
  // EffectiveLr scales every optimizer step and previously had no test at
  // all: pin the warmup ramp, the warmup boundary, the cosine floor at the
  // horizon, and the clamp past it.
  UpdateConfig config = BaseConfig(kBatch);
  config.optimizer.lr = 0.1f;
  config.optimizer.lr_schedule = LrSchedule::kCosineWithWarmup;
  config.optimizer.warmup_steps = 10;
  config.optimizer.min_lr_factor = 0.1f;
  config.default_steps = 100;
  const std::vector<uint8_t> plan = CompilePlan(config);
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));

  engine.SetStep(1);  // linear warmup: base * step / warmup
  EXPECT_NEAR(engine.EffectiveLr(), 0.1f * 1.0f / 10.0f, 1e-7);
  engine.SetStep(10);  // warmup boundary: exactly base
  EXPECT_NEAR(engine.EffectiveLr(), 0.1f, 1e-7);
  engine.SetStep(30);  // cosine decays monotonically between base and floor
  const float mid_early = engine.EffectiveLr();
  engine.SetStep(80);
  const float mid_late = engine.EffectiveLr();
  EXPECT_TRUE(mid_early > mid_late);
  EXPECT_TRUE(mid_early < 0.1f);
  EXPECT_TRUE(mid_late > 0.1f * 0.1f);
  engine.SetStep(100);  // horizon: the floor
  EXPECT_NEAR(engine.EffectiveLr(), 0.1f * 0.1f, 1e-6);
  engine.SetStep(1000);  // past the horizon: clamped at the floor
  EXPECT_NEAR(engine.EffectiveLr(), 0.1f * 0.1f, 1e-6);
}

// --- The run's horizon (E8, #91) ----------------------------------------------

/// A cosine plan with a 1000-step compiled budget and the default floor.
std::vector<uint8_t> CosinePlan(uint64_t warmup = 0) {
  UpdateConfig config = BaseConfig(kBatch);
  config.optimizer.lr = 0.1f;
  config.optimizer.lr_schedule = LrSchedule::kCosineWithWarmup;
  config.optimizer.warmup_steps = warmup;
  config.default_steps = 1000;
  return CompilePlan(config);
}

TEST(UpdateEngineLr, TheHorizonIsTheRunNotTheCompiledBudget) {
  // The same 1000-step plan, run for 20 steps and for 60: each anneals
  // over ITS OWN length and reaches the floor on its last step. Before E8
  // the 20-step run ended at ~99.9% of the base rate (never annealed), and
  // a run longer than the budget idled at the floor for the excess.
  const std::vector<uint8_t> plan = CosinePlan();
  ASSERT_FALSE(plan.empty());
  EXPECT_EQ(HeaderOf(plan).min_lr_factor, 0.1f);  // the default floor
  for (const uint64_t steps : {uint64_t{20}, uint64_t{60}}) {
    UpdateEngine engine;
    ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));
    ASSERT_OK_AND_ASSIGN(Dataset data, MakeClassificationData(64, kInDim, 3));
    ASSERT_OK(engine.Train(data, steps, Quiet()));
    EXPECT_EQ(engine.horizon(), steps);
    EXPECT_EQ(engine.step(), steps);
    EXPECT_NEAR(engine.EffectiveLr(), 0.1f * 0.1f, 1e-7);  // at the floor
    engine.SetStep(steps / 2);  // and genuinely mid-anneal half way
    EXPECT_NEAR(engine.EffectiveLr(), 0.1f * (0.1f + 0.9f * 0.5f), 1e-6);
    engine.SetStep(1);
    EXPECT_TRUE(engine.EffectiveLr() > 0.09f);
  }
  // No run yet: the hook falls back to the compiled budget, as before.
  UpdateEngine idle;
  ASSERT_OK(idle.LoadFromMemory(plan.data(), plan.size()));
  EXPECT_EQ(idle.horizon(), 1000u);
  idle.SetStep(5000);  // past the horizon with the DEFAULT floor: 0.1, not 0
  EXPECT_NEAR(idle.EffectiveLr(), 0.1f * 0.1f, 1e-7);
}

TEST(UpdateEngineLr, ResumeTrainsTheRemainderOnTheSameSchedule) {
  // An update interrupted at step 12 of 30 and resumed with NO step count
  // trains exactly 18 more, anneals to the floor at step 30, and commits
  // the same bits as the uninterrupted run. Before E8 the resume ran a
  // whole extra default budget — 1000 steps — all of it past the horizon.
  const std::vector<uint8_t> plan = CosinePlan(/*warmup=*/4);
  ASSERT_FALSE(plan.empty());
  auto data_at = [&]() -> Dataset {
    auto d = MakeClassificationData(64, kInDim, 9);
    if (!d) std::abort();
    d->EnableShuffle(5);
    return std::move(*d);
  };
  UpdateEngine straight;
  ASSERT_OK(straight.LoadFromMemory(plan.data(), plan.size()));
  Dataset d1 = data_at();
  TrainOptions record = Quiet();
  record.record_loss_curve = true;
  ASSERT_OK_AND_ASSIGN(auto whole, straight.Train(d1, 30, record));
  const std::vector<uint8_t> want(
      straight.arena(), straight.arena() + straight.header().persistent_size);

  ScopedTempDir tmp;
  const std::string ckpt = tmp.File("horizon.ckpt");
  UpdateEngine first;
  ASSERT_OK(first.LoadFromMemory(plan.data(), plan.size()));
  Dataset d2 = data_at();
  TrainOptions interrupted = Quiet();
  interrupted.checkpoint_path = ckpt;
  interrupted.checkpoint_every = 1;
  uint64_t polls = 0;
  interrupted.should_stop = [&] { return polls++ == 12; };  // "power cut"
  ASSERT_OK_AND_ASSIGN(auto cut, first.Train(d2, 30, interrupted));
  EXPECT_TRUE(cut.stopped_early);
  EXPECT_EQ(first.step(), 12u);

  UpdateEngine resumed;
  ASSERT_OK(resumed.LoadFromMemory(plan.data(), plan.size()));
  Dataset d3 = data_at();
  TrainOptions ropt = Quiet();
  ropt.checkpoint_path = ckpt;
  ropt.resume = true;
  ropt.record_loss_curve = true;
  ASSERT_OK_AND_ASSIGN(auto rest, resumed.Train(d3, /*steps=*/0, ropt));
  EXPECT_EQ(rest.steps, 18u);           // the remainder, not a new budget
  EXPECT_EQ(resumed.step(), 30u);
  EXPECT_EQ(resumed.horizon(), 30u);    // restored from the checkpoint
  EXPECT_NEAR(resumed.EffectiveLr(), 0.1f * 0.1f, 1e-7);
  ASSERT_EQ(rest.loss_curve.size(), 18u);
  for (size_t i = 0; i < 18; ++i)
    EXPECT_EQ(rest.loss_curve[i], whole.loss_curve[12 + i]);
  const std::vector<uint8_t> got(
      resumed.arena(), resumed.arena() + resumed.header().persistent_size);
  EXPECT_TRUE(want == got);

  // A finished run resumed again has nothing left: zero steps, no error.
  ASSERT_OK(resumed.SaveCheckpoint(ckpt));
  UpdateEngine done;
  ASSERT_OK(done.LoadFromMemory(plan.data(), plan.size()));
  Dataset d4 = data_at();
  ASSERT_OK_AND_ASSIGN(auto nothing, done.Train(d4, 0, ropt));
  EXPECT_EQ(nothing.steps, 0u);
  EXPECT_EQ(done.step(), 30u);

  // An explicit count on resume stretches the schedule to cover it...
  UpdateEngine more;
  ASSERT_OK(more.LoadFromMemory(plan.data(), plan.size()));
  Dataset d5 = data_at();
  ASSERT_OK(more.Train(d5, 10, ropt));
  EXPECT_EQ(more.step(), 40u);
  EXPECT_EQ(more.horizon(), 40u);
  // ... and an explicit horizon trains the head of a longer schedule.
  UpdateEngine head;
  ASSERT_OK(head.LoadFromMemory(plan.data(), plan.size()));
  Dataset d6 = data_at();
  TrainOptions hopt = Quiet();
  hopt.horizon_steps = 30;
  hopt.record_loss_curve = true;
  ASSERT_OK_AND_ASSIGN(auto early, head.Train(d6, 12, hopt));
  for (size_t i = 0; i < 12; ++i)
    EXPECT_EQ(early.loss_curve[i], whole.loss_curve[i]);
  hopt.horizon_steps = 5;  // a horizon that ends before the run does
  UpdateEngine bad;
  ASSERT_OK(bad.LoadFromMemory(plan.data(), plan.size()));
  Dataset d7 = data_at();
  EXPECT_ERROR_CONTAINS(bad.Train(d7, 12, hopt), "ends before the run");
}

TEST(UpdateEngineLr, V3CheckpointsStillResume) {
  // A pre-E8 (v3) checkpoint has no horizon: it resumes on the plan's
  // compiled default, exactly as it always did. Reproduced by rewriting a
  // v4 file: drop the 8-byte horizon and set the version word back.
  const std::vector<uint8_t> plan = CompilePlan(BaseConfig(kBatch));
  ASSERT_FALSE(plan.empty());
  ScopedTempDir tmp;
  const std::string v4 = tmp.File("v4.ckpt"), v3 = tmp.File("v3.ckpt");
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));
  ASSERT_OK_AND_ASSIGN(Dataset data, MakeClassificationData(64, kInDim, 3));
  ASSERT_OK(engine.Train(data, 4, Quiet()));
  ASSERT_OK(engine.SaveCheckpoint(v4));
  std::ifstream in(v4, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
  const uint32_t version3 = 3;
  std::memcpy(bytes.data() + 4, &version3, sizeof(version3));
  bytes.erase(bytes.begin() + 40, bytes.begin() + 112);  // the v4 + v5 tails
  std::ofstream(v3, std::ios::binary).write(bytes.data(),
                                            static_cast<std::streamsize>(
                                                bytes.size()));
  UpdateEngine old;
  ASSERT_OK(old.LoadFromMemory(plan.data(), plan.size()));
  ASSERT_OK(old.LoadCheckpoint(v3));
  EXPECT_EQ(old.step(), 4u);
  EXPECT_EQ(old.horizon(), old.header().default_steps);
  const std::vector<uint8_t> a(engine.arena(),
                               engine.arena() + engine.header().persistent_size);
  const std::vector<uint8_t> b(old.arena(),
                               old.arena() + old.header().persistent_size);
  EXPECT_TRUE(a == b);
}

// =============================================================================
// The best evaluated state (E9, #92)
// =============================================================================

namespace best_state {

std::vector<uint8_t> Persistent(UpdateEngine& e) {
  return std::vector<uint8_t>(e.arena(),
                              e.arena() + e.header().persistent_size);
}

/// A classification set drawn PORTABLY: mt19937_64's raw output is fixed by
/// the standard, and the floats are derived from it here by plain
/// arithmetic — std::normal_distribution is implementation-defined, and
/// gave libstdc++ and libc++ different corpora (and so different training
/// curves) for one seed. Every sample shares one rule, w_true from `rule`.
Dataset PortableSet(uint64_t n, uint64_t sample_seed, uint64_t rule) {
  auto unit = [](std::mt19937_64& g) {  // [-1, 1), exact in f32
    return static_cast<float>(static_cast<double>(g() >> 40) /
                                  static_cast<double>(1ull << 23) -
                              1.0);
  };
  std::mt19937_64 wr(rule), xr(sample_seed);
  std::vector<float> w(kInDim), x(n * kInDim);
  for (float& v : w) v = unit(wr);
  std::vector<uint8_t> labels(n * sizeof(int32_t));
  auto* lab = reinterpret_cast<int32_t*>(labels.data());
  for (uint64_t i = 0; i < n; ++i) {
    float dot = 0.0f;
    for (int64_t c = 0; c < kInDim; ++c) {
      x[i * kInDim + c] = unit(xr);
      const float prod = x[i * kInDim + c] * w[c];
      dot += prod;
    }
    lab[i] = dot > 0.0f ? 1 : 0;
  }
  auto d = Dataset::FromMemory(std::move(x), std::move(labels), n, kInDim,
                               /*label_kind=*/1, /*label_dim=*/0);
  if (!d) std::abort();
  return std::move(*d);
}

/// A tiny training set and a held-out set drawn from one rule, at a
/// learning rate where the held-out loss dips and then climbs — the
/// small-corpus shape the issue describes.
Dataset TrainSet() {
  Dataset d = PortableSet(8, 101, 7);
  d.EnableShuffle(3);
  return d;
}
Dataset ValSet() { return PortableSet(96, 202, 7); }
std::vector<uint8_t> OverfitPlan() {
  UpdateConfig config = BaseConfig(kBatch);
  config.optimizer.lr = 0.02f;
  return CompilePlan(config, 4);
}

}  // namespace best_state

TEST(UpdateEngineBest, PeriodicEvaluationChangesNoTrainingBits) {
  using namespace best_state;
  const std::vector<uint8_t> plan = OverfitPlan();
  ASSERT_FALSE(plan.empty());
  ScopedTempDir tmp;
  const std::string ckpt = tmp.File("last.ckpt");

  UpdateEngine plain;
  ASSERT_OK(plain.LoadFromMemory(plan.data(), plan.size()));
  Dataset d1 = TrainSet(), v1 = ValSet();
  TrainOptions popt = Quiet();
  popt.validation = &v1;
  popt.record_loss_curve = true;
  ASSERT_OK_AND_ASSIGN(auto untracked, plain.Train(d1, 24, popt));
  EXPECT_FALSE(untracked.best_tracked);
  const std::vector<uint8_t> last = Persistent(plain);

  UpdateEngine tracked;
  ASSERT_OK(tracked.LoadFromMemory(plan.data(), plan.size()));
  Dataset d2 = TrainSet(), v2 = ValSet();
  TrainOptions topt = Quiet();
  topt.validation = &v2;
  topt.record_loss_curve = true;
  topt.eval_every = 3;
  topt.checkpoint_path = ckpt;
  topt.checkpoint_every = 24;  // the last state, saved before the restore
  ASSERT_OK_AND_ASSIGN(auto best, tracked.Train(d2, 24, topt));
  EXPECT_TRUE(best.best_tracked);
  EXPECT_EQ(best.eval_every, 3u);
  EXPECT_EQ(best.evaluations, 7u);  // steps 3..21; 24 is the bracket
  ASSERT_EQ(best.loss_curve.size(), untracked.loss_curve.size());
  for (size_t i = 0; i < best.loss_curve.size(); ++i)
    EXPECT_EQ(best.loss_curve[i], untracked.loss_curve[i]);
  EXPECT_EQ(best.val_initial_loss, untracked.val_initial_loss);
  EXPECT_EQ(best.val_last_loss, untracked.val_final_loss);
  // The endpoint the tracked run reached is the untracked run's, byte for
  // byte: the checkpoint written at step 24 holds it.
  UpdateEngine peek;
  ASSERT_OK(peek.LoadFromMemory(plan.data(), plan.size()));
  ASSERT_OK(peek.LoadCheckpoint(ckpt));
  EXPECT_TRUE(Persistent(peek) == last);
}

TEST(UpdateEngineBest, CommitsTheBestEvaluatedStateNotTheLast) {
  using namespace best_state;
  const std::vector<uint8_t> plan = OverfitPlan();
  ASSERT_FALSE(plan.empty());
  const uint64_t steps = 11;

  // The oracle: the held-out loss of every state from step 0 on, from an
  // untracked run stepped one at a time (each Train call evaluates its
  // endpoint).
  std::vector<float> curve;
  {
    UpdateEngine e;
    ASSERT_OK(e.LoadFromMemory(plan.data(), plan.size()));
    Dataset d = TrainSet(), v = ValSet();
    TrainOptions o = Quiet();
    o.validation = &v;
    for (uint64_t s = 0; s < steps; ++s) {
      ASSERT_OK_AND_ASSIGN(auto r, e.Train(d, 1, o));
      if (s == 0) curve.push_back(r.val_initial_loss);
      curve.push_back(r.val_final_loss);
    }
  }
  size_t best_step = 0;
  for (size_t i = 1; i < curve.size(); ++i)
    if (curve[i] < curve[best_step]) best_step = i;
  // The scenario the issue describes: the endpoint is past the minimum,
  // yet still below step 0 — the old gate committed it.
  ASSERT_GT(best_step, 0u);
  ASSERT_LT(best_step, steps);
  ASSERT_LT(curve[best_step], curve.back());
  ASSERT_LT(curve.back(), curve[0]);

  UpdateEngine tracked;
  ASSERT_OK(tracked.LoadFromMemory(plan.data(), plan.size()));
  Dataset d = TrainSet(), v = ValSet();
  TrainOptions o = Quiet();
  o.validation = &v;
  o.eval_every = 1;
  ASSERT_OK_AND_ASSIGN(auto report, tracked.Train(d, steps, o));
  EXPECT_EQ(report.steps, steps);
  EXPECT_EQ(report.best_step, best_step);
  EXPECT_EQ(report.val_final_loss, curve[best_step]);
  EXPECT_EQ(report.val_last_loss, curve.back());
  EXPECT_TRUE(report.improved());
  EXPECT_LT(report.val_final_loss, report.val_last_loss);

  // What the arena holds is the state of step best_step, bit for bit: a
  // fresh run of exactly that many steps produces it.
  UpdateEngine fresh;
  ASSERT_OK(fresh.LoadFromMemory(plan.data(), plan.size()));
  Dataset d2 = TrainSet();
  ASSERT_OK(fresh.Train(d2, best_step, Quiet()));
  EXPECT_TRUE(Persistent(tracked) == Persistent(fresh));
  // And the step counter still says where the run ended.
  EXPECT_EQ(tracked.step(), steps);
}

TEST(UpdateEngineBest, ResumeKeepsTheSourceModelScoreAndTheBest) {
  using namespace best_state;
  const std::vector<uint8_t> plan = OverfitPlan();
  ASSERT_FALSE(plan.empty());
  ScopedTempDir tmp;
  const std::string ckpt = tmp.File("best.ckpt");

  UpdateEngine straight;
  ASSERT_OK(straight.LoadFromMemory(plan.data(), plan.size()));
  Dataset d1 = TrainSet(), v1 = ValSet();
  TrainOptions o = Quiet();
  o.validation = &v1;
  o.eval_every = 2;
  ASSERT_OK_AND_ASSIGN(auto whole, straight.Train(d1, 11, o));
  // The best comes after the interruption below, at an even step: a
  // resume that keyed the cadence on its own step count (evaluating at 5,
  // 7, 9 instead of 4, 6, 8, 10) would keep a different best (#124 CI).
  ASSERT_GT(whole.best_step, 3u);
  const std::vector<uint8_t> want = Persistent(straight);

  UpdateEngine first;
  ASSERT_OK(first.LoadFromMemory(plan.data(), plan.size()));
  Dataset d2 = TrainSet(), v2 = ValSet();
  TrainOptions cut = o;
  cut.validation = &v2;
  cut.checkpoint_path = ckpt;
  cut.checkpoint_every = 1;
  uint64_t polls = 0;
  cut.should_stop = [&] { return polls++ == 3; };
  ASSERT_OK_AND_ASSIGN(auto head, first.Train(d2, 11, cut));
  EXPECT_TRUE(head.stopped_early);
  EXPECT_EQ(first.step(), 3u);

  UpdateEngine resumed;
  ASSERT_OK(resumed.LoadFromMemory(plan.data(), plan.size()));
  Dataset d3 = TrainSet(), v3 = ValSet();
  TrainOptions ropt = o;
  ropt.validation = &v3;
  ropt.checkpoint_path = ckpt;
  ropt.resume = true;
  ASSERT_OK_AND_ASSIGN(auto rest, resumed.Train(d3, 0, ropt));
  EXPECT_EQ(rest.steps, 8u);
  // The gate's "before" is still the source model's score, not the
  // resumed adapter's; the best and the committed bytes are the whole
  // run's.
  EXPECT_EQ(rest.val_initial_loss, whole.val_initial_loss);
  EXPECT_EQ(rest.best_step, whole.best_step);
  EXPECT_EQ(rest.val_final_loss, whole.val_final_loss);
  EXPECT_EQ(rest.val_last_loss, whole.val_last_loss);
  EXPECT_TRUE(Persistent(resumed) == want);
}

TEST(UpdateEngineBest, AResumedStartIsScoredAsItselfNotAsTheSource) {
  // The first run evaluates but does not track (eval_every 0), so its
  // checkpoint carries the source model's score and no best state. It is
  // interrupted at step 7 — this corpus's best — and the resume tracks.
  // Every later state is worse than step 7 yet better than the source, so
  // a resume that labelled its starting segment with the SOURCE score
  // would let step 8 displace it (ultrareview on #125). The start must be
  // scored as itself, and what the run reports must be what it holds.
  using namespace best_state;
  const std::vector<uint8_t> plan = OverfitPlan();
  ASSERT_FALSE(plan.empty());
  ScopedTempDir tmp;
  const std::string ckpt = tmp.File("untracked.ckpt");
  UpdateEngine first;
  ASSERT_OK(first.LoadFromMemory(plan.data(), plan.size()));
  Dataset d1 = TrainSet(), v1 = ValSet();
  TrainOptions cut = Quiet();
  cut.validation = &v1;
  cut.checkpoint_path = ckpt;
  cut.checkpoint_every = 1;
  uint64_t polls = 0;
  cut.should_stop = [&] { return polls++ == 7; };
  ASSERT_OK_AND_ASSIGN(auto head, first.Train(d1, 11, cut));
  ASSERT_TRUE(head.stopped_early);
  ASSERT_FALSE(head.best_tracked);
  ASSERT_EQ(first.step(), 7u);

  UpdateEngine resumed;
  ASSERT_OK(resumed.LoadFromMemory(plan.data(), plan.size()));
  Dataset d2 = TrainSet(), v2 = ValSet();
  TrainOptions ropt = Quiet();
  ropt.validation = &v2;
  ropt.checkpoint_path = ckpt;
  ropt.resume = true;
  ropt.eval_every = 1;
  ASSERT_OK_AND_ASSIGN(auto rest, resumed.Train(d2, 0, ropt));
  EXPECT_EQ(rest.steps, 4u);
  EXPECT_EQ(rest.val_initial_loss, head.val_initial_loss);  // the source's
  EXPECT_EQ(rest.best_step, 7u);
  EXPECT_LT(rest.val_final_loss, rest.val_last_loss);
  // The reported score is the held state's score.
  Dataset v3 = ValSet();
  ASSERT_OK_AND_ASSIGN(float held, resumed.Evaluate(v3));
  EXPECT_EQ(held, rest.val_final_loss);
  // And the held state is step 7's, byte for byte.
  UpdateEngine fresh;
  ASSERT_OK(fresh.LoadFromMemory(plan.data(), plan.size()));
  Dataset d3 = TrainSet();
  ASSERT_OK(fresh.Train(d3, 7, Quiet()));
  EXPECT_TRUE(Persistent(resumed) == Persistent(fresh));
}

TEST(UpdateEngineBest, ResumeRefusesADifferentSeedOrSplit) {
  using namespace best_state;
  const std::vector<uint8_t> plan = OverfitPlan();
  ASSERT_FALSE(plan.empty());
  ScopedTempDir tmp;
  const std::string ckpt = tmp.File("bound.ckpt");
  {
    UpdateEngine e;
    ASSERT_OK(e.LoadFromMemory(plan.data(), plan.size()));
    Dataset d = TrainSet(), v = ValSet();
    TrainOptions o = Quiet();
    o.validation = &v;
    o.checkpoint_path = ckpt;
    o.checkpoint_every = 4;
    ASSERT_OK(e.Train(d, 8, o));
  }
  TrainOptions ropt = Quiet();
  ropt.checkpoint_path = ckpt;
  ropt.resume = true;

  // Another shuffle seed.
  UpdateEngine seed;
  ASSERT_OK(seed.LoadFromMemory(plan.data(), plan.size()));
  ASSERT_OK_AND_ASSIGN(Dataset d1, MakeClassificationData(8, kInDim, 21));
  d1.EnableShuffle(4);
  Dataset v1 = ValSet();
  ropt.validation = &v1;
  EXPECT_ERROR_CONTAINS(seed.Train(d1, 0, ropt), "resume refused");

  // Another split: a validation set of a different size.
  UpdateEngine split;
  ASSERT_OK(split.LoadFromMemory(plan.data(), plan.size()));
  Dataset d2 = TrainSet();
  ASSERT_OK_AND_ASSIGN(Dataset v2, MakeClassificationData(40, kInDim, 21));
  ropt.validation = &v2;
  EXPECT_ERROR_CONTAINS(split.Train(d2, 0, ropt), "resume refused");

  // No validation set at all, where the run had one.
  UpdateEngine none;
  ASSERT_OK(none.LoadFromMemory(plan.data(), plan.size()));
  Dataset d3 = TrainSet();
  ropt.validation = nullptr;
  EXPECT_ERROR_CONTAINS(none.Train(d3, 0, ropt), "resume refused");

  // The driver's pre-check reads the same binding without touching state.
  ASSERT_OK_AND_ASSIGN(auto peek,
                       seeml::update_rt::PeekCheckpointRecord(ckpt));
  EXPECT_TRUE(peek.has_binding);
  EXPECT_EQ(peek.train_samples, 8u);
  EXPECT_EQ(peek.val_samples, 96u);
  EXPECT_EQ(peek.shuffle_origin, TrainSet().shuffle_origin());
  EXPECT_NE(peek.shuffle_origin, d1.shuffle_origin());

  // The first run's seed and split resume.
  UpdateEngine ok;
  ASSERT_OK(ok.LoadFromMemory(plan.data(), plan.size()));
  Dataset d4 = TrainSet(), v4 = ValSet();
  ropt.validation = &v4;
  ASSERT_OK(ok.Train(d4, 0, ropt));
}

TEST(UpdateEngineBest, PatienceStopsARunThatStoppedImproving) {
  using namespace best_state;
  const std::vector<uint8_t> plan = OverfitPlan();
  ASSERT_FALSE(plan.empty());
  UpdateEngine e;
  ASSERT_OK(e.LoadFromMemory(plan.data(), plan.size()));
  Dataset d = TrainSet(), v = ValSet();
  TrainOptions o = Quiet();
  o.validation = &v;
  o.eval_every = 1;
  o.patience = 3;
  ASSERT_OK_AND_ASSIGN(auto report, e.Train(d, 200, o));
  EXPECT_TRUE(report.stopped_by_patience);
  EXPECT_FALSE(report.stopped_early);
  EXPECT_LT(report.steps, 200u);
  EXPECT_EQ(report.patience, 3u);
  // Three evaluations past the best, none better: the run ends three
  // steps after the best state, which is what it holds.
  EXPECT_EQ(report.steps, report.best_step + 3);
  UpdateEngine fresh;
  ASSERT_OK(fresh.LoadFromMemory(plan.data(), plan.size()));
  Dataset d2 = TrainSet();
  ASSERT_OK(fresh.Train(d2, report.best_step, Quiet()));
  EXPECT_TRUE(Persistent(e) == Persistent(fresh));
}

TEST(UpdateEngineBest, AutoEvaluatesEveryTenthOfTheRun) {
  using namespace best_state;
  const std::vector<uint8_t> plan = OverfitPlan();
  ASSERT_FALSE(plan.empty());
  UpdateEngine e;
  ASSERT_OK(e.LoadFromMemory(plan.data(), plan.size()));
  Dataset d = TrainSet(), v = ValSet();
  TrainOptions o = Quiet();
  o.validation = &v;
  o.eval_every = TrainOptions::kEvalEveryAuto;
  ASSERT_OK_AND_ASSIGN(auto report, e.Train(d, 25, o));
  EXPECT_EQ(report.eval_every, 2u);  // 25 / 10
  EXPECT_EQ(report.evaluations, 12u);  // steps 2..24
  // Without a validation set there is nothing to evaluate: no tracking.
  UpdateEngine bare;
  ASSERT_OK(bare.LoadFromMemory(plan.data(), plan.size()));
  Dataset d2 = TrainSet();
  o.validation = nullptr;
  ASSERT_OK_AND_ASSIGN(auto none, bare.Train(d2, 25, o));
  EXPECT_FALSE(none.best_tracked);
  EXPECT_EQ(none.eval_every, 0u);
}

}  // namespace
