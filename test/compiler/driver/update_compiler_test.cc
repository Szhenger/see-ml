// =============================================================================
// UpdateCompiler tests: plan assembly (header contract, section layout,
// segmented arena binding, debug hooks), per-loss configuration, optimizer
// selection, and the compile-time error surface.
// =============================================================================

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "compiler/driver/update_compiler.h"
#include "source/plan/update_types.h"
#include "source/parallel/parallel_for.h"
#include "source/language/model_format.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"

namespace {

using namespace seeml::update;
using seeml::testing::BaseConfig;
using seeml::testing::MakeMlp;
using seeml::testing::MakeTiedMlp;

constexpr int64_t kInDim = 6;
constexpr int64_t kHidden = 10;
constexpr int64_t kOutDim = 3;
constexpr int64_t kBatch = 4;

PlanHeader HeaderOf(const CompiledUpdate& compiled) {
  PlanHeader h;
  std::memcpy(&h, compiled.plan.data(), sizeof(h));
  return h;
}

std::vector<UpdateInstruction> TrainProgramOf(const CompiledUpdate& compiled) {
  const PlanHeader h = HeaderOf(compiled);
  std::vector<UpdateInstruction> instrs(h.train_instr_count);
  std::memcpy(instrs.data(), compiled.plan.data() + h.train_instr_offset,
              h.train_instr_count * sizeof(UpdateInstruction));
  return instrs;
}

size_t CountOpcode(const std::vector<UpdateInstruction>& instrs, OpCode oc) {
  size_t n = 0;
  for (const UpdateInstruction& ins : instrs)
    if (ins.opcode == static_cast<uint16_t>(oc)) ++n;
  return n;
}

TEST(UpdateCompiler, PlanHeaderContract) {
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 1);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(BaseConfig(kBatch)).Compile(model));

  const PlanHeader h = HeaderOf(compiled);
  EXPECT_EQ(h.magic, kSeeuMagic);
  EXPECT_EQ(h.version, kSeeuVersion);
  EXPECT_EQ(h.batch, static_cast<uint64_t>(kBatch));
  EXPECT_EQ(h.label_kind, 1u);  // softmax cross-entropy: class indices
  EXPECT_EQ(h.input_floats, static_cast<uint64_t>(kBatch * kInDim));
  EXPECT_EQ(h.label_bytes, kBatch * sizeof(int32_t));
  EXPECT_EQ(h.optimizer_kind, 1u);  // AdamW default

  // The memory contract is internally consistent.
  EXPECT_GT(h.arena_size, 0u);
  EXPECT_EQ(h.arena_size % 64, 0u);
  EXPECT_LE(h.persistent_size, h.arena_size);
  EXPECT_EQ(h.persist_init_size, h.persistent_size);
  EXPECT_EQ(h.train_instr_count, compiled.train_instruction_count);
  EXPECT_EQ(h.merge_instr_count, compiled.merge_instruction_count);
  EXPECT_GT(h.rodata_size, 0u);
  EXPECT_EQ(h.emit_count, compiled.adapters.size());

  // Every section lies inside the plan blob.
  EXPECT_LE(h.train_instr_offset +
                h.train_instr_count * sizeof(UpdateInstruction),
            compiled.plan.size());
  EXPECT_LE(h.rodata_offset + h.rodata_size, compiled.plan.size());
  EXPECT_LE(h.emit_table_offset + h.emit_count * sizeof(EmitEntry),
            compiled.plan.size());

  // I/O slots live in the mutable arena.
  EXPECT_FALSE(IsRodataRef(h.input_ref));
  EXPECT_FALSE(IsRodataRef(h.label_ref));
  EXPECT_FALSE(IsRodataRef(h.loss_ref));
}

TEST(UpdateCompiler, DebugHooksDescribeAdaptersAndParams) {
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 2);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(BaseConfig(kBatch)).Compile(model));

  ASSERT_EQ(compiled.adapters.size(), 2u);
  ASSERT_EQ(compiled.params.size(), 4u);  // {A, B} x 2 layers

  for (const AdapterDebugInfo& a : compiled.adapters) {
    EXPECT_EQ(a.r, 4);
    EXPECT_NEAR(a.scale, 2.0f, 1e-6);
    EXPECT_TRUE(IsRodataRef(a.weight_rodata_ref));
    EXPECT_FALSE(IsRodataRef(a.a_ref));
    EXPECT_FALSE(IsRodataRef(a.b_ref));
    // Adapters live in the checkpointed persistent segment.
    EXPECT_LT(RefOffset(a.a_ref), compiled.persistent_size);
    EXPECT_LT(RefOffset(a.b_ref), compiled.persistent_size);
  }
  EXPECT_EQ(compiled.adapters[0].weight_name, "w1");
  EXPECT_EQ(compiled.adapters[1].weight_name, "w2");

  // Params are sorted by id and carry live gradient refs.
  for (size_t i = 1; i < compiled.params.size(); ++i)
    EXPECT_LT(compiled.params[i - 1].id, compiled.params[i].id);
  for (const ParamDebugInfo& p : compiled.params) {
    EXPECT_NE(p.param_ref, kNullRef);
    EXPECT_NE(p.grad_ref, kNullRef);
    EXPECT_GT(p.count, 0u);
  }

  // The SIR dump is a human-readable rendering of both programs.
  EXPECT_STR_CONTAINS(compiled.sir_dump, "sc_high.matmul");
  EXPECT_STR_CONTAINS(compiled.sir_dump, "merge program");
}

TEST(UpdateCompiler, CompilationIsDeterministic) {
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 3);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate a,
                       UpdateCompiler(BaseConfig(kBatch)).Compile(model));
  ASSERT_OK_AND_ASSIGN(CompiledUpdate b,
                       UpdateCompiler(BaseConfig(kBatch)).Compile(model));
  EXPECT_TRUE(a.plan == b.plan);
}

TEST(UpdateCompiler, TiedWeightMaterializesOnce) {
  SmfModel model = MakeTiedMlp(4, 4);
  UpdateConfig config = BaseConfig(kBatch);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));

  // The tied tensor resolves to a single SIR value, and both consuming
  // MatMuls share ONE adapter pair: per-site pairs would train fine but
  // commit W + Δ_1 + Δ_2 to the single file range, polluting every site
  // with every other site's delta. One adapter -> one delta -> one emit
  // entry, and the committed weight is exactly the W + Δ every site
  // computed during training.
  const PlanHeader h = HeaderOf(compiled);
  ASSERT_EQ(compiled.adapters.size(), 1u);
  ASSERT_EQ(h.emit_count, 1u);
}

TEST(UpdateCompiler, MseLossUsesDenseLabels) {
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 5);
  UpdateConfig config = BaseConfig(kBatch);
  config.loss = LossKind::kMse;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));

  const PlanHeader h = HeaderOf(compiled);
  EXPECT_EQ(h.label_kind, 2u);
  EXPECT_EQ(h.label_bytes,
            static_cast<uint64_t>(kBatch * kOutDim) * sizeof(float));

  const auto instrs = TrainProgramOf(compiled);
  EXPECT_EQ(CountOpcode(instrs, OpCode::kMseFwd), 1u);
  EXPECT_EQ(CountOpcode(instrs, OpCode::kSoftmaxXEntFwd), 0u);
}

TEST(UpdateCompiler, DistillationRequiresAndUsesTeacher) {
  SmfModel student = MakeMlp(kInDim, kHidden, kOutDim, 6);
  SmfModel teacher = MakeMlp(kInDim, 14, kOutDim, 7);

  UpdateConfig config = BaseConfig(kBatch);
  config.loss = LossKind::kKLDistill;
  EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(student),
                        "requires a teacher");

  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(student, &teacher));
  const PlanHeader h = HeaderOf(compiled);
  EXPECT_EQ(h.label_kind, 0u);  // the teacher provides the signal in-graph
  EXPECT_EQ(h.label_ref, kNullRef);

  const auto instrs = TrainProgramOf(compiled);
  EXPECT_EQ(CountOpcode(instrs, OpCode::kKLDistillFwd), 1u);
  // Teacher weights ride along frozen: no adapters on them.
  EXPECT_EQ(compiled.adapters.size(), 2u);
}

TEST(UpdateCompiler, RejectsTeacherShapeMismatch) {
  SmfModel student = MakeMlp(kInDim, kHidden, kOutDim, 8);
  UpdateConfig config = BaseConfig(kBatch);
  config.loss = LossKind::kKLDistill;

  SmfModel narrow_input = MakeMlp(kInDim + 2, kHidden, kOutDim, 9);
  EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(student, &narrow_input),
                        "teacher input dimensionality");

  SmfModel wrong_output = MakeMlp(kInDim, kHidden, kOutDim + 1, 10);
  EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(student, &wrong_output),
                        "teacher output dimensionality");
}

TEST(UpdateCompiler, CompositeLossCombinesBothTerms) {
  SmfModel student = MakeMlp(kInDim, kHidden, kOutDim, 11);
  SmfModel teacher = MakeMlp(kInDim, 14, kOutDim, 12);
  UpdateConfig config = BaseConfig(kBatch);
  config.loss = LossKind::kXEntPlusKL;
  config.distill_weight = 0.3f;

  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(student, &teacher));
  const PlanHeader h = HeaderOf(compiled);
  EXPECT_EQ(h.label_kind, 1u);  // the cross-entropy term still needs labels

  const auto instrs = TrainProgramOf(compiled);
  EXPECT_EQ(CountOpcode(instrs, OpCode::kSoftmaxXEntFwd), 1u);
  EXPECT_EQ(CountOpcode(instrs, OpCode::kKLDistillFwd), 1u);
}

TEST(UpdateCompiler, EpilogueFusionShrinksDistillPrograms) {
  // Under distillation the frozen teacher's GEMM -> AddBias -> activation
  // chains fuse into flagged GEMM epilogues; the orphaned ops are DCE'd, so
  // both the train and eval programs shrink and the arena loses their
  // transient slots. The same compile with fusion off is the reference.
  SmfModel student = MakeMlp(kInDim, kHidden, kOutDim, 31);
  SmfModel teacher = MakeMlp(kInDim, 14, kOutDim, 32);
  UpdateConfig config = BaseConfig(kBatch);
  config.loss = LossKind::kKLDistill;
  UpdateConfig unfused_config = config;
  unfused_config.fuse_epilogues = false;

  ASSERT_OK_AND_ASSIGN(CompiledUpdate fused,
                       UpdateCompiler(config).Compile(student, &teacher));
  ASSERT_OK_AND_ASSIGN(
      CompiledUpdate unfused,
      UpdateCompiler(unfused_config).Compile(student, &teacher));

  const PlanHeader fh = HeaderOf(fused);
  const PlanHeader uh = HeaderOf(unfused);
  EXPECT_LT(fh.train_instr_count, uh.train_instr_count);
  EXPECT_LT(fh.eval_instr_count, uh.eval_instr_count);
  // No arena-size assertion: fusion removes transient values, but first-fit
  // offsets are not monotone in the interval set (the fused GEMM's result
  // inherits the chain output's longer liveness), so the high-water mark
  // may move either way by a packing accident.

  // The fused stream carries epilogue flags on forward GEMMs; the unfused
  // stream carries none anywhere (flags == 0 was the pre-v5 invariant).
  size_t flagged = 0;
  for (const UpdateInstruction& ins : TrainProgramOf(fused)) {
    if (ins.flags == 0) continue;
    ++flagged;
    EXPECT_EQ(ins.opcode, static_cast<uint16_t>(OpCode::kGemmNN));
    EXPECT_EQ(ins.flags & static_cast<uint16_t>(~kKnownFlagsMask), 0);
  }
  EXPECT_GT(flagged, 0u);
  for (const UpdateInstruction& ins : TrainProgramOf(unfused))
    EXPECT_EQ(ins.flags, 0);

  // The teacher's hidden layer fused bias+relu, its logits layer bias-only:
  // no teacher AddBias survives, and the student's (LoRA-interposed) ones
  // remain — 2 in the fused stream vs 4 unfused.
  EXPECT_EQ(CountOpcode(TrainProgramOf(fused), OpCode::kAddBias),
            CountOpcode(TrainProgramOf(unfused), OpCode::kAddBias) - 2);
}

TEST(UpdateCompiler, OptimizerSelectionShapesTheProgram) {
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 13);

  UpdateConfig adamw = BaseConfig(kBatch);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate with_adamw,
                       UpdateCompiler(adamw).Compile(model));
  EXPECT_EQ(CountOpcode(TrainProgramOf(with_adamw), OpCode::kAdamWStep), 4u);

  UpdateConfig sgd = BaseConfig(kBatch);
  sgd.optimizer.kind = OptimizerKind::kSgd;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate with_sgd,
                       UpdateCompiler(sgd).Compile(model));
  const auto sgd_instrs = TrainProgramOf(with_sgd);
  EXPECT_EQ(CountOpcode(sgd_instrs, OpCode::kSgdStep), 4u);
  EXPECT_EQ(CountOpcode(sgd_instrs, OpCode::kAdamWStep), 0u);
  // No AdamW moments: SGD's persistent segment holds only the adapters.
  EXPECT_LT(with_sgd.persistent_size, with_adamw.persistent_size);
  EXPECT_EQ(HeaderOf(with_sgd).optimizer_kind, 0u);

  UpdateConfig frozen = BaseConfig(kBatch);
  frozen.emit_optimizer = false;  // gradient-verification builds
  ASSERT_OK_AND_ASSIGN(CompiledUpdate no_opt,
                       UpdateCompiler(frozen).Compile(model));
  const auto no_opt_instrs = TrainProgramOf(no_opt);
  EXPECT_EQ(CountOpcode(no_opt_instrs, OpCode::kAdamWStep), 0u);
  EXPECT_EQ(CountOpcode(no_opt_instrs, OpCode::kSgdStep), 0u);
}

TEST(UpdateCompiler, HyperparametersReachThePlanHeader) {
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 14);
  UpdateConfig config = BaseConfig(kBatch);
  config.optimizer.lr = 0.025f;
  config.optimizer.weight_decay = 0.005f;
  config.default_steps = 123;

  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));
  const PlanHeader h = HeaderOf(compiled);
  EXPECT_NEAR(h.lr, 0.025f, 1e-9);
  EXPECT_NEAR(h.weight_decay, 0.005f, 1e-9);
  EXPECT_EQ(h.default_steps, 123u);
}

TEST(UpdateCompiler, RegatesTheBudgetOnTheExactCompiledFootprint) {
  // The step-0 estimate is a lower bound blind to gradients, optimizer
  // state, and transients; the driver must re-prove the budget against the
  // bytes the runtime actually keeps resident — arena + plan blob.
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 1);
  UpdateConfig config = BaseConfig(kBatch);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));
  const uint64_t need = compiled.arena_size + compiled.plan.size();

  config.memory_budget_bytes = need;
  EXPECT_OK(UpdateCompiler(config).Compile(model));

  // One byte short of the real footprint: the early lower bound admits it,
  // the final gate must not.
  config.memory_budget_bytes = need - 1;
  const auto r = UpdateCompiler(config).Compile(model);
  ASSERT_FALSE(r.has_value());
  EXPECT_STR_CONTAINS(r.error(), "cannot run locally");
}

TEST(UpdateCompiler, RejectsModelWithoutInputMetadata) {
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 15);
  model.input_name = "not_a_tensor";
  EXPECT_ERROR_CONTAINS(UpdateCompiler(BaseConfig(kBatch)).Compile(model),
                        "lacks input metadata");
}

TEST(UpdateCompiler, PropagatesGrafterFailure) {
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 16);
  UpdateConfig config = BaseConfig(kBatch);
  config.lora.target_filters = {"no_such_weight"};
  EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(model), "no eligible");
}

TEST(UpdateCompiler, PlanBytesAreThreadCountInvariant) {
  // Compilation parallelizes its byte-heavy passes (int8 quantization,
  // seeded randn init of the persistent image); the emitted plan must be
  // bit-identical at any pool width — same blob, same integrity hash.
  UpdateConfig config = BaseConfig(kBatch);
  config.quantize_base = true;  // exercise the parallel max-abs scan + pack

  seeml::update::SetParallelThreadCount(1);
  SmfModel model_a = MakeMlp(kInDim, kHidden, kOutDim, 17);
  auto serial = UpdateCompiler(config).Compile(model_a);
  seeml::update::SetParallelThreadCount(8);
  SmfModel model_b = MakeMlp(kInDim, kHidden, kOutDim, 17);
  auto wide = UpdateCompiler(config).Compile(model_b);
  seeml::update::SetParallelThreadCount(0);

  ASSERT_OK(serial);
  ASSERT_OK(wide);
  EXPECT_TRUE(serial->plan == wide->plan);
}

}  // namespace
