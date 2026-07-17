// =============================================================================
// UpdateCompiler tests: plan assembly (header contract, section layout,
// segmented arena binding, debug hooks), per-loss configuration, optimizer
// selection, and the compile-time error surface.
// =============================================================================

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "compiler/backend/update_compiler.h"
#include "source/update_types.h"
#include "source/smf.h"
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

  // The tied tensor resolves to a single SIR value: each consuming MatMul
  // still gets its own adapter, but both share one frozen rodata copy and
  // their emit entries patch the same source byte range.
  const PlanHeader h = HeaderOf(compiled);
  ASSERT_EQ(compiled.adapters.size(), 2u);
  EXPECT_EQ(compiled.adapters[0].weight_rodata_ref,
            compiled.adapters[1].weight_rodata_ref);
  ASSERT_EQ(h.emit_count, 2u);
  std::vector<EmitEntry> emits(h.emit_count);
  std::memcpy(emits.data(), compiled.plan.data() + h.emit_table_offset,
              h.emit_count * sizeof(EmitEntry));
  EXPECT_EQ(emits[0].smf_data_offset, emits[1].smf_data_offset);
  EXPECT_EQ(emits[0].byte_size, emits[1].byte_size);
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

}  // namespace
