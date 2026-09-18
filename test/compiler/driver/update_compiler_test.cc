// =============================================================================
// UpdateCompiler tests: plan assembly (header contract, section layout,
// segmented arena binding, debug hooks), per-loss configuration, optimizer
// selection, and the compile-time error surface.
// =============================================================================

#include <bit>
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

TEST(UpdateCompiler, RopeBaseReachesForwardAndBackwardInstructions) {
  // SMF v5 attr1 → parser "base" attribute → autodiff copy → lowering
  // out[2] on both kRopeFwd and kRopeBwd. A Llama-3-class θ=500k must not
  // silently lower to 10000 anywhere in the train program.
  SmfModel model = seeml::testing::MakeTinyDecoder(8, 2, 4, 12, 3, 9);
  const float theta = 500000.0f;
  for (SmfOp& op : model.ops)
    if (op.kind == SmfOpKind::kRope) op.attr1 = std::bit_cast<uint32_t>(theta);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(BaseConfig(8)).Compile(model));
  const auto instrs = TrainProgramOf(compiled);
  size_t fwd = 0, bwd = 0;
  for (const UpdateInstruction& ins : instrs) {
    const auto op = static_cast<OpCode>(ins.opcode);
    if (op != OpCode::kRopeFwd && op != OpCode::kRopeBwd) continue;
    (op == OpCode::kRopeFwd ? fwd : bwd)++;
    EXPECT_EQ(static_cast<uint32_t>(ins.out[2]),
              std::bit_cast<uint32_t>(theta));
  }
  EXPECT_EQ(fwd, 2u);
  EXPECT_GE(bwd, 1u);
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
  EXPECT_TRUE(compiled.sir_dump.empty());  // debug output is opt-in
  UpdateConfig dumping = BaseConfig(kBatch);
  dumping.dump_sir = true;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate dumped,
                       UpdateCompiler(dumping).Compile(model));
  EXPECT_TRUE(dumped.plan == compiled.plan);  // the dump changes no byte
  compiled = std::move(dumped);
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

  // The objective is decided here: the plan carries T in the low word and
  // the Hinton T^2 scale in the high word of the temperature operand, on
  // both the forward and its VJP (#13).
  const auto bits = [](float f) {
    return uint64_t{std::bit_cast<uint32_t>(f)};
  };
  const float T = config.temperature;
  const uint64_t want = (bits(T * T) << 32) | bits(T);
  size_t fwd = 0, bwd = 0;
  for (const auto& ins : instrs) {
    if (ins.opcode == static_cast<uint16_t>(OpCode::kKLDistillFwd)) {
      EXPECT_EQ(ins.out[2], want);
      ++fwd;
    } else if (ins.opcode == static_cast<uint16_t>(OpCode::kKLDistillBwd)) {
      EXPECT_EQ(ins.out[1], want);
      ++bwd;
    }
  }
  EXPECT_EQ(fwd, 1u);
  EXPECT_EQ(bwd, 1u);
  EXPECT_TRUE(HeaderOf(compiled).version >= seeml::update::kSeeuKlScaleVersion);
}

TEST(UpdateCompiler, GradientAccumulationSplitsTheStream) {
  // G = 4: the train section becomes the grad program (forward, backward,
  // one fold per trainable into a persistent accumulator) and a step
  // section carries clip / step / zero; the seed is 1/G. G = 1 (the
  // default) is the classic one-batch program with no step section.
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 51);
  UpdateConfig config = BaseConfig(kBatch);
  config.optimizer.clip_norm = 1.0f;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate plain,
                       UpdateCompiler(config).Compile(model));
  config.grad_accum_steps = 4;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate accum,
                       UpdateCompiler(config).Compile(model));
  const PlanHeader hp = HeaderOf(plain), ha = HeaderOf(accum);
  EXPECT_EQ(hp.grad_accum_steps, 1u);
  EXPECT_EQ(hp.step_instr_count, 0u);
  EXPECT_EQ(ha.grad_accum_steps, 4u);
  EXPECT_GT(ha.step_instr_count, 0u);
  EXPECT_TRUE(ha.version >= kSeeuGradAccumVersion);
  const size_t trainables = 2 * accum.adapters.size();

  const auto grad = TrainProgramOf(accum);
  EXPECT_EQ(CountOpcode(grad, OpCode::kAccumulate), trainables);
  EXPECT_EQ(CountOpcode(grad, OpCode::kAdamWStep), 0u);
  EXPECT_EQ(CountOpcode(grad, OpCode::kClipNorm), 0u);
  EXPECT_EQ(CountOpcode(TrainProgramOf(plain), OpCode::kAccumulate), 0u);
  EXPECT_EQ(CountOpcode(TrainProgramOf(plain), OpCode::kAdamWStep), trainables);

  std::vector<UpdateInstruction> step(ha.step_instr_count);
  std::memcpy(step.data(), accum.plan.data() + ha.step_instr_offset,
              step.size() * sizeof(UpdateInstruction));
  // The clip rides the step (plan v12): no standalone pass over the
  // accumulator, its threshold in every step's out[1].
  EXPECT_EQ(CountOpcode(step, OpCode::kClipNorm), 0u);
  EXPECT_EQ(CountOpcode(step, OpCode::kAdamWStep), trainables);
  for (const auto& ins : step)
    if (ins.opcode == static_cast<uint16_t>(OpCode::kAdamWStep))
      EXPECT_EQ(std::bit_cast<float>(static_cast<uint32_t>(ins.out[1])), 1.0f);
  EXPECT_EQ(CountOpcode(step, OpCode::kFill), trainables);  // the zeroes
  EXPECT_EQ(CountOpcode(step, OpCode::kAccumulate), 0u);
  // The seed carries 1/G: exactly one fill of 0.25 in the grad program.
  size_t quarter_seeds = 0;
  for (const auto& ins : grad)
    if (ins.opcode == static_cast<uint16_t>(OpCode::kFill) &&
        std::bit_cast<float>(static_cast<uint32_t>(ins.in[1])) == 0.25f)
      ++quarter_seeds;
  EXPECT_EQ(quarter_seeds, 1u);
  // One extra grad-sized set in the persistent segment, activations
  // unchanged (the arena beyond it is the same micro-batch workspace).
  uint64_t grad_bytes = 0;  // each accumulator is a 64-byte-aligned slot
  for (const auto& p : accum.params) {
    grad_bytes += (p.count * sizeof(float) + 63) & ~uint64_t{63};
    EXPECT_NE(p.acc_ref, kNullRef);
  }
  for (const auto& p : plain.params) EXPECT_EQ(p.acc_ref, kNullRef);
  EXPECT_EQ(accum.persistent_size, plain.persistent_size + grad_bytes);
}

TEST(UpdateCompiler, GradientAccumulationSplitsAnSgdStream) {
  // SGD has no moments: per trainable the step program is step + zero with
  // the clip folded into the step (plan v12) — or, with fuse_clip off, the
  // v11 shape clip + step + zero — and the grad program's folds precede it.
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 52);
  UpdateConfig config = BaseConfig(kBatch);
  config.optimizer.kind = OptimizerKind::kSgd;
  config.optimizer.clip_norm = 0.5f;
  config.grad_accum_steps = 2;
  for (const bool fuse : {true, false}) {
    config.fuse_clip = fuse;
    ASSERT_OK_AND_ASSIGN(CompiledUpdate c,
                         UpdateCompiler(config).Compile(model));
    const PlanHeader h = HeaderOf(c);
    const size_t trainables = 2 * c.adapters.size();
    std::vector<UpdateInstruction> step(h.step_instr_count);
    std::memcpy(step.data(), c.plan.data() + h.step_instr_offset,
                step.size() * sizeof(UpdateInstruction));
    EXPECT_EQ(step.size(), (fuse ? 2 : 3) * trainables);
    EXPECT_EQ(CountOpcode(step, OpCode::kClipNorm), fuse ? 0u : trainables);
    EXPECT_EQ(CountOpcode(step, OpCode::kSgdStep), trainables);
    EXPECT_EQ(CountOpcode(step, OpCode::kFill), trainables);
    EXPECT_EQ(static_cast<OpCode>(step[0].opcode),
              fuse ? OpCode::kSgdStep : OpCode::kClipNorm);
    for (const auto& ins : step)
      if (ins.opcode == static_cast<uint16_t>(OpCode::kSgdStep))
        EXPECT_EQ(ins.out[1],
                  fuse ? uint64_t{std::bit_cast<uint32_t>(0.5f)} : 0u);
    const auto grad = TrainProgramOf(c);
    EXPECT_EQ(CountOpcode(grad, OpCode::kAccumulate), trainables);
    EXPECT_EQ(CountOpcode(grad, OpCode::kSgdStep), 0u);
    EXPECT_EQ(CountOpcode(grad, OpCode::kClipNorm), 0u);
  }
}

TEST(UpdateCompiler, Bf16BaseHalvesRodataAndLowersTheWideningGemms) {
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 53);
  UpdateConfig f32 = BaseConfig(kBatch);
  UpdateConfig bf16 = BaseConfig(kBatch);
  bf16.bf16_base = true;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate a, UpdateCompiler(f32).Compile(model));
  ASSERT_OK_AND_ASSIGN(CompiledUpdate b, UpdateCompiler(bf16).Compile(model));
  // Declared at the current version, which is at least the one that
  // introduced the widening opcodes (the validator gates them on it).
  EXPECT_EQ(HeaderOf(b).version, kSeeuVersion);
  EXPECT_GE(kSeeuVersion, kSeeuBf16Version);
  // Both matmul weights are adapted: a forward NN per weight, and a
  // backward NT (dX through the frozen weight) for every weight but the
  // first — nothing trainable sits before layer 1's input, so autodiff
  // prunes that adjoint.
  const auto instrs = TrainProgramOf(b);
  EXPECT_EQ(CountOpcode(instrs, OpCode::kGemmNNBF16), b.adapters.size());
  EXPECT_EQ(CountOpcode(instrs, OpCode::kGemmNTBF16), b.adapters.size() - 1);
  // The adapter GEMMs (X@A, t@B and their backward) stay f32: A and B are
  // trainable arena parameters, not frozen rodata.
  EXPECT_EQ(CountOpcode(instrs, OpCode::kGemmNN),
            CountOpcode(TrainProgramOf(a), OpCode::kGemmNN) - b.adapters.size());
  EXPECT_EQ(CountOpcode(instrs, OpCode::kGemmNNQ8), 0u);
  for (const auto& ad : b.adapters) {
    EXPECT_TRUE(ad.bf16);
    EXPECT_EQ(ad.quant_scale, 0.0f);
  }
  for (const auto& ad : a.adapters) EXPECT_FALSE(ad.bf16);
  // Weights halve; the biases stay f32 (they are AddBias operands, not
  // matmul weights), so rodata shrinks by exactly the weight bytes / 2.
  uint64_t weight_bytes = 0;
  for (const auto& ad : b.adapters)
    weight_bytes += static_cast<uint64_t>(ad.k * ad.m) * sizeof(float);
  EXPECT_LT(b.rodata_size, a.rodata_size);
  EXPECT_GE(a.rodata_size - b.rodata_size, weight_bytes / 2 - 64 * b.adapters.size());
  // One storage precision per weight.
  UpdateConfig both = BaseConfig(kBatch);
  both.bf16_base = true;
  both.quantize_base = true;
  EXPECT_ERROR_CONTAINS(UpdateCompiler(both).Compile(model),
                        "mutually exclusive");
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

TEST(UpdateCompiler, LoraSitesCostNoActivationSizedPassOfTheirOwn) {
  // E10 (#93). Per adapted matmul the program used to spend two
  // activation-sized elementwise passes forward (scale, add) and two
  // backward (the scale's VJP over dC, the fan-out add into dX). Now the
  // scale lives on the rank-r activation and both adds ride their GEMM.
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 31);
  UpdateConfig config = BaseConfig(kBatch);  // rank 4, alpha 8
  UpdateConfig unfolded = config;
  unfolded.fuse_gemm_addend = false;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate folded, UpdateCompiler(config).Compile(model));
  ASSERT_OK_AND_ASSIGN(CompiledUpdate plain,
                       UpdateCompiler(unfolded).Compile(model));
  ASSERT_EQ(folded.adapters.size(), 2u);

  // Every scale in either program is rank-sized: [N, r] = 4 * 4 elements.
  const uint64_t rank_sized =
      static_cast<uint64_t>(kBatch) * static_cast<uint64_t>(config.lora.rank);
  for (const CompiledUpdate* c : {&folded, &plain}) {
    size_t scales = 0;
    for (const UpdateInstruction& ins : TrainProgramOf(*c)) {
      if (ins.opcode != static_cast<uint16_t>(OpCode::kScale)) continue;
      ++scales;
      EXPECT_EQ(ins.out[0], rank_sized);
    }
    EXPECT_EQ(scales, 2 * folded.adapters.size());  // forward t, backward dt
  }

  // Folded: one forward addend per site (C' = C + ts@B), and one backward
  // wherever dX is needed at all — the first layer's input takes no
  // gradient — each naming a real addend; no standalone add is left.
  size_t nn = 0, nt = 0;
  for (const UpdateInstruction& ins : TrainProgramOf(folded)) {
    if (!(ins.flags & kFlagGemmAddend)) continue;
    EXPECT_EQ(ins.flags, kFlagGemmAddend);
    EXPECT_NE(ins.in[3], kNullRef);
    if (ins.opcode == static_cast<uint16_t>(OpCode::kGemmNN)) ++nn;
    if (ins.opcode == static_cast<uint16_t>(OpCode::kGemmNT)) ++nt;
  }
  EXPECT_EQ(nn, 2u);
  EXPECT_EQ(nt, 1u);
  EXPECT_EQ(CountOpcode(TrainProgramOf(folded), OpCode::kAddEW), 0u);

  // Unfolded: the same sums as instructions of their own, and no flag.
  for (const UpdateInstruction& ins : TrainProgramOf(plain))
    EXPECT_EQ(ins.flags & kFlagGemmAddend, 0);
  EXPECT_EQ(CountOpcode(TrainProgramOf(plain), OpCode::kAddEW), 3u);
  EXPECT_EQ(HeaderOf(plain).train_instr_count,
            HeaderOf(folded).train_instr_count + 3);
  EXPECT_EQ(HeaderOf(plain).eval_instr_count,
            HeaderOf(folded).eval_instr_count + 2);
}

TEST(UpdateCompiler, NarrowWeightGemmsKeepTheirOwnInstruction) {
  // The addend exists for the f32 GEMMs only. At a LoRA site over an int8
  // or bf16 frozen weight the rank-r GEMM is the one that folds, so the
  // fold loses nothing — and the validator would refuse the other choice.
  for (const bool bf16 : {false, true}) {
    SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 32);
    UpdateConfig config = BaseConfig(kBatch);
    config.quantize_base = !bf16;
    config.bf16_base = bf16;
    ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                         UpdateCompiler(config).Compile(model));
    size_t folded = 0;
    for (const UpdateInstruction& ins : TrainProgramOf(compiled)) {
      if (!(ins.flags & kFlagGemmAddend)) continue;
      ++folded;
      EXPECT_TRUE(ins.opcode == static_cast<uint16_t>(OpCode::kGemmNN) ||
                  ins.opcode == static_cast<uint16_t>(OpCode::kGemmNT));
      EXPECT_FALSE(IsRodataRef(ins.in[1]));  // the adapter, never the base
    }
    EXPECT_EQ(folded, 3u);
  }
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
  // (The GEMM addend, v14, is a different pass's flag and rides both.)
  size_t flagged = 0;
  for (const UpdateInstruction& ins : TrainProgramOf(fused)) {
    EXPECT_EQ(ins.flags & static_cast<uint16_t>(~kKnownFlagsMask), 0);
    if ((ins.flags & kEpilogueFlagsMask) == 0) continue;
    ++flagged;
    EXPECT_EQ(ins.opcode, static_cast<uint16_t>(OpCode::kGemmNN));
  }
  EXPECT_GT(flagged, 0u);
  for (const UpdateInstruction& ins : TrainProgramOf(unfused))
    EXPECT_EQ(ins.flags & kEpilogueFlagsMask, 0);

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

TEST(UpdateCompiler, ATargetThatNamesOnlyATiedHeadIsAnError) {
  // P7 (#96): the tied embedding / LM head is ineligible (the gather reads
  // the same tensor), so --targets emb cannot be honoured: a compile error
  // that says why, not a silently unmet filter. A filter naming nothing,
  // beside one that does match, is an error too.
  SmfModel tied = seeml::testing::MakeTiedTokenDecoder(8, 2, 4, 12, 3);
  UpdateConfig config = BaseConfig(8);
  config.lora.target_filters = {"emb"};
  auto r = UpdateCompiler(config).Compile(tied);
  ASSERT_FALSE(r.has_value());
  EXPECT_STR_CONTAINS(r.error(), "--targets 'emb'");
  EXPECT_STR_CONTAINS(r.error(), "tied embedding / LM head");
  config.lora.target_filters = {"wq", "no_such_weight"};
  EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(tied),
                        "--targets 'no_such_weight' matches no eligible");
  // Without a filter the tied model compiles, adapting everything else.
  config.lora.target_filters = {};
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(tied));
  for (const auto& a : compiled.adapters) EXPECT_NE(a.weight_name, "emb");
  EXPECT_FALSE(compiled.adapters.empty());
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

// --- Single weight residency (E2, #81) ---------------------------------------

TEST(UpdateCompiler, TheConsumingCompileIsTheSamePlanAndReleasesWhatItPacked) {
  // Every storage form, a teacher, and the student/teacher split: the
  // consuming overload must assemble the identical blob — it changes WHEN
  // payloads die, nothing else — and leave no packed payload resident.
  SmfModel student = MakeMlp(kInDim, kHidden, kOutDim, 71);
  SmfModel teacher = MakeMlp(kInDim, 2 * kHidden, kOutDim, 72);
  for (int storage = 0; storage < 3; ++storage) {
    UpdateConfig config = BaseConfig(kBatch);
    config.loss = LossKind::kXEntPlusKL;
    config.quantize_base = storage == 1;
    config.bf16_base = storage == 2;
    ASSERT_OK_AND_ASSIGN(CompiledUpdate kept,
                         UpdateCompiler(config).Compile(student, &teacher));
    SmfModel s = student, t = teacher;
    const uint64_t hash = s.content_hash;
    ASSERT_OK_AND_ASSIGN(CompiledUpdate consumed,
                         UpdateCompiler(config).Compile(std::move(s), &t));
    EXPECT_TRUE(kept.plan == consumed.plan);
    EXPECT_EQ(s.content_hash, hash);  // identity survives the release
    for (const SmfModel* m : {&s, &t}) {
      size_t released = 0;
      for (const SmfTensor& tensor : m->tensors) {
        if (!tensor.is_const) continue;
        EXPECT_GT(tensor.byte_size, 0u);  // metadata is kept
        if (tensor.data.empty()) ++released;
      }
      EXPECT_GT(released, 0u);
    }
    // The caller's originals were never touched.
    for (const SmfTensor& tensor : student.tensors)
      if (tensor.is_const) EXPECT_EQ(tensor.data.size(), tensor.byte_size);
  }
}

// --- The schedule triple is validated at compile time (E8, #91) ---------------

TEST(UpdateCompiler, RefusesSchedulesThatWouldMisbehaveSilently) {
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 91);
  auto compile = [&](auto&& edit) {
    UpdateConfig config = BaseConfig(kBatch);
    config.default_steps = 1000;
    edit(config);
    return UpdateCompiler(config).Compile(model);
  };
  auto cosine = [](UpdateConfig& c) {
    c.optimizer.lr_schedule = LrSchedule::kCosineWithWarmup;
  };
  EXPECT_ERROR_CONTAINS(compile([](UpdateConfig& c) { c.default_steps = 0; }),
                        "step budget must be positive");
  EXPECT_ERROR_CONTAINS(compile([&](UpdateConfig& c) {
                          cosine(c);
                          c.optimizer.warmup_steps = 1000;  // == the budget
                        }),
                        "must be shorter than the step budget");
  EXPECT_ERROR_CONTAINS(
      compile([](UpdateConfig& c) { c.optimizer.warmup_steps = 10; }),
      "cosine schedule only");
  EXPECT_ERROR_CONTAINS(compile([&](UpdateConfig& c) {
                          cosine(c);
                          c.optimizer.min_lr_factor = 0.0f;
                        }),
                        "learning rate of zero");
  ASSERT_OK_AND_ASSIGN(CompiledUpdate zero, compile([&](UpdateConfig& c) {
                         cosine(c);
                         c.optimizer.min_lr_factor = 0.0f;
                         c.optimizer.allow_zero_lr = true;  // asked for twice
                       }));
  EXPECT_EQ(HeaderOf(zero).min_lr_factor, 0.0f);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate normal, compile([&](UpdateConfig& c) {
                         cosine(c);
                         c.optimizer.warmup_steps = 999;
                       }));
  EXPECT_EQ(HeaderOf(normal).min_lr_factor, 0.1f);  // the default floor
  // A constant-schedule plan records no floor: byte-identical to the plans
  // compiled before the default moved.
  ASSERT_OK_AND_ASSIGN(CompiledUpdate constant, compile([](UpdateConfig&) {}));
  EXPECT_EQ(HeaderOf(constant).min_lr_factor, 0.0f);
  EXPECT_EQ(HeaderOf(constant).warmup_steps, 0u);
}

}  // namespace
