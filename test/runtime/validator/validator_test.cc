// =============================================================================
// validator/ unit tests: the load-time bounds proof the executor's blind
// dispatch rests on — per-opcode operand extents, the write-only-to-arena
// rule, quantized-B pinning to rodata, overflow-safe bounds math, and the
// regression that every instruction the compiler emits validates.
// =============================================================================

#include <cstdint>
#include <cstring>
#include <vector>

#include "compiler/driver/update_compiler.h"
#include "runtime/engine/contract.h"
#include "runtime/validator/plan_validator.h"
#include "source/plan/update_types.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"

namespace {

using namespace seeml::update_rt;
namespace up = seeml::update;
using seeml::testing::BaseConfig;
using seeml::testing::MakeMlp;

constexpr uint64_t kArena = 1024;
constexpr uint64_t kRodata = 256;

up::UpdateInstruction AddEw(uint64_t x, uint64_t y, uint64_t out,
                            uint64_t count) {
  up::UpdateInstruction ins;
  ins.opcode = static_cast<uint16_t>(up::OpCode::kAddEW);
  ins.in[0] = x;
  ins.in[1] = y;
  ins.in[2] = out;
  ins.out[0] = count;
  return ins;
}

TEST(PlanValidator, AcceptsInBoundsOperands) {
  EXPECT_OK(ValidateInstruction(AddEw(up::MakeArenaRef(0),
                                      up::MakeArenaRef(256),
                                      up::MakeArenaRef(512), 64),
                                kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, TransformerOpcodesAreVersionGated) {
  // A valid attention instruction: q/k/v/o at disjoint arena offsets
  // (spaced 64 floats apart), probs cache beyond them. B=1, S=4, H=2,
  // d=4 -> T*D = 32 floats each, probs 32 floats.
  up::UpdateInstruction attn;
  attn.opcode = static_cast<uint16_t>(up::OpCode::kAttnFwd);
  attn.in[0] = up::MakeArenaRef(0);
  attn.in[1] = up::MakeArenaRef(64 * 4);
  attn.in[2] = up::MakeArenaRef(128 * 4);
  attn.in[3] = up::MakeArenaRef(192 * 4);
  attn.out[0] = up::MakeArenaRef(256 * 4);
  attn.out[1] = (uint64_t{1} << 32) | 4;  // B<<32|S
  attn.out[2] = (uint64_t{2} << 32) | 4;  // H<<32|d
  EXPECT_OK(ValidateInstruction(attn, 2048, kRodata, up::kSeeuVersion));
  // The identical instruction inside a pre-v6 plan is corruption: no pre-v6
  // compiler emits transformer opcodes.
  EXPECT_ERROR_CONTAINS(
      ValidateInstruction(attn, 2048, kRodata, up::kSeeuTransformerVersion - 1),
      "pre-v6 plan");
  // Overlapping q and o must fail the written-range discipline.
  attn.in[3] = attn.in[0];
  EXPECT_ERROR(ValidateInstruction(attn, 2048, kRodata, up::kSeeuVersion));
  // Zero heads is malformed geometry.
  attn.in[3] = up::MakeArenaRef(192 * 4);
  attn.out[2] = 4;  // H == 0
  EXPECT_ERROR(ValidateInstruction(attn, 2048, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, RopeRequiresEvenHeadWidth) {
  up::UpdateInstruction rope;
  rope.opcode = static_cast<uint16_t>(up::OpCode::kRopeFwd);
  rope.in[0] = up::MakeArenaRef(0);
  rope.in[1] = up::MakeArenaRef(96);
  rope.out[0] = (uint64_t{1} << 32) | 2;  // B=1, S=2
  rope.out[1] = (uint64_t{2} << 32) | 3;  // H=2, d=3: odd head width
  EXPECT_ERROR(ValidateInstruction(rope, kArena, kRodata, up::kSeeuVersion));
  rope.out[1] = (uint64_t{2} << 32) | 4;  // d=4 with disjoint in/out
  rope.in[1] = up::MakeArenaRef(512);
  EXPECT_OK(ValidateInstruction(rope, kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, RejectsReadsPastTheArena) {
  const auto r = ValidateInstruction(
      AddEw(up::MakeArenaRef(900), up::MakeArenaRef(0), up::MakeArenaRef(256),
            64),  // 900 + 256 B > 1024
      kArena, kRodata, up::kSeeuVersion);
  ASSERT_FALSE(r.has_value());
  EXPECT_STR_CONTAINS(r.error(), "instruction operand out of bounds");
  EXPECT_TRUE(WellFormedDiagnostic(r.error()));
}

TEST(PlanValidator, RejectsWritesIntoRodata) {
  // The frozen weights must be physically unwritable from the stream.
  EXPECT_ERROR(ValidateInstruction(AddEw(up::MakeArenaRef(0),
                                         up::MakeArenaRef(256),
                                         up::MakeRodataRef(0), 16),
                                   kArena, kRodata, up::kSeeuVersion));
  up::UpdateInstruction fill;
  fill.opcode = static_cast<uint16_t>(up::OpCode::kFill);
  fill.in[0] = up::MakeRodataRef(0);
  fill.out[0] = 16;
  EXPECT_ERROR(ValidateInstruction(fill, kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, RejectsNullRefsAndUnknownOpcodes) {
  up::UpdateInstruction null_ref = AddEw(up::kNullRef, up::MakeArenaRef(0),
                                         up::MakeArenaRef(256), 16);
  EXPECT_ERROR(ValidateInstruction(null_ref, kArena, kRodata, up::kSeeuVersion));

  up::UpdateInstruction bogus;
  bogus.opcode = 0xFFFF;
  const auto r = ValidateInstruction(bogus, kArena, kRodata, up::kSeeuVersion);
  ASSERT_FALSE(r.has_value());
  EXPECT_STR_CONTAINS(r.error(), "unknown opcode");
}

TEST(PlanValidator, QuantizedWeightsMustLiveInRodata) {
  up::UpdateInstruction q8;
  q8.opcode = static_cast<uint16_t>(up::OpCode::kGemmNNQ8);
  q8.in[0] = up::MakeArenaRef(0);
  q8.in[1] = up::MakeArenaRef(64);  // int8 B in the mutable arena: forbidden
  q8.in[2] = up::MakeArenaRef(512);
  q8.out[0] = 2;
  q8.out[1] = 2;
  q8.out[2] = 2;
  EXPECT_ERROR(ValidateInstruction(q8, kArena, kRodata, up::kSeeuVersion));
  q8.in[1] = up::MakeRodataRef(0);
  EXPECT_OK(ValidateInstruction(q8, kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, RejectsWriteRangesAliasingOtherOperands) {
  // The kernels are compiled with SEEML_RESTRICT: a written range that
  // overlaps another operand is undefined behavior, so a plan carrying one
  // must be a load error, never a dispatch.
  const auto r = ValidateInstruction(
      AddEw(up::MakeArenaRef(0), up::MakeArenaRef(512),
            up::MakeArenaRef(128), 64),  // out 128..384 overlaps in0 0..256
      kArena, kRodata, up::kSeeuVersion);
  ASSERT_FALSE(r.has_value());
  EXPECT_STR_CONTAINS(r.error(), "alias");
  EXPECT_TRUE(WellFormedDiagnostic(r.error()));
}

TEST(PlanValidator, AllowsReadOnlyOperandsToAlias) {
  // in0 == in1 (x + x): reads may share a range; only writes make an alias.
  EXPECT_OK(ValidateInstruction(AddEw(up::MakeArenaRef(0), up::MakeArenaRef(0),
                                      up::MakeArenaRef(512), 64),
                                kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, InPlaceOptimizerStepsAliasOnlyThroughOneRef) {
  // SGD updates the param through a single read-write ref — legal. The same
  // range surfacing again as the gradient operand is an alias — rejected.
  up::UpdateInstruction sgd;
  sgd.opcode = static_cast<uint16_t>(up::OpCode::kSgdStep);
  sgd.in[0] = up::MakeArenaRef(0);
  sgd.in[1] = up::MakeArenaRef(256);
  sgd.out[0] = 64;
  EXPECT_OK(ValidateInstruction(sgd, kArena, kRodata, up::kSeeuVersion));
  sgd.in[1] = up::MakeArenaRef(128);  // grad overlaps the updated param
  EXPECT_ERROR(ValidateInstruction(sgd, kArena, kRodata, up::kSeeuVersion));

  up::UpdateInstruction adamw;
  adamw.opcode = static_cast<uint16_t>(up::OpCode::kAdamWStep);
  adamw.in[0] = up::MakeArenaRef(0);    // param (in place)
  adamw.in[1] = up::MakeArenaRef(256);  // grad
  adamw.in[2] = up::MakeArenaRef(512);  // m (in place)
  adamw.in[3] = up::MakeArenaRef(768);  // v (in place)
  adamw.out[0] = 64;
  EXPECT_OK(ValidateInstruction(adamw, kArena, kRodata, up::kSeeuVersion));
  adamw.in[3] = up::MakeArenaRef(512);  // v aliases m: two written ranges
  EXPECT_ERROR(ValidateInstruction(adamw, kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, BoundsMathIsOverflowSafe) {
  uint64_t out = 0;
  EXPECT_TRUE(MulOk(1u << 20, 1u << 20, &out));
  EXPECT_FALSE(MulOk(UINT64_MAX, 2, &out));
  EXPECT_TRUE(RangeOk(0, 1024, 1024));
  EXPECT_FALSE(RangeOk(1, 1024, 1024));
  // off + bytes wraps u64 — must reject, not wrap.
  EXPECT_FALSE(RangeOk(UINT64_MAX - 8, 64, 1024));

  // A count chosen so elems * 4 wraps to something tiny.
  EXPECT_ERROR(ValidateInstruction(
      AddEw(up::MakeArenaRef(0), up::MakeArenaRef(0), up::MakeArenaRef(0),
            UINT64_MAX / 2),
      kArena, kRodata, up::kSeeuVersion));
}

up::UpdateInstruction GemmNN(uint64_t a, uint64_t b, uint64_t c, uint64_t m,
                             uint64_t n, uint64_t k) {
  up::UpdateInstruction ins;
  ins.opcode = static_cast<uint16_t>(up::OpCode::kGemmNN);
  ins.in[0] = a;
  ins.in[1] = b;
  ins.in[2] = c;
  ins.out[0] = m;
  ins.out[1] = n;
  ins.out[2] = k;
  return ins;
}

TEST(PlanValidator, RejectsFlagsOnPreFlagsPlans) {
  // A pre-v5 plan predates the flags vocabulary: any nonzero word there is
  // corruption, not a feature.
  up::UpdateInstruction gemm = GemmNN(up::MakeArenaRef(0), up::MakeArenaRef(64),
                                      up::MakeArenaRef(512), 4, 4, 4);
  gemm.flags = up::kFlagEpilogueBias;
  gemm.in[3] = up::MakeArenaRef(256);
  const auto r = ValidateInstruction(gemm, kArena, kRodata,
                                     up::kSeeuFlagsVersion - 1);
  ASSERT_FALSE(r.has_value());
  EXPECT_STR_CONTAINS(r.error(), "flags");
  EXPECT_TRUE(WellFormedDiagnostic(r.error()));
}

TEST(PlanValidator, RejectsUnknownFlagBits) {
  up::UpdateInstruction gemm = GemmNN(up::MakeArenaRef(0), up::MakeArenaRef(64),
                                      up::MakeArenaRef(512), 4, 4, 4);
  gemm.flags = static_cast<uint16_t>(1u << 15);  // future flag, unknown today
  const auto r = ValidateInstruction(gemm, kArena, kRodata, up::kSeeuVersion);
  ASSERT_FALSE(r.has_value());
  EXPECT_STR_CONTAINS(r.error(), "flags");
}

TEST(PlanValidator, RejectsEpilogueFlagsOnNonGemmOpcodes) {
  // The epilogue vocabulary is defined for the forward GEMMs only; a
  // bias/act bit on any other opcode would be silently ignored by
  // Execute() — exactly what the validator must never allow.
  up::UpdateInstruction add = AddEw(up::MakeArenaRef(0), up::MakeArenaRef(256),
                                    up::MakeArenaRef(512), 16);
  add.flags = up::MakeEpilogueFlags(false, up::EpilogueAct::kRelu);
  EXPECT_ERROR(ValidateInstruction(add, kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, FusedBiasJoinsBoundsAndOverlapDiscipline) {
  up::UpdateInstruction gemm = GemmNN(up::MakeArenaRef(0), up::MakeArenaRef(64),
                                      up::MakeArenaRef(512), 4, 4, 4);
  gemm.flags = up::MakeEpilogueFlags(true, up::EpilogueAct::kGelu);
  gemm.in[3] = up::MakeArenaRef(256);
  EXPECT_OK(ValidateInstruction(gemm, kArena, kRodata, up::kSeeuVersion));

  // Bias read out of bounds: 1020 + 4*4 B > 1024.
  gemm.in[3] = up::MakeArenaRef(1020);
  EXPECT_ERROR(ValidateInstruction(gemm, kArena, kRodata, up::kSeeuVersion));

  // Bias overlapping the written C range is an alias, not a layout.
  gemm.in[3] = up::MakeArenaRef(512);
  const auto r = ValidateInstruction(gemm, kArena, kRodata, up::kSeeuVersion);
  ASSERT_FALSE(r.has_value());
  EXPECT_STR_CONTAINS(r.error(), "alias");
}

TEST(PlanValidator, RejectsFusedBiasOnQuantizedGemm) {
  // The q8 GEMM's in[3] carries the dequant scale — a bias flag there
  // would make Execute() read the scale bits as an arena ref.
  up::UpdateInstruction q8;
  q8.opcode = static_cast<uint16_t>(up::OpCode::kGemmNNQ8);
  q8.in[0] = up::MakeArenaRef(0);
  q8.in[1] = up::MakeRodataRef(0);
  q8.in[2] = up::MakeArenaRef(512);
  q8.out[0] = 2;
  q8.out[1] = 2;
  q8.out[2] = 2;
  q8.flags = up::MakeEpilogueFlags(false, up::EpilogueAct::kSilu);
  EXPECT_OK(ValidateInstruction(q8, kArena, kRodata, up::kSeeuVersion));
  q8.flags = up::MakeEpilogueFlags(true, up::EpilogueAct::kSilu);
  EXPECT_ERROR(ValidateInstruction(q8, kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, RejectsZeroExtentOperands) {
  // Every kernel dereferences element 0 unconditionally, so a zero-extent
  // operand is an unchecked pointer, not an empty loop. The canonical
  // exploit was a zero-class softmax: all extents collapse to 0, every
  // range check passes, and the backward kernel then writes through a raw
  // dataset label. Both the packed-width and the plain-count forms must
  // reject.
  up::UpdateInstruction sm;
  sm.opcode = static_cast<uint16_t>(up::OpCode::kSoftmaxXEntFwd);
  sm.in[0] = up::MakeArenaRef(0);
  sm.in[1] = up::MakeArenaRef(256);
  sm.in[2] = up::MakeArenaRef(512);
  sm.in[3] = up::MakeArenaRef(640);
  sm.out[0] = 4;  // N
  sm.out[1] = 0;  // C == 0: the exploit
  EXPECT_ERROR(ValidateInstruction(sm, kArena, kRodata, up::kSeeuVersion));
  sm.out[1] = 4;
  EXPECT_OK(ValidateInstruction(sm, kArena, kRodata, up::kSeeuVersion));

  EXPECT_ERROR(ValidateInstruction(
      AddEw(up::MakeArenaRef(0), up::MakeArenaRef(256), up::MakeArenaRef(512),
            0),
      kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, RejectsMisalignedRefs) {
  // Kernels cast arena + offset straight to float*/int32_t*: an unaligned
  // offset is UB on strict-alignment targets, so it must be a load error.
  EXPECT_ERROR(ValidateInstruction(AddEw(up::MakeArenaRef(2),
                                         up::MakeArenaRef(256),
                                         up::MakeArenaRef(512), 16),
                                   kArena, kRodata, up::kSeeuVersion));
  // Quantized B is 1-byte-per-element rodata: any offset is fine there.
  up::UpdateInstruction q8;
  q8.opcode = static_cast<uint16_t>(up::OpCode::kGemmNNQ8);
  q8.in[0] = up::MakeArenaRef(0);
  q8.in[1] = up::MakeRodataRef(3);
  q8.in[2] = up::MakeArenaRef(512);
  q8.out[0] = 2;
  q8.out[1] = 2;
  q8.out[2] = 2;
  EXPECT_OK(ValidateInstruction(q8, kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, RejectsAliasingWriteAndReadOperands) {
  // The kernels carry no-alias (restrict) qualifiers on the arena binder's
  // no-overlap guarantee; a foreign plan must prove disjointness or blind
  // dispatch is UB. Read-read overlap stays legal.
  EXPECT_ERROR(ValidateInstruction(
      AddEw(up::MakeArenaRef(0), up::MakeArenaRef(256), up::MakeArenaRef(32),
            16),  // write [32,96) overlaps read [0,64)
      kArena, kRodata, up::kSeeuVersion));
  EXPECT_OK(ValidateInstruction(
      AddEw(up::MakeArenaRef(0), up::MakeArenaRef(0), up::MakeArenaRef(512),
            16),  // x and y share storage: reads may alias
      kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, EveryCompiledInstructionValidates) {
  // Regression: the compiler must never emit an instruction the validator
  // rejects — the exact contract the engine's load re-proves on device.
  up::SmfModel model = MakeMlp(6, 10, 3, 1);
  ASSERT_OK_AND_ASSIGN(up::CompiledUpdate compiled,
                       up::UpdateCompiler(BaseConfig(4)).Compile(model));
  up::PlanHeader h;
  std::memcpy(&h, compiled.plan.data(), sizeof(h));

  auto check = [&](uint64_t offset, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
      up::UpdateInstruction ins;
      std::memcpy(&ins,
                  compiled.plan.data() + offset + i * sizeof(ins),
                  sizeof(ins));
      EXPECT_OK(ValidateInstruction(ins, h.arena_size, h.rodata_size, h.version));
    }
  };
  check(h.train_instr_offset, h.train_instr_count);
  check(h.eval_instr_offset, h.eval_instr_count);
  check(h.merge_instr_offset, h.merge_instr_count);
}

}  // namespace
