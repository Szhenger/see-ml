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
                                kArena, kRodata));
}

TEST(PlanValidator, RejectsReadsPastTheArena) {
  const auto r = ValidateInstruction(
      AddEw(up::MakeArenaRef(900), up::MakeArenaRef(0), up::MakeArenaRef(256),
            64),  // 900 + 256 B > 1024
      kArena, kRodata);
  ASSERT_FALSE(r.has_value());
  EXPECT_STR_CONTAINS(r.error(), "instruction operand out of bounds");
  EXPECT_TRUE(WellFormedDiagnostic(r.error()));
}

TEST(PlanValidator, RejectsWritesIntoRodata) {
  // The frozen weights must be physically unwritable from the stream.
  EXPECT_ERROR(ValidateInstruction(AddEw(up::MakeArenaRef(0),
                                         up::MakeArenaRef(256),
                                         up::MakeRodataRef(0), 16),
                                   kArena, kRodata));
  up::UpdateInstruction fill;
  fill.opcode = static_cast<uint16_t>(up::OpCode::kFill);
  fill.in[0] = up::MakeRodataRef(0);
  fill.out[0] = 16;
  EXPECT_ERROR(ValidateInstruction(fill, kArena, kRodata));
}

TEST(PlanValidator, RejectsNullRefsAndUnknownOpcodes) {
  up::UpdateInstruction null_ref = AddEw(up::kNullRef, up::MakeArenaRef(0),
                                         up::MakeArenaRef(256), 16);
  EXPECT_ERROR(ValidateInstruction(null_ref, kArena, kRodata));

  up::UpdateInstruction bogus;
  bogus.opcode = 0xFFFF;
  const auto r = ValidateInstruction(bogus, kArena, kRodata);
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
  EXPECT_ERROR(ValidateInstruction(q8, kArena, kRodata));
  q8.in[1] = up::MakeRodataRef(0);
  EXPECT_OK(ValidateInstruction(q8, kArena, kRodata));
}

TEST(PlanValidator, RejectsWriteRangesAliasingOtherOperands) {
  // The kernels are compiled with SEEML_RESTRICT: a written range that
  // overlaps another operand is undefined behavior, so a plan carrying one
  // must be a load error, never a dispatch.
  const auto r = ValidateInstruction(
      AddEw(up::MakeArenaRef(0), up::MakeArenaRef(512),
            up::MakeArenaRef(128), 64),  // out 128..384 overlaps in0 0..256
      kArena, kRodata);
  ASSERT_FALSE(r.has_value());
  EXPECT_STR_CONTAINS(r.error(), "alias");
  EXPECT_TRUE(WellFormedDiagnostic(r.error()));
}

TEST(PlanValidator, AllowsReadOnlyOperandsToAlias) {
  // in0 == in1 (x + x): reads may share a range; only writes make an alias.
  EXPECT_OK(ValidateInstruction(AddEw(up::MakeArenaRef(0), up::MakeArenaRef(0),
                                      up::MakeArenaRef(512), 64),
                                kArena, kRodata));
}

TEST(PlanValidator, InPlaceOptimizerStepsAliasOnlyThroughOneRef) {
  // SGD updates the param through a single read-write ref — legal. The same
  // range surfacing again as the gradient operand is an alias — rejected.
  up::UpdateInstruction sgd;
  sgd.opcode = static_cast<uint16_t>(up::OpCode::kSgdStep);
  sgd.in[0] = up::MakeArenaRef(0);
  sgd.in[1] = up::MakeArenaRef(256);
  sgd.out[0] = 64;
  EXPECT_OK(ValidateInstruction(sgd, kArena, kRodata));
  sgd.in[1] = up::MakeArenaRef(128);  // grad overlaps the updated param
  EXPECT_ERROR(ValidateInstruction(sgd, kArena, kRodata));

  up::UpdateInstruction adamw;
  adamw.opcode = static_cast<uint16_t>(up::OpCode::kAdamWStep);
  adamw.in[0] = up::MakeArenaRef(0);    // param (in place)
  adamw.in[1] = up::MakeArenaRef(256);  // grad
  adamw.in[2] = up::MakeArenaRef(512);  // m (in place)
  adamw.in[3] = up::MakeArenaRef(768);  // v (in place)
  adamw.out[0] = 64;
  EXPECT_OK(ValidateInstruction(adamw, kArena, kRodata));
  adamw.in[3] = up::MakeArenaRef(512);  // v aliases m: two written ranges
  EXPECT_ERROR(ValidateInstruction(adamw, kArena, kRodata));
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
      kArena, kRodata));
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
      EXPECT_OK(ValidateInstruction(ins, h.arena_size, h.rodata_size));
    }
  };
  check(h.train_instr_offset, h.train_instr_count);
  check(h.eval_instr_offset, h.eval_instr_count);
  check(h.merge_instr_offset, h.merge_instr_count);
}

}  // namespace
