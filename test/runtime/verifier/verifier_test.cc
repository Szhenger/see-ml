// =============================================================================
// validator/ unit tests: the load-time bounds proof the executor's blind
// dispatch rests on — per-opcode operand extents, the write-only-to-arena
// rule, quantized-B pinning to rodata, overflow-safe bounds math, and the
// regression that every instruction the compiler emits validates.
// =============================================================================

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "compiler/driver/update_compiler.h"
#include "runtime/dispatcher/contract.h"
#include "runtime/verifier/plan_validator.h"
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

TEST(PlanValidator, Bf16GemmsAreVersionGatedAndRodataPinned) {
  // M=4, N=8, K=16: A 64 floats (arena), B 128 bf16 = 256 bytes (rodata),
  // C 32 floats (arena), bias 8 floats (rodata) under the fused flag.
  up::UpdateInstruction g;
  g.opcode = static_cast<uint16_t>(up::OpCode::kGemmNNBF16);
  g.in[0] = up::MakeArenaRef(0);
  g.in[1] = up::MakeRodataRef(0);
  g.in[2] = up::MakeArenaRef(512);
  g.out[0] = 4;
  g.out[1] = 8;
  g.out[2] = 16;
  EXPECT_OK(ValidateInstruction(g, kArena, kRodata, up::kSeeuVersion));
  EXPECT_ERROR(ValidateInstruction(g, kArena, kRodata,
                                   up::kSeeuBf16Version - 1));
  // bf16 B is two bytes per element: 128 elements fit in 256 rodata bytes
  // and would not as f32 — an f32 extent check would over-reject.
  g.out[1] = 8;
  g.out[2] = 16;  // 128 elements x 2 B = 256 B == kRodata, exactly in bounds
  EXPECT_OK(ValidateInstruction(g, kArena, kRodata, up::kSeeuVersion));
  g.out[2] = 17;  // 136 elements x 2 B > kRodata
  EXPECT_ERROR(ValidateInstruction(g, kArena, kRodata, up::kSeeuVersion));
  g.out[2] = 16;
  g.in[1] = up::MakeArenaRef(768);  // bf16 B must be rodata
  EXPECT_ERROR(ValidateInstruction(g, kArena, kRodata, up::kSeeuVersion));
  g.in[1] = up::MakeRodataRef(1);  // ...and 2-byte aligned
  g.out[2] = 8;                    // 64 elements, plenty of room
  EXPECT_ERROR(ValidateInstruction(g, kArena, kRodata, up::kSeeuVersion));
  g.in[1] = up::MakeRodataRef(2);
  EXPECT_OK(ValidateInstruction(g, kArena, kRodata, up::kSeeuVersion));
  g.out[2] = 16;
  g.in[1] = up::MakeRodataRef(0);
  // The fused bias epilogue is allowed on the bf16 NN GEMM, like kGemmNN.
  g.flags = up::MakeEpilogueFlags(true, up::EpilogueAct::kRelu);
  g.in[3] = up::MakeArenaRef(768);
  EXPECT_OK(ValidateInstruction(g, kArena, kRodata, up::kSeeuVersion));
  // ...and not on the NT variant, whose B must be rodata as well.
  g.opcode = static_cast<uint16_t>(up::OpCode::kGemmNTBF16);
  EXPECT_ERROR(ValidateInstruction(g, kArena, kRodata, up::kSeeuVersion));
  g.flags = 0;
  EXPECT_OK(ValidateInstruction(g, kArena, kRodata, up::kSeeuVersion));
  g.in[1] = up::MakeArenaRef(768);
  EXPECT_ERROR(ValidateInstruction(g, kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, AccumulateIsVersionGatedAndInPlace) {
  up::UpdateInstruction acc;
  acc.opcode = static_cast<uint16_t>(up::OpCode::kAccumulate);
  acc.in[0] = up::MakeArenaRef(0);    // dst: read + written, one operand
  acc.in[1] = up::MakeArenaRef(256);  // src
  acc.out[0] = 32;
  EXPECT_OK(ValidateInstruction(acc, kArena, kRodata, up::kSeeuVersion));
  EXPECT_ERROR(ValidateInstruction(acc, kArena, kRodata,
                                   up::kSeeuGradAccumVersion - 1));
  acc.in[1] = up::MakeArenaRef(64);  // src overlaps the written dst
  EXPECT_ERROR(ValidateInstruction(acc, kArena, kRodata, up::kSeeuVersion));
  acc.in[1] = up::MakeRodataRef(0);  // a rodata source is fine
  EXPECT_OK(ValidateInstruction(acc, kArena, kRodata, up::kSeeuVersion));
  acc.in[0] = up::MakeRodataRef(0);  // a rodata destination is not
  EXPECT_ERROR(ValidateInstruction(acc, kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, KlLossScaleIsVersionGated) {
  // N=2, C=3 -> 6 floats per logit/prob tensor, disjoint arena slots; the
  // loss is one float. T = 2 in the low word; the v8 scale in the high word.
  const auto f32 = [](float f) {
    return uint64_t{std::bit_cast<uint32_t>(f)};
  };
  up::UpdateInstruction fwd;
  fwd.opcode = static_cast<uint16_t>(up::OpCode::kKLDistillFwd);
  fwd.in[0] = up::MakeArenaRef(0);
  fwd.in[1] = up::MakeArenaRef(64);
  fwd.in[2] = up::MakeArenaRef(128);
  fwd.in[3] = up::MakeArenaRef(192);
  fwd.out[0] = up::MakeArenaRef(256);
  fwd.out[1] = (uint64_t{2} << 32) | 3;
  fwd.out[2] = f32(2.0f);  // no scale: means 1.0 at any version
  EXPECT_OK(ValidateInstruction(fwd, kArena, kRodata, up::kSeeuVersion));
  EXPECT_OK(ValidateInstruction(fwd, kArena, kRodata,
                                up::kSeeuKlScaleVersion - 1));

  fwd.out[2] = (f32(4.0f) << 32) | f32(2.0f);  // T^2 scale, the v8 form
  EXPECT_OK(ValidateInstruction(fwd, kArena, kRodata, up::kSeeuVersion));
  // A pre-v8 plan never wrote the high word: corruption, not a feature.
  EXPECT_ERROR(ValidateInstruction(fwd, kArena, kRodata,
                                   up::kSeeuKlScaleVersion - 1));
  // A written scale must be a finite positive float — 0 or NaN would train
  // on nothing, silently.
  // (an all-zero high word is "absent" and accepted; a signed zero is not)
  fwd.out[2] = (f32(-0.0f) << 32) | f32(2.0f);
  EXPECT_ERROR(ValidateInstruction(fwd, kArena, kRodata, up::kSeeuVersion));
  fwd.out[2] = (f32(std::bit_cast<float>(0x7FC00000u)) << 32) | f32(2.0f);
  EXPECT_ERROR(ValidateInstruction(fwd, kArena, kRodata, up::kSeeuVersion));

  up::UpdateInstruction bwd;
  bwd.opcode = static_cast<uint16_t>(up::OpCode::kKLDistillBwd);
  bwd.in[0] = up::MakeArenaRef(0);
  bwd.in[1] = up::MakeArenaRef(64);
  bwd.in[2] = up::MakeArenaRef(128);
  bwd.in[3] = up::MakeArenaRef(192);
  bwd.out[0] = (uint64_t{2} << 32) | 3;
  bwd.out[1] = (f32(4.0f) << 32) | f32(2.0f);
  EXPECT_OK(ValidateInstruction(bwd, kArena, kRodata, up::kSeeuVersion));
  EXPECT_ERROR(ValidateInstruction(bwd, kArena, kRodata,
                                   up::kSeeuKlScaleVersion - 1));
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

TEST(PlanValidator, RmsNormBackwardProvesTheStatsRef) {
  // kRmsNormBwd carries the rstd cache as a REF in out[0]; its extent
  // (rows floats) must be proven like any operand, not trusted.
  up::UpdateInstruction bwd;
  bwd.opcode = static_cast<uint16_t>(up::OpCode::kRmsNormBwd);
  bwd.in[0] = up::MakeArenaRef(0);    // dy [4 x 8]
  bwd.in[1] = up::MakeArenaRef(128);  // x
  bwd.in[2] = up::MakeArenaRef(256);  // gamma [8]
  bwd.in[3] = up::MakeArenaRef(512);  // dx (write)
  bwd.out[0] = up::MakeArenaRef(768); // rstd [4]
  bwd.out[1] = (uint64_t{4} << 32) | 8;
  EXPECT_OK(ValidateInstruction(bwd, kArena, kRodata, up::kSeeuVersion));
  bwd.out[0] = up::MakeArenaRef(kArena - 4);  // one float short of 4
  EXPECT_ERROR(ValidateInstruction(bwd, kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, TheTiledAttentionFamilyIsVersionedAndBounded) {
  // B = 1, S = 4, H = 2, d = 4: activations are 32 floats (128 B), the
  // stats row B*H*S*4 = 32 floats. Slots: q 0, k 128, v 256, o/dO 384,
  // stats 512, result 640 — all inside the 1024-byte arena.
  const uint64_t bs = (uint64_t{1} << 32) | 4, hd = (uint64_t{2} << 32) | 4;
  up::UpdateInstruction fwd;
  fwd.opcode = static_cast<uint16_t>(up::OpCode::kAttnFwdTiled);
  fwd.in[0] = up::MakeArenaRef(0);
  fwd.in[1] = up::MakeArenaRef(128);
  fwd.in[2] = up::MakeArenaRef(256);
  fwd.in[3] = up::MakeArenaRef(384);
  fwd.out[0] = up::MakeArenaRef(512);
  fwd.out[1] = bs;
  fwd.out[2] = hd;
  EXPECT_OK(ValidateInstruction(fwd, kArena, kRodata, up::kSeeuVersion));
  const auto old = ValidateInstruction(fwd, kArena, kRodata,
                                       up::kSeeuTiledAttentionVersion - 1);
  ASSERT_FALSE(old.has_value());
  EXPECT_STR_CONTAINS(old.error(), "tiled attention");
  fwd.out[0] = up::MakeArenaRef(1024 - 64);  // stats row runs off the arena
  EXPECT_ERROR(ValidateInstruction(fwd, kArena, kRodata, up::kSeeuVersion));

  for (const up::OpCode op : {up::OpCode::kAttnDQTiled,
                              up::OpCode::kAttnDKTiled,
                              up::OpCode::kAttnDVTiled}) {
    up::UpdateInstruction bwd;
    bwd.opcode = static_cast<uint16_t>(op);
    bwd.in[0] = up::MakeArenaRef(0);
    bwd.in[1] = up::MakeArenaRef(128);
    bwd.in[2] = up::MakeArenaRef(256);
    bwd.in[3] = up::MakeArenaRef(384);
    bwd.out[0] = up::MakeArenaRef(512);
    bwd.out[1] = up::MakeArenaRef(640);
    bwd.out[2] = up::PackAttnGeometry(1, 4, 2, 4);
    EXPECT_OK(ValidateInstruction(bwd, kArena, kRodata, up::kSeeuVersion));
    bwd.out[2] = up::PackAttnGeometry(1, 4, 0, 4);  // a zero field
    EXPECT_ERROR(ValidateInstruction(bwd, kArena, kRodata, up::kSeeuVersion));
    bwd.out[2] = up::PackAttnGeometry(1, 4, 2, 4);
    bwd.out[1] = up::MakeArenaRef(512);  // the result aliases the stats row
    EXPECT_ERROR(ValidateInstruction(bwd, kArena, kRodata, up::kSeeuVersion));
  }
}

TEST(PlanValidator, AttentionBackwardExtentsAreProven) {
  // kAttnDP writes the [B*H*S, S] probability-shaped dP; the write extent
  // derives from the packed geometry and must stay inside the arena, and
  // it must not alias the reads.
  up::UpdateInstruction dp;
  dp.opcode = static_cast<uint16_t>(up::OpCode::kAttnDP);
  dp.in[0] = up::MakeArenaRef(0);    // dO [4 x 8] = 128 B
  dp.in[1] = up::MakeArenaRef(128);  // v
  dp.in[2] = up::MakeArenaRef(256);  // dP: B*H*S*S = 32 floats = 128 B
  dp.out[0] = (uint64_t{1} << 32) | 4;  // B=1, S=4
  dp.out[1] = (uint64_t{2} << 32) | 4;  // H=2, d=4
  EXPECT_OK(ValidateInstruction(dp, kArena, kRodata, up::kSeeuVersion));
  dp.in[2] = up::MakeArenaRef(kArena - 64);  // write range spills out
  EXPECT_ERROR(ValidateInstruction(dp, kArena, kRodata, up::kSeeuVersion));
  dp.in[2] = up::MakeArenaRef(64);  // write overlaps the dO read
  EXPECT_ERROR(ValidateInstruction(dp, kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, SoftmaxRowsBackwardOverflowIsRejected) {
  // rows and cols each fill their 32-bit half; their product must go
  // through MulOk — 2^32 * 2^32 wraps u64 to zero-ish garbage.
  up::UpdateInstruction sm;
  sm.opcode = static_cast<uint16_t>(up::OpCode::kSoftmaxRowsBwd);
  sm.in[0] = up::MakeArenaRef(0);
  sm.in[1] = up::MakeArenaRef(256);
  sm.in[2] = up::MakeArenaRef(512);
  sm.out[0] = (uint64_t{0xFFFFFFFFu} << 32) | 0xFFFFFFFFu;
  EXPECT_ERROR(ValidateInstruction(sm, kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, EmbedProvesBuffersAndDemandsRodataTable) {
  // T=4 tokens (i32), table [V=8, D=4] in RODATA, out [4, 4] in the arena.
  up::UpdateInstruction emb;
  emb.opcode = static_cast<uint16_t>(up::OpCode::kEmbedFwd);
  emb.in[0] = up::MakeArenaRef(0);
  emb.in[1] = up::MakeRodataRef(0);  // 8*4*4 = 128 B <= kRodata
  emb.in[2] = up::MakeArenaRef(64);
  emb.out[0] = 4;
  emb.out[1] = (uint64_t{8} << 32) | 4;
  EXPECT_OK(ValidateInstruction(emb, kArena, kRodata, up::kSeeuVersion));
  // The gather's index bound is the FEEDER contract's job, but the buffers
  // are this validator's: an arena-resident table is rejected (only the
  // compiler's rodata packing produces one), as is a pre-v7 plan carrying
  // the opcode, an out-of-bounds output, and an output aliasing the tokens.
  emb.in[1] = up::MakeArenaRef(256);
  EXPECT_ERROR(ValidateInstruction(emb, kArena, kRodata, up::kSeeuVersion));
  emb.in[1] = up::MakeRodataRef(0);
  EXPECT_ERROR_CONTAINS(
      ValidateInstruction(emb, kArena, kRodata, up::kSeeuTokenVersion - 1),
      "token opcode");
  emb.in[2] = up::MakeArenaRef(kArena - 32);
  EXPECT_ERROR(ValidateInstruction(emb, kArena, kRodata, up::kSeeuVersion));
  emb.in[2] = up::MakeArenaRef(0);  // write over the tokens read
  EXPECT_ERROR(ValidateInstruction(emb, kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, RopeRequiresEvenHeadWidth) {
  up::UpdateInstruction rope;
  rope.opcode = static_cast<uint16_t>(up::OpCode::kRopeFwd);
  rope.in[0] = up::MakeArenaRef(0);
  rope.in[1] = up::MakeArenaRef(96);
  rope.out[0] = (uint64_t{1} << 32) | 2;  // B=1, S=2
  rope.out[1] = (uint64_t{2} << 32) | 3;  // H=2, d=3: odd head width
  rope.out[2] = std::bit_cast<uint32_t>(10000.0f);
  EXPECT_ERROR(ValidateInstruction(rope, kArena, kRodata, up::kSeeuVersion));
  rope.out[1] = (uint64_t{2} << 32) | 4;  // d=4 with disjoint in/out
  rope.in[1] = up::MakeArenaRef(512);
  EXPECT_OK(ValidateInstruction(rope, kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, RopeBaseMustBeAUsableFloat) {
  // out[2] carries the rotary base as f32 bits (SMF v5 attr1 → lowering).
  // A zero, NaN, infinite, or <= 1 base makes every angle degenerate; a
  // stray high word means the word was not written by this lowering.
  up::UpdateInstruction rope;
  rope.opcode = static_cast<uint16_t>(up::OpCode::kRopeBwd);
  rope.in[0] = up::MakeArenaRef(0);
  rope.in[1] = up::MakeArenaRef(512);
  rope.out[0] = (uint64_t{1} << 32) | 2;  // B=1, S=2
  rope.out[1] = (uint64_t{2} << 32) | 4;  // H=2, d=4
  for (const float ok_base : {10000.0f, 500000.0f, 1000000.0f, 1.5f}) {
    rope.out[2] = std::bit_cast<uint32_t>(ok_base);
    EXPECT_OK(ValidateInstruction(rope, kArena, kRodata, up::kSeeuVersion));
  }
  for (const uint64_t bad : {uint64_t{0}, uint64_t{0x7FC00000u},
                             uint64_t{0x7F800000u},
                             uint64_t{std::bit_cast<uint32_t>(1.0f)},
                             uint64_t{std::bit_cast<uint32_t>(-10000.0f)},
                             (uint64_t{1} << 32) |
                                 std::bit_cast<uint32_t>(10000.0f)}) {
    rope.out[2] = bad;
    EXPECT_ERROR(ValidateInstruction(rope, kArena, kRodata, up::kSeeuVersion));
  }
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

TEST(PlanValidator, ASourceRefIsReadOnlyAndBelongsToTheEvalProgram) {
  // v17 (E12, #95): bit 62 addresses the source model file. The eval
  // program may read f32 weights through it; no other program may, no
  // plan below v17 may, and nothing may write through it.
  up::UpdateInstruction gemm = GemmNN(up::MakeArenaRef(0),
                                      up::MakeSourceRef(4096),
                                      up::MakeArenaRef(512), 4, 4, 4);
  EXPECT_OK(ValidateInstruction(gemm, kArena, kRodata, up::kSeeuVersion,
                                /*allow_source=*/true));
  auto ex = DescribeInstruction(gemm, kArena, kRodata, up::kSeeuVersion,
                                /*allow_source=*/true);
  ASSERT_TRUE(ex.has_value());
  bool saw = false;
  for (size_t i = 0; i < ex->count; ++i)
    if (ex->ranges[i].source) {
      saw = true;
      EXPECT_EQ(ex->ranges[i].off, 4096u);
      EXPECT_EQ(ex->ranges[i].bytes, 4u * 4u * sizeof(float));
      EXPECT_FALSE(ex->ranges[i].write);
    }
  EXPECT_TRUE(saw);
  EXPECT_ERROR(ValidateInstruction(gemm, kArena, kRodata, up::kSeeuVersion));
  EXPECT_ERROR(ValidateInstruction(gemm, kArena, kRodata,
                                   up::kSeeuShippedEvalVersion - 1, true));
  up::UpdateInstruction into = GemmNN(up::MakeArenaRef(0), up::MakeArenaRef(64),
                                      up::MakeSourceRef(0), 4, 4, 4);
  EXPECT_ERROR(ValidateInstruction(into, kArena, kRodata, up::kSeeuVersion,
                                   true));
  // A source extent never aliases an arena write at the same offset.
  up::UpdateInstruction same = GemmNN(up::MakeArenaRef(0),
                                      up::MakeSourceRef(512),
                                      up::MakeArenaRef(512), 4, 4, 4);
  EXPECT_OK(ValidateInstruction(same, kArena, kRodata, up::kSeeuVersion,
                                true));
}

TEST(PlanValidator, PerColumnInt8ScalesAreARodataVectorFromV17) {
  // kFlagQ8ColScale: in[3] is a rodata ref to N floats (NN) or K floats
  // (NT), only on the q8 GEMMs, only from v17.
  const size_t M = 4, N = 8, K = 16;
  up::UpdateInstruction q8;
  q8.opcode = static_cast<uint16_t>(up::OpCode::kGemmNNQ8);
  q8.flags = up::kFlagQ8ColScale;
  q8.in[0] = up::MakeArenaRef(0);
  q8.in[1] = up::MakeRodataRef(0);          // K*N int8 = 128 B
  q8.in[2] = up::MakeArenaRef(512);
  q8.in[3] = up::MakeRodataRef(128);        // N floats = 32 B
  q8.out[0] = M; q8.out[1] = N; q8.out[2] = K;
  EXPECT_OK(ValidateInstruction(q8, kArena, kRodata, up::kSeeuVersion));
  EXPECT_ERROR(ValidateInstruction(q8, kArena, kRodata,
                                   up::kSeeuShippedEvalVersion - 1));
  q8.in[3] = up::MakeArenaRef(256);  // the scales must be rodata
  EXPECT_ERROR(ValidateInstruction(q8, kArena, kRodata, up::kSeeuVersion));
  q8.in[3] = up::MakeRodataRef(kRodata - 16);  // runs off the section
  EXPECT_ERROR(ValidateInstruction(q8, kArena, kRodata, up::kSeeuVersion));
  q8.opcode = static_cast<uint16_t>(up::OpCode::kGemmNTQ8);
  q8.in[3] = up::MakeRodataRef(128);        // K floats = 64 B for NT
  EXPECT_OK(ValidateInstruction(q8, kArena, kRodata, up::kSeeuVersion));
  q8.opcode = static_cast<uint16_t>(up::OpCode::kGemmNN);
  EXPECT_ERROR(ValidateInstruction(q8, kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, TheRelaxedBitIsAFrozenGemmPermissionFromV18) {
  // kFlagRelaxed (v18): the six frozen-weight GEMMs, train and step
  // programs only — never the eval program (allow_source), never TN, never
  // below v18. Rodata is 256 B here: K*N f32 = 128 B.
  const size_t M = 4, N = 4, K = 8;
  up::UpdateInstruction g;
  g.opcode = static_cast<uint16_t>(up::OpCode::kGemmNN);
  g.flags = up::kFlagRelaxed;
  g.in[0] = up::MakeArenaRef(0);     // M*K f32 = 128 B
  g.in[1] = up::MakeRodataRef(0);    // K*N f32 = 128 B
  g.in[2] = up::MakeArenaRef(512);   // M*N f32 = 64 B
  g.out[0] = M; g.out[1] = N; g.out[2] = K;
  EXPECT_OK(ValidateInstruction(g, kArena, kRodata, up::kSeeuVersion));
  EXPECT_ERROR(ValidateInstruction(g, kArena, kRodata,
                                   up::kSeeuRelaxedVersion - 1));
  // At the instruction level the bit is an opcode permission — which
  // program may carry it is the executor contract's rule (contract.cc).
  EXPECT_OK(ValidateInstruction(g, kArena, kRodata, up::kSeeuVersion,
                                /*allow_source=*/true));
  // It composes with the epilogue on NN and with the addend on NT.
  g.flags = up::kFlagRelaxed | up::MakeEpilogueFlags(false,
                                                     up::EpilogueAct::kGelu);
  EXPECT_OK(ValidateInstruction(g, kArena, kRodata, up::kSeeuVersion));
  up::UpdateInstruction nt = g;
  nt.opcode = static_cast<uint16_t>(up::OpCode::kGemmNT);
  nt.flags = up::kFlagRelaxed | up::kFlagGemmAddend;
  nt.in[3] = up::MakeArenaRef(768);  // M*N f32 = 64 B
  EXPECT_OK(ValidateInstruction(nt, kArena, kRodata, up::kSeeuVersion));
  // The int8 and bf16 forms take it; TN and everything else do not.
  up::UpdateInstruction q8 = g;
  q8.flags = up::kFlagRelaxed | up::kFlagQ8ColScale;
  q8.opcode = static_cast<uint16_t>(up::OpCode::kGemmNNQ8);
  q8.in[3] = up::MakeRodataRef(128);  // N floats = 16 B
  EXPECT_OK(ValidateInstruction(q8, kArena, kRodata, up::kSeeuVersion));
  up::UpdateInstruction bf = g;
  bf.flags = up::kFlagRelaxed;
  bf.opcode = static_cast<uint16_t>(up::OpCode::kGemmNTBF16);  // N*K bf16 = 64 B
  EXPECT_OK(ValidateInstruction(bf, kArena, kRodata, up::kSeeuVersion));
  up::UpdateInstruction tn = g;
  tn.flags = up::kFlagRelaxed;
  tn.opcode = static_cast<uint16_t>(up::OpCode::kGemmTN);
  tn.in[1] = up::MakeArenaRef(256);  // K*N f32 = 128 B
  EXPECT_ERROR(ValidateInstruction(tn, kArena, kRodata, up::kSeeuVersion));
  up::UpdateInstruction add;
  add.opcode = static_cast<uint16_t>(up::OpCode::kAddEW);
  add.flags = up::kFlagRelaxed;
  add.in[0] = up::MakeArenaRef(0); add.in[1] = up::MakeArenaRef(64);
  add.in[2] = up::MakeArenaRef(128); add.out[0] = 16;
  EXPECT_ERROR(ValidateInstruction(add, kArena, kRodata, up::kSeeuVersion));
}

TEST(PlanValidator, TheImmWordIsANormEpsilonAndNothingElse) {
  // v16 (P7, #96): the former pad word carries a normalization forward's
  // epsilon; anywhere else, or below v16, a nonzero imm is corruption.
  // RMSNorm over 4 rows x 4 cols: x at 0, gamma at 64, y at 128, rstd 192.
  up::UpdateInstruction rms;
  rms.opcode = static_cast<uint16_t>(up::OpCode::kRmsNormFwd);
  rms.in[0] = up::MakeArenaRef(0);
  rms.in[1] = up::MakeArenaRef(64);
  rms.in[2] = up::MakeArenaRef(128);
  rms.in[3] = up::MakeArenaRef(192);
  rms.out[0] = (uint64_t{4} << 32) | 4;
  EXPECT_OK(ValidateInstruction(rms, kArena, kRodata, up::kSeeuVersion));
  rms.imm = std::bit_cast<uint32_t>(1e-6f);
  EXPECT_OK(ValidateInstruction(rms, kArena, kRodata, up::kSeeuVersion));
  const auto old = ValidateInstruction(rms, kArena, kRodata,
                                       up::kSeeuNormEpsVersion - 1);
  ASSERT_FALSE(old.has_value());
  EXPECT_STR_CONTAINS(old.error(), "imm");
  for (const float bad : {-1e-6f, INFINITY, NAN}) {
    rms.imm = std::bit_cast<uint32_t>(bad);
    EXPECT_ERROR(ValidateInstruction(rms, kArena, kRodata, up::kSeeuVersion));
  }
  up::UpdateInstruction add = AddEw(up::MakeArenaRef(0), up::MakeArenaRef(256),
                                    up::MakeArenaRef(512), 16);
  add.imm = std::bit_cast<uint32_t>(1e-6f);
  const auto r = ValidateInstruction(add, kArena, kRodata, up::kSeeuVersion);
  ASSERT_FALSE(r.has_value());
  EXPECT_STR_CONTAINS(r.error(), "imm");
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

TEST(PlanValidator, TheGemmAddendIsAVersionedReadOfTheWholeResultShape) {
  // kFlagGemmAddend (v14): C = D + A@B on the three f32 GEMMs, D's ref in
  // in[3]. M = N = K = 4: A at 0, B at 64, C at 512, D at 256 (64 B each).
  for (const up::OpCode op :
       {up::OpCode::kGemmNN, up::OpCode::kGemmNT, up::OpCode::kGemmTN}) {
    up::UpdateInstruction gemm = GemmNN(
        up::MakeArenaRef(0), up::MakeArenaRef(64), up::MakeArenaRef(512), 4, 4, 4);
    gemm.opcode = static_cast<uint16_t>(op);
    gemm.flags = up::kFlagGemmAddend;
    gemm.in[3] = up::MakeArenaRef(256);
    EXPECT_OK(ValidateInstruction(gemm, kArena, kRodata, up::kSeeuVersion));

    // A v13 runtime's vocabulary does not hold the bit: corruption there.
    const auto old = ValidateInstruction(gemm, kArena, kRodata,
                                         up::kSeeuGemmAddendVersion - 1);
    ASSERT_FALSE(old.has_value());
    EXPECT_STR_CONTAINS(old.error(), "flags");

    // The addend is M*N floats: 964 + 64 B > 1024.
    gemm.in[3] = up::MakeArenaRef(964);
    EXPECT_ERROR(ValidateInstruction(gemm, kArena, kRodata, up::kSeeuVersion));

    // The kernels hold C restrict: D may not alias it (GemmAccNN is the
    // in-place form, and rounds differently).
    gemm.in[3] = up::MakeArenaRef(512);
    const auto alias =
        ValidateInstruction(gemm, kArena, kRodata, up::kSeeuVersion);
    ASSERT_FALSE(alias.has_value());
    EXPECT_STR_CONTAINS(alias.error(), "alias");
  }
}

TEST(PlanValidator, TheGemmAddendSharesItsSlotWithNothing) {
  // in[3] holds one ref: an addend excludes the bias / activation epilogue
  // outright, and the narrow-weight GEMMs (in[3] = a dequant scale, or no
  // addend kernel) and every non-GEMM opcode refuse the bit.
  up::UpdateInstruction gemm = GemmNN(up::MakeArenaRef(0), up::MakeArenaRef(64),
                                      up::MakeArenaRef(512), 4, 4, 4);
  gemm.in[3] = up::MakeArenaRef(256);
  for (const uint16_t epilogue :
       {up::MakeEpilogueFlags(true, up::EpilogueAct::kNone),
        up::MakeEpilogueFlags(false, up::EpilogueAct::kRelu)}) {
    gemm.flags = static_cast<uint16_t>(up::kFlagGemmAddend | epilogue);
    const auto r = ValidateInstruction(gemm, kArena, kRodata, up::kSeeuVersion);
    ASSERT_FALSE(r.has_value());
    EXPECT_STR_CONTAINS(r.error(), "addend");
  }
  gemm.flags = up::kFlagGemmAddend;
  for (const up::OpCode op :
       {up::OpCode::kGemmAccNN, up::OpCode::kGemmNNQ8, up::OpCode::kGemmNTQ8,
        up::OpCode::kGemmNNBF16, up::OpCode::kGemmNTBF16}) {
    gemm.opcode = static_cast<uint16_t>(op);
    EXPECT_ERROR(ValidateInstruction(gemm, kArena, kRodata, up::kSeeuVersion));
  }
  up::UpdateInstruction add = AddEw(up::MakeArenaRef(0), up::MakeArenaRef(256),
                                    up::MakeArenaRef(512), 16);
  add.flags = up::kFlagGemmAddend;
  EXPECT_ERROR(ValidateInstruction(add, kArena, kRodata, up::kSeeuVersion));
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

// --- kFusedMap (plan v13, E4): the micro-program is proven like any operand ---

up::UpdateInstruction FusedMap(uint64_t stages, uint64_t imms = 0) {
  up::UpdateInstruction ins;
  ins.opcode = static_cast<uint16_t>(up::OpCode::kFusedMap);
  ins.in[0] = up::MakeArenaRef(0);
  ins.in[3] = up::MakeArenaRef(768);
  ins.out[0] = 32;
  ins.out[1] = stages;
  ins.out[2] = imms;
  return ins;
}

TEST(PlanValidator, FusedMapProgramsAreProven) {
  using up::FusedStage;
  using up::MakeFusedStage;
  const uint64_t scale0 = MakeFusedStage(FusedStage::kScale, 0);
  const uint64_t add1 = MakeFusedStage(FusedStage::kAdd, 1);
  const uint64_t mul2r = MakeFusedStage(FusedStage::kMul, 2, true);
  const uint64_t silu = MakeFusedStage(FusedStage::kSilu);
  const uint64_t half = std::bit_cast<uint32_t>(0.5f);
  auto check = [&](up::UpdateInstruction ins, uint32_t version = up::kSeeuVersion) {
    return ValidateInstruction(ins, kArena, kRodata, version);
  };

  // scale -> add: the LoRA forward chain.
  up::UpdateInstruction lora = FusedMap(scale0 | add1 << 8, half);
  lora.in[1] = up::MakeArenaRef(256);
  EXPECT_OK(check(lora));
  EXPECT_ERROR_CONTAINS(check(lora, up::kSeeuFusedMapVersion - 1), "pre-v13");

  // Four stages, both slots, both immediates.
  up::UpdateInstruction full = FusedMap(
      silu | add1 << 8 |
          (uint64_t{MakeFusedStage(FusedStage::kScale, 1)} << 16) |
          mul2r << 24,
      half << 32);
  full.in[1] = up::MakeArenaRef(256);
  full.in[2] = up::MakeRodataRef(0);
  // rodata slot 2 is 256 bytes = 64 floats >= 32: in bounds.
  EXPECT_OK(check(full));

  // Every way a program can lie:
  EXPECT_ERROR(check(FusedMap(0)));                        // empty
  EXPECT_ERROR(check(FusedMap(7)));                        // unknown kind
  EXPECT_ERROR(check(FusedMap(silu | uint64_t{1} << 40)));  // bytes past 4 stages
  EXPECT_ERROR(check(FusedMap(silu | silu << 16)));        // data after the end
  EXPECT_ERROR(check(FusedMap(silu | 0x10)));              // arg on a unary
  EXPECT_ERROR(check(FusedMap(MakeFusedStage(FusedStage::kAdd, 0))));  // slot 0
  EXPECT_ERROR(check(FusedMap(add1)));                     // slot 1 named, absent
  up::UpdateInstruction stray = FusedMap(silu);
  stray.in[2] = up::MakeArenaRef(512);                     // slot 2 present, unnamed
  EXPECT_ERROR(check(stray));
  EXPECT_ERROR(check(FusedMap(silu, half)));               // immediate set, unused
  EXPECT_ERROR(check(FusedMap(
      scale0, std::bit_cast<uint32_t>(std::numeric_limits<float>::infinity()))));
  EXPECT_ERROR(check(FusedMap(scale0 | 0x80, half)));      // run-is-right on scale
  up::UpdateInstruction alias = FusedMap(silu);
  alias.in[3] = alias.in[0];                               // in-place is not the ABI
  EXPECT_ERROR_CONTAINS(check(alias), "alias");
  up::UpdateInstruction oob = FusedMap(silu);
  oob.out[0] = 1 << 20;
  EXPECT_ERROR(check(oob));
  up::UpdateInstruction ro_out = FusedMap(silu);
  ro_out.in[3] = up::MakeRodataRef(0);
  EXPECT_ERROR(check(ro_out));
}

}  // namespace
