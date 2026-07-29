// =============================================================================
// trainer/ unit tests: arena alignment, the liveness-driven transient
// allocator (correct reuse — the efficiency regression guard — and pinning),
// and instruction lowering (opcode mapping, storage-op elision, q8
// selection, unloweable-op rejection).
// =============================================================================

#include <bit>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "compiler/backend/trainer/arena_binder.h"
#include "compiler/backend/trainer/instruction_lowering.h"
#include "compiler/frontend/representation/sir.h"
#include "source/plan/update_types.h"
#include "test/framework/seetest.h"

namespace {

using namespace seeml::update;
namespace sir = seeml::sir;

TEST(ArenaBinder, AlignUpRoundsToTheArenaAlignment) {
  EXPECT_EQ(AlignUp(0), 0u);
  EXPECT_EQ(AlignUp(1), 64u);
  EXPECT_EQ(AlignUp(64), 64u);
  EXPECT_EQ(AlignUp(65), 128u);
}

/// A three-op chain a -> b -> c of 1024-float values: `a` dies when `b` is
/// born from it, so `c` can reuse `a`'s slot.
struct Chain {
  sir::Block block;
  sir::Value* a;
  sir::Value* b;
  sir::Value* c;

  Chain() {
    sir::Operation* o1 = block.appendOp("sc_high.relu");
    a = o1->addResult("a", sir::DataType::F32, sir::Shape{1024});
    sir::Operation* o2 = block.appendOp("sc_high.relu");
    o2->addOperand(a);
    b = o2->addResult("b", sir::DataType::F32, sir::Shape{1024});
    sir::Operation* o3 = block.appendOp("sc_high.relu");
    o3->addOperand(b);
    c = o3->addResult("c", sir::DataType::F32, sir::Shape{1024});
  }
};

TEST(ArenaBinder, TransientScanReusesDeadSlots) {
  Chain chain;
  std::unordered_map<const sir::Value*, uint64_t> bound, refs;
  std::unordered_set<const sir::Value*> pinned;

  const uint64_t high =
      LinearScanTransients(chain.block, /*base=*/0, bound, pinned, refs);

  // Efficiency regression: `c` reuses `a`'s offset, so the high-water mark
  // is two slots, not three (a slot is 1024 f32 = 4096 B).
  EXPECT_EQ(refs.at(chain.c), refs.at(chain.a));
  EXPECT_EQ(high, 2u * 4096u);
}

TEST(ArenaBinder, PinnedValuesAreNeverReclaimed) {
  Chain chain;
  std::unordered_map<const sir::Value*, uint64_t> bound, refs;
  std::unordered_set<const sir::Value*> pinned{chain.a};

  const uint64_t high =
      LinearScanTransients(chain.block, /*base=*/0, bound, pinned, refs);

  // `a` stays live forever, so `c` cannot take its slot.
  EXPECT_NE(refs.at(chain.c), refs.at(chain.a));
  EXPECT_EQ(high, 3u * 4096u);
}

TEST(ArenaBinder, AlreadyBoundValuesAreSkipped) {
  Chain chain;
  std::unordered_map<const sir::Value*, uint64_t> bound{
      {chain.a, MakeArenaRef(9000)}};
  std::unordered_map<const sir::Value*, uint64_t> refs;
  std::unordered_set<const sir::Value*> pinned;

  LinearScanTransients(chain.block, /*base=*/0, bound, pinned, refs);
  EXPECT_FALSE(refs.contains(chain.a));  // owned by the caller's binding
  EXPECT_TRUE(refs.contains(chain.b));
  EXPECT_TRUE(refs.contains(chain.c));
}

// =============================================================================
// Instruction lowering
// =============================================================================

struct MatmulFixture {
  sir::Block block;
  sir::Value* x;
  sir::Value* w;
  sir::Value* y;
  std::unordered_map<const sir::Value*, uint64_t> refs;

  MatmulFixture() {
    x = block.addArgument(sir::DataType::F32, sir::Shape{4, 6});
    sir::Operation* wop = block.appendOp("sc_mem.weight");
    w = wop->addResult("w", sir::DataType::F32, sir::Shape{6, 3});
    sir::Operation* mm = block.appendOp("sc_high.matmul");
    mm->addOperand(x);
    mm->addOperand(w);
    y = mm->addResult("y", sir::DataType::F32, sir::Shape{4, 3});
    refs = {{x, MakeArenaRef(0)},
            {w, MakeRodataRef(0)},
            {y, MakeArenaRef(128)}};
  }

  std::vector<sir::Operation*> Ops() {
    std::vector<sir::Operation*> ops;
    block.walk([&](sir::Operation* op) { ops.push_back(op); });
    return ops;
  }

  ResolveFn Resolve() {
    return [this](const sir::Value* v) -> std::expected<uint64_t, std::string> {
      return refs.at(v);
    };
  }
};

TEST(InstructionLowering, StorageOpsEmitNoCodeAndMatmulMapsToGemm) {
  MatmulFixture fx;
  ASSERT_OK_AND_ASSIGN(std::vector<UpdateInstruction> instrs,
                       LowerOps(fx.Ops(), fx.Resolve(), {}));
  // sc_mem.weight is a declaration; only the matmul lowers.
  ASSERT_EQ(instrs.size(), 1u);
  EXPECT_EQ(instrs[0].opcode, static_cast<uint16_t>(OpCode::kGemmNN));
  EXPECT_EQ(instrs[0].in[0], MakeArenaRef(0));
  EXPECT_EQ(instrs[0].in[1], MakeRodataRef(0));
  EXPECT_EQ(instrs[0].in[2], MakeArenaRef(128));
}

TEST(InstructionLowering, QuantizedWeightsLowerToQ8WithTheirScale) {
  MatmulFixture fx;
  const std::unordered_map<const sir::Value*, float> quant{{fx.w, 0.5f}};
  ASSERT_OK_AND_ASSIGN(std::vector<UpdateInstruction> instrs,
                       LowerOps(fx.Ops(), fx.Resolve(), quant));
  ASSERT_EQ(instrs.size(), 1u);
  EXPECT_EQ(instrs[0].opcode, static_cast<uint16_t>(OpCode::kGemmNNQ8));
  // The dequant scale rides in in[3] as f32 bits.
  EXPECT_EQ(instrs[0].in[3],
            static_cast<uint64_t>(std::bit_cast<uint32_t>(0.5f)));
}

TEST(InstructionLowering, RejectsOpsWithoutAnEncoding) {
  sir::Block block;
  sir::Value* x = block.addArgument(sir::DataType::F32, sir::Shape{4});
  sir::Operation* op = block.appendOp("sc_high.tanh");
  op->addOperand(x);
  sir::Value* y = op->addResult("y", sir::DataType::F32, sir::Shape{4});

  std::unordered_map<const sir::Value*, uint64_t> refs{
      {x, MakeArenaRef(0)}, {y, MakeArenaRef(64)}};
  auto resolve =
      [&](const sir::Value* v) -> std::expected<uint64_t, std::string> {
    return refs.at(v);
  };
  const auto r = LowerOps({op}, resolve, {});
  ASSERT_FALSE(r.has_value());
  EXPECT_STR_CONTAINS(r.error(), "InstructionLowering: cannot lower");
}

}  // namespace
