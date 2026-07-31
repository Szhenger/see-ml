// =============================================================================
// updater/ tests: PassManager (ordering, the per-pass Block::verify gate)
// and ConvLowering (conv2d -> im2col-GEMM rewrite, bias handling, geometry
// restrictions, no-op on convolution-free blocks).
// =============================================================================

#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "compiler/analysis/update_passes.h"
#include "compiler/frontend/operator/op_builder.h"
#include "compiler/frontend/representation/sir.h"
#include "test/framework/seetest.h"

namespace {

using namespace seeml::update;
namespace sir = seeml::sir;

size_t CountOps(sir::Block& block, std::string_view mnemonic) {
  size_t n = 0;
  block.walk([&](sir::Operation* op) {
    if (op->mnemonic() == mnemonic) ++n;
  });
  return n;
}

/// Appends a conv2d over a fresh NCHW input to `block` and returns the conv
/// op. Input [2, 3, 8, 8], filter [4, 3, 3, 3], stride 1, pad 1 -> output
/// [2, 4, 8, 8].
sir::Operation* BuildConvGraph(sir::Block& block, bool with_bias) {
  sir::Value* x =
      block.addArgument(sir::DataType::F32, sir::Shape{2, 3, 8, 8});
  sir::Operation* w_op = block.appendOp("sc_mem.weight");
  sir::Value* f = w_op->addResult("conv.w", sir::DataType::F32,
                                  sir::Shape{4, 3, 3, 3});
  sir::Value* bias = nullptr;
  if (with_bias) {
    sir::Operation* b_op = block.appendOp("sc_mem.weight");
    bias = b_op->addResult("conv.b", sir::DataType::F32, sir::Shape{4});
  }
  auto conv = sir::OpBuilder::conv2d(x, f, bias, {1, 1}, {1, 1, 1, 1});
  return block.appendOp(std::move(conv));
}

// =============================================================================
// PassManager
// =============================================================================

TEST(PassManager, RunsPassesInRegistrationOrder) {
  sir::Block block;
  std::vector<std::string> ran;

  PassManager pm;
  pm.Add("first", [&](sir::Block&) -> std::expected<void, std::string> {
    ran.push_back("first");
    return {};
  });
  pm.Add("second", [&](sir::Block&) -> std::expected<void, std::string> {
    ran.push_back("second");
    return {};
  });

  ASSERT_TRUE(pm.Run(block).has_value());
  ASSERT_EQ(ran.size(), 2u);
  EXPECT_EQ(ran[0], "first");
  EXPECT_EQ(ran[1], "second");
}

TEST(PassManager, PassErrorsPropagateVerbatim) {
  sir::Block block;
  PassManager pm;
  pm.Add("doomed", [](sir::Block&) -> std::expected<void, std::string> {
    return std::unexpected("DoomedPass: reasons");
  });
  auto r = pm.Run(block);
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error(), "DoomedPass: reasons");
}

TEST(PassManager, NamesThePassThatViolatesInvariants) {
  sir::Block block;
  sir::Operation* fill = block.appendOp("sc_low.fill");
  fill->addResult("x", sir::DataType::F32, sir::Shape{2});

  PassManager pm;
  pm.Add("fine", [](sir::Block&) -> std::expected<void, std::string> {
    return {};
  });
  // Corrupts the block by duplicating an existing value id — the invariant
  // gate must attribute the violation to THIS pass, not fail later.
  pm.Add("corruptor", [](sir::Block& b) -> std::expected<void, std::string> {
    sir::Operation* dup = b.appendOp("sc_low.fill");
    dup->addResult("x", sir::DataType::F32, sir::Shape{2});
    return {};
  });

  auto r = pm.Run(block);
  ASSERT_FALSE(r.has_value());
  EXPECT_NE(r.error().find("after pass 'corruptor'"), std::string::npos);
  EXPECT_NE(r.error().find("duplicate value id"), std::string::npos);
}

// =============================================================================
// ConvLowering
// =============================================================================

TEST(DeadCodeElimination, SweepsDeadChainsKeepsRootsUsesAndEffects) {
  sir::Block block;
  sir::Value* x = block.addArgument(sir::DataType::F32, sir::Shape{2, 4});

  // Live: rooted result.
  sir::Operation* live = block.appendOp("sc_high.relu");
  live->addOperand(x);
  sir::Value* rooted =
      live->addResult("live.y", sir::DataType::F32, sir::Shape{2, 4});

  // Dead chain: a -> b, b unused and unrooted; the backward sweep must
  // remove b first and then the now-useless a in the same run.
  sir::Operation* a = block.appendOp("sc_high.gelu");
  a->addOperand(x);
  sir::Value* av =
      a->addResult("dead.a", sir::DataType::F32, sir::Shape{2, 4});
  sir::Operation* b = block.appendOp("sc_high.silu");
  b->addOperand(av);
  b->addResult("dead.b", sir::DataType::F32, sir::Shape{2, 4});

  // Effectful with an unused result: must survive on the effect alone.
  sir::Operation* step = block.appendOp("sc_low.sgd_step");
  step->addOperand(rooted);
  step->addResult("step.tick", sir::DataType::F32, sir::Shape{});

  // Storage declaration with no users: kept — the binder holds its name.
  sir::Operation* mem = block.appendOp("sc_mem.param");
  mem->addResult("orphan.param", sir::DataType::F32, sir::Shape{4});

  std::unordered_set<const sir::Value*> roots{rooted};
  ASSERT_OK_AND_ASSIGN(size_t removed,
                       DeadCodeElimination().Run(block, roots));
  EXPECT_EQ(removed, 2u);
  EXPECT_EQ(CountOps(block, "sc_high.gelu"), 0u);
  EXPECT_EQ(CountOps(block, "sc_high.silu"), 0u);
  EXPECT_EQ(CountOps(block, "sc_high.relu"), 1u);
  EXPECT_EQ(CountOps(block, "sc_low.sgd_step"), 1u);
  EXPECT_EQ(CountOps(block, "sc_mem.param"), 1u);
  ASSERT_OK(block.verify());

  // Idempotent: a second sweep finds nothing.
  ASSERT_OK_AND_ASSIGN(size_t again, DeadCodeElimination().Run(block, roots));
  EXPECT_EQ(again, 0u);
}

TEST(ConvLowering, LowersConvToIm2colGemm) {
  sir::Block block;
  sir::Operation* conv = BuildConvGraph(block, /*with_bias=*/true);
  sir::Value* y = conv->result(0);
  EXPECT_TRUE(y->shape() == sir::Shape({2, 4, 8, 8}));

  sir::Operation* relu = block.appendOp("sc_high.relu");
  relu->addOperand(y);
  relu->addResult("out", sir::DataType::F32, y->shape());

  ASSERT_TRUE(ConvLowering().Run(block).has_value());
  EXPECT_TRUE(block.validate());

  EXPECT_EQ(CountOps(block, "sc_high.conv2d"), 0u);
  EXPECT_EQ(CountOps(block, "sc_low.im2col"), 1u);
  EXPECT_EQ(CountOps(block, "sc_low.filter_matrix"), 1u);
  EXPECT_EQ(CountOps(block, "sc_high.matmul"), 1u);
  EXPECT_EQ(CountOps(block, "sc_high.add_bias"), 1u);
  EXPECT_EQ(CountOps(block, "sc_low.col2im"), 1u);

  // The consumer was rewired onto the col2im result, same NCHW geometry.
  sir::Value* replacement = relu->operand(0);
  ASSERT_NE(replacement->definingOp(), nullptr);
  EXPECT_EQ(replacement->definingOp()->mnemonic(), "sc_low.col2im");
  EXPECT_TRUE(replacement->shape() == sir::Shape({2, 4, 8, 8}));

  // GEMM geometry: cols [N*OH*OW, Cin*KH*KW] @ wmat [Cin*KH*KW, Cout].
  block.walk([&](sir::Operation* op) {
    if (op->mnemonic() == "sc_low.im2col")
      EXPECT_TRUE(op->result(0)->shape() == sir::Shape({128, 27}));
    if (op->mnemonic() == "sc_low.filter_matrix")
      EXPECT_TRUE(op->result(0)->shape() == sir::Shape({27, 4}));
    if (op->mnemonic() == "sc_high.matmul")
      EXPECT_TRUE(op->result(0)->shape() == sir::Shape({128, 4}));
  });
}

TEST(ConvLowering, BiaslessConvSkipsAddBias) {
  sir::Block block;
  BuildConvGraph(block, /*with_bias=*/false);

  ASSERT_TRUE(ConvLowering().Run(block).has_value());
  EXPECT_TRUE(block.validate());
  EXPECT_EQ(CountOps(block, "sc_high.conv2d"), 0u);
  EXPECT_EQ(CountOps(block, "sc_high.add_bias"), 0u);
  EXPECT_EQ(CountOps(block, "sc_low.col2im"), 1u);
}

TEST(ConvLowering, ConvFreeBlockIsUntouched) {
  sir::Block block;
  sir::Value* x = block.addArgument(sir::DataType::F32, sir::Shape{2, 4});
  sir::Operation* relu = block.appendOp("sc_high.relu");
  relu->addOperand(x);
  relu->addResult("out", sir::DataType::F32, x->shape());
  const size_t ops_before = block.numOps();

  ASSERT_TRUE(ConvLowering().Run(block).has_value());
  EXPECT_EQ(block.numOps(), ops_before);
}

TEST(ConvLowering, RejectsDilatedConvolutions) {
  sir::Block block;
  sir::Value* x =
      block.addArgument(sir::DataType::F32, sir::Shape{1, 3, 8, 8});
  sir::Operation* w_op = block.appendOp("sc_mem.weight");
  sir::Value* f = w_op->addResult("conv.w", sir::DataType::F32,
                                  sir::Shape{4, 3, 3, 3});
  auto conv =
      sir::OpBuilder::conv2d(x, f, nullptr, {1, 1}, {1, 1, 1, 1}, {2, 2});
  block.appendOp(std::move(conv));

  auto r = ConvLowering().Run(block);
  ASSERT_FALSE(r.has_value());
  EXPECT_NE(r.error().find("group/dilation"), std::string::npos);
}

}  // namespace
