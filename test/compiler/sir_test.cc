// =============================================================================
// SIR core tests: shapes, SSA use-def bookkeeping, block surgery (the graph
// rewriting primitives the LoRA grafter depends on), validation, and the
// OpBuilder helpers.
// =============================================================================

#include <memory>
#include <string>
#include <vector>

#include "compiler/frontend/op_builder.h"
#include "compiler/frontend/sir.h"
#include "test/framework/seetest.h"

namespace {

using namespace seeml::sir;

TEST(Shape, VolumeAndStaticness) {
  EXPECT_EQ(Shape({2, 3, 4}).volume(), 24);
  EXPECT_TRUE(Shape({2, 3}).isFullyStatic());
  EXPECT_EQ(Shape({2, Shape::kDynamic}).volume(), Shape::kDynamic);
  EXPECT_FALSE(Shape({2, Shape::kDynamic}).isFullyStatic());

  const Shape scalar = Shape::scalar();
  EXPECT_TRUE(scalar.isScalar());
  EXPECT_EQ(scalar.rank(), 0);
  EXPECT_EQ(scalar.volume(), 1);
}

TEST(Shape, ByteSize) {
  EXPECT_EQ(Shape({2, 3}).byteSize(DataType::F32), 24u);
  EXPECT_EQ(Shape({2, 3}).byteSize(DataType::F64), 48u);
  EXPECT_EQ(Shape({2, 3}).byteSize(DataType::I8), 6u);
  // Dynamic shapes have no meaningful byte size.
  EXPECT_EQ(Shape({Shape::kDynamic, 3}).byteSize(DataType::F32), 0u);
}

TEST(DataTypes, WidthsAndNames) {
  EXPECT_EQ(dtypeByteWidth(DataType::F32), 4u);
  EXPECT_EQ(dtypeByteWidth(DataType::F16), 2u);
  EXPECT_EQ(dtypeByteWidth(DataType::I64), 8u);
  EXPECT_EQ(dtypeByteWidth(DataType::Bool), 1u);
  EXPECT_EQ(dtypeName(DataType::F32), "f32");
  EXPECT_EQ(dtypeName(DataType::BF16), "bf16");
}

TEST(Value, UseDefBookkeeping) {
  Block block;
  Value* x = block.addArgument(DataType::F32, Shape{2, 2});
  EXPECT_TRUE(x->isBlockArgument());
  EXPECT_TRUE(x->hasNoUses());

  Operation* op = block.appendOp("sc_high.relu");
  op->addOperand(x);
  Value* y = op->addResult("y", DataType::F32, Shape{2, 2});

  EXPECT_TRUE(x->hasOneUse());
  EXPECT_EQ(x->users()[0], op);
  EXPECT_EQ(y->definingOp(), op);
  EXPECT_FALSE(y->isBlockArgument());
  EXPECT_EQ(op->numOperands(), 1u);
  EXPECT_EQ(op->operand(0), x);
}

TEST(Value, SetOperandRewiresUserLists) {
  Block block;
  Value* a = block.addArgument(DataType::F32, Shape{4});
  Value* b = block.addArgument(DataType::F32, Shape{4});

  Operation* op = block.appendOp("sc_high.relu");
  op->addOperand(a);
  op->setOperand(0, b);

  EXPECT_TRUE(a->hasNoUses());
  EXPECT_TRUE(b->hasOneUse());
  EXPECT_EQ(op->operand(0), b);
}

TEST(Value, ReplaceAllUsesWith) {
  Block block;
  Value* old_v = block.addArgument(DataType::F32, Shape{4});
  Value* new_v = block.addArgument(DataType::F32, Shape{4});

  Operation* u1 = block.appendOp("sc_high.relu");
  u1->addOperand(old_v);
  Operation* u2 = block.appendOp("sc_high.add");
  u2->addOperand(old_v);
  u2->addOperand(old_v);  // multiple operand slots on one user

  old_v->replaceAllUsesWith(new_v);

  EXPECT_TRUE(old_v->hasNoUses());
  EXPECT_EQ(new_v->users().size(), 3u);
  EXPECT_EQ(u1->operand(0), new_v);
  EXPECT_EQ(u2->operand(0), new_v);
  EXPECT_EQ(u2->operand(1), new_v);
}

TEST(Operation, Attributes) {
  Operation op("sc_high.gemm");
  op.setAttribute("alpha", 2.5f);
  op.setAttribute("name", std::string("w1"));
  op.setAttribute("dims", std::vector<int64_t>{2, 3});

  EXPECT_TRUE(op.hasAttribute("alpha"));
  EXPECT_FALSE(op.hasAttribute("beta"));
  EXPECT_EQ(op.getAttrAs<float>("alpha").value_or(0.0f), 2.5f);
  EXPECT_EQ(op.getAttrAs<std::string>("name").value_or(""), "w1");
  // Wrong-type access yields nullopt, not a crash.
  EXPECT_FALSE(op.getAttrAs<int64_t>("alpha").has_value());
  EXPECT_FALSE(op.getAttrAs<float>("missing").has_value());
}

TEST(Operation, MnemonicClassification) {
  EXPECT_TRUE(Operation("sc_high.matmul").isHighLevel());
  EXPECT_TRUE(Operation("sc_low.matmul_nt").isLowLevel());
  EXPECT_TRUE(Operation("sc_mem.param").isMemoryOp());
  EXPECT_TRUE(Operation("sc_ctrl.branch").isControlFlow());
  EXPECT_FALSE(Operation("sc_low.fill").isHighLevel());
}

TEST(Operation, ToStringShowsStructure) {
  Block block;
  Value* x = block.addArgument(DataType::F32, Shape{2, 3});
  Operation* op = block.appendOp("sc_high.relu");
  op->addOperand(x);
  op->addResult("y", DataType::F32, Shape{2, 3});
  op->setAttribute("alpha", 1.5f);

  const std::string s = op->toString();
  EXPECT_STR_CONTAINS(s, "sc_high.relu");
  EXPECT_STR_CONTAINS(s, "y : f32<2x3>");
  EXPECT_STR_CONTAINS(s, "alpha");
}

TEST(Block, InsertOpsAfterPreservesOrder) {
  Block block;
  Operation* first = block.appendOp("first");
  block.appendOp("last");

  std::vector<std::unique_ptr<Operation>> mids;
  mids.push_back(std::make_unique<Operation>("mid1"));
  mids.push_back(std::make_unique<Operation>("mid2"));
  block.insertOpsAfter(first, std::move(mids));

  std::vector<std::string> order;
  block.walk([&](Operation* op) { order.emplace_back(op->mnemonic()); });
  ASSERT_EQ(order.size(), 4u);
  EXPECT_EQ(order[0], "first");
  EXPECT_EQ(order[1], "mid1");
  EXPECT_EQ(order[2], "mid2");
  EXPECT_EQ(order[3], "last");
  // Inserted ops are owned by the block now.
  EXPECT_EQ(block.operations()[1]->parentBlock(), &block);
}

TEST(Block, RemoveOpDropsUserEntries) {
  Block block;
  Value* x = block.addArgument(DataType::F32, Shape{4});
  Operation* op = block.appendOp("sc_high.relu");
  op->addOperand(x);
  EXPECT_TRUE(x->hasOneUse());

  std::unique_ptr<Operation> removed = block.removeOp(op);
  EXPECT_TRUE(x->hasNoUses());
  EXPECT_EQ(block.numOps(), 0u);
  EXPECT_EQ(removed->parentBlock(), nullptr);
}

TEST(Block, ValidateAcceptsSsaOrder) {
  Block block;
  Value* x = block.addArgument(DataType::F32, Shape{4});
  Operation* a = block.appendOp("sc_high.relu");
  a->addOperand(x);
  Value* y = a->addResult("y", DataType::F32, Shape{4});
  Operation* b = block.appendOp("sc_high.relu");
  b->addOperand(y);
  b->addResult("z", DataType::F32, Shape{4});

  EXPECT_TRUE(block.validate());
}

TEST(Block, ValidateRejectsUseBeforeDef) {
  Block block;
  Operation* user = block.appendOp("sc_high.relu");
  Operation* def = block.appendOp("sc_high.relu");
  Value* v = def->addResult("v", DataType::F32, Shape{4});
  user->addOperand(v);  // user precedes def in the op list

  EXPECT_FALSE(block.validate());
}

TEST(Block, WalkOrders) {
  Block block;
  block.appendOp("a");
  block.appendOp("b");
  block.appendOp("c");

  std::string forward, backward;
  block.walk([&](Operation* op) { forward += op->mnemonic(); });
  block.walkReverse([&](Operation* op) { backward += op->mnemonic(); });
  EXPECT_EQ(forward, "abc");
  EXPECT_EQ(backward, "cba");
}

TEST(Region, EntryBlock) {
  Region region;
  Block* b0 = region.addBlock();
  region.addBlock();
  EXPECT_EQ(region.entryBlock(), b0);
  EXPECT_EQ(region.blocks().size(), 2u);
}

TEST(OpBuilder, GemmCarriesTransposeFlags) {
  Block block;
  Value* a = block.addArgument(DataType::F32, Shape{2, 3});
  Value* b = block.addArgument(DataType::F32, Shape{3, 4});

  std::unique_ptr<Operation> gemm =
      OpBuilder::gemm(a, b, nullptr, /*trans_a=*/false, /*trans_b=*/true);
  EXPECT_EQ(gemm->mnemonic(), "sc_high.gemm");
  EXPECT_EQ(gemm->numOperands(), 2u);
  EXPECT_EQ(gemm->getAttrAs<int64_t>("trans_a").value_or(-1), 0);
  EXPECT_EQ(gemm->getAttrAs<int64_t>("trans_b").value_or(-1), 1);
  EXPECT_EQ(gemm->numResults(), 1u);
}

TEST(OpBuilder, ReluPropagatesShape) {
  Block block;
  Value* x = block.addArgument(DataType::F32, Shape{2, 5});
  std::unique_ptr<Operation> relu = OpBuilder::relu(x);
  EXPECT_EQ(relu->mnemonic(), "sc_high.relu");
  EXPECT_TRUE(relu->result(0)->shape() == Shape({2, 5}));
  EXPECT_EQ(relu->result(0)->dtype(), DataType::F32);
}

TEST(OpBuilder, Conv2dCarriesGeometry) {
  Block block;
  Value* x = block.addArgument(DataType::F32, Shape{1, 3, 8, 8});
  Value* w = block.addArgument(DataType::F32, Shape{4, 3, 3, 3});
  std::unique_ptr<Operation> conv = OpBuilder::conv2d(x, w, nullptr, {2, 2});
  EXPECT_EQ(conv->mnemonic(), "sc_high.conv2d");
  EXPECT_EQ(conv->numOperands(), 2u);
  const auto strides = conv->getAttrAs<std::vector<int64_t>>("strides");
  ASSERT_TRUE(strides.has_value());
  EXPECT_TRUE(*strides == (std::vector<int64_t>{2, 2}));
}

TEST(OpBuilder, Conv2dInfersOutputShape) {
  // [1,3,8,8] * [4,3,3,3], stride 2, no pad: OH = OW = (8-3)/2 + 1 = 3.
  Block block;
  Value* x = block.addArgument(DataType::F32, Shape{1, 3, 8, 8});
  Value* w = block.addArgument(DataType::F32, Shape{4, 3, 3, 3});
  auto conv = OpBuilder::conv2d(x, w, nullptr, {2, 2});
  EXPECT_TRUE(conv->result(0)->shape() == Shape({1, 4, 3, 3}));

  // A dynamic batch propagates; static spatial math is unaffected.
  Value* xd = block.addArgument(DataType::F32,
                                Shape{Shape::kDynamic, 3, 8, 8});
  auto convd = OpBuilder::conv2d(xd, w, nullptr, {2, 2});
  EXPECT_TRUE(convd->result(0)->shape() ==
              Shape({Shape::kDynamic, 4, 3, 3}));
}

TEST(OpBuilder, GemmInfersResultShape) {
  Block block;
  Value* a = block.addArgument(DataType::F32, Shape{2, 3});
  Value* b = block.addArgument(DataType::F32, Shape{4, 3});
  auto gemm = OpBuilder::gemm(a, b, nullptr, /*trans_a=*/false,
                              /*trans_b=*/true);
  // op(B) = B^T is [3, 4]: result is [2, 4].
  EXPECT_TRUE(gemm->result(0)->shape() == Shape({2, 4}));
}

TEST(OpBuilder, Im2colInfersPatchMatrix) {
  Block block;
  Value* x = block.addArgument(DataType::F32, Shape{1, 3, 8, 8});
  auto op = OpBuilder::im2col(x, {3, 3}, {2, 2}, {0, 0, 0, 0});
  // 3x3 output positions, 3*3*3-element patches: [1*3*3, 3*3*3].
  EXPECT_TRUE(op->result(0)->shape() == Shape({9, 27}));
}

TEST(Shape, VolumeOverflowSaturatesToDynamic) {
  EXPECT_EQ(Shape({INT64_MAX / 2, 3}).volume(), Shape::kDynamic);
  EXPECT_EQ(Shape({INT64_MAX / 2, 3}).byteSize(DataType::F32), 0u);
  EXPECT_EQ(Shape({-7, 4}).volume(), Shape::kDynamic);  // invalid negative
  EXPECT_EQ(Shape({4, 0, 9}).volume(), 0);              // empty is exact
}

TEST(Block, VerifyExplainsUseBeforeDef) {
  Block block;
  Operation* user = block.appendOp("sc_high.relu");
  Operation* def = block.appendOp("sc_high.relu");
  Value* v = def->addResult("v", DataType::F32, Shape{4});
  user->addOperand(v);

  auto verdict = block.verify();
  ASSERT_FALSE(verdict.has_value());
  EXPECT_TRUE(verdict.error().find("before its definition") !=
              std::string::npos);
}

TEST(Block, VerifyRejectsDuplicateValueIds) {
  Block block;
  Value* x = block.addArgument(DataType::F32, Shape{4});
  Operation* a = block.appendOp("sc_high.relu");
  a->addOperand(x);
  a->addResult("y", DataType::F32, Shape{4});
  Operation* b = block.appendOp("sc_high.relu");
  b->addOperand(x);
  b->addResult("y", DataType::F32, Shape{4});

  auto verdict = block.verify();
  ASSERT_FALSE(verdict.has_value());
  EXPECT_TRUE(verdict.error().find("duplicate value id") !=
              std::string::npos);
}

TEST(Block, VerifyDetectsUseListDrift) {
  // An op that references a block value without living in the block leaves
  // the value's use-list pointing at an operation verify cannot account
  // for — exactly the drift a buggy rewrite would introduce.
  Block block;
  Operation* def = block.appendOp("sc_high.relu");
  Value* v = def->addResult("v", DataType::F32, Shape{4});
  EXPECT_TRUE(block.validate());

  Operation stray("sc_high.relu");
  stray.addOperand(v);

  auto verdict = block.verify();
  ASSERT_FALSE(verdict.has_value());
  EXPECT_TRUE(verdict.error().find("disagrees") != std::string::npos);
}

}  // namespace
