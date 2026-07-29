// =============================================================================
// operator/ unit tests: the OpBuilder factories — operand wiring (use
// registration), attribute recording, and per-dimension result-shape
// inference, including dynamic dimensions. Regression guards for the shape
// formulas the conv lowering and parser lean on.
// =============================================================================

#include <cstdint>
#include <memory>
#include <vector>

#include "compiler/frontend/operator/op_builder.h"
#include "compiler/frontend/representation/sir.h"
#include "test/framework/seetest.h"

namespace {

namespace sir = seeml::sir;
using sir::DataType;
using sir::Shape;

TEST(OpBuilder, Conv2dInfersSpatialGeometry) {
  sir::Block block;
  sir::Value* x = block.addArgument(DataType::F32, Shape{2, 3, 8, 8});
  sir::Value* w = block.addArgument(DataType::F32, Shape{4, 3, 3, 3});
  sir::Value* b = block.addArgument(DataType::F32, Shape{4});

  auto conv = sir::OpBuilder::conv2d(x, w, b, /*strides=*/{2, 2},
                                     /*pads=*/{1, 1, 1, 1});
  ASSERT_NE(conv.get(), nullptr);
  EXPECT_EQ(conv->mnemonic(), "sc_high.conv2d");
  ASSERT_EQ(conv->numOperands(), 3u);

  // OH = (8 + 1 + 1 - 3) / 2 + 1 = 4, same for OW.
  const std::vector<int64_t> expect{2, 4, 4, 4};
  EXPECT_TRUE(conv->result(0)->shape().dims == expect);

  // The geometry attributes the lowering reads back.
  EXPECT_TRUE(conv->getAttrAs<std::vector<int64_t>>("strides").has_value());
  EXPECT_TRUE(conv->getAttrAs<std::vector<int64_t>>("pads").has_value());
  EXPECT_EQ(conv->getAttrAs<int64_t>("group").value_or(-1), 1);

  block.appendOp(std::move(conv));
  EXPECT_OK(block.verify());
}

TEST(OpBuilder, Conv2dPropagatesDynamicBatch) {
  sir::Block block;
  sir::Value* x = block.addArgument(
      DataType::F32, Shape{sir::Shape::kDynamic, 3, 8, 8});
  sir::Value* w = block.addArgument(DataType::F32, Shape{4, 3, 3, 3});

  auto conv = sir::OpBuilder::conv2d(x, w, /*bias=*/nullptr, {1, 1});
  ASSERT_NE(conv.get(), nullptr);
  EXPECT_EQ(conv->numOperands(), 2u);  // biasless form
  EXPECT_EQ(conv->result(0)->shape().dims.at(0), sir::Shape::kDynamic);
  EXPECT_EQ(conv->result(0)->shape().dims.at(1), 4);
  block.appendOp(std::move(conv));
}

TEST(OpBuilder, Im2colInfersPatchMatrix) {
  sir::Block block;
  sir::Value* x = block.addArgument(DataType::F32, Shape{2, 3, 8, 8});

  auto cols = sir::OpBuilder::im2col(x, /*kernel_shape=*/{3, 3},
                                     /*strides=*/{2, 2},
                                     /*pads=*/{1, 1, 1, 1});
  ASSERT_NE(cols.get(), nullptr);
  // [N * OH * OW, C * KH * KW] = [2 * 4 * 4, 3 * 3 * 3].
  const std::vector<int64_t> expect{32, 27};
  EXPECT_TRUE(cols->result(0)->shape().dims == expect);
  block.appendOp(std::move(cols));
}

TEST(OpBuilder, GemmHonorsTransposeFlags) {
  sir::Block block;
  sir::Value* a = block.addArgument(DataType::F32, Shape{4, 6});
  sir::Value* b = block.addArgument(DataType::F32, Shape{6, 3});
  sir::Value* at = block.addArgument(DataType::F32, Shape{6, 4});
  sir::Value* bt = block.addArgument(DataType::F32, Shape{3, 6});
  sir::Value* bias = block.addArgument(DataType::F32, Shape{3});

  const std::vector<int64_t> expect{4, 3};
  auto nn = sir::OpBuilder::gemm(a, b);
  EXPECT_TRUE(nn->result(0)->shape().dims == expect);
  EXPECT_EQ(nn->numOperands(), 2u);

  auto tn = sir::OpBuilder::gemm(at, b, nullptr, /*trans_a=*/true);
  EXPECT_TRUE(tn->result(0)->shape().dims == expect);

  auto nt = sir::OpBuilder::gemm(a, bt, nullptr, /*trans_a=*/false,
                                 /*trans_b=*/true);
  EXPECT_TRUE(nt->result(0)->shape().dims == expect);

  auto biased = sir::OpBuilder::gemm(a, b, bias);
  EXPECT_EQ(biased->numOperands(), 3u);

  block.appendOp(std::move(nn));
  block.appendOp(std::move(tn));
  block.appendOp(std::move(nt));
  block.appendOp(std::move(biased));
  EXPECT_OK(block.verify());
}

TEST(OpBuilder, ReluPreservesShapeAndRegistersUse) {
  sir::Block block;
  sir::Value* x = block.addArgument(DataType::F32, Shape{4, 6});
  auto relu = sir::OpBuilder::relu(x);
  EXPECT_TRUE(relu->result(0)->shape().dims == x->shape().dims);
  // The factory wires the operand, so the use-list already sees the op.
  bool found = false;
  for (const sir::Operation* user : x->users())
    if (user == relu.get()) found = true;
  EXPECT_TRUE(found);
  block.appendOp(std::move(relu));
}

TEST(OpBuilder, BatchNormPreservesShapeAndRecordsEpsilon) {
  sir::Block block;
  sir::Value* x = block.addArgument(DataType::F32, Shape{2, 3, 8, 8});
  sir::Value* scale = block.addArgument(DataType::F32, Shape{3});
  sir::Value* bias = block.addArgument(DataType::F32, Shape{3});
  sir::Value* mean = block.addArgument(DataType::F32, Shape{3});
  sir::Value* var = block.addArgument(DataType::F32, Shape{3});

  auto bn = sir::OpBuilder::batchNorm(x, scale, bias, mean, var, 1e-3f);
  ASSERT_EQ(bn->numOperands(), 5u);
  EXPECT_TRUE(bn->result(0)->shape().dims == x->shape().dims);
  EXPECT_NEAR(bn->getAttrAs<float>("epsilon").value_or(0.0f), 1e-3f, 1e-9);
  block.appendOp(std::move(bn));
}

}  // namespace
