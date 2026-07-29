// =============================================================================
// reviewer/ unit tests: int8 quantization selection — eligibility (weights
// consumed exclusively as the weight operand of matmul kernels), the
// per-tensor symmetric scale, the all-zero guard, and the deterministic
// chunked max-abs sweep on tensors larger than one sweep grain.
// =============================================================================

#include <cstdint>
#include <cstring>
#include <vector>

#include "compiler/analysis/reviewer/quantization.h"
#include "compiler/frontend/parser/graph_build.h"
#include "compiler/frontend/representation/sir.h"
#include "source/language/model_format.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"

namespace {

using namespace seeml::update;
namespace sir = seeml::sir;
using seeml::testing::AsBytes;

/// Declares one frozen weight in `block`, backed by `tensor` (whose data and
/// byte_size are filled from `data`), and registers it in `build`.
sir::Value* AddWeight(sir::Block& block, GraphBuild& build, SmfTensor& tensor,
                      const char* name, std::vector<int64_t> dims,
                      const std::vector<float>& data) {
  tensor.name = name;
  tensor.dims = dims;
  tensor.is_const = true;
  tensor.data = AsBytes(data);
  tensor.byte_size = tensor.data.size();
  sir::Operation* op = block.appendOp("sc_mem.weight");
  sir::Value* v =
      op->addResult(name, sir::DataType::F32, sir::Shape(std::move(dims)));
  build.weight_sources[v] = &tensor;
  return v;
}

sir::Value* AddMatmul(sir::Block& block, const char* out, sir::Value* x,
                      sir::Value* w) {
  sir::Operation* mm = block.appendOp("sc_high.matmul");
  mm->addOperand(x);
  mm->addOperand(w);
  return mm->addResult(out, sir::DataType::F32,
                       sir::Shape{x->shape().dims.at(0),
                                  w->shape().dims.at(1)});
}

TEST(Reviewer, SelectsMatmulOnlyWeightsAtMaxAbsScale) {
  sir::Block block;
  GraphBuild build;
  sir::Value* x = block.addArgument(sir::DataType::F32, sir::Shape{4, 2});
  SmfTensor tensor;
  sir::Value* w =
      AddWeight(block, build, tensor, "w", {2, 2}, {1.0f, -3.0f, 2.0f, 0.5f});
  AddMatmul(block, "y", x, w);

  const auto scales = SelectQuantizedWeights(block, build);
  ASSERT_EQ(scales.size(), 1u);
  ASSERT_TRUE(scales.contains(w));
  EXPECT_NEAR(scales.at(w), 3.0f / 127.0f, 1e-7);
}

TEST(Reviewer, RejectsWeightsConsumedAsActivations) {
  sir::Block block;
  GraphBuild build;
  sir::Value* x = block.addArgument(sir::DataType::F32, sir::Shape{2, 2});
  SmfTensor tensor;
  sir::Value* w =
      AddWeight(block, build, tensor, "w", {2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
  AddMatmul(block, "y", x, w);
  // The same weight also feeds an elementwise add — dequantizing on the fly
  // would be wrong there, so the review must exclude it entirely.
  sir::Operation* add = block.appendOp("sc_high.add");
  add->addOperand(w);
  add->addOperand(x);
  add->addResult("s", sir::DataType::F32, sir::Shape{2, 2});

  EXPECT_EQ(SelectQuantizedWeights(block, build).size(), 0u);
}

TEST(Reviewer, RejectsWeightsOnTheActivationSideOfMatmul) {
  sir::Block block;
  GraphBuild build;
  SmfTensor a_tensor, b_tensor;
  sir::Value* wa =
      AddWeight(block, build, a_tensor, "wa", {2, 2}, {1, 1, 1, 1});
  sir::Value* wb =
      AddWeight(block, build, b_tensor, "wb", {2, 2}, {2, 2, 2, 2});
  // wa is operand 0 (the activation side) — ineligible; wb stays eligible.
  AddMatmul(block, "y", wa, wb);

  const auto scales = SelectQuantizedWeights(block, build);
  EXPECT_FALSE(scales.contains(wa));
  EXPECT_TRUE(scales.contains(wb));
}

TEST(Reviewer, AllZeroWeightGetsUnitScale) {
  sir::Block block;
  GraphBuild build;
  sir::Value* x = block.addArgument(sir::DataType::F32, sir::Shape{2, 2});
  SmfTensor tensor;
  sir::Value* w =
      AddWeight(block, build, tensor, "w", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f});
  AddMatmul(block, "y", x, w);

  const auto scales = SelectQuantizedWeights(block, build);
  ASSERT_TRUE(scales.contains(w));
  EXPECT_NEAR(scales.at(w), 1.0f, 1e-9);
}

TEST(Reviewer, SweepFindsMaxBeyondTheFirstChunk) {
  // A tensor larger than kWeightSweepGrain with its extremum in the last
  // chunk: the parallel partial-max reduction must still find it.
  const size_t n = kWeightSweepGrain + 1024;
  std::vector<float> data(n, 0.25f);
  data[n - 1] = -8.0f;

  sir::Block block;
  GraphBuild build;
  sir::Value* x = block.addArgument(sir::DataType::F32, sir::Shape{1, 1});
  SmfTensor tensor;
  sir::Value* w = AddWeight(block, build, tensor, "w",
                            {1, static_cast<int64_t>(n)}, data);
  AddMatmul(block, "y", x, w);

  const auto scales = SelectQuantizedWeights(block, build);
  ASSERT_TRUE(scales.contains(w));
  EXPECT_NEAR(scales.at(w), 8.0f / 127.0f, 1e-7);
}

}  // namespace
