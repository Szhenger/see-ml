// =============================================================================
// ForwardBuilder tests: SMF -> forward SIR import, semantic analysis
// (rank / inner-dimension / bias-width checks), weight caching, and the
// teacher-prefix namespacing used by distillation.
// =============================================================================

#include <cstdint>
#include <string>

#include "compiler/frontend/forward_builder.h"
#include "compiler/frontend/sir.h"
#include "source/smf.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"

namespace {

using namespace seeml::update;
namespace sir = seecpp::sir;
using seeml::testing::AsBytes;
using seeml::testing::MakeMlp;
using seeml::testing::MakeTiedMlp;

constexpr int64_t kBatch = 4;

sir::Value* AddInput(sir::Block& block, int64_t in_dim) {
  return block.addArgument(sir::DataType::F32, sir::Shape{kBatch, in_dim});
}

size_t CountOps(sir::Block& block, std::string_view mnemonic) {
  size_t n = 0;
  block.walk([&](sir::Operation* op) {
    if (op->mnemonic() == mnemonic) ++n;
  });
  return n;
}

TEST(ForwardBuilder, BuildsMlpGraph) {
  SmfModel model = MakeMlp(6, 10, 3, 1);
  sir::Block block;
  GraphBuild build;
  build.input = AddInput(block, 6);

  ASSERT_OK_AND_ASSIGN(sir::Value * out,
                       BuildForward(block, model, "", build.input, kBatch,
                                    build));
  EXPECT_TRUE(out->shape() == sir::Shape({kBatch, 3}));
  EXPECT_EQ(out->id(), "logits");
  EXPECT_TRUE(block.validate());

  // 5 SMF ops import 1:1 plus one sc_mem.weight per constant tensor.
  EXPECT_EQ(CountOps(block, "sc_high.matmul"), 2u);
  EXPECT_EQ(CountOps(block, "sc_high.add_bias"), 2u);
  EXPECT_EQ(CountOps(block, "sc_high.relu"), 1u);
  EXPECT_EQ(CountOps(block, "sc_mem.weight"), 4u);
  EXPECT_EQ(build.weight_sources.size(), 4u);

  // Every materialized weight tracks its SMF tensor for rodata packing.
  for (const auto& [value, tensor] : build.weight_sources) {
    EXPECT_TRUE(tensor->is_const);
    EXPECT_EQ(value->id(), tensor->name);
  }
}

TEST(ForwardBuilder, CachesTiedWeights) {
  SmfModel model = MakeTiedMlp(4, 2);
  sir::Block block;
  GraphBuild build;
  build.input = AddInput(block, 4);

  ASSERT_OK(BuildForward(block, model, "", build.input, kBatch, build));
  // Both MatMuls reference "w"; it must materialize exactly once so the
  // compiler emits one adapter / one rodata copy / one emit entry.
  EXPECT_EQ(CountOps(block, "sc_mem.weight"), 1u);
  EXPECT_EQ(build.weight_sources.size(), 1u);
}

TEST(ForwardBuilder, PrefixNamespacesValueIds) {
  SmfModel model = MakeMlp(6, 10, 3, 3);
  sir::Block block;
  GraphBuild build;
  build.input = AddInput(block, 6);

  ASSERT_OK_AND_ASSIGN(sir::Value * out,
                       BuildForward(block, model, "t::", build.input, kBatch,
                                    build));
  EXPECT_EQ(out->id(), "t::logits");
  block.walk([&](sir::Operation* op) {
    if (op->mnemonic() == "sc_mem.weight")
      EXPECT_TRUE(op->result(0)->id().starts_with("t::"));
  });
}

TEST(ForwardBuilder, SharedInputSupportsStudentPlusTeacher) {
  SmfModel student = MakeMlp(6, 10, 3, 4);
  SmfModel teacher = MakeMlp(6, 14, 3, 5);
  sir::Block block;
  GraphBuild build;
  build.input = AddInput(block, 6);

  ASSERT_OK(BuildForward(block, student, "", build.input, kBatch, build));
  ASSERT_OK(BuildForward(block, teacher, "t::", build.input, kBatch, build));
  EXPECT_TRUE(block.validate());
  EXPECT_EQ(build.weight_sources.size(), 8u);
  EXPECT_EQ(build.input->users().size(), 2u);  // one MatMul per subgraph
}

TEST(ForwardBuilder, RejectsNonConstWeight) {
  SmfModel model;
  model.input_name = "x";
  model.output_name = "y";
  model.tensors.push_back({.name = "x", .dims = {-1, 4}, .is_const = false});
  // "w" is missing entirely.
  model.ops.push_back({SmfOpKind::kMatMul, "mm", {"x", "w"}, "y"});

  sir::Block block;
  GraphBuild build;
  build.input = AddInput(block, 4);
  EXPECT_ERROR_CONTAINS(BuildForward(block, model, "", build.input, kBatch,
                                     build),
                        "not a constant tensor");
}

TEST(ForwardBuilder, RejectsRank1MatMulOperand) {
  SmfModel model;
  model.input_name = "x";
  model.output_name = "y";
  model.tensors.push_back({.name = "x", .dims = {-1, 4}, .is_const = false});
  model.tensors.push_back({.name = "w",
                           .dims = {4},
                           .is_const = true,
                           .data = AsBytes({1, 2, 3, 4})});
  model.tensors.back().byte_size = model.tensors.back().data.size();
  model.ops.push_back({SmfOpKind::kMatMul, "mm", {"x", "w"}, "y"});

  sir::Block block;
  GraphBuild build;
  build.input = AddInput(block, 4);
  EXPECT_ERROR_CONTAINS(BuildForward(block, model, "", build.input, kBatch,
                                     build),
                        "rank-2");
}

TEST(ForwardBuilder, RejectsInnerDimensionMismatch) {
  SmfModel model;
  model.input_name = "x";
  model.output_name = "y";
  model.tensors.push_back({.name = "x", .dims = {-1, 4}, .is_const = false});
  model.tensors.push_back(
      {.name = "w",
       .dims = {3, 2},
       .is_const = true,
       .data = AsBytes({1, 2, 3, 4, 5, 6})});
  model.tensors.back().byte_size = model.tensors.back().data.size();
  model.ops.push_back({SmfOpKind::kMatMul, "mm", {"x", "w"}, "y"});

  sir::Block block;
  GraphBuild build;
  build.input = AddInput(block, 4);
  EXPECT_ERROR_CONTAINS(BuildForward(block, model, "", build.input, kBatch,
                                     build),
                        "inner dimensions disagree");
}

TEST(ForwardBuilder, RejectsBiasWidthMismatch) {
  SmfModel model;
  model.input_name = "x";
  model.output_name = "y";
  model.tensors.push_back({.name = "x", .dims = {-1, 4}, .is_const = false});
  model.tensors.push_back({.name = "b",
                           .dims = {3},  // input's last dim is 4
                           .is_const = true,
                           .data = AsBytes({1, 2, 3})});
  model.tensors.back().byte_size = model.tensors.back().data.size();
  model.ops.push_back({SmfOpKind::kAddBias, "ab", {"x", "b"}, "y"});

  sir::Block block;
  GraphBuild build;
  build.input = AddInput(block, 4);
  EXPECT_ERROR_CONTAINS(BuildForward(block, model, "", build.input, kBatch,
                                     build),
                        "bias width");
}

TEST(ForwardBuilder, RejectsWrongOperandCount) {
  SmfModel model;
  model.input_name = "x";
  model.output_name = "y";
  model.tensors.push_back({.name = "x", .dims = {-1, 4}, .is_const = false});
  model.ops.push_back({SmfOpKind::kMatMul, "mm", {"x"}, "y"});

  sir::Block block;
  GraphBuild build;
  build.input = AddInput(block, 4);
  EXPECT_ERROR_CONTAINS(BuildForward(block, model, "", build.input, kBatch,
                                     build),
                        "needs 2 inputs");
}

TEST(ForwardBuilder, RejectsMissingOutput) {
  SmfModel model;
  model.input_name = "x";
  model.output_name = "never_produced";
  model.tensors.push_back({.name = "x", .dims = {-1, 4}, .is_const = false});
  model.ops.push_back({SmfOpKind::kRelu, "r", {"x"}, "y"});

  sir::Block block;
  GraphBuild build;
  build.input = AddInput(block, 4);
  EXPECT_ERROR_CONTAINS(BuildForward(block, model, "", build.input, kBatch,
                                     build),
                        "was never produced");
}

}  // namespace
