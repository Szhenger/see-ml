// =============================================================================
// ForwardBuilder tests: SMF -> forward SIR import, semantic analysis
// (rank / inner-dimension / bias-width checks), weight caching, and the
// teacher-prefix namespacing used by distillation.
// =============================================================================

#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "compiler/frontend/parser/parser.h"
#include "compiler/frontend/representation/sir.h"
#include "source/language/model_format.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"

namespace {

using namespace seeml::update;
namespace sir = seeml::sir;
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

TEST(ForwardBuilder, RejectsNonPositiveBatch) {
  SmfModel model = MakeMlp(6, 10, 3, 1);
  sir::Block block;
  GraphBuild build;
  build.input = AddInput(block, 6);
  EXPECT_ERROR_CONTAINS(BuildForward(block, model, "", build.input, 0, build),
                        "batch must be at least 1");
}

TEST(ForwardBuilder, RejectsDuplicateOutputName) {
  SmfModel model;
  model.input_name = "x";
  model.output_name = "y";
  model.tensors.push_back({.name = "x", .dims = {-1, 4}, .is_const = false});
  model.ops.push_back({SmfOpKind::kRelu, "r1", {"x"}, "y"});
  model.ops.push_back({SmfOpKind::kRelu, "r2", {"x"}, "y"});

  sir::Block block;
  GraphBuild build;
  build.input = AddInput(block, 4);
  EXPECT_ERROR_CONTAINS(BuildForward(block, model, "", build.input, kBatch,
                                     build),
                        "redefines an existing value");
}

TEST(ForwardBuilder, RejectsOutputShadowingTensorName) {
  SmfModel model;
  model.input_name = "x";
  model.output_name = "w";
  model.tensors.push_back({.name = "x", .dims = {-1, 4}, .is_const = false});
  model.tensors.push_back({.name = "w", .dims = {4}, .is_const = true});
  model.ops.push_back({SmfOpKind::kRelu, "r1", {"x"}, "w"});

  sir::Block block;
  GraphBuild build;
  build.input = AddInput(block, 4);
  EXPECT_ERROR_CONTAINS(BuildForward(block, model, "", build.input, kBatch,
                                     build),
                        "redefines an existing value");
}

TEST(ForwardBuilder, RejectsUseBeforeProduction) {
  // "h" is produced by the second op but consumed by the first: the parser
  // must name the topological-order violation, not claim "h" is missing.
  SmfModel model;
  model.input_name = "x";
  model.output_name = "y";
  model.tensors.push_back({.name = "x", .dims = {-1, 4}, .is_const = false});
  model.ops.push_back({SmfOpKind::kRelu, "r1", {"h"}, "y"});
  model.ops.push_back({SmfOpKind::kRelu, "r2", {"x"}, "h"});

  sir::Block block;
  GraphBuild build;
  build.input = AddInput(block, 4);
  EXPECT_ERROR_CONTAINS(BuildForward(block, model, "", build.input, kBatch,
                                     build),
                        "before it is produced");
}

TEST(ForwardBuilder, RejectsWeightAsModelOutput) {
  SmfModel model;
  model.input_name = "x";
  model.output_name = "w";
  model.tensors.push_back({.name = "x", .dims = {-1, 4}, .is_const = false});
  model.tensors.push_back({.name = "w", .dims = {4}, .is_const = true});
  model.ops.push_back({SmfOpKind::kRelu, "r1", {"x"}, "y"});

  sir::Block block;
  GraphBuild build;
  build.input = AddInput(block, 4);
  EXPECT_ERROR_CONTAINS(BuildForward(block, model, "", build.input, kBatch,
                                     build),
                        "was never produced by an operation");
}

TEST(ForwardBuilder, RejectsInputAsModelOutput) {
  SmfModel model;
  model.input_name = "x";
  model.output_name = "x";
  model.tensors.push_back({.name = "x", .dims = {-1, 4}, .is_const = false});
  model.ops.push_back({SmfOpKind::kRelu, "r1", {"x"}, "y"});

  sir::Block block;
  GraphBuild build;
  build.input = AddInput(block, 4);
  EXPECT_ERROR_CONTAINS(BuildForward(block, model, "", build.input, kBatch,
                                     build),
                        "was never produced by an operation");
}

TEST(ForwardBuilder, ConcurrentDisjointBuildsAreSafe) {
  // The documented contract: one shared const model, but each thread owns its
  // (block, build, input). Threads only record; assertions run after the
  // join (the test framework's recorders are not thread-safe).
  SmfModel model = MakeMlp(6, 10, 3, 1);
  constexpr size_t kThreads = 4;
  bool built[kThreads] = {};
  bool valid[kThreads] = {};
  size_t op_count[kThreads] = {};

  std::vector<std::thread> builders;
  for (size_t t = 0; t < kThreads; ++t)
    builders.emplace_back([&, t] {
      sir::Block block;
      GraphBuild build;
      build.input = AddInput(block, 6);
      auto out = BuildForward(block, model, "", build.input, kBatch, build);
      built[t] = out.has_value();
      valid[t] = block.validate();
      op_count[t] = block.numOps();
    });
  for (std::thread& t : builders) t.join();

  for (size_t t = 0; t < kThreads; ++t) {
    EXPECT_TRUE(built[t]);
    EXPECT_TRUE(valid[t]);
    EXPECT_EQ(op_count[t], op_count[0]);
  }
  EXPECT_TRUE(op_count[0] > 0);
}

// --- Transformer vocabulary (SMF v3) -----------------------------------------

TEST(ForwardBuilder, BuildsTransformerDecoder) {
  // batch 8 = 2 sequences of seq_len 4; dim 8, 2 heads (head width 4, even).
  SmfModel model = seeml::testing::MakeTinyDecoder(8, 2, 4, 12, 3, 7);
  sir::Block block;
  GraphBuild build;
  build.input = block.addArgument(sir::DataType::F32, sir::Shape{8, 8});

  ASSERT_OK_AND_ASSIGN(sir::Value * out,
                       BuildForward(block, model, "", build.input, 8, build));
  EXPECT_TRUE(out->shape() == sir::Shape({8, 3}));
  EXPECT_TRUE(block.validate());
  EXPECT_EQ(CountOps(block, "sc_high.rms_norm"), 3u);
  EXPECT_EQ(CountOps(block, "sc_high.rope"), 2u);
  EXPECT_EQ(CountOps(block, "sc_high.attention"), 1u);
  EXPECT_EQ(CountOps(block, "sc_high.add"), 2u);

  // The attention op carries the probability cache as a second result:
  // [rows * heads, seq_len] = [16, 4].
  block.walk([&](sir::Operation* op) {
    if (op->mnemonic() != "sc_high.attention") return;
    EXPECT_EQ(op->numResults(), 2u);
    EXPECT_TRUE(op->result(1)->shape() == sir::Shape({16, 4}));
    EXPECT_EQ(op->getAttrAs<int64_t>("heads").value_or(0), 2);
    EXPECT_EQ(op->getAttrAs<int64_t>("seq").value_or(0), 4);
  });
}

TEST(ForwardBuilder, RejectsAttentionWithIndivisibleHeads) {
  SmfModel model = seeml::testing::MakeTinyDecoder(8, 2, 4, 12, 3, 7);
  for (auto& op : model.ops)
    if (op.kind == SmfOpKind::kAttention) op.attr0 = 3;  // 3 does not divide 8
  sir::Block block;
  GraphBuild build;
  build.input = block.addArgument(sir::DataType::F32, sir::Shape{8, 8});
  EXPECT_ERROR_CONTAINS(BuildForward(block, model, "", build.input, 8, build),
                        "not divisible by num_heads");
}

TEST(ForwardBuilder, RejectsSequenceOpsWithoutSeqLen) {
  SmfModel model = seeml::testing::MakeTinyDecoder(8, 2, 4, 12, 3, 7);
  model.seq_len = 0;  // the model forgot to declare its sequence geometry
  sir::Block block;
  GraphBuild build;
  build.input = block.addArgument(sir::DataType::F32, sir::Shape{8, 8});
  EXPECT_ERROR_CONTAINS(BuildForward(block, model, "", build.input, 8, build),
                        "declares no seq_len");
}

TEST(ForwardBuilder, BuildsTokenDecoderThroughEmbedding) {
  SmfModel model = seeml::testing::MakeTinyTokenDecoder(16, 8, 2, 4, 12, 7);
  sir::Block block;
  GraphBuild build;
  build.input = block.addArgument(sir::DataType::I32, sir::Shape{8});
  ASSERT_OK_AND_ASSIGN(sir::Value * out,
                       BuildForward(block, model, "", build.input, 8, build));
  EXPECT_TRUE(out->shape() == sir::Shape({8, 16}));  // [rows, vocab]
  EXPECT_TRUE(block.validate());
  EXPECT_EQ(CountOps(block, "sc_high.embedding"), 1u);
}

TEST(ForwardBuilder, RejectsFloatConsumerOfTokenInput) {
  SmfModel model = seeml::testing::MakeTinyTokenDecoder(16, 8, 2, 4, 12, 7);
  // A residual add reading the raw token input would treat ids as floats.
  for (auto& op : model.ops)
    if (op.name == "res1") op.inputs[0] = "x";
  sir::Block block;
  GraphBuild build;
  build.input = block.addArgument(sir::DataType::I32, sir::Shape{8});
  EXPECT_ERROR_CONTAINS(BuildForward(block, model, "", build.input, 8, build),
                        "only Embedding may read");
}

TEST(ForwardBuilder, RejectsEmbeddingWithoutTokenInput) {
  SmfModel model = seeml::testing::MakeTinyTokenDecoder(16, 8, 2, 4, 12, 7);
  model.tensors[0].dims = {-1, 8};  // input demoted back to feature rows
  sir::Block block;
  GraphBuild build;
  build.input = block.addArgument(sir::DataType::F32, sir::Shape{8, 8});
  EXPECT_ERROR_CONTAINS(BuildForward(block, model, "", build.input, 8, build),
                        "rank-1 dynamic");
}

TEST(ForwardBuilder, RejectsSeqLenPastInt64) {
  // Hostile u64 seq_len past INT64_MAX would wrap negative through the
  // int64 casts (rows % -1 == 0 accepts every batch) and reach the parser
  // as a negative probs dim; sema must refuse it in unsigned arithmetic.
  SmfModel model = seeml::testing::MakeTinyDecoder(8, 2, 4, 12, 3, 7);
  model.seq_len = ~0ULL;
  sir::Block block;
  GraphBuild build;
  build.input = block.addArgument(sir::DataType::F32, sir::Shape{8, 8});
  EXPECT_ERROR_CONTAINS(BuildForward(block, model, "", build.input, 8, build),
                        "does not fit a signed 64-bit");
}

TEST(ForwardBuilder, RejectsBatchNotWholeSequences) {
  SmfModel model = seeml::testing::MakeTinyDecoder(8, 2, 4, 12, 3, 7);
  sir::Block block;
  GraphBuild build;
  // 6 rows cannot be sequences of 4.
  build.input = block.addArgument(sir::DataType::F32, sir::Shape{6, 8});
  EXPECT_ERROR_CONTAINS(BuildForward(block, model, "", build.input, 6, build),
                        "whole number of sequences");
}

TEST(ForwardBuilder, RejectsRopeWithOddHeadWidth) {
  // dim 6 with 2 heads gives head width 3: attention would accept it, but
  // RoPE's rotation pairs cannot.
  SmfModel model;
  model.input_name = "x";
  model.output_name = "y";
  model.seq_len = 2;
  model.tensors.push_back({.name = "x", .dims = {-1, 6}, .is_const = false});
  model.ops.push_back({SmfOpKind::kRope, "r", {"x"}, "y", 2});
  sir::Block block;
  GraphBuild build;
  build.input = block.addArgument(sir::DataType::F32, sir::Shape{4, 6});
  EXPECT_ERROR_CONTAINS(BuildForward(block, model, "", build.input, 4, build),
                        "head width must be even");
}

}  // namespace
