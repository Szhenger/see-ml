// =============================================================================
// SIR-to-SIR pass tests: LoRA grafting (structure, filters, teacher
// exclusion), trainable-set reverse autodiff (pruning, fan-out accumulation,
// error paths), optimizer synthesis, and the merge-program builder.
// =============================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "compiler/frontend/forward_builder.h"
#include "compiler/frontend/sir.h"
#include "compiler/trainer/update_passes.h"
#include "source/smf.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"

namespace {

using namespace seeml::update;
namespace sir = seecpp::sir;
using seeml::testing::MakeMlp;

constexpr int64_t kBatch = 4;

/// Imports the standard test MLP (6 -> 10 -> 3) into `block`.
sir::Value* BuildMlpGraph(sir::Block& block, GraphBuild& build,
                          uint64_t seed = 1) {
  SmfModel model = MakeMlp(6, 10, 3, seed);
  build.input = block.addArgument(sir::DataType::F32, sir::Shape{kBatch, 6});
  auto out = BuildForward(block, model, "", build.input, kBatch, build);
  if (!out) return nullptr;
  return *out;
}

size_t CountOps(sir::Block& block, std::string_view mnemonic) {
  size_t n = 0;
  block.walk([&](sir::Operation* op) {
    if (op->mnemonic() == mnemonic) ++n;
  });
  return n;
}

LoRASpec Spec(int64_t rank = 4, float alpha = 8.0f, uint64_t seed = 7) {
  LoRASpec spec;
  spec.rank = rank;
  spec.alpha = alpha;
  spec.seed = seed;
  return spec;
}

// =============================================================================
// LoraGrafter
// =============================================================================

TEST(LoraGrafter, GraftsEveryEligibleMatMul) {
  sir::Block block;
  GraphBuild build;
  ASSERT_NE(BuildMlpGraph(block, build), nullptr);
  const size_t ops_before = block.numOps();

  ASSERT_OK_AND_ASSIGN(std::vector<GraftedAdapter> adapters,
                       LoraGrafter(Spec()).Run(block));
  ASSERT_EQ(adapters.size(), 2u);
  EXPECT_TRUE(block.validate());
  // 6 new ops per adapter: A, B, two matmuls, scale, add.
  EXPECT_EQ(block.numOps(), ops_before + 12);

  // Adapter 0 wraps w1 [6, 10]: A [6, r] randn-init, B [r, 10] zero-init.
  const GraftedAdapter& a0 = adapters[0];
  EXPECT_EQ(a0.frozen_weight->id(), "w1");
  EXPECT_TRUE(a0.A->shape() == sir::Shape({6, 4}));
  EXPECT_TRUE(a0.B->shape() == sir::Shape({4, 10}));
  EXPECT_NEAR(a0.scale, 8.0f / 4.0f, 1e-6);

  const sir::Operation* a_def = a0.A->definingOp();
  ASSERT_NE(a_def, nullptr);
  EXPECT_EQ(a_def->mnemonic(), "sc_mem.param");
  EXPECT_EQ(a_def->getAttrAs<int64_t>("trainable").value_or(0), 1);
  EXPECT_EQ(a_def->getAttrAs<std::string>("init").value_or(""), "randn");
  EXPECT_NEAR(a_def->getAttrAs<float>("std").value_or(0.0f),
              1.0f / std::sqrt(6.0f), 1e-6);

  const sir::Operation* b_def = a0.B->definingOp();
  ASSERT_NE(b_def, nullptr);
  EXPECT_EQ(b_def->getAttrAs<std::string>("init").value_or(""), "zeros");

  // Per-adapter seeds diverge so A matrices are not clones of each other.
  EXPECT_EQ(a_def->getAttrAs<int64_t>("seed").value_or(-1), 7);
  EXPECT_EQ(adapters[1].A->definingOp()->getAttrAs<int64_t>("seed").value_or(
                -1),
            8);
}

TEST(LoraGrafter, RewiresConsumersOntoAdapterOutput) {
  sir::Block block;
  GraphBuild build;
  ASSERT_NE(BuildMlpGraph(block, build), nullptr);
  ASSERT_OK(LoraGrafter(Spec()).Run(block));

  // Every add_bias must now consume a ".lora_out" value instead of the raw
  // MatMul result — the graft is transparent to downstream consumers.
  block.walk([&](sir::Operation* op) {
    if (op->mnemonic() != "sc_high.add_bias") return;
    EXPECT_TRUE(op->operand(0)->id().ends_with(".lora_out"));
  });
}

TEST(LoraGrafter, TargetFiltersRestrictGrafting) {
  sir::Block block;
  GraphBuild build;
  ASSERT_NE(BuildMlpGraph(block, build), nullptr);

  LoRASpec spec = Spec();
  spec.target_filters = {"w2"};
  ASSERT_OK_AND_ASSIGN(std::vector<GraftedAdapter> adapters,
                       LoraGrafter(spec).Run(block));
  ASSERT_EQ(adapters.size(), 1u);
  EXPECT_EQ(adapters[0].frozen_weight->id(), "w2");
}

TEST(LoraGrafter, SkipsTeacherSubgraph) {
  sir::Block block;
  GraphBuild build;
  ASSERT_NE(BuildMlpGraph(block, build), nullptr);
  SmfModel teacher = MakeMlp(6, 14, 3, 9);
  ASSERT_OK(BuildForward(block, teacher, "t::", build.input, kBatch, build));

  ASSERT_OK_AND_ASSIGN(std::vector<GraftedAdapter> adapters,
                       LoraGrafter(Spec()).Run(block));
  ASSERT_EQ(adapters.size(), 2u);  // student only
  for (const GraftedAdapter& a : adapters)
    EXPECT_FALSE(a.frozen_weight->id().starts_with("t::"));
}

TEST(LoraGrafter, RejectsNonPositiveRank) {
  sir::Block block;
  GraphBuild build;
  ASSERT_NE(BuildMlpGraph(block, build), nullptr);
  EXPECT_ERROR_CONTAINS(LoraGrafter(Spec(/*rank=*/0)).Run(block),
                        "rank must be positive");
}

TEST(LoraGrafter, RejectsTiedFrozenWeight) {
  sir::Block block;
  GraphBuild build;
  SmfModel model = seeml::testing::MakeTiedMlp(4, 1);
  build.input = block.addArgument(sir::DataType::F32, sir::Shape{kBatch, 4});
  ASSERT_OK(BuildForward(block, model, "", build.input, kBatch, build));

  // Both MatMuls multiply the same frozen weight Value; per-site adapters
  // cannot be merged back into the single on-disk copy.
  EXPECT_ERROR_CONTAINS(LoraGrafter(Spec()).Run(block), "weight tying");
}

TEST(LoraGrafter, RejectsWhenNoTargetMatches) {
  sir::Block block;
  GraphBuild build;
  ASSERT_NE(BuildMlpGraph(block, build), nullptr);
  LoRASpec spec = Spec();
  spec.target_filters = {"no_such_weight"};
  EXPECT_ERROR_CONTAINS(LoraGrafter(spec).Run(block), "no eligible");
}

// =============================================================================
// TrainableAutodiff
// =============================================================================

/// Builds:  y = x @ W;  loss = mse(y, target)  with trainable W [3, 2].
struct TinyGraph {
  sir::Block block;
  sir::Value* x = nullptr;
  sir::Value* target = nullptr;
  sir::Value* w = nullptr;
  sir::Value* y = nullptr;
  sir::Value* loss = nullptr;
};

void BuildTinyGraph(TinyGraph& g) {
  g.x = g.block.addArgument(sir::DataType::F32, sir::Shape{2, 3});
  g.target = g.block.addArgument(sir::DataType::F32, sir::Shape{2, 2});

  sir::Operation* w_op = g.block.appendOp("sc_mem.param");
  w_op->setAttribute("trainable", int64_t{1});
  g.w = w_op->addResult("W", sir::DataType::F32, sir::Shape{3, 2});

  sir::Operation* mm = g.block.appendOp("sc_high.matmul");
  mm->addOperand(g.x);
  mm->addOperand(g.w);
  g.y = mm->addResult("y", sir::DataType::F32, sir::Shape{2, 2});

  sir::Operation* mse = g.block.appendOp("sc_high.mse");
  mse->addOperand(g.y);
  mse->addOperand(g.target);
  g.loss = mse->addResult("loss", sir::DataType::F32, sir::Shape{});
}

TEST(TrainableAutodiff, SynthesizesGradientForTrainable) {
  TinyGraph g;
  BuildTinyGraph(g);

  TrainableAutodiff ad;
  ASSERT_OK_AND_ASSIGN(auto grads, ad.Run(g.block, g.loss, {g.w}));
  ASSERT_EQ(grads.size(), 1u);
  ASSERT_TRUE(grads.contains(g.w));
  EXPECT_TRUE(grads[g.w]->shape() == g.w->shape());
  EXPECT_TRUE(g.block.validate());

  // dW = X^T @ dY lowers through the transposed-GEMM adjoint...
  EXPECT_EQ(CountOps(g.block, "sc_low.matmul_tn"), 1u);
  // ...but x is frozen, so no dX (matmul_nt) is ever synthesized.
  EXPECT_EQ(CountOps(g.block, "sc_low.matmul_nt"), 0u);
  // The dL/dL = 1 seed is materialized exactly once.
  EXPECT_EQ(CountOps(g.block, "sc_low.fill"), 1u);
}

TEST(TrainableAutodiff, AccumulatesFanOutGradients) {
  // c = x @ W;  d = c + c;  loss = mse(d, target): the two adjoint paths
  // into c must be summed (multivariable chain rule).
  sir::Block block;
  sir::Value* x = block.addArgument(sir::DataType::F32, sir::Shape{2, 3});
  sir::Value* target =
      block.addArgument(sir::DataType::F32, sir::Shape{2, 2});

  sir::Operation* w_op = block.appendOp("sc_mem.param");
  w_op->setAttribute("trainable", int64_t{1});
  sir::Value* w = w_op->addResult("W", sir::DataType::F32, sir::Shape{3, 2});

  sir::Operation* mm = block.appendOp("sc_high.matmul");
  mm->addOperand(x);
  mm->addOperand(w);
  sir::Value* c = mm->addResult("c", sir::DataType::F32, sir::Shape{2, 2});

  sir::Operation* add = block.appendOp("sc_high.add");
  add->addOperand(c);
  add->addOperand(c);
  sir::Value* d = add->addResult("d", sir::DataType::F32, sir::Shape{2, 2});

  sir::Operation* mse = block.appendOp("sc_high.mse");
  mse->addOperand(d);
  mse->addOperand(target);
  sir::Value* loss = mse->addResult("loss", sir::DataType::F32, sir::Shape{});

  TrainableAutodiff ad;
  ASSERT_OK_AND_ASSIGN(auto grads, ad.Run(block, loss, {w}));
  ASSERT_TRUE(grads.contains(w));
  EXPECT_TRUE(block.validate());

  bool has_accumulation = false;
  block.walk([&](sir::Operation* op) {
    for (const auto& res : op->results())
      if (res->id().find(".grad_acc") != std::string_view::npos)
        has_accumulation = true;
  });
  EXPECT_TRUE(has_accumulation);
}

TEST(TrainableAutodiff, RejectsEmptyTrainableSet) {
  TinyGraph g;
  BuildTinyGraph(g);
  TrainableAutodiff ad;
  EXPECT_ERROR_CONTAINS(ad.Run(g.block, g.loss, {}), "empty trainable set");
}

TEST(TrainableAutodiff, RejectsNullLoss) {
  TinyGraph g;
  BuildTinyGraph(g);
  TrainableAutodiff ad;
  EXPECT_ERROR_CONTAINS(ad.Run(g.block, nullptr, {g.w}), "null loss");
}

TEST(TrainableAutodiff, RejectsLossDisconnectedFromTrainables) {
  // The loss is computed purely from block arguments; W never feeds it.
  sir::Block block;
  sir::Value* x = block.addArgument(sir::DataType::F32, sir::Shape{2, 2});
  sir::Value* target =
      block.addArgument(sir::DataType::F32, sir::Shape{2, 2});

  sir::Operation* w_op = block.appendOp("sc_mem.param");
  w_op->setAttribute("trainable", int64_t{1});
  sir::Value* w = w_op->addResult("W", sir::DataType::F32, sir::Shape{3, 2});

  sir::Operation* mse = block.appendOp("sc_high.mse");
  mse->addOperand(x);
  mse->addOperand(target);
  sir::Value* loss = mse->addResult("loss", sir::DataType::F32, sir::Shape{});

  TrainableAutodiff ad;
  EXPECT_ERROR_CONTAINS(ad.Run(block, loss, {w}),
                        "does not depend on any trainable");
}

TEST(TrainableAutodiff, RejectsOpWithoutVjpRule) {
  sir::Block block;
  sir::Value* target =
      block.addArgument(sir::DataType::F32, sir::Shape{2, 2});

  sir::Operation* w_op = block.appendOp("sc_mem.param");
  w_op->setAttribute("trainable", int64_t{1});
  sir::Value* w = w_op->addResult("W", sir::DataType::F32, sir::Shape{2, 2});

  sir::Operation* mystery = block.appendOp("sc_high.mystery");
  mystery->addOperand(w);
  sir::Value* y =
      mystery->addResult("y", sir::DataType::F32, sir::Shape{2, 2});

  sir::Operation* mse = block.appendOp("sc_high.mse");
  mse->addOperand(y);
  mse->addOperand(target);
  sir::Value* loss = mse->addResult("loss", sir::DataType::F32, sir::Shape{});

  TrainableAutodiff ad;
  EXPECT_ERROR_CONTAINS(ad.Run(block, loss, {w}), "no VJP rule");
}

// =============================================================================
// OptimizerSynthesizer
// =============================================================================

/// Appends a fake parameter + gradient pair to `block`.
std::pair<sir::Value*, sir::Value*> AddParamAndGrad(sir::Block& block,
                                                    const std::string& id) {
  sir::Operation* p_op = block.appendOp("sc_mem.param");
  p_op->setAttribute("trainable", int64_t{1});
  sir::Value* p = p_op->addResult(id, sir::DataType::F32, sir::Shape{4});
  sir::Operation* g_op = block.appendOp("sc_low.fill");
  g_op->setAttribute("value", 0.0f);
  sir::Value* g =
      g_op->addResult(id + ".d", sir::DataType::F32, sir::Shape{4});
  return {p, g};
}

TEST(OptimizerSynthesizer, AdamWDeclaresMomentStatePerParam) {
  sir::Block block;
  auto [p1, g1] = AddParamAndGrad(block, "a");
  auto [p2, g2] = AddParamAndGrad(block, "b");
  const size_t params_before = CountOps(block, "sc_mem.param");

  OptimizerSpec spec;
  spec.kind = OptimizerKind::kAdamW;
  ASSERT_OK(OptimizerSynthesizer(spec).Run(block, {{p1, g1}, {p2, g2}}));

  // Two zero-initialized moment tensors per parameter, plus the fused step.
  EXPECT_EQ(CountOps(block, "sc_mem.param"), params_before + 4);
  EXPECT_EQ(CountOps(block, "sc_low.adamw_step"), 2u);
  EXPECT_TRUE(block.validate());

  // Emission is sorted by parameter id: "a" state precedes "b" state.
  std::vector<std::string> step_params;
  block.walk([&](sir::Operation* op) {
    if (op->mnemonic() == "sc_low.adamw_step")
      step_params.emplace_back(op->operand(0)->id());
  });
  ASSERT_EQ(step_params.size(), 2u);
  EXPECT_EQ(step_params[0], "a");
  EXPECT_EQ(step_params[1], "b");

  bool found_moment = false;
  block.walk([&](sir::Operation* op) {
    if (op->mnemonic() != "sc_mem.param") return;
    if (op->result(0)->id() == "a.adam_m") {
      found_moment = true;
      EXPECT_EQ(op->getAttrAs<int64_t>("trainable").value_or(-1), 0);
      EXPECT_EQ(op->getAttrAs<std::string>("init").value_or(""), "zeros");
    }
  });
  EXPECT_TRUE(found_moment);
}

TEST(OptimizerSynthesizer, SgdAddsNoState) {
  sir::Block block;
  auto [p, g] = AddParamAndGrad(block, "a");
  const size_t params_before = CountOps(block, "sc_mem.param");

  OptimizerSpec spec;
  spec.kind = OptimizerKind::kSgd;
  ASSERT_OK(OptimizerSynthesizer(spec).Run(block, {{p, g}}));

  EXPECT_EQ(CountOps(block, "sc_mem.param"), params_before);
  EXPECT_EQ(CountOps(block, "sc_low.sgd_step"), 1u);
  EXPECT_EQ(CountOps(block, "sc_low.adamw_step"), 0u);
}

// =============================================================================
// MergeBuilder
// =============================================================================

TEST(MergeBuilder, BuildsCopyPlusGemmAccPerAdapter) {
  sir::Block train_block;
  GraphBuild build;
  ASSERT_NE(BuildMlpGraph(train_block, build), nullptr);
  ASSERT_OK_AND_ASSIGN(std::vector<GraftedAdapter> adapters,
                       LoraGrafter(Spec()).Run(train_block));

  ASSERT_OK_AND_ASSIGN(MergeProgram program, MergeBuilder().Run(adapters));
  ASSERT_NE(program.block, nullptr);
  EXPECT_TRUE(program.block->validate());
  ASSERT_EQ(program.outputs.size(), adapters.size());
  // Three storage mirrors (W, A, B) per adapter.
  EXPECT_EQ(program.aliases.size(), 3 * adapters.size());
  EXPECT_EQ(CountOps(*program.block, "sc_low.copy"), adapters.size());
  EXPECT_EQ(CountOps(*program.block, "sc_low.gemm_acc"), adapters.size());

  for (size_t i = 0; i < adapters.size(); ++i) {
    const auto& [merged, adapter] = program.outputs[i];
    EXPECT_EQ(adapter, &adapters[i]);
    // W' has W's shape; the gemm_acc folds with the adapter's alpha/r scale.
    EXPECT_TRUE(merged->shape() == adapters[i].frozen_weight->shape());
  }

  program.block->walk([&](sir::Operation* op) {
    if (op->mnemonic() == "sc_low.gemm_acc")
      EXPECT_NEAR(op->getAttrAs<float>("alpha").value_or(0.0f),
                  adapters[0].scale, 1e-6);
  });

  // Every alias mirror maps back to a value owned by the training block.
  for (const auto& [mirror, original] : program.aliases) {
    EXPECT_NE(mirror, original);
    EXPECT_TRUE(mirror->shape() == original->shape());
  }
}

TEST(MergeBuilder, RejectsEmptyAdapterSet) {
  EXPECT_ERROR_CONTAINS(MergeBuilder().Run({}), "no adapters");
}

}  // namespace
