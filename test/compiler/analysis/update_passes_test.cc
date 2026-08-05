// =============================================================================
// SIR-to-SIR pass tests: LoRA grafting (structure, filters, teacher
// exclusion), trainable-set reverse autodiff (pruning, fan-out accumulation,
// error paths), optimizer synthesis, and the merge-program builder.
// =============================================================================

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "compiler/frontend/parser/parser.h"
#include "compiler/frontend/representation/sir.h"
#include "compiler/analysis/update_passes.h"
#include "source/language/model_format.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"

namespace {

using namespace seeml::update;
namespace sir = seeml::sir;
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

TEST(MergeBuilder, BuildsZeroedDeltaPlusGemmAccPerAdapter) {
  sir::Block train_block;
  GraphBuild build;
  ASSERT_NE(BuildMlpGraph(train_block, build), nullptr);
  ASSERT_OK_AND_ASSIGN(std::vector<GraftedAdapter> adapters,
                       LoraGrafter(Spec()).Run(train_block));

  ASSERT_OK_AND_ASSIGN(MergeProgram program, MergeBuilder().Run(adapters));
  ASSERT_NE(program.block, nullptr);
  EXPECT_TRUE(program.block->validate());
  ASSERT_EQ(program.outputs.size(), adapters.size());
  // Two storage mirrors (A, B) per adapter: the frozen base never enters the
  // merge — commit adds each delta onto the model file's own weights.
  EXPECT_EQ(program.aliases.size(), 2 * adapters.size());
  EXPECT_EQ(CountOps(*program.block, "sc_low.fill"), adapters.size());
  EXPECT_EQ(CountOps(*program.block, "sc_low.gemm_acc"), adapters.size());

  for (size_t i = 0; i < adapters.size(); ++i) {
    const auto& [delta, adapter] = program.outputs[i];
    EXPECT_EQ(adapter, &adapters[i]);
    // Δ has W's shape; the gemm_acc folds with the adapter's alpha/r scale.
    EXPECT_TRUE(delta->shape() == adapters[i].frozen_weight->shape());
  }

  program.block->walk([&](sir::Operation* op) {
    if (op->mnemonic() == "sc_low.gemm_acc")
      EXPECT_NEAR(op->getAttrAs<float>("alpha").value_or(0.0f),
                  adapters[0].scale, 1e-6);
    // Each delta buffer starts from exactly zero.
    if (op->mnemonic() == "sc_low.fill")
      EXPECT_NEAR(op->getAttrAs<float>("value").value_or(1.0f), 0.0f, 0.0f);
  });

  // Every alias mirror maps back to a value owned by the training block.
  for (const auto& [mirror, original] : program.aliases) {
    EXPECT_NE(mirror, original);
    EXPECT_TRUE(mirror->shape() == original->shape());
  }
}

// =============================================================================
// GemmEpilogueFuser
// =============================================================================

/// One layer as the parser emits it: weights materialized at first use, so
/// the bias declaration sits AFTER the matmul (the SSA-hoist case).
struct Chain {
  sir::Block block;
  sir::Value* x = nullptr;
  sir::Value* c = nullptr;      // matmul result
  sir::Value* z = nullptr;      // chain output (act, or bias sum)
  sir::Operation* gemm = nullptr;
  sir::Operation* add_bias = nullptr;
  sir::Operation* act = nullptr;
  sir::Value* consumer_out = nullptr;  // scale op reading the chain output
};

void BuildChain(Chain& g, const char* act_mnemonic /* nullptr = bias only */) {
  g.x = g.block.addArgument(sir::DataType::F32, sir::Shape{2, 3});

  sir::Operation* w_op = g.block.appendOp("sc_mem.weight");
  sir::Value* w = w_op->addResult("w", sir::DataType::F32, sir::Shape{3, 2});

  g.gemm = g.block.appendOp("sc_high.matmul");
  g.gemm->addOperand(g.x);
  g.gemm->addOperand(w);
  g.c = g.gemm->addResult("c", sir::DataType::F32, sir::Shape{2, 2});

  sir::Operation* b_op = g.block.appendOp("sc_mem.weight");
  sir::Value* b = b_op->addResult("b", sir::DataType::F32, sir::Shape{2});

  g.add_bias = g.block.appendOp("sc_high.add_bias");
  g.add_bias->addOperand(g.c);
  g.add_bias->addOperand(b);
  sir::Value* y = g.add_bias->addResult("y", sir::DataType::F32,
                                        sir::Shape{2, 2});
  g.z = y;

  if (act_mnemonic) {
    g.act = g.block.appendOp(act_mnemonic);
    g.act->addOperand(y);
    g.z = g.act->addResult("z", sir::DataType::F32, sir::Shape{2, 2});
  }

  sir::Operation* consumer = g.block.appendOp("sc_high.scale");
  consumer->setAttribute("alpha", 1.0f);
  consumer->addOperand(g.z);
  g.consumer_out = consumer->addResult("out", sir::DataType::F32,
                                       sir::Shape{2, 2});
}

TEST(GemmEpilogueFuser, FusesBiasAndActivationChain) {
  Chain g;
  BuildChain(g, "sc_high.relu");
  ASSERT_OK_AND_ASSIGN(EpilogueFusion fusion,
                       GemmEpilogueFuser().Run(g.block, {}, {}));
  EXPECT_EQ(fusion.fused_chains, 1u);
  EXPECT_EQ(fusion.fused_away.size(), 2u);

  // The GEMM absorbed the bias operand and the activation attribute, and
  // the chain output's identity migrated onto its result.
  EXPECT_EQ(g.gemm->numOperands(), 3u);
  const std::string act_attr =
      g.gemm->getAttrAs<std::string>("epilogue_act").value_or("");
  EXPECT_EQ(act_attr, "relu");
  EXPECT_TRUE(g.z->hasNoUses());
  ASSERT_EQ(g.consumer_out->definingOp()->operand(0), g.c);

  // The hoisted bias declaration keeps the block SSA-verifiable, and DCE
  // (with the live chain output rooted) sweeps exactly the orphaned
  // AddBias + activation.
  EXPECT_OK(g.block.verify());
  ASSERT_OK_AND_ASSIGN(size_t removed,
                       DeadCodeElimination().Run(g.block, {g.consumer_out}));
  EXPECT_EQ(removed, 2u);
  EXPECT_OK(g.block.verify());
  EXPECT_EQ(CountOps(g.block, "sc_high.add_bias"), 0u);
  EXPECT_EQ(CountOps(g.block, "sc_high.relu"), 0u);
}

TEST(GemmEpilogueFuser, FusesBiasOnlyWhenNoActivationFollows) {
  Chain g;
  BuildChain(g, nullptr);
  ASSERT_OK_AND_ASSIGN(EpilogueFusion fusion,
                       GemmEpilogueFuser().Run(g.block, {}, {}));
  EXPECT_EQ(fusion.fused_chains, 1u);
  EXPECT_EQ(fusion.fused_away.size(), 1u);
  EXPECT_EQ(g.gemm->numOperands(), 3u);
  EXPECT_FALSE(g.gemm->hasAttribute("epilogue_act"));
  EXPECT_OK(g.block.verify());
}

TEST(GemmEpilogueFuser, SkipsWhenIntermediateHasAnotherReader) {
  // A second reader of the pre-activation sum — exactly what a backward
  // relu_grad is — must block the activation fold; the bias still fuses
  // (the raw GEMM output has no other reader).
  Chain g;
  BuildChain(g, "sc_high.relu");
  sir::Operation* extra = g.block.appendOp("sc_high.scale");
  extra->setAttribute("alpha", 2.0f);
  extra->addOperand(g.add_bias->result(0));
  extra->addResult("extra", sir::DataType::F32, sir::Shape{2, 2});

  ASSERT_OK_AND_ASSIGN(EpilogueFusion fusion,
                       GemmEpilogueFuser().Run(g.block, {}, {}));
  EXPECT_EQ(fusion.fused_chains, 1u);
  EXPECT_EQ(fusion.fused_away.size(), 1u);  // AddBias only
  EXPECT_FALSE(g.gemm->hasAttribute("epilogue_act"));
  EXPECT_EQ(CountOps(g.block, "sc_high.relu"), 1u);  // act survives
  EXPECT_OK(g.block.verify());
}

TEST(GemmEpilogueFuser, SkipsBiasFusionOnQuantizedWeights) {
  // The q8 GEMM's in[3] carries the dequant scale — a fused bias has
  // nowhere to ride, so the chain must keep its standalone AddBias.
  Chain g;
  BuildChain(g, "sc_high.relu");
  std::unordered_set<const sir::Value*> quantized{g.gemm->operand(1)};
  ASSERT_OK_AND_ASSIGN(EpilogueFusion fusion,
                       GemmEpilogueFuser().Run(g.block, quantized, {}));
  EXPECT_EQ(fusion.fused_chains, 0u);
  EXPECT_EQ(g.gemm->numOperands(), 2u);
  EXPECT_OK(g.block.verify());
}

TEST(GemmEpilogueFuser, NeverRewiresProtectedValues) {
  Chain g;
  BuildChain(g, "sc_high.relu");
  std::unordered_set<const sir::Value*> protected_values{g.z};
  ASSERT_OK_AND_ASSIGN(EpilogueFusion fusion,
                       GemmEpilogueFuser().Run(g.block, {}, protected_values));
  // The activation output is read outside the program; the bias may still
  // fuse but z's producing op must survive with its identity intact.
  for (const sir::Operation* dead : fusion.fused_away)
    EXPECT_TRUE(dead != g.act);
  EXPECT_OK(g.block.verify());
}

TEST(GemmEpilogueFuser, ActOnlyChainFusesWithoutBias) {
  sir::Block block;
  sir::Value* x = block.addArgument(sir::DataType::F32, sir::Shape{2, 3});
  sir::Operation* w_op = block.appendOp("sc_mem.weight");
  sir::Value* w = w_op->addResult("w", sir::DataType::F32, sir::Shape{3, 2});
  sir::Operation* mm = block.appendOp("sc_high.matmul");
  mm->addOperand(x);
  mm->addOperand(w);
  sir::Value* c = mm->addResult("c", sir::DataType::F32, sir::Shape{2, 2});
  sir::Operation* act = block.appendOp("sc_high.gelu");
  act->addOperand(c);
  sir::Value* z = act->addResult("z", sir::DataType::F32, sir::Shape{2, 2});
  sir::Operation* consumer = block.appendOp("sc_high.scale");
  consumer->setAttribute("alpha", 1.0f);
  consumer->addOperand(z);
  consumer->addResult("out", sir::DataType::F32, sir::Shape{2, 2});

  ASSERT_OK_AND_ASSIGN(EpilogueFusion fusion,
                       GemmEpilogueFuser().Run(block, {}, {}));
  EXPECT_EQ(fusion.fused_chains, 1u);
  EXPECT_EQ(mm->numOperands(), 2u);  // no bias operand
  const std::string act_attr =
      mm->getAttrAs<std::string>("epilogue_act").value_or("");
  EXPECT_EQ(act_attr, "gelu");
  EXPECT_OK(block.verify());
}

TEST(MergeBuilder, RejectsEmptyAdapterSet) {
  EXPECT_ERROR_CONTAINS(MergeBuilder().Run({}), "no adapters");
}

}  // namespace
