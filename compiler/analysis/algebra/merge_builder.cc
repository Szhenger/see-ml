#include "compiler/analysis/algebra/merge_builder.h"

#include <unordered_set>

#include "compiler/diagnostics/updating/error.h"

namespace seeml::update {

namespace sir = seeml::sir;
namespace updating = seeml::diag::updating;

std::expected<MergeProgram, std::string> MergeBuilder::Run(
    const std::vector<GraftedAdapter>& adapters) {
  if (adapters.empty())
    return updating::Error(updating::kMergeBuilder, "no adapters to merge");

  // Commit applies every delta additively to its weight's file bytes, so
  // two adapters over one frozen weight would commit W + Δ_1 + Δ_2 — a model
  // the training graph never computed. LoraGrafter shares one adapter pair
  // per tied weight precisely so this cannot happen; reject rather than
  // silently corrupt if a future grafter change reintroduces duplicates.
  std::unordered_set<const sir::Value*> seen_weights;
  for (const GraftedAdapter& adapter : adapters)
    if (!seen_weights.insert(adapter.frozen_weight).second)
      return updating::Error(
          updating::kMergeBuilder,
          "two adapters share frozen weight '" +
              std::string(adapter.frozen_weight->id()) +
              "'; tied weights must share one adapter pair");

  MergeProgram program;
  program.block = std::make_unique<sir::Block>();

  for (size_t adapter_index = 0; adapter_index < adapters.size();
       ++adapter_index) {
    const GraftedAdapter& adapter = adapters[adapter_index];
    // Mirror declarations aliasing the training program's persistent storage.
    sir::Operation* a_op = program.block->appendOp("sc_mem.param");
    sir::Value* a = a_op->addResult(std::string(adapter.A->id()) + ".merge_a",
                                    sir::DataType::F32, adapter.A->shape());
    program.aliases[a] = adapter.A;

    sir::Operation* b_op = program.block->appendOp("sc_mem.param");
    sir::Value* b = b_op->addResult(std::string(adapter.B->id()) + ".merge_b",
                                    sir::DataType::F32, adapter.B->shape());
    program.aliases[b] = adapter.B;

    // Δ = 0; Δ += (α/r) · A @ B — pure linear algebra, no epochs. Commit
    // adds Δ to the model file's own f32 weights (see EmitEntry).
    sir::Operation* fill = program.block->appendOp("sc_low.fill");
    fill->setAttribute("value", 0.0f);
    // Named from the adapter's unique id stem; one delta per unique frozen
    // weight (tied weights share a single adapter pair).
    sir::Value* delta = fill->addResult(adapter.id_base + ".delta",
                                        sir::DataType::F32,
                                        adapter.frozen_weight->shape());

    sir::Operation* acc = program.block->appendOp("sc_low.gemm_acc");
    acc->setAttribute("alpha", adapter.scale);
    acc->addOperand(a);
    acc->addOperand(b);
    acc->addOperand(delta);

    program.outputs.emplace_back(delta, adapter_index);
  }

  return program;
}

}  // namespace seeml::update
