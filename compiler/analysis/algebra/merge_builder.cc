#include "compiler/analysis/algebra/merge_builder.h"

#include "compiler/diagnostics/updating/error.h"

namespace seeml::update {

namespace sir = seeml::sir;
namespace updating = seeml::diag::updating;

std::expected<MergeProgram, std::string> MergeBuilder::Run(
    const std::vector<GraftedAdapter>& adapters) {
  if (adapters.empty())
    return updating::Error(updating::kMergeBuilder, "no adapters to merge");

  MergeProgram program;
  program.block = std::make_unique<sir::Block>();

  for (const GraftedAdapter& adapter : adapters) {
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
    // Named from the adapter's unique site stem, not the frozen weight: a
    // tied weight has one delta per graft site.
    sir::Value* delta = fill->addResult(adapter.id_base + ".delta",
                                        sir::DataType::F32,
                                        adapter.frozen_weight->shape());

    sir::Operation* acc = program.block->appendOp("sc_low.gemm_acc");
    acc->setAttribute("alpha", adapter.scale);
    acc->addOperand(a);
    acc->addOperand(b);
    acc->addOperand(delta);

    program.outputs.emplace_back(delta, &adapter);
  }

  return program;
}

}  // namespace seeml::update
