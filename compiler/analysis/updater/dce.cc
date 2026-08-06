#include "compiler/analysis/updater/dce.h"

#include <vector>

namespace seeml::update {

namespace {

/// Ops whose execution is a side effect on buffers rather than a produced
/// value: removing one because its results look unused would drop the very
/// mutation the program exists to perform.
bool IsEffectful(const sir::Operation& op) {
  if (op.isMemoryOp()) return true;  // storage declarations, bound by name
  const std::string_view m = op.mnemonic();
  return m == "sc_low.clip_norm" || m == "sc_low.sgd_step" ||
         m == "sc_low.adamw_step" || m == "sc_low.gemm_acc";
}

}  // namespace

std::expected<size_t, std::string> DeadCodeElimination::Run(
    sir::Block& block, const std::unordered_set<const sir::Value*>& roots) {
  std::vector<sir::Operation*> ops;
  ops.reserve(block.numOps());
  block.walk([&](sir::Operation* op) { ops.push_back(op); });

  size_t removed = 0;
  // Backward sweep: removing a dead consumer drops its operands' use-list
  // entries first, so a chain of dead ops disappears in one pass and
  // removeOp's no-remaining-uses contract holds at every step.
  for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
    sir::Operation* op = *it;
    if (IsEffectful(*op)) continue;
    bool live = false;
    for (const auto& r : op->results())
      if (!r->hasNoUses() || roots.contains(r.get())) {
        live = true;
        break;
      }
    if (live) continue;
    block.removeOp(op);
    ++removed;
  }
  return removed;
}

}  // namespace seeml::update
