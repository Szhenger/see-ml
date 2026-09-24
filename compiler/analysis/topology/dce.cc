#include "compiler/analysis/topology/dce.h"

#include <unordered_map>
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
         m == "sc_low.adamw_step" || m == "sc_low.gemm_acc" ||
         m == "sc_low.accumulate" || m == "sc_low.zero" ||
         m == "sc_low.attn_dq_tiled";  // writes delta into the stats row
}

}  // namespace

std::expected<size_t, std::string> DeadCodeElimination::Run(
    sir::Block& block, const std::unordered_set<const sir::Value*>& roots) {
  std::vector<sir::Operation*> ops;
  ops.reserve(block.numOps());
  block.walk([&](sir::Operation* op) { ops.push_back(op); });

  // Mark, then compact once (E5, #84). The backward sweep decides
  // liveness on a private use count — a dead consumer releases its
  // operands' counts, so a whole dead chain is found in one pass — and the
  // doomed ops leave the block together (Block::removeOps). Removing them
  // one removeOp at a time was O(removed x ops): free while DCE only swept
  // fusion orphans, a quadratic cliff the first time a pass dead-codes a
  // real fraction of the graph.
  std::unordered_map<const sir::Value*, size_t> uses;
  auto uses_of = [&](const sir::Value* v) -> size_t& {
    auto [it, fresh] = uses.try_emplace(v, 0);
    if (fresh) it->second = v->users().size();
    return it->second;
  };
  std::vector<sir::Operation*> dead;  // consumers before producers
  for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
    sir::Operation* op = *it;
    if (IsEffectful(*op)) continue;
    bool live = false;
    for (const auto& r : op->results())
      if (uses_of(r.get()) != 0 || roots.contains(r.get())) {
        live = true;
        break;
      }
    if (live) continue;
    for (size_t i = 0; i < op->numOperands(); ++i) --uses_of(op->operand(i));
    dead.push_back(op);
  }
  block.removeOps(dead);
  return dead.size();
}

}  // namespace seeml::update
