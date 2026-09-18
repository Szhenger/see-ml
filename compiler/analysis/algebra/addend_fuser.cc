#include "compiler/analysis/algebra/addend_fuser.h"

#include <string_view>
#include <unordered_map>

#include "compiler/diagnostics/updating/error.h"

namespace seeml::update {

namespace sir = seeml::sir;
namespace updating = seeml::diag::updating;

namespace {

/// "nn" | "nt" | "tn" for a plain two-operand GEMM, else empty.
std::string_view GemmForm(const sir::Operation& op) {
  if (op.numOperands() != 2 || op.numResults() != 1 ||
      op.hasAttribute("epilogue_act"))
    return {};
  const std::string_view m = op.mnemonic();
  if (m == "sc_high.matmul") return "nn";
  if (m == "sc_low.matmul_nt") return "nt";
  if (m == "sc_low.matmul_tn") return "tn";
  return {};
}

/// The GEMM's inner (summed-over) dimension K.
int64_t InnerDim(const sir::Operation& gemm, std::string_view form) {
  const auto& a = gemm.operand(0)->shape().dims;
  return form == "tn" ? a.at(0) : a.at(1);
}

/// Ops that rewrite storage in place. Mirrors DeadCodeElimination's
/// effectful set, minus storage declarations (which execute nothing).
bool MutatesInPlace(const sir::Operation& op) {
  const std::string_view m = op.mnemonic();
  return m == "sc_low.clip_norm" || m == "sc_low.sgd_step" ||
         m == "sc_low.adamw_step" || m == "sc_low.gemm_acc" ||
         m == "sc_low.accumulate" || m == "sc_low.zero";
}

}  // namespace

std::expected<AddendFusion, std::string> GemmAddendFuser::Run(
    sir::Block& block,
    const std::unordered_set<const sir::Value*>& narrow_weights,
    const std::unordered_set<const sir::Value*>& protected_values) {
  // Snapshot with positions, and a prefix count of in-place mutators so
  // "nothing mutates between the GEMM and the add" is two lookups.
  std::vector<sir::Operation*> ops;
  ops.reserve(block.numOps());
  block.walk([&](sir::Operation* op) { ops.push_back(op); });
  std::unordered_map<const sir::Operation*, size_t> position;
  position.reserve(ops.size());
  std::vector<size_t> mutators_before(ops.size() + 1, 0);
  for (size_t i = 0; i < ops.size(); ++i) {
    position[ops[i]] = i;
    mutators_before[i + 1] =
        mutators_before[i] + (MutatesInPlace(*ops[i]) ? 1 : 0);
  }

  AddendFusion fusion;
  for (sir::Operation* add : ops) {
    if (add->mnemonic() != "sc_high.add" || add->numOperands() != 2 ||
        add->numResults() != 1 || add->hasAttribute("fused_stages") ||
        add->operand(0) == add->operand(1))
      continue;
    const sir::Shape& shape = add->result(0)->shape();
    if (shape.dims.size() != 2) continue;

    // The better of the (up to) two candidates: smaller inner dimension.
    sir::Operation* best = nullptr;
    std::string_view best_form;
    size_t best_index = 0;
    for (size_t i = 0; i < 2; ++i) {
      sir::Value* v = add->operand(i);
      sir::Operation* gemm = v->definingOp();
      if (!gemm || !v->hasOneUse() || protected_values.contains(v)) continue;
      const std::string_view form = GemmForm(*gemm);
      if (form.empty() || v->shape().dims != shape.dims ||
          add->operand(1 - i)->shape().dims != shape.dims)
        continue;
      if (narrow_weights.contains(gemm->operand(0)) ||
          narrow_weights.contains(gemm->operand(1)))
        continue;
      const auto at = position.find(gemm);
      const auto to = position.find(add);
      if (at == position.end() || to == position.end() ||
          at->second >= to->second)
        return updating::Error(updating::kAddendFuser,
                               "a GEMM does not precede the add reading it");
      if (mutators_before[to->second] != mutators_before[at->second + 1])
        continue;
      if (!best || InnerDim(*gemm, form) < InnerDim(*best, best_form)) {
        best = gemm;
        best_form = form;
        best_index = i;
      }
    }
    if (!best) continue;

    sir::Value* addend = add->operand(1 - best_index);
    add->setOperand(0, addend);
    add->setOperand(1, best->operand(0));
    add->addOperand(best->operand(1));
    add->setAttribute("gemm_addend", std::string(best_form));
    fusion.fused_away.push_back(best);
    ++fusion.fused;
  }

  if (fusion.fused > 0)
    seeml::diag::Note(updating::kAddendFuser,
                      "folded " + std::to_string(fusion.fused) +
                          " GEMM(s) into the add reading them, orphaning "
                          "them for the DCE sweep");
  return fusion;
}

}  // namespace seeml::update
