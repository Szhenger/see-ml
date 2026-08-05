#include "compiler/analysis/algebra/epilogue_fuser.h"

#include <string_view>

#include "compiler/diagnostics/updating/error.h"

namespace seeml::update {

namespace sir = seeml::sir;
namespace updating = seeml::diag::updating;

namespace {

/// The activation mnemonics an epilogue can absorb, mapped to the attribute
/// vocabulary instruction lowering encodes into the v5 flag bits.
const char* EpilogueActName(std::string_view mnemonic) {
  if (mnemonic == "sc_high.relu") return "relu";
  if (mnemonic == "sc_high.gelu") return "gelu";
  if (mnemonic == "sc_high.silu") return "silu";
  return nullptr;
}

/// True iff `a` appears before `b` in `block`'s program order. Recomputed
/// per query — earlier hoists shift positions, so a cached index would lie.
bool OpPrecedes(const sir::Block& block, const sir::Operation* a,
                const sir::Operation* b) {
  for (const auto& op : block.operations()) {
    if (op.get() == a) return true;
    if (op.get() == b) return false;
  }
  return false;
}

}  // namespace

std::expected<EpilogueFusion, std::string> GemmEpilogueFuser::Run(
    sir::Block& block,
    const std::unordered_set<const sir::Value*>& quantized_weights,
    const std::unordered_set<const sir::Value*>& protected_values) {
  // Snapshot: fusion rewires use-lists but never adds or removes ops (the
  // orphans stay in place for DCE), so a pre-walk op list stays valid.
  std::vector<sir::Operation*> ops;
  ops.reserve(block.numOps());
  block.walk([&](sir::Operation* op) { ops.push_back(op); });

  const auto is_protected = [&](const sir::Value* v) {
    return protected_values.contains(v);
  };

  EpilogueFusion fusion;
  for (sir::Operation* gemm : ops) {
    if (gemm->mnemonic() != "sc_high.matmul" || gemm->numOperands() != 2 ||
        gemm->numResults() != 1)
      continue;
    sir::Value* c = gemm->result(0);
    // The chain output's identity migrates onto C, so C must have exactly
    // one reader (the chain) and no reader outside the program. Any value
    // the backward program consumes fails this by carrying its adjoint
    // producer as an extra user — legality falls out of SSA bookkeeping.
    if (!c->hasOneUse() || is_protected(c)) continue;
    sir::Operation* user = c->users().front();

    if (user->mnemonic() == "sc_high.add_bias" && user->numOperands() == 2 &&
        user->operand(0) == c && user->numResults() == 1 &&
        !quantized_weights.contains(gemm->operand(1))) {
      sir::Value* y = user->result(0);
      sir::Value* bias = user->operand(1);

      // SSA order for the new operand: the parser materializes a weight at
      // its first use, so the bias declaration typically sits after the
      // GEMM. An operand-less storage declaration hoists above it soundly;
      // a bias produced by real computation defined after the GEMM makes
      // the chain unfusable.
      sir::Operation* bias_def = bias->definingOp();
      if (bias_def && !OpPrecedes(block, bias_def, gemm)) {
        if (!bias_def->isMemoryOp()) continue;
        block.moveOpBefore(bias_def, gemm);
      }

      // Chain step 2: fold the activation too when Y's only reader is one.
      sir::Operation* act = y->hasOneUse() ? y->users().front() : nullptr;
      const char* act_name =
          act && act->numOperands() == 1 && act->numResults() == 1
              ? EpilogueActName(act->mnemonic())
              : nullptr;
      if (act_name && !is_protected(y) && !is_protected(act->result(0))) {
        gemm->addOperand(bias);
        gemm->setAttribute("epilogue_act", std::string(act_name));
        act->result(0)->replaceAllUsesWith(c);
        fusion.fused_away.push_back(user);
        fusion.fused_away.push_back(act);
        ++fusion.fused_chains;
      } else if (!is_protected(y)) {
        gemm->addOperand(bias);
        y->replaceAllUsesWith(c);
        fusion.fused_away.push_back(user);
        ++fusion.fused_chains;
      }
    } else if (const char* act_name = EpilogueActName(user->mnemonic());
               act_name && user->numOperands() == 1 &&
               user->numResults() == 1 && !is_protected(user->result(0))) {
      // Bias-less layer: the activation alone becomes the epilogue. Valid
      // for quantized weights too — the activation needs no operand slot.
      gemm->setAttribute("epilogue_act", std::string(act_name));
      user->result(0)->replaceAllUsesWith(c);
      fusion.fused_away.push_back(user);
      ++fusion.fused_chains;
    }
  }

  if (fusion.fused_chains > 0)
    seeml::diag::Note(updating::kEpilogueFuser,
                      "fused " + std::to_string(fusion.fused_chains) +
                          " GEMM epilogue chain(s), orphaning " +
                          std::to_string(fusion.fused_away.size()) +
                          " op(s) for the DCE sweep");
  return fusion;
}

}  // namespace seeml::update
