#include "compiler/analysis/algebra/chain_fuser.h"

#include <bit>
#include <cstdint>
#include <optional>
#include <string_view>

#include "compiler/diagnostics/updating/error.h"
#include "source/plan/instruction.h"

namespace seeml::update {

namespace sir = seeml::sir;
namespace updating = seeml::diag::updating;

namespace {

/// The kFusedMap stage an op lowers to, if it is one the chain can hold.
std::optional<FusedStage> StageKindOf(const sir::Operation& op) {
  if (op.numResults() != 1 || op.hasAttribute("fused_stages"))
    return std::nullopt;
  const std::string_view m = op.mnemonic();
  if (m == "sc_high.add" && op.numOperands() == 2) return FusedStage::kAdd;
  if ((m == "sc_high.mul" || m == "sc_low.mul") && op.numOperands() == 2)
    return FusedStage::kMul;
  if ((m == "sc_high.scale" || m == "sc_low.scale") && op.numOperands() == 1)
    return FusedStage::kScale;
  if (op.numOperands() != 1) return std::nullopt;
  if (m == "sc_high.relu") return FusedStage::kRelu;
  if (m == "sc_high.gelu") return FusedStage::kGelu;
  if (m == "sc_high.silu") return FusedStage::kSilu;
  return std::nullopt;
}

bool IsBinary(FusedStage kind) {
  return kind == FusedStage::kAdd || kind == FusedStage::kMul;
}

/// A chain under construction: the stage bytes, and the operands they name.
struct Chain {
  std::vector<sir::Operation*> ops;
  sir::Value* x = nullptr;            // the running value's origin
  std::vector<sir::Value*> others;    // slots 1..2
  std::vector<float> imms;            // immediates 0..1
  uint64_t stages = 0;

  /// Appends `op`, whose running input is operand `run_index`. False —
  /// with the chain unchanged — if the micro-program has no room for it.
  bool Append(sir::Operation* op, FusedStage kind, size_t run_index) {
    if (ops.size() == kFusedMapMaxStages) return false;
    uint8_t arg = 0;
    if (IsBinary(kind)) {
      sir::Value* other = op->operand(1 - run_index);
      size_t slot = 0;
      for (; slot < others.size(); ++slot)
        if (others[slot] == other) break;
      if (slot == others.size()) {
        if (others.size() == 2) return false;
        others.push_back(other);
      }
      arg = static_cast<uint8_t>(slot + 1);
    } else if (kind == FusedStage::kScale) {
      const float alpha = op->getAttrAs<float>("alpha").value_or(1.0f);
      size_t i = 0;
      for (; i < imms.size(); ++i)
        if (std::bit_cast<uint32_t>(imms[i]) == std::bit_cast<uint32_t>(alpha))
          break;
      if (i == imms.size()) {
        if (imms.size() == 2) return false;
        imms.push_back(alpha);
      }
      arg = static_cast<uint8_t>(i);
    }
    stages |= static_cast<uint64_t>(MakeFusedStage(
                  kind, arg, IsBinary(kind) && run_index == 1))
              << (8 * ops.size());
    ops.push_back(op);
    return true;
  }
};

}  // namespace

std::expected<ChainFusion, std::string> ElementwiseChainFuser::Run(
    sir::Block& block,
    const std::unordered_set<const sir::Value*>& protected_values) {
  // Snapshot: fusion rewires operands and attributes but never adds or
  // removes ops (the orphans wait for DCE), so the list stays valid.
  std::vector<sir::Operation*> ops;
  ops.reserve(block.numOps());
  block.walk([&](sir::Operation* op) { ops.push_back(op); });

  std::unordered_set<const sir::Operation*> absorbed;
  ChainFusion fusion;
  for (sir::Operation* head : ops) {
    if (absorbed.contains(head)) continue;
    const auto head_kind = StageKindOf(*head);
    if (!head_kind) continue;
    const sir::Shape& shape = head->result(0)->shape();
    // Foldable: one shape throughout, and no operand that is mutable
    // storage. Fusion moves every read of the chain to the tail's position;
    // a parameter or accumulator read there could see an in-place step,
    // fold or zero that ran in between. Activations and gradients are
    // single-assignment, so reading them later reads the same bytes.
    auto same_shape = [&](const sir::Operation& op) {
      if (op.result(0)->shape().dims != shape.dims) return false;
      for (sir::Value* v : op.operands()) {
        if (v->shape().dims != shape.dims) return false;
        const sir::Operation* def = v->definingOp();
        if (def && def->mnemonic() == "sc_mem.param") return false;
      }
      return true;
    };
    if (!same_shape(*head)) continue;
    // x + x (or x * x) has no "other" operand to name; leave it alone.
    if (IsBinary(*head_kind) && head->operand(0) == head->operand(1)) continue;

    Chain chain;
    chain.x = head->operand(0);
    chain.Append(head, *head_kind, /*run_index=*/0);
    for (sir::Operation* cur = head;;) {
      sir::Value* v = cur->result(0);
      if (!v->hasOneUse() || protected_values.contains(v)) break;
      sir::Operation* user = v->users().front();
      const auto kind = StageKindOf(*user);
      if (!kind || absorbed.contains(user) || !same_shape(*user)) break;
      size_t run_index = 0;
      if (IsBinary(*kind)) {
        if (user->operand(0) == user->operand(1)) break;
        run_index = user->operand(0) == v ? 0 : 1;
      }
      if (!chain.Append(user, *kind, run_index)) break;
      cur = user;
    }
    if (chain.ops.size() < 2) continue;

    // Rewrite the tail in place: operands [x, others...], the program as
    // attributes. The tail never has MORE operands than that — a binary
    // tail contributed one of `others` itself — so set-then-add suffices.
    sir::Operation* tail = chain.ops.back();
    std::vector<sir::Value*> operands{chain.x};
    operands.insert(operands.end(), chain.others.begin(), chain.others.end());
    if (operands.size() < tail->numOperands())
      return updating::Error(updating::kChainFuser,
                             "fused operand list shorter than the tail's");
    for (size_t i = 0; i < operands.size(); ++i) {
      if (i < tail->numOperands())
        tail->setOperand(i, operands[i]);
      else
        tail->addOperand(operands[i]);
    }
    tail->setAttribute("fused_stages", static_cast<int64_t>(chain.stages));
    for (size_t i = 0; i < chain.imms.size(); ++i)
      tail->setAttribute(i == 0 ? "fused_imm0" : "fused_imm1", chain.imms[i]);
    for (size_t i = 0; i + 1 < chain.ops.size(); ++i) {
      absorbed.insert(chain.ops[i]);
      fusion.fused_away.push_back(chain.ops[i]);
    }
    absorbed.insert(tail);
    ++fusion.fused_chains;
  }

  if (fusion.fused_chains > 0)
    seeml::diag::Note(updating::kChainFuser,
                      "fused " + std::to_string(fusion.fused_chains) +
                          " elementwise chain(s), orphaning " +
                          std::to_string(fusion.fused_away.size()) +
                          " op(s) for the DCE sweep");
  return fusion;
}

}  // namespace seeml::update
