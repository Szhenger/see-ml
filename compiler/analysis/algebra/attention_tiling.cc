#include "compiler/analysis/algebra/attention_tiling.h"

#include <vector>

#include "compiler/diagnostics/updating/error.h"
#include "source/plan/instruction.h"

namespace seeml::update {

namespace sir = seeml::sir;
namespace updating = seeml::diag::updating;

std::expected<AttilingDecision, std::string> AttentionTiling::Run(
    sir::Block& block) {
  std::vector<sir::Operation*> ops;
  block.walk([&](sir::Operation* op) {
    if (op->mnemonic() == "sc_high.attention" && !op->hasAttribute("tiled"))
      ops.push_back(op);
  });
  AttilingDecision decision;
  decision.attention_ops = ops.size();
  for (const sir::Operation* op : ops) {
    if (op->numResults() != 2)
      return updating::Error(updating::kAttentionTiling,
                             "attention op without a cache result");
    const auto& dims = op->result(1)->shape().dims;
    if (dims.size() != 2)
      return updating::Error(updating::kAttentionTiling,
                             "attention cache is not [B*H*S, S]");
    decision.probs_cache_bytes +=
        static_cast<uint64_t>(dims[0]) * static_cast<uint64_t>(dims[1]) * 4;
  }
  decision.tiled = kind_ == AttentionKind::kTiled ||
                   (kind_ == AttentionKind::kAuto &&
                    decision.probs_cache_bytes > budget_);
  if (!decision.tiled || ops.empty()) return decision;

  for (sir::Operation* op : ops) {
    // The tiled backward's one geometry word holds 16 bits per field.
    const int64_t heads = op->getAttrAs<int64_t>("heads").value_or(0);
    const int64_t seq = op->getAttrAs<int64_t>("seq").value_or(0);
    const auto& q = op->operand(0)->shape().dims;
    const int64_t rows = q.at(0), width = q.at(1);
    if (heads <= 0 || seq <= 0 || rows % seq != 0 || width % heads != 0 ||
        rows / seq > 0xFFFF || seq > 0xFFFF || heads > 0xFFFF ||
        width / heads > 0xFFFF)
      return updating::Error(updating::kAttentionTiling,
                             "attention geometry does not fit the tiled "
                             "family's 16-bit fields");
    op->setAttribute("tiled", int64_t{1});
    op->result(1)->setShape(sir::Shape{
        rows * heads, static_cast<int64_t>(kAttnStatsWidth)});
  }
  seeml::diag::Note(updating::kAttentionTiling,
                    "tiled " + std::to_string(ops.size()) +
                        " attention op(s): a probability cache of " +
                        std::to_string(decision.probs_cache_bytes) +
                        " B becomes " +
                        std::to_string(ops.size()) + " stats row(s)");
  return decision;
}

}  // namespace seeml::update
