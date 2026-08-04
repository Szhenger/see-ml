#include "compiler/analysis/calculus/optimizer.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace seeml::update {

namespace sir = seeml::sir;

std::expected<void, std::string> OptimizerSynthesizer::Run(
    sir::Block& block,
    const std::unordered_map<sir::Value*, sir::Value*>& param_grads) {
  // Deterministic emission order: sort by parameter id.
  std::vector<std::pair<sir::Value*, sir::Value*>> ordered(param_grads.begin(),
                                                           param_grads.end());
  std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
    return a.first->id() < b.first->id();
  });

  for (auto& [p, g] : ordered) {
    // Per-tensor L2 clipping precedes the step: one bad batch must not be
    // able to blow up the parameters (or poison AdamW's moment state).
    if (clip_norm_ > 0.0f) {
      sir::Operation* clip = block.appendOp("sc_low.clip_norm");
      clip->setAttribute("max_norm", clip_norm_);
      clip->addOperand(g);
    }
    if (kind_ == OptimizerKind::kSgd) {
      sir::Operation* step = block.appendOp("sc_low.sgd_step");
      step->addOperand(p);
      step->addOperand(g);
      continue;
    }
    // AdamW: declare persistent first/second moment state (zero-initialized,
    // checkpointed with the adapters), then the in-place fused step.
    sir::Operation* m_op = block.appendOp("sc_mem.param");
    m_op->setAttribute("trainable", int64_t{0});
    m_op->setAttribute("init", std::string("zeros"));
    sir::Value* m = m_op->addResult(std::string(p->id()) + ".adam_m",
                                    sir::DataType::F32, p->shape());

    sir::Operation* v_op = block.appendOp("sc_mem.param");
    v_op->setAttribute("trainable", int64_t{0});
    v_op->setAttribute("init", std::string("zeros"));
    sir::Value* v = v_op->addResult(std::string(p->id()) + ".adam_v",
                                    sir::DataType::F32, p->shape());

    sir::Operation* step = block.appendOp("sc_low.adamw_step");
    step->addOperand(p);
    step->addOperand(g);
    step->addOperand(m);
    step->addOperand(v);
  }
  return {};
}

}  // namespace seeml::update
