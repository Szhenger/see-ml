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

  // Gradient accumulation: the gradient the step consumes is the
  // persistent accumulator, folded once per micro-batch. Every fold is
  // appended BEFORE any step op, so the lowered stream splits cleanly into
  // the grad program (everything through the last fold) and the step
  // program (everything after).
  const bool accumulate = grad_accum_steps_ > 1;
  std::vector<sir::Value*> accs;
  if (accumulate) {
    for (auto& [p, g] : ordered) {
      sir::Operation* acc_op = block.appendOp("sc_mem.param");
      acc_op->setAttribute("trainable", int64_t{0});
      acc_op->setAttribute("init", std::string("zeros"));
      sir::Value* acc = acc_op->addResult(std::string(p->id()) + ".grad_acc",
                                          sir::DataType::F32, p->shape());
      sir::Operation* fold = block.appendOp("sc_low.accumulate");
      fold->addOperand(acc);
      fold->addOperand(g);
      accs.push_back(acc);
    }
  }
  if (!emit_step_) return {};

  for (size_t i = 0; i < ordered.size(); ++i) {
    sir::Value* p = ordered[i].first;
    sir::Value* g = accumulate ? accs[i] : ordered[i].second;
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
      if (accumulate) block.appendOp("sc_low.zero")->addOperand(g);
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
    // The accumulator restarts from zero for the next G micro-batches.
    if (accumulate) block.appendOp("sc_low.zero")->addOperand(g);
  }
  return {};
}

}  // namespace seeml::update
