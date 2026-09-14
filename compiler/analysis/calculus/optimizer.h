#ifndef SEEML_COMPILER_ANALYSIS_CALCULUS_OPTIMIZER_H_
#define SEEML_COMPILER_ANALYSIS_CALCULUS_OPTIMIZER_H_

#include <expected>
#include <string>
#include <unordered_map>

#include "compiler/frontend/representation/sir.h"
#include "source/plan/update_types.h"

// =============================================================================
// OptimizerSynthesizer — appends the optimizer step (SGD / AdamW) as
// ordinary SIR ops, declaring persistent moment state so one program
// execution = one full training step (fwd + bwd + update). Part of
// analysis/calculus/ — the descent half of SGD (gradients come from
// autodiff.h).
//
// The synthesizer shapes the program; it does not carry hyperparameters.
// It consumes exactly the two spec fields that change the program's
// structure — the optimizer kind (which step op, what moment state) and
// clip_norm (whether a clip op precedes the steps) — so the constructor
// takes exactly those. Numeric hyperparameters (lr, betas, eps, weight
// decay, the LR schedule) are a PlanHeader concern: the runtime reads them
// from the header at dispatch, and baking them here as well would create a
// second source of truth.
// =============================================================================

namespace seeml::update {

// Gradient accumulation (roadmap 2a): with grad_accum_steps > 1 the
// synthesizer declares one persistent accumulator per parameter, folds
// the gradient into it (sc_low.accumulate, the tail of the GRAD program)
// and then clips, steps and zeroes the accumulator instead of the gradient
// (the STEP program). The driver splits the lowered stream at the first
// step-program instruction. `emit_step` false (the finite-difference test
// hook) keeps the accumulators and the folds but emits no step.
class OptimizerSynthesizer {
 public:
  OptimizerSynthesizer(OptimizerKind kind, float clip_norm,
                       uint32_t grad_accum_steps = 1, bool emit_step = true)
      : kind_(kind),
        clip_norm_(clip_norm),
        grad_accum_steps_(grad_accum_steps),
        emit_step_(emit_step) {}

  [[nodiscard]] std::expected<void, std::string> Run(
      seeml::sir::Block& block,
      const std::unordered_map<seeml::sir::Value*, seeml::sir::Value*>&
          param_grads);

 private:
  OptimizerKind kind_;
  float clip_norm_;
  uint32_t grad_accum_steps_;
  bool emit_step_;
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_CALCULUS_OPTIMIZER_H_
