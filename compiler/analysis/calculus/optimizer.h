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

class OptimizerSynthesizer {
 public:
  OptimizerSynthesizer(OptimizerKind kind, float clip_norm)
      : kind_(kind), clip_norm_(clip_norm) {}

  [[nodiscard]] std::expected<void, std::string> Run(
      seeml::sir::Block& block,
      const std::unordered_map<seeml::sir::Value*, seeml::sir::Value*>&
          param_grads);

 private:
  OptimizerKind kind_;
  float clip_norm_;
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_CALCULUS_OPTIMIZER_H_
