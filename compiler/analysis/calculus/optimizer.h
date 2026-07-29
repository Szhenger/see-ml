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
// =============================================================================

namespace seeml::update {

class OptimizerSynthesizer {
 public:
  explicit OptimizerSynthesizer(OptimizerSpec spec) : spec_(spec) {}

  [[nodiscard]] std::expected<void, std::string> Run(
      seeml::sir::Block& block,
      const std::unordered_map<seeml::sir::Value*, seeml::sir::Value*>&
          param_grads);

 private:
  OptimizerSpec spec_;
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_CALCULUS_OPTIMIZER_H_
