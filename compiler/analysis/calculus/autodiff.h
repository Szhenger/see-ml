#ifndef SEEML_COMPILER_ANALYSIS_CALCULUS_AUTODIFF_H_
#define SEEML_COMPILER_ANALYSIS_CALCULUS_AUTODIFF_H_

#include <expected>
#include <string>
#include <unordered_map>
#include <vector>

#include "compiler/frontend/representation/sir.h"

// =============================================================================
// TrainableAutodiff — reverse-mode AD pruned to the trainable set: adjoints
// are synthesized only along paths that reach a LoRA parameter; the frozen
// base model generates no backward computation. Part of analysis/calculus/ —
// the differentiation half of SGD (the descent step is optimizer.h).
// =============================================================================

namespace seeml::update {

class TrainableAutodiff {
 public:
  /// `seed_value` is dL/dL: 1.0 for a whole-batch step, 1/G under
  /// gradient accumulation so that G accumulated micro-batch gradients
  /// (each a mean over its own rows) sum to the mean over the effective
  /// batch. A power-of-two G makes the scaling exact per element.
  explicit TrainableAutodiff(float seed_value = 1.0f)
      : seed_value_(seed_value) {}

  /// Weaves the backward pass into `block`. `loss` must be a scalar value in
  /// the block; `trainables` defines the set of parameters that need
  /// gradients. Returns the map param -> synthesized gradient value.
  [[nodiscard]] std::expected<
      std::unordered_map<seeml::sir::Value*, seeml::sir::Value*>, std::string>
  Run(seeml::sir::Block& block, seeml::sir::Value* loss,
      const std::vector<seeml::sir::Value*>& trainables);

 private:
  float seed_value_ = 1.0f;
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_CALCULUS_AUTODIFF_H_
