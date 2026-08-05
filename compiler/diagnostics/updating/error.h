#ifndef SEEML_COMPILER_DIAGNOSTICS_UPDATING_ERROR_H_
#define SEEML_COMPILER_DIAGNOSTICS_UPDATING_ERROR_H_

#include <expected>
#include <string>
#include <string_view>

#include "compiler/diagnostics/diagnostic.h"

// =============================================================================
// updating/ — errors formed by the analytic methods that synthesize the
// update program (compiler/analysis/calculus/ + compiler/analysis/algebra/):
// reverse-mode autodiff, LoRA grafting, optimizer synthesis, and the merge
// program. These are the mathematics of the compilation; what can fail is
// analytic — a missing VJP rule, an empty trainable set, no graft targets.
// Failure discipline: name the analytic unit and the op or set that broke
// the derivation, so the fix points at the model or the spec, not the IR.
// =============================================================================

namespace seeml::diag::updating {

inline constexpr std::string_view kAutodiff = "TrainableAutodiff";
inline constexpr std::string_view kLoraGrafter = "LoraGrafter";
inline constexpr std::string_view kMergeBuilder = "MergeBuilder";
inline constexpr std::string_view kOptimizer = "OptimizerSynthesizer";
inline constexpr std::string_view kEpilogueFuser = "GemmEpilogueFuser";

/// Analytic failure in `unit`: "<unit>: <message>".
[[nodiscard]] inline std::unexpected<std::string> Error(std::string_view unit,
                                                        std::string_view message) {
  return Fail(unit, message);
}

}  // namespace seeml::diag::updating

#endif  // SEEML_COMPILER_DIAGNOSTICS_UPDATING_ERROR_H_
