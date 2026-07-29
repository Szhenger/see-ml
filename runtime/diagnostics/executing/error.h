#ifndef SEEML_RUNTIME_DIAGNOSTICS_EXECUTING_ERROR_H_
#define SEEML_RUNTIME_DIAGNOSTICS_EXECUTING_ERROR_H_

#include <expected>
#include <string>
#include <string_view>

#include "runtime/diagnostics/diagnostic.h"

// =============================================================================
// executing/ — errors formed by the engine's lifecycle and dispatch
// (runtime/engine/): plan identity and integrity at load, training-loop
// guards (the non-finite-loss abort), gate and merge sequencing, and the
// engine's boundary contracts. The executor's kernels themselves cannot
// fail — every input was bounds-proven by the validating process — so every
// executing error is a lifecycle error, and the engine speaks for all of
// them under its own unit.
// =============================================================================

namespace seeml::update_rt::diag::executing {

inline constexpr std::string_view kEngine = "UpdateEngine";

/// Lifecycle failure: "UpdateEngine: <message>".
[[nodiscard]] inline std::unexpected<std::string> Error(std::string_view message) {
  return Fail(kEngine, message);
}

}  // namespace seeml::update_rt::diag::executing

#endif  // SEEML_RUNTIME_DIAGNOSTICS_EXECUTING_ERROR_H_
