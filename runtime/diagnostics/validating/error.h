#ifndef SEEML_RUNTIME_DIAGNOSTICS_VALIDATING_ERROR_H_
#define SEEML_RUNTIME_DIAGNOSTICS_VALIDATING_ERROR_H_

#include <expected>
#include <string>
#include <string_view>

#include "runtime/diagnostics/diagnostic.h"

// =============================================================================
// validating/ — errors formed during load-time verification of a .seeu
// plan's instruction streams (runtime/validator/). After this process
// accepts a plan, the executor dispatches it blindly — so the discipline is
// total coverage: every operand ref of every instruction is bounds-checked
// against its address space, writes may only target the mutable arena, and
// unknown opcodes are load errors, never silent skips.
// =============================================================================

namespace seeml::update_rt::diag::validating {

inline constexpr std::string_view kPlanValidator = "PlanValidator";

/// Verification failure: "PlanValidator: <message>".
[[nodiscard]] inline std::unexpected<std::string> Error(std::string_view message) {
  return Fail(kPlanValidator, message);
}

}  // namespace seeml::update_rt::diag::validating

#endif  // SEEML_RUNTIME_DIAGNOSTICS_VALIDATING_ERROR_H_
