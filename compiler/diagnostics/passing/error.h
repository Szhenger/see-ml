#ifndef SEEML_COMPILER_DIAGNOSTICS_PASSING_ERROR_H_
#define SEEML_COMPILER_DIAGNOSTICS_PASSING_ERROR_H_

#include <cstddef>
#include <expected>
#include <source_location>
#include <string>
#include <string_view>

#include "compiler/diagnostics/diagnostic.h"

// =============================================================================
// passing/ — errors formed while orchestrating and lowering SIR in the
// analysis subsystem, outside the analytic methods themselves
// (compiler/analysis/updater/): the pass manager's invariant gate and the
// legality checks of structural lowerings such as im2col conv rewriting.
// Failure discipline: a pass's own error is propagated verbatim (tests match
// it exactly); corruption found *after* a pass is attributed to that pass by
// name so the offender is never ambiguous.
// =============================================================================

namespace seeml::diag::passing {

inline constexpr std::string_view kPassManager = "PassManager";
inline constexpr std::string_view kConvLowering = "ConvLowering";

/// Post-pass verification failure, attributed to the offending pass:
/// "PassManager: SIR invariants violated after pass '<pass>': <why>".
[[nodiscard]] inline std::unexpected<std::string> InvariantsViolated(
    std::string_view pass, std::string_view why) {
  std::string m;
  m.reserve(pass.size() + why.size() + 48);
  m.append("SIR invariants violated after pass '")
      .append(pass)
      .append("': ")
      .append(why);
  return Fail(kPassManager, m);
}

/// Lowering legality failure: "ConvLowering: '<op>' <what>".
[[nodiscard]] inline std::unexpected<std::string> LoweringError(std::string_view op,
                                                                std::string_view what) {
  std::string m;
  m.reserve(op.size() + what.size() + 3);
  m.append("'").append(op).append("' ").append(what);
  return Fail(kConvLowering, m);
}

/// Progress note after a pass runs clean:
/// "PassManager: pass '<pass>' ok (<ops> ops)".
inline void PassNote(std::string_view pass, size_t num_ops,
                     const std::source_location loc = std::source_location::current()) {
  std::string m;
  m.reserve(pass.size() + 24);
  m.append("pass '").append(pass).append("' ok (")
      .append(std::to_string(num_ops)).append(" ops)");
  Note(kPassManager, m, loc);
}

}  // namespace seeml::diag::passing

#endif  // SEEML_COMPILER_DIAGNOSTICS_PASSING_ERROR_H_
