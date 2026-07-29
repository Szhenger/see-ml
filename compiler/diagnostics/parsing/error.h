#ifndef SEEML_COMPILER_DIAGNOSTICS_PARSING_ERROR_H_
#define SEEML_COMPILER_DIAGNOSTICS_PARSING_ERROR_H_

#include <expected>
#include <string>
#include <string_view>

#include "compiler/diagnostics/diagnostic.h"

// =============================================================================
// parsing/ — errors formed while turning a decoded SMF graph into forward
// SIR (compiler/frontend/parser/: value resolution, per-op semantic checks,
// graph construction). By this stage the bytes are trusted; what can fail is
// meaning — an undefined tensor name, an operand-shape mismatch, a dangling
// model output. Failure discipline: name the op (and its SMF kind) so the
// user can find it in the model, and stop at the first semantic error.
// =============================================================================

namespace seeml::diag::parsing {

/// The one unit active in this process: sema + value_resolver + parser all
/// speak as the parser.
inline constexpr std::string_view kUnit = "Parser";

/// Graph-level failure: "Parser: <message>".
[[nodiscard]] inline std::unexpected<std::string> Error(std::string_view message) {
  return Fail(kUnit, message);
}

/// Op-shaped failure: "Parser: <kind> '<name>' <what>".
[[nodiscard]] inline std::unexpected<std::string> OpError(std::string_view kind,
                                                          std::string_view name,
                                                          std::string_view what) {
  std::string m;
  m.reserve(kind.size() + name.size() + what.size() + 4);
  m.append(kind).append(" '").append(name).append("' ").append(what);
  return Fail(kUnit, m);
}

}  // namespace seeml::diag::parsing

#endif  // SEEML_COMPILER_DIAGNOSTICS_PARSING_ERROR_H_
