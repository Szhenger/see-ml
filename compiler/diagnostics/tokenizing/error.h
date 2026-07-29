#ifndef SEEML_COMPILER_DIAGNOSTICS_TOKENIZING_ERROR_H_
#define SEEML_COMPILER_DIAGNOSTICS_TOKENIZING_ERROR_H_

#include <expected>
#include <string>
#include <string_view>

#include "compiler/diagnostics/diagnostic.h"

// =============================================================================
// tokenizing/ — errors formed while decoding or encoding the SMF byte
// stream (compiler/frontend/ingressor/). This is the lexical layer of the
// compilation: nothing here understands the graph, only the container —
// magic, version, section geometry, tensor payload bounds. Failure discipline:
// reject the file with the path in hand; never read past a bound.
// =============================================================================

namespace seeml::diag::tokenizing {

/// The SMF container reader/writer.
inline constexpr std::string_view kContainer = "SMF";
/// The ingress feasibility gate (resource_analyzer).
inline constexpr std::string_view kIngressor = "Ingressor";

/// Container-level failure: "SMF: <message>".
[[nodiscard]] inline std::unexpected<std::string> Error(std::string_view message) {
  return Fail(kContainer, message);
}

/// File-shaped failure: "SMF: <what> '<path>'".
[[nodiscard]] inline std::unexpected<std::string> FileError(std::string_view what,
                                                            std::string_view path) {
  std::string m;
  m.reserve(what.size() + path.size() + 3);
  m.append(what).append(" '").append(path).append("'");
  return Fail(kContainer, m);
}

/// Tensor-shaped failure: "SMF: tensor '<name>' <what>".
[[nodiscard]] inline std::unexpected<std::string> TensorError(std::string_view name,
                                                              std::string_view what) {
  std::string m;
  m.reserve(name.size() + what.size() + 10);
  m.append("tensor '").append(name).append("' ").append(what);
  return Fail(kContainer, m);
}

/// Feasibility failure from the ingress gate: "Ingressor: <message>".
[[nodiscard]] inline std::unexpected<std::string> IngressError(std::string_view message) {
  return Fail(kIngressor, message);
}

}  // namespace seeml::diag::tokenizing

#endif  // SEEML_COMPILER_DIAGNOSTICS_TOKENIZING_ERROR_H_
