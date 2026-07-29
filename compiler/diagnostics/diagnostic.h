#ifndef SEEML_COMPILER_DIAGNOSTICS_DIAGNOSTIC_H_
#define SEEML_COMPILER_DIAGNOSTICS_DIAGNOSTIC_H_

#include <expected>
#include <source_location>
#include <string>
#include <string_view>

#include "compiler/diagnostics/logger.h"

// =============================================================================
// Diagnostics — error handling partitioned by the active processes of the
// compilation system. Every compiler diagnostic is one line,
//
//     "<unit>: <message>"
//
// where the *unit* names the component that spoke and the *process module*
// that formed the message names the compilation stage it belongs to:
//
//   tokenizing/    decoding and encoding the SMF byte stream (ingressor/)
//   parsing/       SMF graph -> forward SIR construction (parser/)
//   passing/       pass orchestration and lowering legality (analysis/updater/)
//   updating/      the analytic methods — autodiff, LoRA grafting, merge
//                  synthesis (analysis/calculus/ + analysis/algebra/)
//   architecting/  local device analysis — ISA/cache detection and the
//                  tiling contract (backend/architecture/)
//   generating/    code generation — arena binding, instruction lowering,
//                  native packaging, and the AOT driver (backend/trainer/)
//
// Each process module is header-only: it holds the registry of unit names
// active in that process plus builders for the message shapes the process
// repeats. Errors travel as std::expected<T, std::string> (the codebase
// idiom); builders return std::unexpected so they slot into any such return.
// The Logger below the builders is the only stateful piece.
// =============================================================================

namespace seeml::diag {

/// Forms the canonical one-line diagnostic for `unit`.
[[nodiscard]] inline std::unexpected<std::string> Fail(std::string_view unit,
                                                       std::string_view message) {
  std::string s;
  s.reserve(unit.size() + 2 + message.size());
  s.append(unit).append(": ").append(message);
  return std::unexpected(std::move(s));
}

/// Progress note from `unit` (INFO). The call site's location is preserved.
inline void Note(std::string_view unit, std::string_view message,
                 const std::source_location loc = std::source_location::current()) {
  std::string s;
  s.reserve(unit.size() + 2 + message.size());
  s.append(unit).append(": ").append(message);
  Logger::Info(s, loc);
}

/// Degraded-but-continuing condition in `unit` (WARN) — used where a process
/// falls back to a conservative default instead of failing.
inline void Fallback(std::string_view unit, std::string_view message,
                     const std::source_location loc = std::source_location::current()) {
  std::string s;
  s.reserve(unit.size() + 2 + message.size());
  s.append(unit).append(": ").append(message);
  Logger::Warn(s, loc);
}

}  // namespace seeml::diag

#endif  // SEEML_COMPILER_DIAGNOSTICS_DIAGNOSTIC_H_
