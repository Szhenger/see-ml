#ifndef SEEML_COMPILER_DIAGNOSTICS_ARCHITECTING_ERROR_H_
#define SEEML_COMPILER_DIAGNOSTICS_ARCHITECTING_ERROR_H_

#include <expected>
#include <source_location>
#include <string>
#include <string_view>

#include "compiler/diagnostics/diagnostic.h"

// =============================================================================
// architecting/ — handling for local device analysis
// (compiler/backend/architecture/ and the tuner that refines its hints).
// Detection can never hard-fail — a machine that hides its cache geometry
// must still compile — so this process has two disciplines:
//
//   * Fallback (WARN): a sysctl/sysconf probe came back empty and a
//     conservative default was assumed. The compilation continues; the user
//     learns why the tiling may be conservative.
//   * Error: a tiling *contract* violation — a hint claimed to fit the
//     detected cache hierarchy but does not. These are hard errors because a
//     bad hint silently costs every training step.
// =============================================================================

namespace seeml::diag::architecting {

inline constexpr std::string_view kHostArch = "HostArch";
inline constexpr std::string_view kAutotuner = "Autotuner";

/// Contract failure in device analysis: "<unit>: <message>".
[[nodiscard]] inline std::unexpected<std::string> Error(std::string_view unit,
                                                        std::string_view message) {
  return Fail(unit, message);
}

/// Detection fell back to a conservative default (WARN, compilation
/// continues): "<unit>: <message>".
inline void DetectionFallback(std::string_view unit, std::string_view message,
                              const std::source_location loc =
                                  std::source_location::current()) {
  Fallback(unit, message, loc);
}

}  // namespace seeml::diag::architecting

#endif  // SEEML_COMPILER_DIAGNOSTICS_ARCHITECTING_ERROR_H_
