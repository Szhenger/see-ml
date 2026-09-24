#ifndef SEEML_COMPILER_DIAGNOSTICS_ARCHITECTING_ERROR_H_
#define SEEML_COMPILER_DIAGNOSTICS_ARCHITECTING_ERROR_H_

#include <expected>
#include <source_location>
#include <string>
#include <string_view>

#include "compiler/diagnostics/diagnostic.h"

// =============================================================================
// architecting/ — handling for local device analysis
// (compiler/backend/architecture/) and the kernel-policy table the offline
// tuner measured for it (compiler/backend/architecture/). Detection can never
// hard-fail — a machine that hides its cache geometry must still compile —
// so this process has two disciplines:
//
//   * Fallback (WARN): a sysctl/sysconf probe came back empty and a
//     conservative default was assumed, or a policy table names no entry
//     for this host and the kernel defaults apply. The compilation
//     continues; the user learns why.
//   * Error: a *contract* violation — a tiling that lies about fitting the
//     detected cache hierarchy, a policy table that does not parse or
//     carries a geometry the kernels reject. Hard errors, because a bad
//     policy silently costs every training step.
// =============================================================================

namespace seeml::diag::architecting {

inline constexpr std::string_view kHostArch = "HostArch";
inline constexpr std::string_view kKernelPolicy = "KernelPolicy";

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
