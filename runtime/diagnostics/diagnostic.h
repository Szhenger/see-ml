#ifndef SEEML_RUNTIME_DIAGNOSTICS_DIAGNOSTIC_H_
#define SEEML_RUNTIME_DIAGNOSTICS_DIAGNOSTIC_H_

#include <expected>
#include <string>
#include <string_view>

// =============================================================================
// Runtime diagnostics — error handling partitioned by the active processes
// of the on-device update, mirroring compiler/diagnostics/. Every runtime
// diagnostic is one line,
//
//     "<unit>: <message>"
//
// where the *unit* names the component that spoke and the *process module*
// that formed the message names the update stage it belongs to:
//
//   feeding/     dataset decode and batch staging (feeder/)
//   validating/  load-time plan verification (validator/)
//   executing/   the engine's lifecycle and dispatch (engine/)
//   persisting/  durable state — checkpoints, atomic commits (custodian/)
//
// Each process module is header-only: the registry of unit names active in
// that process plus builders for its message shapes. Errors travel as
// std::expected<T, std::string>. Unlike the compiler's core there is no
// logger here — the runtime is zero-dependency and reports progress through
// the engine's own stderr lines — so the core is message formation only.
// =============================================================================

namespace seeml::update_rt::diag {

/// Forms the canonical one-line diagnostic for `unit`.
[[nodiscard]] inline std::unexpected<std::string> Fail(std::string_view unit,
                                                       std::string_view message) {
  std::string s;
  s.reserve(unit.size() + 2 + message.size());
  s.append(unit).append(": ").append(message);
  return std::unexpected(std::move(s));
}

}  // namespace seeml::update_rt::diag

#endif  // SEEML_RUNTIME_DIAGNOSTICS_DIAGNOSTIC_H_
