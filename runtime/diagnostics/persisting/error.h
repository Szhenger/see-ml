#ifndef SEEML_RUNTIME_DIAGNOSTICS_PERSISTING_ERROR_H_
#define SEEML_RUNTIME_DIAGNOSTICS_PERSISTING_ERROR_H_

#include <expected>
#include <string>
#include <string_view>

#include "runtime/diagnostics/diagnostic.h"

// =============================================================================
// persisting/ — errors formed while putting durable state on disk or taking
// it back (runtime/custodian/): the fsync'd atomic write path shared by
// model commits and checkpoints, and the checkpoint container's binding
// checks. Failure discipline: name the path for I/O failures (the fix is on
// disk), and reject a foreign or corrupt checkpoint before a single byte
// reaches the arena.
// =============================================================================

namespace seeml::update_rt::diag::persisting {

inline constexpr std::string_view kDurableIo = "DurableIO";
inline constexpr std::string_view kCheckpoint = "Checkpoint";

/// Persistence failure in `unit`: "<unit>: <message>".
[[nodiscard]] inline std::unexpected<std::string> Error(std::string_view unit,
                                                        std::string_view message) {
  return Fail(unit, message);
}

/// I/O-shaped failure: "<unit>: <what> '<path>'".
[[nodiscard]] inline std::unexpected<std::string> FileError(std::string_view unit,
                                                            std::string_view what,
                                                            std::string_view path) {
  std::string m;
  m.reserve(what.size() + path.size() + 3);
  m.append(what).append(" '").append(path).append("'");
  return Fail(unit, m);
}

}  // namespace seeml::update_rt::diag::persisting

#endif  // SEEML_RUNTIME_DIAGNOSTICS_PERSISTING_ERROR_H_
