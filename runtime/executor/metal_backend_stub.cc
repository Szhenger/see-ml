#include "runtime/executor/metal_backend.h"

// =============================================================================
// The Metal backend where Metal does not exist: `--backend metal` is a
// loud error, `--backend auto` degrades to the CPU. Compiled on every
// non-Apple host (and on Apple hosts built with SEEML_NO_METAL); the real
// backend lives in metal_backend.mm.
// =============================================================================

namespace seeml::update_rt {

bool MetalBackendAvailable() { return false; }

std::expected<std::unique_ptr<ExecutorBackend>, std::string>
CreateMetalBackend() {
  return std::unexpected(
      "the Metal backend is not built into this runtime (Apple platforms "
      "only; SEEML_NO_METAL=1 disables it there)");
}

}  // namespace seeml::update_rt
