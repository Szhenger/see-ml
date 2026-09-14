#ifndef SEEML_RUNTIME_EXECUTOR_METAL_BACKEND_H_
#define SEEML_RUNTIME_EXECUTOR_METAL_BACKEND_H_

#include <expected>
#include <memory>
#include <string>

#include "runtime/executor/backend.h"

// =============================================================================
// The Metal backend's factory (roadmap Project 5, G1b). Two translation
// units define these symbols: metal_backend.mm on Apple platforms (the real
// backend — zero-copy arena residency, batched command encoding, the GPU
// kernel library) and metal_backend_stub.cc everywhere else, where Metal
// does not exist. No other translation unit references a Metal symbol, so
// a package still builds with `c++` alone on Linux and Windows; the Metal
// path is an `#ifdef __APPLE__` plus a run-time device check, never a
// build requirement.
// =============================================================================

namespace seeml::update_rt {

/// Whether this runtime carries the Metal backend AND a device exists.
bool MetalBackendAvailable();

/// Builds the Metal backend, JIT-compiling the kernel library for the
/// default device. Fails with a one-line reason where unavailable.
[[nodiscard]] std::expected<std::unique_ptr<ExecutorBackend>, std::string>
CreateMetalBackend();

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_EXECUTOR_METAL_BACKEND_H_
