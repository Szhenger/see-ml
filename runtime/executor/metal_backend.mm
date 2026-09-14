#include "runtime/executor/metal_backend.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Placeholder for phase G1b-1: the device probe is real; the backend itself
// lands with G1b-2/3/4 (zero-copy residency, batched encoding, the kernel
// library). Until then `--backend metal` reports why it cannot run and
// `--backend auto` degrades to the CPU with that reason.

namespace seeml::update_rt {

bool MetalBackendAvailable() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device != nil;
  }
}

std::expected<std::unique_ptr<ExecutorBackend>, std::string>
CreateMetalBackend() {
  return std::unexpected("the Metal backend's kernel library is not built yet");
}

}  // namespace seeml::update_rt
