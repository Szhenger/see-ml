#include "runtime/executor/metal_gemm.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>

// =============================================================================
// Objective-C++ implementation of the Metal GEMM harness (see the header
// for scope and the determinism contract). Compiled only on Apple hosts;
// every entry point reports a one-line error string on failure instead of
// throwing across the language boundary.
// =============================================================================

namespace seeml::update_rt {

namespace {

std::string NsError(NSError* err, const char* what) {
  std::string s = what;
  if (err && err.localizedDescription)
    s += std::string(": ") + err.localizedDescription.UTF8String;
  return s;
}

}  // namespace

struct MetalGemmRunner::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline[4] = {nil, nil, nil, nil};
  size_t tile_m = 0, tile_n = 0;
};

bool MetalGemmRunner::Available() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device != nil;
  }
}

std::expected<std::unique_ptr<MetalGemmRunner>, std::string>
MetalGemmRunner::Create(const std::string& msl_source, size_t tile_m,
                        size_t tile_n) {
  @autoreleasepool {
    auto impl = std::make_unique<Impl>();
    impl->tile_m = tile_m;
    impl->tile_n = tile_n;
    impl->device = MTLCreateSystemDefaultDevice();
    if (!impl->device) return std::unexpected("no Metal device available");
    impl->queue = [impl->device newCommandQueue];
    if (!impl->queue) return std::unexpected("cannot create command queue");

    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:msl_source.c_str()];
    id<MTLLibrary> lib = [impl->device newLibraryWithSource:src
                                                    options:nil
                                                      error:&err];
    if (!lib)
      return std::unexpected(NsError(err, "MSL compilation failed"));

    const char* names[4] = {"seeml_matmul", "seeml_matmul_nt",
                            "seeml_matmul_tn", "seeml_gemm_acc"};
    for (int i = 0; i < 4; ++i) {
      id<MTLFunction> fn =
          [lib newFunctionWithName:[NSString stringWithUTF8String:names[i]]];
      if (!fn)
        return std::unexpected(std::string("missing kernel '") + names[i] +
                               "' in emitted source");
      impl->pipeline[i] = [impl->device newComputePipelineStateWithFunction:fn
                                                                      error:&err];
      if (!impl->pipeline[i])
        return std::unexpected(NsError(err, "pipeline creation failed"));
      if (impl->pipeline[i].maxTotalThreadsPerThreadgroup <
          tile_m * tile_n)
        return std::unexpected(
            "threadgroup " + std::to_string(tile_m) + "x" +
            std::to_string(tile_n) + " exceeds the device limit of " +
            std::to_string(impl->pipeline[i].maxTotalThreadsPerThreadgroup));
    }
    return std::unique_ptr<MetalGemmRunner>(
        new MetalGemmRunner(std::move(impl)));
  }
}

MetalGemmRunner::MetalGemmRunner(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MetalGemmRunner::~MetalGemmRunner() = default;

std::expected<void, std::string> MetalGemmRunner::Run(Kind kind,
                                                      const float* a,
                                                      const float* b, float* c,
                                                      size_t m, size_t n,
                                                      size_t k, float alpha) {
  @autoreleasepool {
    // Element counts per variant: A always carries M*K values, B always
    // K*N, C M*N — the transposes only change the index expressions.
    const size_t a_bytes = m * k * sizeof(float);
    const size_t b_bytes = k * n * sizeof(float);
    const size_t c_bytes = m * n * sizeof(float);
    id<MTLBuffer> ab = [impl_->device newBufferWithBytes:a
                                                  length:a_bytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [impl_->device newBufferWithBytes:b
                                                  length:b_bytes
                                                 options:MTLResourceStorageModeShared];
    // kAccNN reads C's prior contents; the pure GEMMs overwrite every
    // element, but uploading is cheaper than reasoning about it.
    id<MTLBuffer> cb = [impl_->device newBufferWithBytes:c
                                                  length:c_bytes
                                                 options:MTLResourceStorageModeShared];
    if (!ab || !bb || !cb) return std::unexpected("buffer allocation failed");

    struct {
      uint32_t m, n, k;
      float alpha;
    } dims = {static_cast<uint32_t>(m), static_cast<uint32_t>(n),
              static_cast<uint32_t>(k), alpha};

    id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:impl_->pipeline[static_cast<int>(kind)]];
    [enc setBuffer:ab offset:0 atIndex:0];
    [enc setBuffer:bb offset:0 atIndex:1];
    [enc setBuffer:cb offset:0 atIndex:2];
    [enc setBytes:&dims length:sizeof(dims) atIndex:3];
    // One threadgroup per (tile_n x tile_m) output tile; the kernel guards
    // the ragged edge, but the staging loops need full threadgroups.
    const MTLSize tg = MTLSizeMake(impl_->tile_n, impl_->tile_m, 1);
    const MTLSize grid = MTLSizeMake((n + impl_->tile_n - 1) / impl_->tile_n,
                                     (m + impl_->tile_m - 1) / impl_->tile_m,
                                     1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status == MTLCommandBufferStatusError)
      return std::unexpected(NsError(cmd.error, "GPU dispatch failed"));

    std::memcpy(c, cb.contents, c_bytes);
    return {};
  }
}

}  // namespace seeml::update_rt
