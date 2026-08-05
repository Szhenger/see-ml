#ifndef SEEML_RUNTIME_EXECUTOR_METAL_GEMM_H_
#define SEEML_RUNTIME_EXECUTOR_METAL_GEMM_H_

#include <cstddef>
#include <expected>
#include <memory>
#include <string>

// =============================================================================
// Metal GEMM dispatch (G1a, Apple platforms only): JIT-compiles the kernel
// emitter's .metal source for the local device and dispatches the four
// GEMM-family kernels against it. This is the correctness harness that
// proves the emitted MSL on real hardware — copy-in/copy-out buffers, one
// synchronous command per call. The throughput path (zero-copy arena
// buffers, batched command encoding, engine integration behind a backend
// switch) is roadmap Project 5 phases b/c; nothing here is vendored into
// emitted packages yet.
//
// Determinism contract: dispatch is bitwise-reproducible run-to-run on the
// same device (fixed grid geometry, fixed K-loop order), but NOT bitwise-
// equal to the CPU kernels — the GPU contracts FMA differently. Cross-
// backend comparisons are tolerance-based; the per-backend determinism
// doctrine is documented in docs/roadmap.md.
// =============================================================================

namespace seeml::update_rt {

class MetalGemmRunner {
 public:
  /// Whether a Metal device exists on this machine. False on non-Apple
  /// builds (the stub translation unit) and on Metal-less hosts.
  static bool Available();

  /// JIT-compiles `msl_source` (EmitMetalKernels output) and builds the
  /// four compute pipelines. tile_m/tile_n must match the GpuTiling the
  /// source was emitted with — they define the threadgroup shape.
  static std::expected<std::unique_ptr<MetalGemmRunner>, std::string> Create(
      const std::string& msl_source, size_t tile_m, size_t tile_n);

  enum class Kind {
    kNN,     // C = A[M,K] @ B[K,N]
    kNT,     // C = A[M,K] @ B[N,K]^T
    kTN,     // C = A[K,M]^T @ B[K,N]
    kAccNN,  // C += alpha * A @ B
  };

  /// Synchronous dispatch; C is read back before returning. For kAccNN the
  /// prior contents of C are uploaded first.
  [[nodiscard]] std::expected<void, std::string> Run(Kind kind, const float* a,
                                                     const float* b, float* c,
                                                     size_t m, size_t n,
                                                     size_t k,
                                                     float alpha = 1.0f);

  ~MetalGemmRunner();
  MetalGemmRunner(const MetalGemmRunner&) = delete;
  MetalGemmRunner& operator=(const MetalGemmRunner&) = delete;

 private:
  struct Impl;
  explicit MetalGemmRunner(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_EXECUTOR_METAL_GEMM_H_
