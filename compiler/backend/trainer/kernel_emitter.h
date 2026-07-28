#ifndef SEEML_COMPILER_BACKEND_TRAINER_KERNEL_EMITTER_H_
#define SEEML_COMPILER_BACKEND_TRAINER_KERNEL_EMITTER_H_

#include <string>

#include "compiler/backend/architecture/host_arch.h"

// =============================================================================
// Kernel emitter — the GPU half of the trainer's code generation. Emits
// Metal Shading Language source for the GEMM-family kernels the training
// program spends its time in (forward matmul, the dX/dW backward variants,
// and the merge's scaled GEMM-accumulate), threadgroup-tiled from the same
// tiling geometry the architecture analysis / tuner selected for the host.
// Text generation only: the emitted .metal file is a build input for a GPU
// runtime, compiled by the device toolchain — this module never executes it.
// =============================================================================

namespace seeml::update {

/// Threadgroup tile geometry for the emitted kernels: the host tiling
/// re-clamped to GPU threadgroup limits (tiles are square-ish and small;
/// a threadgroup covers tile_m x tile_n outputs and marches over K in
/// tile_k-deep panels staged through threadgroup memory).
struct GpuTiling {
  size_t tile_m = 16;
  size_t tile_n = 16;
  size_t tile_k = 16;
};

/// Deterministic mapping from the host tiling: each dimension clamped to
/// [8, 32] and rounded down to a multiple of 8 (SIMD-group friendly).
GpuTiling GpuTilingFromHost(const GemmTiling& host);

/// The .metal translation unit: seeml_matmul (C = A@B), seeml_matmul_nt
/// (C = A@B^T), seeml_matmul_tn (C = A^T@B), and seeml_gemm_acc
/// (C += alpha * A@B), all f32, all tiled per `tiling`.
std::string EmitMetalKernels(const GpuTiling& tiling);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_BACKEND_TRAINER_KERNEL_EMITTER_H_
