#ifndef SEEML_RUNTIME_EXECUTOR_KERNEL_POLICY_H_
#define SEEML_RUNTIME_EXECUTOR_KERNEL_POLICY_H_

#include <cmath>
#include <cstddef>

// =============================================================================
// Shared execution policy for the kernel family units (gemm / elementwise /
// activation / normalization / loss / optimizer).
//
// Parallel decomposition: every kernel splits its output across ParallelFor
// chunks whose boundaries are a pure function of the problem shape — never
// of the thread count — and each chunk writes only its own slice (or its own
// partial-reduction slot). Consequences, by construction:
//   - no data races: every output element has exactly one writer;
//   - bitwise determinism: per-element arithmetic order is that of the
//     serial loop within a chunk, and reductions combine per-chunk partials
//     in chunk order, so any thread count computes identical bits.
//
// Aliasing: no kernel is ever invoked with overlapping source/destination
// buffers (the compiler's arena allocator guarantees it), so
// SEEML_RESTRICT-qualified pointers let the inner loops vectorize without
// runtime alias checks.
// =============================================================================

#if defined(_MSC_VER)
#define SEEML_RESTRICT __restrict
#else
#define SEEML_RESTRICT __restrict__
#endif

namespace seeml::update_rt::kernels {

// Minimum inner-loop iterations per chunk for cheap (add/mul-class) bodies
// and for transcendental (exp/tanh/sqrt-class) bodies, so small tensors
// (LoRA adapters, loss scalars) never leave the calling thread.
inline constexpr size_t kGrainCheap = 32768;
inline constexpr size_t kGrainMath = 4096;

// Rows per chunk for a kernel whose per-row cost is `per_row` operations.
inline size_t RowGrain(size_t per_row, size_t budget) {
  return per_row == 0 ? budget : (budget / per_row > 0 ? budget / per_row : 1);
}

inline size_t MinZ(size_t a, size_t b) { return a < b ? a : b; }

// --- Activation expressions --------------------------------------------------
// One definition each, shared by the standalone activation kernels and the
// fused GEMM epilogues. Sharing the expression (not just the formula) is
// what makes fusion bitwise-neutral: a fused chain evaluates exactly the
// floats the unfused instruction sequence would.

inline float SigmoidExpr(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// gelu(x) = 0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³))) — the tanh
// approximation.
inline constexpr float kGeluC = 0.7978845608028654f;  // √(2/π)
inline constexpr float kGeluA = 0.044715f;

inline float ReluExpr(float x) { return x > 0.0f ? x : 0.0f; }

inline float GeluExpr(float x) {
  const float t = std::tanh(kGeluC * (x + kGeluA * x * x * x));
  return 0.5f * x * (1.0f + t);
}

inline float SiluExpr(float x) { return x * SigmoidExpr(x); }

}  // namespace seeml::update_rt::kernels

#endif  // SEEML_RUNTIME_EXECUTOR_KERNEL_POLICY_H_
