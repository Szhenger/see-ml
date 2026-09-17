#ifndef SEEML_RUNTIME_EXECUTOR_KERNEL_POLICY_H_
#define SEEML_RUNTIME_EXECUTOR_KERNEL_POLICY_H_

#include <cmath>
#include <cstddef>

#include "source/plan/schema.h"  // kDefaultGemmPanelFloats

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

// --- GEMM cache tiles --------------------------------------------------------
// The blocked CPU GEMM cores walk K in tiles of `k` and N in tiles of `n`
// (gemm.cc). The geometry is a throughput knob only, never a bit: the N
// tile picks traversal order, not reduction grouping, and the K tile keeps
// the 4-wide unroll groups aligned as long as it stays a multiple of the
// unroll width — which the validator proves on the plan header before any
// kernel runs, and the compiled-in default asserts below. The compiler
// decides the geometry (a measured, host-keyed kernel-policy table via
// tool/autotune.py, or the defaults here) and writes it into the plan
// header (schema.h, v11); zero there selects these defaults, which a
// package's build.sh may still override with -DSEEML_GEMM_TILE_K/N.
#ifndef SEEML_GEMM_TILE_K
#define SEEML_GEMM_TILE_K 64
#endif
#ifndef SEEML_GEMM_TILE_N
#define SEEML_GEMM_TILE_N 256
#endif
static_assert(SEEML_GEMM_TILE_K > 0 && SEEML_GEMM_TILE_K % 4 == 0,
              "the K tile must be a positive multiple of the 4-wide unroll "
              "so reduction grouping — and therefore every bit of every "
              "result — is independent of the tiling");
static_assert(SEEML_GEMM_TILE_N > 0, "the N tile must be positive");

struct GemmTiles {
  size_t k = SEEML_GEMM_TILE_K;
  size_t n = SEEML_GEMM_TILE_N;
  bool operator==(const GemmTiles&) const = default;
};
inline constexpr GemmTiles kDefaultGemmTiles{};

/// Whether a tile pair is one the kernels accept (the header contract).
inline bool GemmTilesValid(const GemmTiles& t) {
  return t.k > 0 && t.k % 4 == 0 && t.n > 0;
}

// The packed B panel of the blocked NN core (gemm.cc): a fixed local
// buffer the active tile is copied — and, for int8 / bf16 weights, widened
// — into, once per tile. 32 KiB by default: within any thread stack the
// runtime targets, and measured flat in throughput against 64 KiB. A
// smaller-stack target may shrink it (-DSEEML_GEMM_PANEL_FLOATS=...); like
// the tiles, it is traversal only, never bits.
#ifndef SEEML_GEMM_PANEL_FLOATS
#define SEEML_GEMM_PANEL_FLOATS seeml::update::kDefaultGemmPanelFloats
#endif
inline constexpr size_t kGemmPanelFloats = SEEML_GEMM_PANEL_FLOATS;
static_assert(kGemmPanelFloats >= 4,
              "the packed B panel must hold at least one k-quad column");

/// The tile geometry the NN core actually walks for a header's tiles: K
/// shrunk to a multiple of the quad that fits the panel (quads stay
/// aligned to k = 0), then N to what is left. A pure function of the
/// tiles — what seeml-bench records as the effective geometry and what the
/// compiler's analytic model (SuggestGemmTiling) is derived against.
inline GemmTiles FitToPanel(const GemmTiles& tiles) {
  GemmTiles fit;
  fit.k = MinZ(tiles.k, kGemmPanelFloats / 4 * 4) / 4 * 4;
  fit.n = MinZ(tiles.n, kGemmPanelFloats / fit.k);
  return fit;
}

/// Everything a backend may be told about how to run a plan's kernels
/// beyond the instruction stream itself. Set by the engine from the plan
/// header after Bind (ExecutorBackend::Configure).
struct KernelPolicy {
  GemmTiles gemm_tiles = kDefaultGemmTiles;
};

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
