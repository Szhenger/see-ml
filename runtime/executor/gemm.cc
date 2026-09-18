#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "runtime/executor/kernel_policy.h"
#include "runtime/executor/update_kernels.h"
#include "source/parallel/parallel_for.h"

// =============================================================================
// GEMM family: the four f32 variants and the dequantizing q8 / widening
// bf16 variants, all reduced to two blocked cores. The parallel wrappers
// partition C into a (row band x column band) grid of tasks, so every
// worker owns a disjoint cell of C and each element's arithmetic order
// matches the serial loop exactly.
// =============================================================================

namespace seeml::update_rt::kernels {

namespace up = seeml::update;

namespace {

// Tile geometry (kernel_policy.h): the K tile bounds the k0 panel a pass
// over C folds in, the N tile the column sweep; both arrive per call from
// the plan header through the CPU backend (the compiled-in defaults when
// the header says zero). Throughput only, never bits — see GemmTiles.

// Shared blocked core over the C-row range [m_begin, m_end):
// C[m,N] (+)= alpha * A[m,K] @ B[K,N] with B row-major. GemmNN/GemmAccNN/
// GemmTN and the q8 / bf16 variants all reduce to this loop nest. A/B/C
// must not overlap (guaranteed by the arena allocator, which never reuses
// an operand's slot for a result born at the same instruction).
//
// THE BITS. Per element the reduction is, and has always been,
//     c += a0*b0 + a1*b1 + a2*b2 + a3*b3
// over consecutive k-quads aligned to k = 0 (the K tile is a multiple of
// 4), then single terms for the K % 4 tail. Everything below is traversal
// around that one expression, written out identically wherever it appears,
// so every form of this kernel — any tiling, any row grouping, packed or
// not, any thread count — computes the same floats.
//
// THE SPEED (E1, #80). Two things the first blocked nest left on the table:
//
//   Register blocking over M. Four rows of C share each pass over a B
//   quad: the four B loads that fed 4 FMAs now feed 16, and four C rows
//   ride the sweep together. Rows are independent, so grouping them cannot
//   change a bit — a chunk's leftover rows take the one-row form.
//
//   A packed panel. The active B tile is copied once per (k0, n0) into a
//   contiguous stack panel and then swept by every row block of the chunk.
//   B's rows are N apart in memory, so the four streams of a quad (and C's
//   four) sit at strides of N floats: at power-of-two widths they alias in
//   L1 (the lm-head shape lost 35% to it), and int8 / bf16 weights were
//   re-widened on every one of those sweeps. The panel is unit-stride and
//   f32: widening happens once per tile, exactly (a widened int8 or bf16 is
//   the same float whenever it is widened), and the sweep never sees N.
//
// The panel is a fixed 32 KiB local — no allocation, within any thread
// stack the runtime targets — and the N tile is clamped so a tile always
// fits it; a geometry change is a traversal change, never a bit. (The one
// buffer a GEMM kernel owns is GemmNTQ8's per-column form: a per-thread
// vector, at most 1 MiB, that grows once and is reused — see there.) Measured
// on one Apple M5 core, bit-identical on every shape: 1.6-3.2x on f32
// (30 -> 54 GFLOP/s at D=512 projections, 14 -> 46 at SmolLM's 49k-vocab
// head) and 2.2-2.5x on int8 weights.
inline constexpr size_t kGemmRowBlock = 4;
// A's orientation is a template parameter, not a run-time flag: the row
// block reads sixteen A scalars per quad, and a branch in each of them
// (even a perfectly predicted one) keeps the compiler from hoisting the
// row addresses — measured, it cost the whole register-blocking gain.
template <bool kTransposedA, typename BType>
void BlockedNNImpl(const float* SEEML_RESTRICT A,
                   const BType* SEEML_RESTRICT B, float* SEEML_RESTRICT C,
                   size_t m_begin, size_t m_end, size_t n_begin, size_t n_end,
                   size_t N, size_t K, float alpha, size_t a_stride,
                   const GemmTiles& tiles,
                   const float* SEEML_RESTRICT col_scale) {
  auto a_at = [&](size_t k, size_t m) {
    if constexpr (kTransposedA)
      return A[k * a_stride + m];
    else
      return A[m * a_stride + k];
  };
  const GemmTiles fit = FitToPanel(tiles);
  const size_t tile_k = fit.k, tile_n = fit.n;
  float panel[kGemmPanelFloats];

  for (size_t k0 = 0; k0 < K; k0 += tile_k) {
    const size_t k1 = MinZ(k0 + tile_k, K);
    for (size_t n0 = n_begin; n0 < n_end; n0 += tile_n) {
      const size_t w = MinZ(n0 + tile_n, n_end) - n0;
      // Pack (and widen) B[k0:k1, n0:n0+w] row by row, unit stride. With
      // per-column int8 scales (v17) the scale joins the widening: once per
      // panel element, reused by every row the panel serves.
      if (col_scale) {
        const float* SEEML_RESTRICT cs = col_scale + n0;
        for (size_t k = k0; k < k1; ++k) {
          const BType* SEEML_RESTRICT src = B + k * N + n0;
          float* SEEML_RESTRICT dst = panel + (k - k0) * w;
          for (size_t n = 0; n < w; ++n)
            dst[n] = static_cast<float>(src[n]) * cs[n];
        }
      } else {
        for (size_t k = k0; k < k1; ++k) {
          const BType* SEEML_RESTRICT src = B + k * N + n0;
          float* SEEML_RESTRICT dst = panel + (k - k0) * w;
          for (size_t n = 0; n < w; ++n) dst[n] = static_cast<float>(src[n]);
        }
      }

      size_t m = m_begin;
      for (; m + kGemmRowBlock <= m_end; m += kGemmRowBlock) {
        float* SEEML_RESTRICT c0 = C + (m + 0) * N + n0;
        float* SEEML_RESTRICT c1 = C + (m + 1) * N + n0;
        float* SEEML_RESTRICT c2 = C + (m + 2) * N + n0;
        float* SEEML_RESTRICT c3 = C + (m + 3) * N + n0;
        size_t k = k0;
        for (; k + 4 <= k1; k += 4) {
          const float a00 = alpha * a_at(k + 0, m + 0);
          const float a01 = alpha * a_at(k + 1, m + 0);
          const float a02 = alpha * a_at(k + 2, m + 0);
          const float a03 = alpha * a_at(k + 3, m + 0);
          const float a10 = alpha * a_at(k + 0, m + 1);
          const float a11 = alpha * a_at(k + 1, m + 1);
          const float a12 = alpha * a_at(k + 2, m + 1);
          const float a13 = alpha * a_at(k + 3, m + 1);
          const float a20 = alpha * a_at(k + 0, m + 2);
          const float a21 = alpha * a_at(k + 1, m + 2);
          const float a22 = alpha * a_at(k + 2, m + 2);
          const float a23 = alpha * a_at(k + 3, m + 2);
          const float a30 = alpha * a_at(k + 0, m + 3);
          const float a31 = alpha * a_at(k + 1, m + 3);
          const float a32 = alpha * a_at(k + 2, m + 3);
          const float a33 = alpha * a_at(k + 3, m + 3);
          const float* SEEML_RESTRICT b0 = panel + (k - k0) * w;
          const float* SEEML_RESTRICT b1 = b0 + w;
          const float* SEEML_RESTRICT b2 = b1 + w;
          const float* SEEML_RESTRICT b3 = b2 + w;
          for (size_t n = 0; n < w; ++n) {
            const float v0 = b0[n], v1 = b1[n], v2 = b2[n], v3 = b3[n];
            c0[n] += a00 * v0 + a01 * v1 + a02 * v2 + a03 * v3;
            c1[n] += a10 * v0 + a11 * v1 + a12 * v2 + a13 * v3;
            c2[n] += a20 * v0 + a21 * v1 + a22 * v2 + a23 * v3;
            c3[n] += a30 * v0 + a31 * v1 + a32 * v2 + a33 * v3;
          }
        }
        for (; k < k1; ++k) {  // the K % 4 tail, one term at a time
          const float a0 = alpha * a_at(k, m + 0);
          const float a1 = alpha * a_at(k, m + 1);
          const float a2 = alpha * a_at(k, m + 2);
          const float a3 = alpha * a_at(k, m + 3);
          const float* SEEML_RESTRICT b = panel + (k - k0) * w;
          for (size_t n = 0; n < w; ++n) {
            const float v = b[n];
            c0[n] += a0 * v;
            c1[n] += a1 * v;
            c2[n] += a2 * v;
            c3[n] += a3 * v;
          }
        }
      }
      for (; m < m_end; ++m) {  // the chunk's leftover rows, one at a time
        float* SEEML_RESTRICT c_row = C + m * N + n0;
        size_t k = k0;
        for (; k + 4 <= k1; k += 4) {
          const float a0 = alpha * a_at(k + 0, m);
          const float a1 = alpha * a_at(k + 1, m);
          const float a2 = alpha * a_at(k + 2, m);
          const float a3 = alpha * a_at(k + 3, m);
          const float* SEEML_RESTRICT b0 = panel + (k - k0) * w;
          const float* SEEML_RESTRICT b1 = b0 + w;
          const float* SEEML_RESTRICT b2 = b1 + w;
          const float* SEEML_RESTRICT b3 = b2 + w;
          for (size_t n = 0; n < w; ++n)
            c_row[n] += a0 * b0[n] + a1 * b1[n] + a2 * b2[n] + a3 * b3[n];
        }
        for (; k < k1; ++k) {
          const float a = alpha * a_at(k, m);
          const float* SEEML_RESTRICT b_row = panel + (k - k0) * w;
          for (size_t n = 0; n < w; ++n) c_row[n] += a * b_row[n];
        }
      }
    }
  }
}

template <typename BType>
void BlockedNN(const float* SEEML_RESTRICT A, const BType* SEEML_RESTRICT B,
               float* SEEML_RESTRICT C, size_t m_begin, size_t m_end,
               size_t n_begin, size_t n_end, size_t N, size_t K, float alpha,
               size_t a_stride, bool a_transposed, const GemmTiles& tiles,
               const float* col_scale = nullptr) {
  if (a_transposed)
    BlockedNNImpl<true>(A, B, C, m_begin, m_end, n_begin, n_end, N, K, alpha,
                        a_stride, tiles, col_scale);
  else
    BlockedNNImpl<false>(A, B, C, m_begin, m_end, n_begin, n_end, N, K, alpha,
                         a_stride, tiles, col_scale);
}

// The 2D partition of C (E1 stage 3). The first nest split C by rows alone,
// with a grain that handed each task a single row of a large GEMM — which
// defeats the row block and re-packs the panel once per row — and left a
// small-M GEMM (a LoRA dB: M = rank) with M tasks at most, one core's worth
// at rank 4. Tasks here are cells of a (row band x column band) grid:
//
//   row bands of kGemmBandRows rows (a multiple of the row block, so every
//   full band is all four-row blocks, and one panel pack serves the band);
//   column bands only when the row bands alone are too few to occupy the
//   pool, each a whole number of N tiles, so a task packs whole panels.
//
// The grid is a pure function of (M, N, K, tiles) — never of the worker
// count — and tasks are claimed dynamically, the ParallelFor contract.
// Each element of C belongs to exactly one task and its reduction runs
// over k alone, so no partition can change a bit.
inline constexpr size_t kGemmBandRows = 64;
inline constexpr size_t kGemmTaskWork = size_t{1} << 20;  // MACs per task, min
inline constexpr size_t kGemmWantTasks = 64;  // enough to balance any pool
static_assert(kGemmBandRows % kGemmRowBlock == 0);

struct GemmGrid {
  size_t band_rows = 1, bands_m = 1;
  size_t band_cols = 1, bands_n = 1;
  size_t tasks() const { return bands_m * bands_n; }
};

inline GemmGrid PlanGemmGrid(size_t M, size_t N, size_t K,
                             const GemmTiles& tiles) {
  GemmGrid g;
  g.band_rows = kGemmBandRows;
  // Never more tasks than the pool's chunk table holds.
  const size_t min_rows =
      (M + up::kMaxParallelChunks - 1) / up::kMaxParallelChunks;
  if (g.band_rows < min_rows)
    g.band_rows = (min_rows + kGemmRowBlock - 1) / kGemmRowBlock *
                  kGemmRowBlock;
  g.bands_m = (M + g.band_rows - 1) / g.band_rows;
  g.band_cols = N;
  g.bands_n = 1;
  // How many tasks the work is worth, and whether the rows fall short.
  const size_t per_row = N * K;
  const size_t work = per_row != 0 && M > SIZE_MAX / per_row
                          ? SIZE_MAX
                          : M * per_row;
  const size_t want = MinZ(kGemmWantTasks, work / kGemmTaskWork);
  const size_t tile_n = FitToPanel(tiles).n;
  if (g.bands_m < want && N > tile_n) {
    const size_t max_bands = (N + tile_n - 1) / tile_n;
    const size_t bands =
        MinZ(MinZ(max_bands, (want + g.bands_m - 1) / g.bands_m),
             up::kMaxParallelChunks / g.bands_m);
    if (bands > 1) {
      g.band_cols = ((N + bands - 1) / bands + tile_n - 1) / tile_n * tile_n;
      g.bands_n = (N + g.band_cols - 1) / g.band_cols;
    }
  }
  // Still short (a narrow, short C — an adapter's dB): thinner row bands,
  // down to one row block.
  if (g.tasks() < want && M > kGemmRowBlock) {
    const size_t bands = MinZ((want + g.bands_n - 1) / g.bands_n,
                              (M + kGemmRowBlock - 1) / kGemmRowBlock);
    const size_t rows = ((M + bands - 1) / bands + kGemmRowBlock - 1) /
                        kGemmRowBlock * kGemmRowBlock;
    if (rows < g.band_rows) {
      g.band_rows = rows;
      g.bands_m = (M + rows - 1) / rows;
    }
  }
  return g;
}

/// Runs body(m0, m1, n0, n1) over every cell of the grid.
template <typename Body>
void ForEachGemmTask(size_t M, size_t N, size_t K, const GemmTiles& tiles,
                     Body&& body) {
  if (M == 0 || N == 0) return;
  const GemmGrid g = PlanGemmGrid(M, N, K, tiles);
  up::ParallelFor(g.tasks(), 1, [&](size_t t0, size_t t1, size_t) {
    for (size_t t = t0; t < t1; ++t) {
      const size_t m0 = (t / g.bands_n) * g.band_rows;
      const size_t n0 = (t % g.bands_n) * g.band_cols;
      body(m0, MinZ(m0 + g.band_rows, M), n0, MinZ(n0 + g.band_cols, N));
    }
  });
}

/// Zeroes C[m0:m1, n0:n1] — a task's cell, ahead of its accumulation.
inline void ZeroCell(float* C, size_t N, size_t m0, size_t m1, size_t n0,
                     size_t n1) {
  if (n0 == 0 && n1 == N) {
    std::memset(C + m0 * N, 0, (m1 - m0) * N * sizeof(float));
    return;
  }
  for (size_t m = m0; m < m1; ++m)
    std::memset(C + m * N + n0, 0, (n1 - n0) * sizeof(float));
}

// Dot-product core shared by GemmNT and GemmNTQ8 over the C-row range
// [m_begin, m_end): C[m,n] = A row · B row. Four output columns per pass
// reuse the streamed A row from L1 four times.
//
// The K reduction runs in eight fixed lanes — lane l accumulates every
// k ≡ l (mod 8) — combined in lane order once the row is done (#66). A
// single serial chain per column cannot vectorize under the bitwise
// contract (the compiler may not reassociate), and ran 3–5x slower than
// GemmNN at equal FLOPs; eight independent chains give the vectorizer two
// full NEON/SSE registers per column with no reassociation at all: the
// lane assignment and the combine order are pure functions of K, never of
// the tiling or the thread count, so every width computes identical bits.
// The reduction order changed exactly once, here, by design — results
// differ from the pre-#66 serial chain, as any kernel change does.
//
// For int8 and bf16 B, each eight-wide block is widened to f32 by its own
// lane loop before the multiply-accumulate lane loop consumes it (#104).
// That split is the difference between a vectorized kernel and a scalar
// one: a static_cast inside the multiply-accumulate makes the vectorizer
// give up on the whole loop and work the accumulators in memory (the int8
// form ran 7x below the f32 form on every shape measured). Widening an
// int8 or a bf16 to f32 is exact, so widening first and multiplying second
// is the same arithmetic on the same values in the same lane order —
// bit-identical to the unsplit loop. The widening loop is written inline
// on purpose (through a helper taking the block by pointer, clang keeps
// the widened block in memory), and the f32 instantiation keeps the
// single-loop body it has always compiled: a no-op widening pass through
// a local array costs it 3x on the same compiler, so the split exists only
// where the element type needs it.
inline constexpr size_t kNtLanes = 8;
// Two rows of C ride each pass over a four-row B group when K is long
// enough to pay for the wider accumulator set (E1, #80): eight dot products
// share two A-row and four B-row streams — and, for int8 / bf16 B, one
// widening of each block — where four shared one and four. Every product
// still lands in lane k mod 8 of its own accumulator and every accumulator
// combines in lane order, so the pairing is invisible in the result; below
// the threshold the setup and combine of sixteen accumulators costs more
// than the sharing saves (measured 0.87x at K = 16, 1.5-1.7x from K = 203).
inline constexpr size_t kNtPairMinK = 64;

/// One (row, four columns) or (row, one column) group of the dot-product
/// core; the unpaired rows and columns of BlockedNT.
template <typename BType>
#if defined(__GNUC__) || defined(__clang__)
__attribute__((always_inline))
#endif
inline void NtRow(const float* SEEML_RESTRICT a_row,
                  const BType* SEEML_RESTRICT B, float* SEEML_RESTRICT c_row,
                  size_t n0, size_t n1, size_t K, float alpha) {
  size_t n = n0;
  for (; n + 4 <= n1; n += 4) {
    const BType* SEEML_RESTRICT b0 = B + (n + 0) * K;
    const BType* SEEML_RESTRICT b1 = B + (n + 1) * K;
    const BType* SEEML_RESTRICT b2 = B + (n + 2) * K;
    const BType* SEEML_RESTRICT b3 = B + (n + 3) * K;
    float acc0[kNtLanes] = {}, acc1[kNtLanes] = {}, acc2[kNtLanes] = {},
          acc3[kNtLanes] = {};
    size_t k = 0;
    for (; k + kNtLanes <= K; k += kNtLanes) {
      if constexpr (std::is_same_v<BType, float>) {
        for (size_t l = 0; l < kNtLanes; ++l) {
          const float a = a_row[k + l];
          acc0[l] += a * b0[k + l];
          acc1[l] += a * b1[k + l];
          acc2[l] += a * b2[k + l];
          acc3[l] += a * b3[k + l];
        }
      } else {
        float w0[kNtLanes], w1[kNtLanes], w2[kNtLanes], w3[kNtLanes];
        for (size_t l = 0; l < kNtLanes; ++l) {
          w0[l] = static_cast<float>(b0[k + l]);
          w1[l] = static_cast<float>(b1[k + l]);
          w2[l] = static_cast<float>(b2[k + l]);
          w3[l] = static_cast<float>(b3[k + l]);
        }
        for (size_t l = 0; l < kNtLanes; ++l) {
          const float a = a_row[k + l];
          acc0[l] += a * w0[l];
          acc1[l] += a * w1[l];
          acc2[l] += a * w2[l];
          acc3[l] += a * w3[l];
        }
      }
    }
    for (; k < K; ++k) {  // tail: the same k -> lane rule
      const float a = a_row[k];
      const size_t l = k % kNtLanes;
      acc0[l] += a * static_cast<float>(b0[k]);
      acc1[l] += a * static_cast<float>(b1[k]);
      acc2[l] += a * static_cast<float>(b2[k]);
      acc3[l] += a * static_cast<float>(b3[k]);
    }
    float s0 = acc0[0], s1 = acc1[0], s2 = acc2[0], s3 = acc3[0];
    for (size_t l = 1; l < kNtLanes; ++l) {  // fixed combine order
      s0 += acc0[l];
      s1 += acc1[l];
      s2 += acc2[l];
      s3 += acc3[l];
    }
    c_row[n + 0] = alpha * s0;
    c_row[n + 1] = alpha * s1;
    c_row[n + 2] = alpha * s2;
    c_row[n + 3] = alpha * s3;
  }
  for (; n < n1; ++n) {
    const BType* SEEML_RESTRICT b_row = B + n * K;
    float acc[kNtLanes] = {};
    size_t k = 0;
    for (; k + kNtLanes <= K; k += kNtLanes) {
      if constexpr (std::is_same_v<BType, float>) {
        for (size_t l = 0; l < kNtLanes; ++l)
          acc[l] += a_row[k + l] * b_row[k + l];
      } else {
        float w[kNtLanes];
        for (size_t l = 0; l < kNtLanes; ++l)
          w[l] = static_cast<float>(b_row[k + l]);
        for (size_t l = 0; l < kNtLanes; ++l) acc[l] += a_row[k + l] * w[l];
      }
    }
    for (; k < K; ++k)
      acc[k % kNtLanes] += a_row[k] * static_cast<float>(b_row[k]);
    float sum = acc[0];
    for (size_t l = 1; l < kNtLanes; ++l) sum += acc[l];
    c_row[n] = alpha * sum;
  }
}

template <typename BType>
void BlockedNT(const float* SEEML_RESTRICT A, const BType* SEEML_RESTRICT B,
               float* SEEML_RESTRICT C, size_t m_begin, size_t m_end,
               size_t n_begin, size_t n_end, size_t N, size_t K, float alpha,
               size_t tile_n) {
  const bool pair = K >= kNtPairMinK;
  for (size_t n0 = n_begin; n0 < n_end; n0 += tile_n) {
    const size_t n1 = MinZ(n0 + tile_n, n_end);
    size_t m = m_begin;
    for (; pair && m + 2 <= m_end; m += 2) {
      const float* SEEML_RESTRICT a0 = A + (m + 0) * K;
      const float* SEEML_RESTRICT a1 = A + (m + 1) * K;
      float* SEEML_RESTRICT c0 = C + (m + 0) * N;
      float* SEEML_RESTRICT c1 = C + (m + 1) * N;
      size_t n = n0;
      for (; n + 4 <= n1; n += 4) {
        const BType* SEEML_RESTRICT b0 = B + (n + 0) * K;
        const BType* SEEML_RESTRICT b1 = B + (n + 1) * K;
        const BType* SEEML_RESTRICT b2 = B + (n + 2) * K;
        const BType* SEEML_RESTRICT b3 = B + (n + 3) * K;
        float p0[kNtLanes] = {}, p1[kNtLanes] = {}, p2[kNtLanes] = {},
              p3[kNtLanes] = {}, q0[kNtLanes] = {}, q1[kNtLanes] = {},
              q2[kNtLanes] = {}, q3[kNtLanes] = {};
        size_t k = 0;
        for (; k + kNtLanes <= K; k += kNtLanes) {
          // Eight products per lane: two A rows against four B rows. One
          // body for both storage forms — f32 reads B in place; int8 and
          // bf16 widen the block first (their own loop, as in NtRow) and
          // only those instantiations declare the widened block.
          auto fold = [&](size_t l, float v0, float v1, float v2, float v3) {
            const float x = a0[k + l], y = a1[k + l];
            p0[l] += x * v0;
            p1[l] += x * v1;
            p2[l] += x * v2;
            p3[l] += x * v3;
            q0[l] += y * v0;
            q1[l] += y * v1;
            q2[l] += y * v2;
            q3[l] += y * v3;
          };
          if constexpr (std::is_same_v<BType, float>) {
            for (size_t l = 0; l < kNtLanes; ++l)
              fold(l, b0[k + l], b1[k + l], b2[k + l], b3[k + l]);
          } else {
            float w0[kNtLanes], w1[kNtLanes], w2[kNtLanes], w3[kNtLanes];
            for (size_t l = 0; l < kNtLanes; ++l) {
              w0[l] = static_cast<float>(b0[k + l]);
              w1[l] = static_cast<float>(b1[k + l]);
              w2[l] = static_cast<float>(b2[k + l]);
              w3[l] = static_cast<float>(b3[k + l]);
            }
            for (size_t l = 0; l < kNtLanes; ++l)
              fold(l, w0[l], w1[l], w2[l], w3[l]);
          }
        }
        for (; k < K; ++k) {  // tail: the same k -> lane rule
          const size_t l = k % kNtLanes;
          const float x = a0[k], y = a1[k];
          const float v0 = static_cast<float>(b0[k]);
          const float v1 = static_cast<float>(b1[k]);
          const float v2 = static_cast<float>(b2[k]);
          const float v3 = static_cast<float>(b3[k]);
          p0[l] += x * v0;
          p1[l] += x * v1;
          p2[l] += x * v2;
          p3[l] += x * v3;
          q0[l] += y * v0;
          q1[l] += y * v1;
          q2[l] += y * v2;
          q3[l] += y * v3;
        }
        float s0 = p0[0], s1 = p1[0], s2 = p2[0], s3 = p3[0];
        float t0 = q0[0], t1 = q1[0], t2 = q2[0], t3 = q3[0];
        for (size_t l = 1; l < kNtLanes; ++l) {  // fixed combine order
          s0 += p0[l];
          s1 += p1[l];
          s2 += p2[l];
          s3 += p3[l];
          t0 += q0[l];
          t1 += q1[l];
          t2 += q2[l];
          t3 += q3[l];
        }
        c0[n + 0] = alpha * s0;
        c0[n + 1] = alpha * s1;
        c0[n + 2] = alpha * s2;
        c0[n + 3] = alpha * s3;
        c1[n + 0] = alpha * t0;
        c1[n + 1] = alpha * t1;
        c1[n + 2] = alpha * t2;
        c1[n + 3] = alpha * t3;
      }
      if (n < n1) {  // the tile's leftover columns, one row at a time
        NtRow(a0, B, c0, n, n1, K, alpha);
        NtRow(a1, B, c1, n, n1, K, alpha);
      }
    }
    for (; m < m_end; ++m)  // short K, or the band's odd row
      NtRow(A + m * K, B, C + m * N, n0, n1, K, alpha);
  }
}

// Fused epilogue over the finished C-row range [m_begin, m_end):
// C[m,n] = act(C[m,n] + bias[n]), applied after the row's accumulation
// completes — the rows are still hot from the write-back. Per element this
// is the exact expression the standalone kAddBias and k<Act>Fwd kernels
// evaluate (the Expr functions of kernel_policy.h), in the same order, so a
// fused program is bitwise-identical to its unfused form. The act switch is
// hoisted out of the row loop; each case is one tight vectorizable pass.
void EpilogueRows(float* SEEML_RESTRICT C, const float* SEEML_RESTRICT bias,
                  size_t m_begin, size_t m_end, size_t n_begin, size_t n_end,
                  size_t N, up::EpilogueAct act) {
  auto rows = [&](auto&& per_element) {
    for (size_t m = m_begin; m < m_end; ++m) {
      float* SEEML_RESTRICT c_row = C + m * N;
      for (size_t n = n_begin; n < n_end; ++n)
        c_row[n] = per_element(bias ? c_row[n] + bias[n] : c_row[n]);
    }
  };
  switch (act) {
    case up::EpilogueAct::kNone:
      if (bias) rows([](float v) { return v; });
      break;
    case up::EpilogueAct::kRelu:
      rows([](float v) { return ReluExpr(v); });
      break;
    case up::EpilogueAct::kGelu:
      rows([](float v) { return GeluExpr(v); });
      break;
    case up::EpilogueAct::kSilu:
      rows([](float v) { return SiluExpr(v); });
      break;
  }
}

/// The addend epilogue of the f32 GEMMs (plan v14, E10): C = D + A@B,
/// applied to a task's cell right after the core wrote it — while the cell
/// is still cache-resident — instead of as a separate kAddEW over the whole
/// tensor. Each element is `d + s` with s the COMPLETE dot product the core
/// just produced from a zeroed (NN, TN) or fresh (NT) accumulator: the
/// expression, and so the bits, of the GEMM + add pair it replaces. Seeding
/// the accumulator with D instead — the kGemmAccNN form — would round
/// differently, which is why this is not that.
void AddendRows(float* SEEML_RESTRICT C, const float* SEEML_RESTRICT D,
                size_t m_begin, size_t m_end, size_t n_begin, size_t n_end,
                size_t N) {
  for (size_t m = m_begin; m < m_end; ++m) {
    float* SEEML_RESTRICT c_row = C + m * N;
    const float* SEEML_RESTRICT d_row = D + m * N;
    for (size_t n = n_begin; n < n_end; ++n) c_row[n] = d_row[n] + c_row[n];
  }
}

}  // namespace

void GemmNN(const float* A, const float* B, float* C, size_t M, size_t N,
            size_t K, const float* bias, up::EpilogueAct act,
            const GemmTiles& tiles, const float* addend) {
  ForEachGemmTask(M, N, K, tiles,
                  [&](size_t m0, size_t m1, size_t n0, size_t n1) {
                    ZeroCell(C, N, m0, m1, n0, n1);
                    BlockedNN(A, B, C, m0, m1, n0, n1, N, K, 1.0f, K,
                              /*a_transposed=*/false, tiles);
                    if (addend) AddendRows(C, addend, m0, m1, n0, n1, N);
                    EpilogueRows(C, bias, m0, m1, n0, n1, N, act);
                  });
}

void GemmNT(const float* A, const float* B, float* C, size_t M, size_t N,
            size_t K, const GemmTiles& tiles, const float* addend) {
  ForEachGemmTask(M, N, K, tiles,
                  [&](size_t m0, size_t m1, size_t n0, size_t n1) {
                    BlockedNT(A, B, C, m0, m1, n0, n1, N, K, 1.0f, tiles.n);
                    if (addend) AddendRows(C, addend, m0, m1, n0, n1, N);
                  });
}

void GemmTN(const float* A, const float* B, float* C, size_t M, size_t N,
            size_t K, const GemmTiles& tiles, const float* addend) {
  ForEachGemmTask(M, N, K, tiles,
                  [&](size_t m0, size_t m1, size_t n0, size_t n1) {
                    ZeroCell(C, N, m0, m1, n0, n1);
                    BlockedNN(A, B, C, m0, m1, n0, n1, N, K, 1.0f, M,
                              /*a_transposed=*/true, tiles);
                    if (addend) AddendRows(C, addend, m0, m1, n0, n1, N);
                  });
}

void GemmAccNN(const float* A, const float* B, float* C, size_t M, size_t N,
               size_t K, float alpha, const GemmTiles& tiles) {
  ForEachGemmTask(M, N, K, tiles,
                  [&](size_t m0, size_t m1, size_t n0, size_t n1) {
                    BlockedNN(A, B, C, m0, m1, n0, n1, N, K, alpha, K,
                              /*a_transposed=*/false, tiles);
                  });
}

void GemmNNQ8(const float* A, const int8_t* B, float* C, size_t M, size_t N,
              size_t K, float scale, up::EpilogueAct act,
              const GemmTiles& tiles, const float* col_scale) {
  ForEachGemmTask(M, N, K, tiles,
                  [&](size_t m0, size_t m1, size_t n0, size_t n1) {
                    ZeroCell(C, N, m0, m1, n0, n1);
                    // Per-column scales (v17) join B's panel widening; the
                    // per-tensor form folds its one scale into A.
                    BlockedNN(A, B, C, m0, m1, n0, n1, N, K,
                              col_scale ? 1.0f : scale, K,
                              /*a_transposed=*/false, tiles, col_scale);
                    EpilogueRows(C, /*bias=*/nullptr, m0, m1, n0, n1, N, act);
                  });
}

// Per-column int8 scales in the dX GEMM (v17) sit on the reduction axis:
// sum_k A[m, k] * q[n, k] * s[k] = sum_k (A[m, k] * s[k]) * q[n, k]. Each
// task scales its A rows once into a per-thread buffer (bounded: a
// vocabulary-wide K goes in row chunks) and runs the unchanged NT core on
// them — the per-tensor path is the committed kernel, byte for byte.
inline constexpr size_t kNtScaledFloatsMax = size_t{1} << 18;  // 1 MiB

void GemmNTQ8(const float* A, const int8_t* B, float* C, size_t M, size_t N,
              size_t K, float scale, const GemmTiles& tiles,
              const float* k_scale) {
  if (!k_scale) {
    ForEachGemmTask(M, N, K, tiles,
                    [&](size_t m0, size_t m1, size_t n0, size_t n1) {
                      BlockedNT(A, B, C, m0, m1, n0, n1, N, K, scale,
                                tiles.n);
                    });
    return;
  }
  const size_t chunk_rows =
      std::max<size_t>(2, (kNtScaledFloatsMax / std::max<size_t>(1, K)) & ~size_t{1});
  ForEachGemmTask(M, N, K, tiles,
                  [&](size_t m0, size_t m1, size_t n0, size_t n1) {
                    thread_local std::vector<float> scaled;
                    for (size_t r0 = m0; r0 < m1; r0 += chunk_rows) {
                      const size_t rows = MinZ(chunk_rows, m1 - r0);
                      if (scaled.size() < rows * K) scaled.resize(rows * K);
                      for (size_t r = 0; r < rows; ++r)
                        for (size_t k = 0; k < K; ++k)
                          scaled[r * K + k] = A[(r0 + r) * K + k] * k_scale[k];
                      BlockedNT(scaled.data(), B, C + r0 * N, 0, rows, n0, n1,
                                N, K, 1.0f, tiles.n);
                    }
                  });
}

// bf16 B: the same blocked cores over up::Bf16 elements, whose
// static_cast<float> is the exact widening — bit-identical to GemmNN /
// GemmNT over the widened f32 matrix.
void GemmNNBF16(const float* A, const uint16_t* B, float* C, size_t M,
                size_t N, size_t K, const float* bias, up::EpilogueAct act,
                const GemmTiles& tiles) {
  const auto* Bh = reinterpret_cast<const up::Bf16*>(B);
  ForEachGemmTask(M, N, K, tiles,
                  [&](size_t m0, size_t m1, size_t n0, size_t n1) {
                    ZeroCell(C, N, m0, m1, n0, n1);
                    BlockedNN(A, Bh, C, m0, m1, n0, n1, N, K, 1.0f, K,
                              /*a_transposed=*/false, tiles);
                    EpilogueRows(C, bias, m0, m1, n0, n1, N, act);
                  });
}

void GemmNTBF16(const float* A, const uint16_t* B, float* C, size_t M,
                size_t N, size_t K, const GemmTiles& tiles) {
  const auto* Bh = reinterpret_cast<const up::Bf16*>(B);
  ForEachGemmTask(M, N, K, tiles,
                  [&](size_t m0, size_t m1, size_t n0, size_t n1) {
                    BlockedNT(A, Bh, C, m0, m1, n0, n1, N, K, 1.0f, tiles.n);
                  });
}

}  // namespace seeml::update_rt::kernels
