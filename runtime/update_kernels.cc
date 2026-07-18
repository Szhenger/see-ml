#include "runtime/update_kernels.h"

#include <cmath>
#include <cstring>

#include "source/parallel_for.h"

// No kernel is ever invoked with overlapping source/destination buffers
// (the compiler's arena allocator guarantees it), so tell the optimizer:
// restrict-qualified pointers let the inner loops vectorize without runtime
// alias checks.
#if defined(_MSC_VER)
#define SEEML_RESTRICT __restrict
#else
#define SEEML_RESTRICT __restrict__
#endif

namespace seeml::update_rt::kernels {

namespace up = seeml::update;

namespace {

// =============================================================================
// Parallel decomposition policy.
//
// Every kernel splits its output across ParallelFor chunks whose boundaries
// are a pure function of the problem shape — never of the thread count — and
// each chunk writes only its own slice (or its own partial-reduction slot).
// Consequences, by construction:
//   - no data races: every output element has exactly one writer;
//   - bitwise determinism: per-element arithmetic order is that of the
//     serial loop within a chunk, and reductions combine per-chunk partials
//     in chunk order, so any thread count computes identical bits.
// The grains below set the minimum work per chunk so small tensors (LoRA
// adapters, loss scalars) never leave the calling thread.
// =============================================================================

// Minimum inner-loop iterations per chunk for cheap (add/mul-class) bodies
// and for transcendental (exp/tanh/sqrt-class) bodies.
constexpr size_t kGrainCheap = 32768;
constexpr size_t kGrainMath = 4096;

// Rows per chunk for a kernel whose per-row cost is `per_row` operations.
inline size_t RowGrain(size_t per_row, size_t budget) {
  return per_row == 0 ? budget : (budget / per_row > 0 ? budget / per_row : 1);
}

// Cache-blocking tile sizes for the GEMM family. The K/N tiles keep the
// working set (one A panel + one B panel + one C panel) inside L1/L2 for
// typical embedded cache geometries; the row-major inner loops vectorize
// under -O2 without intrinsics, keeping the reference kernels portable.
constexpr size_t kTileK = 64;
constexpr size_t kTileN = 256;

inline size_t MinZ(size_t a, size_t b) { return a < b ? a : b; }

// Shared blocked core over the C-row range [m_begin, m_end):
// C[m,N] (+)= alpha * A[m,K] @ B[K,N] with B row-major. GemmNN/GemmAccNN/
// GemmTN and the q8 variants all reduce to this loop nest; the parallel
// wrappers partition C's rows, so every worker owns a disjoint row slice and
// each row's arithmetic order matches the serial loop exactly.
// The k loop is unrolled 4-wide so each pass over the C row folds in four B
// rows: 4x fewer C load/store round-trips per FLOP, and four independent
// multiply chains for the vectorizer to interleave. A/B/C must not overlap
// (guaranteed by the arena allocator, which never reuses an operand's slot
// for a result born at the same instruction).
template <typename BType>
void BlockedNN(const float* SEEML_RESTRICT A, const BType* SEEML_RESTRICT B,
               float* SEEML_RESTRICT C, size_t m_begin, size_t m_end, size_t N,
               size_t K, float alpha, size_t a_stride, bool a_transposed) {
  auto a_at = [&](size_t k, size_t m) {
    return a_transposed ? A[k * a_stride + m] : A[m * a_stride + k];
  };
  for (size_t k0 = 0; k0 < K; k0 += kTileK) {
    const size_t k1 = MinZ(k0 + kTileK, K);
    for (size_t n0 = 0; n0 < N; n0 += kTileN) {
      const size_t n1 = MinZ(n0 + kTileN, N);
      for (size_t m = m_begin; m < m_end; ++m) {
        float* SEEML_RESTRICT c_row = C + m * N;
        size_t k = k0;
        for (; k + 4 <= k1; k += 4) {
          const float a0 = alpha * a_at(k + 0, m);
          const float a1 = alpha * a_at(k + 1, m);
          const float a2 = alpha * a_at(k + 2, m);
          const float a3 = alpha * a_at(k + 3, m);
          const BType* SEEML_RESTRICT b0 = B + (k + 0) * N;
          const BType* SEEML_RESTRICT b1 = B + (k + 1) * N;
          const BType* SEEML_RESTRICT b2 = B + (k + 2) * N;
          const BType* SEEML_RESTRICT b3 = B + (k + 3) * N;
          for (size_t n = n0; n < n1; ++n)
            c_row[n] += a0 * static_cast<float>(b0[n]) +
                        a1 * static_cast<float>(b1[n]) +
                        a2 * static_cast<float>(b2[n]) +
                        a3 * static_cast<float>(b3[n]);
        }
        for (; k < k1; ++k) {
          const float a = alpha * a_at(k, m);
          const BType* SEEML_RESTRICT b_row = B + k * N;
          for (size_t n = n0; n < n1; ++n)
            c_row[n] += a * static_cast<float>(b_row[n]);
        }
      }
    }
  }
}

// Dot-product core shared by GemmNT and GemmNTQ8 over the C-row range
// [m_begin, m_end): C[m,n] = A row · B row. Four output columns per pass
// reuse the streamed A row from L1 four times and run four independent
// accumulator chains.
template <typename BType>
void BlockedNT(const float* SEEML_RESTRICT A, const BType* SEEML_RESTRICT B,
               float* SEEML_RESTRICT C, size_t m_begin, size_t m_end, size_t N,
               size_t K, float alpha) {
  for (size_t n0 = 0; n0 < N; n0 += kTileN) {
    const size_t n1 = MinZ(n0 + kTileN, N);
    for (size_t m = m_begin; m < m_end; ++m) {
      const float* SEEML_RESTRICT a_row = A + m * K;
      size_t n = n0;
      for (; n + 4 <= n1; n += 4) {
        const BType* SEEML_RESTRICT b0 = B + (n + 0) * K;
        const BType* SEEML_RESTRICT b1 = B + (n + 1) * K;
        const BType* SEEML_RESTRICT b2 = B + (n + 2) * K;
        const BType* SEEML_RESTRICT b3 = B + (n + 3) * K;
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (size_t k = 0; k < K; ++k) {
          const float a = a_row[k];
          acc0 += a * static_cast<float>(b0[k]);
          acc1 += a * static_cast<float>(b1[k]);
          acc2 += a * static_cast<float>(b2[k]);
          acc3 += a * static_cast<float>(b3[k]);
        }
        float* c_at = C + m * N + n;
        c_at[0] = alpha * acc0;
        c_at[1] = alpha * acc1;
        c_at[2] = alpha * acc2;
        c_at[3] = alpha * acc3;
      }
      for (; n < n1; ++n) {
        const BType* SEEML_RESTRICT b_row = B + n * K;
        float acc = 0.0f;
        for (size_t k = 0; k < K; ++k)
          acc += a_row[k] * static_cast<float>(b_row[k]);
        C[m * N + n] = alpha * acc;
      }
    }
  }
}

}  // namespace

void GemmNN(const float* A, const float* B, float* C, size_t M, size_t N,
            size_t K) {
  up::ParallelFor(M, RowGrain(N * K, kGrainCheap),
                  [&](size_t m0, size_t m1, size_t) {
                    std::memset(C + m0 * N, 0, (m1 - m0) * N * sizeof(float));
                    BlockedNN(A, B, C, m0, m1, N, K, 1.0f, K,
                              /*a_transposed=*/false);
                  });
}

void GemmNT(const float* A, const float* B, float* C, size_t M, size_t N,
            size_t K) {
  up::ParallelFor(M, RowGrain(N * K, kGrainCheap),
                  [&](size_t m0, size_t m1, size_t) {
                    BlockedNT(A, B, C, m0, m1, N, K, 1.0f);
                  });
}

void GemmTN(const float* A, const float* B, float* C, size_t M, size_t N,
            size_t K) {
  up::ParallelFor(M, RowGrain(N * K, kGrainCheap),
                  [&](size_t m0, size_t m1, size_t) {
                    std::memset(C + m0 * N, 0, (m1 - m0) * N * sizeof(float));
                    BlockedNN(A, B, C, m0, m1, N, K, 1.0f, M,
                              /*a_transposed=*/true);
                  });
}

void GemmAccNN(const float* A, const float* B, float* C, size_t M, size_t N,
               size_t K, float alpha) {
  up::ParallelFor(M, RowGrain(N * K, kGrainCheap),
                  [&](size_t m0, size_t m1, size_t) {
                    BlockedNN(A, B, C, m0, m1, N, K, alpha, K,
                              /*a_transposed=*/false);
                  });
}

void GemmNNQ8(const float* A, const int8_t* B, float* C, size_t M, size_t N,
              size_t K, float scale) {
  up::ParallelFor(M, RowGrain(N * K, kGrainCheap),
                  [&](size_t m0, size_t m1, size_t) {
                    std::memset(C + m0 * N, 0, (m1 - m0) * N * sizeof(float));
                    BlockedNN(A, B, C, m0, m1, N, K, scale, K,
                              /*a_transposed=*/false);
                  });
}

void GemmNTQ8(const float* A, const int8_t* B, float* C, size_t M, size_t N,
              size_t K, float scale) {
  up::ParallelFor(M, RowGrain(N * K, kGrainCheap),
                  [&](size_t m0, size_t m1, size_t) {
                    BlockedNT(A, B, C, m0, m1, N, K, scale);
                  });
}

void AddEW(const float* x, const float* y, float* out, size_t n) {
  up::ParallelFor(n, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) out[i] = x[i] + y[i];
  });
}

void MulEW(const float* x, const float* y, float* out, size_t n) {
  up::ParallelFor(n, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) out[i] = x[i] * y[i];
  });
}

void AddBias(const float* x, const float* b, float* out, size_t rows,
             size_t cols) {
  up::ParallelFor(rows, RowGrain(cols, kGrainCheap),
                  [&](size_t r0, size_t r1, size_t) {
                    for (size_t r = r0; r < r1; ++r)
                      for (size_t c = 0; c < cols; ++c)
                        out[r * cols + c] = x[r * cols + c] + b[c];
                  });
}

void ReluFwd(const float* x, float* out, size_t n) {
  up::ParallelFor(n, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
  });
}

void ReluBwd(const float* dy, const float* x, float* dx, size_t n) {
  up::ParallelFor(n, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;
  });
}

namespace {

// gelu(x) = 0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³))) — the tanh
// approximation. The backward kernel differentiates exactly this expression,
// keeping the fwd/bwd pair consistent for gradient verification.
constexpr float kGeluC = 0.7978845608028654f;  // √(2/π)
constexpr float kGeluA = 0.044715f;

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

void GeluFwd(const float* x, float* out, size_t n) {
  up::ParallelFor(n, kGrainMath, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) {
      const float v = x[i];
      const float t = std::tanh(kGeluC * (v + kGeluA * v * v * v));
      out[i] = 0.5f * v * (1.0f + t);
    }
  });
}

void GeluBwd(const float* dy, const float* x, float* dx, size_t n) {
  up::ParallelFor(n, kGrainMath, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) {
      const float v = x[i];
      const float u = kGeluC * (v + kGeluA * v * v * v);
      const float t = std::tanh(u);
      const float du = kGeluC * (1.0f + 3.0f * kGeluA * v * v);
      // d/dv [0.5 v (1 + tanh(u))] = 0.5 (1 + t) + 0.5 v (1 - t²) u'
      dx[i] = dy[i] * (0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * du);
    }
  });
}

void SiluFwd(const float* x, float* out, size_t n) {
  up::ParallelFor(n, kGrainMath, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) out[i] = x[i] * Sigmoid(x[i]);
  });
}

void SiluBwd(const float* dy, const float* x, float* dx, size_t n) {
  up::ParallelFor(n, kGrainMath, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) {
      const float s = Sigmoid(x[i]);
      dx[i] = dy[i] * (s * (1.0f + x[i] * (1.0f - s)));
    }
  });
}

void Scale(const float* x, float* out, float alpha, size_t n) {
  up::ParallelFor(n, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) out[i] = alpha * x[i];
  });
}

void ReduceRows(const float* dy, float* db, size_t rows, size_t cols) {
  // Partitioned over db's columns: each chunk owns a column slice and walks
  // the rows in order, so per-column accumulation order — and therefore the
  // result — is bit-identical to the serial loop.
  up::ParallelFor(cols, RowGrain(rows, kGrainCheap),
                  [&](size_t c0, size_t c1, size_t) {
                    std::memset(db + c0, 0, (c1 - c0) * sizeof(float));
                    for (size_t r = 0; r < rows; ++r) {
                      const float* SEEML_RESTRICT row = dy + r * cols;
                      for (size_t c = c0; c < c1; ++c) db[c] += row[c];
                    }
                  });
}

void LayerNormFwd(const float* x, const float* gamma, const float* beta,
                  float* y, float* mean, float* rstd, size_t rows,
                  size_t cols) {
  constexpr float kEps = 1e-5f;
  up::ParallelFor(rows, RowGrain(cols, kGrainMath), [&](size_t r0, size_t r1,
                                                        size_t) {
    for (size_t r = r0; r < r1; ++r) {
      const float* xr = x + r * cols;
      double sum = 0.0;
      for (size_t c = 0; c < cols; ++c) sum += xr[c];
      const float mu = static_cast<float>(sum / static_cast<double>(cols));
      double var = 0.0;
      for (size_t c = 0; c < cols; ++c) {
        const double d = static_cast<double>(xr[c]) - mu;
        var += d * d;
      }
      const float rs = 1.0f / std::sqrt(
          static_cast<float>(var / static_cast<double>(cols)) + kEps);
      mean[r] = mu;
      rstd[r] = rs;
      float* yr = y + r * cols;
      for (size_t c = 0; c < cols; ++c)
        yr[c] = (xr[c] - mu) * rs * gamma[c] + beta[c];
    }
  });
}

void LayerNormBwd(const float* dy, const float* x, const float* gamma,
                  const float* mean, const float* rstd, float* dx, size_t rows,
                  size_t cols) {
  // With x̂ = (x - μ)·rstd and g = dy·γ:
  //   dx = rstd · (g - mean(g) - x̂ · mean(g·x̂))
  const float inv_d = 1.0f / static_cast<float>(cols);
  up::ParallelFor(rows, RowGrain(cols, kGrainMath), [&](size_t r0, size_t r1,
                                                        size_t) {
    for (size_t r = r0; r < r1; ++r) {
      const float* dyr = dy + r * cols;
      const float* xr = x + r * cols;
      float* dxr = dx + r * cols;
      const float mu = mean[r], rs = rstd[r];
      double sum_g = 0.0, sum_gx = 0.0;
      for (size_t c = 0; c < cols; ++c) {
        const float xhat = (xr[c] - mu) * rs;
        const float g = dyr[c] * gamma[c];
        sum_g += g;
        sum_gx += static_cast<double>(g) * xhat;
      }
      const float mg = static_cast<float>(sum_g) * inv_d;
      const float mgx = static_cast<float>(sum_gx) * inv_d;
      for (size_t c = 0; c < cols; ++c) {
        const float xhat = (xr[c] - mu) * rs;
        const float g = dyr[c] * gamma[c];
        dxr[c] = rs * (g - mg - xhat * mgx);
      }
    }
  });
}

namespace {

// Numerically stable in-place row softmax with optional temperature.
void RowSoftmax(const float* logits, float* probs, size_t C, float inv_T) {
  float max_v = logits[0] * inv_T;
  for (size_t c = 1; c < C; ++c) max_v = std::fmax(max_v, logits[c] * inv_T);
  float sum = 0.0f;
  for (size_t c = 0; c < C; ++c) {
    probs[c] = std::exp(logits[c] * inv_T - max_v);
    sum += probs[c];
  }
  const float inv_sum = 1.0f / sum;
  for (size_t c = 0; c < C; ++c) probs[c] *= inv_sum;
}

}  // namespace

void SoftmaxXEntFwd(const float* logits, const int32_t* labels, float* loss,
                    float* probs, size_t N, size_t C) {
  // One partial per chunk, combined in chunk order: the loss is
  // bitwise-deterministic for every thread count (chunk geometry is a pure
  // function of the shape).
  const size_t grain = RowGrain(C, kGrainMath);
  double partials[up::kMaxParallelChunks] = {};
  up::ParallelFor(N, grain, [&](size_t r0, size_t r1, size_t chunk) {
    double total = 0.0;
    for (size_t n = r0; n < r1; ++n) {
      RowSoftmax(logits + n * C, probs + n * C, C, 1.0f);
      const float p = probs[n * C + static_cast<size_t>(labels[n])];
      // fmax would silently scrub a NaN probability (fmax(NaN, x) == x) and
      // report a large-but-finite loss while the gradients poison the
      // parameters. Propagate NaN so the engine's finite-loss guard trips;
      // the clamp only rescues genuine underflow.
      const double safe = std::isnan(p)
                              ? static_cast<double>(p)
                              : static_cast<double>(std::fmax(p, 1e-12f));
      total -= std::log(safe);
    }
    partials[chunk] = total;
  });
  double total = 0.0;
  const size_t chunks = up::ParallelChunkCount(N, grain);
  for (size_t c = 0; c < chunks; ++c) total += partials[c];
  *loss = static_cast<float>(total / static_cast<double>(N));
}

void SoftmaxXEntBwd(const float* probs, const int32_t* labels,
                    const float* seed, float* dlogits, size_t N, size_t C) {
  const float scale = *seed / static_cast<float>(N);
  up::ParallelFor(N, RowGrain(C, kGrainCheap),
                  [&](size_t r0, size_t r1, size_t) {
                    for (size_t n = r0; n < r1; ++n) {
                      for (size_t c = 0; c < C; ++c)
                        dlogits[n * C + c] = scale * probs[n * C + c];
                      dlogits[n * C + static_cast<size_t>(labels[n])] -= scale;
                    }
                  });
}

void MseFwd(const float* pred, const float* target, float* loss, size_t n) {
  const size_t grain = kGrainCheap;
  double partials[up::kMaxParallelChunks] = {};
  up::ParallelFor(n, grain, [&](size_t b, size_t e, size_t chunk) {
    double total = 0.0;
    for (size_t i = b; i < e; ++i) {
      const double d = static_cast<double>(pred[i]) - target[i];
      total += d * d;
    }
    partials[chunk] = total;
  });
  double total = 0.0;
  const size_t chunks = up::ParallelChunkCount(n, grain);
  for (size_t c = 0; c < chunks; ++c) total += partials[c];
  *loss = static_cast<float>(total / static_cast<double>(n));
}

void MseBwd(const float* pred, const float* target, const float* seed,
            float* dpred, size_t n) {
  const float scale = 2.0f * *seed / static_cast<float>(n);
  up::ParallelFor(n, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) dpred[i] = scale * (pred[i] - target[i]);
  });
}

void KLDistillFwd(const float* s_logits, const float* t_logits, float* loss,
                  float* p_s, float* p_t, size_t N, size_t C, float T) {
  const float inv_T = 1.0f / T;
  const size_t grain = RowGrain(C, kGrainMath);
  double partials[up::kMaxParallelChunks] = {};
  up::ParallelFor(N, grain, [&](size_t r0, size_t r1, size_t chunk) {
    double total = 0.0;
    for (size_t n = r0; n < r1; ++n) {
      RowSoftmax(s_logits + n * C, p_s + n * C, C, inv_T);
      RowSoftmax(t_logits + n * C, p_t + n * C, C, inv_T);
      for (size_t c = 0; c < C; ++c) {
        const float pt = p_t[n * C + c];
        if (pt <= 0.0f) continue;
        const float ps = std::fmax(p_s[n * C + c], 1e-12f);
        total += static_cast<double>(pt) *
                 (std::log(static_cast<double>(pt)) -
                  std::log(static_cast<double>(ps)));
      }
    }
    partials[chunk] = total;
  });
  double total = 0.0;
  const size_t chunks = up::ParallelChunkCount(N, grain);
  for (size_t c = 0; c < chunks; ++c) total += partials[c];
  *loss = static_cast<float>(total / static_cast<double>(N));
}

void KLDistillBwd(const float* p_s, const float* p_t, const float* seed,
                  float* dlogits, size_t N, size_t C, float T) {
  const float scale = *seed / (static_cast<float>(N) * T);
  up::ParallelFor(N * C, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) dlogits[i] = scale * (p_s[i] - p_t[i]);
  });
}

void ClipNorm(float* g, size_t n, float max_norm) {
  const size_t grain = kGrainCheap;
  double partials[up::kMaxParallelChunks] = {};
  up::ParallelFor(n, grain, [&](size_t b, size_t e, size_t chunk) {
    double sq = 0.0;
    for (size_t i = b; i < e; ++i)
      sq += static_cast<double>(g[i]) * static_cast<double>(g[i]);
    partials[chunk] = sq;
  });
  double sq = 0.0;
  const size_t chunks = up::ParallelChunkCount(n, grain);
  for (size_t c = 0; c < chunks; ++c) sq += partials[c];
  const double norm = std::sqrt(sq);
  if (norm <= static_cast<double>(max_norm) || norm == 0.0) return;
  const float s = static_cast<float>(static_cast<double>(max_norm) / norm);
  up::ParallelFor(n, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) g[i] *= s;
  });
}

void SgdStep(float* p, const float* g, size_t n, float lr,
             float weight_decay) {
  up::ParallelFor(n, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) p[i] -= lr * (g[i] + weight_decay * p[i]);
  });
}

void AdamWStep(float* SEEML_RESTRICT p, const float* SEEML_RESTRICT g,
               float* SEEML_RESTRICT m, float* SEEML_RESTRICT v, size_t n,
               float lr, float beta1, float beta2, float eps,
               float weight_decay, uint64_t step) {
  // Hoist everything that is per-step, not per-element: the bias-correction
  // divides become two multiplies, and the (1-beta) blend factors are
  // computed once instead of n times.
  const float inv_bc1 =
      1.0f / (1.0f - std::pow(beta1, static_cast<float>(step)));
  const float inv_bc2 =
      1.0f / (1.0f - std::pow(beta2, static_cast<float>(step)));
  const float om_b1 = 1.0f - beta1;
  const float om_b2 = 1.0f - beta2;
  up::ParallelFor(n, kGrainMath, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) {
      const float gi = g[i];
      const float mi = beta1 * m[i] + om_b1 * gi;
      const float vi = beta2 * v[i] + om_b2 * gi * gi;
      m[i] = mi;
      v[i] = vi;
      const float m_hat = mi * inv_bc1;
      const float v_hat = vi * inv_bc2;
      p[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * p[i]);
    }
  });
}

// Fill and Copy stay serial: memset/memcpy-class loops saturate memory
// bandwidth from one core on the device classes the runtime targets, so
// splitting them buys contention, not throughput.
void Fill(float* dst, float value, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] = value;
}

void Copy(const float* src, float* dst, size_t n) {
  std::memcpy(dst, src, n * sizeof(float));
}

}  // namespace seeml::update_rt::kernels
