#include <cstring>

#include "runtime/executor/kernel_policy.h"
#include "runtime/executor/update_kernels.h"
#include "source/parallel/parallel_for.h"

// =============================================================================
// GEMM family: the four f32 variants and the two dequantizing q8 variants,
// all reduced to two blocked cores. The parallel wrappers partition C's
// rows, so every worker owns a disjoint row slice and each row's arithmetic
// order matches the serial loop exactly.
// =============================================================================

namespace seeml::update_rt::kernels {

namespace up = seeml::update;

namespace {

// Cache-blocking tile sizes. The K/N tiles keep the working set (one A panel
// + one B panel + one C panel) inside L1/L2 for typical embedded cache
// geometries; the row-major inner loops vectorize under -O2 without
// intrinsics, keeping the reference kernels portable.
constexpr size_t kTileK = 64;
constexpr size_t kTileN = 256;

// Shared blocked core over the C-row range [m_begin, m_end):
// C[m,N] (+)= alpha * A[m,K] @ B[K,N] with B row-major. GemmNN/GemmAccNN/
// GemmTN and the q8 variants all reduce to this loop nest.
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

}  // namespace seeml::update_rt::kernels
