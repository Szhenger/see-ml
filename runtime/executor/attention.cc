#include <cmath>

#include "runtime/executor/kernel_policy.h"
#include "runtime/executor/update_kernels.h"
#include "source/parallel/parallel_for.h"

// =============================================================================
// Transformer family (plan v6): rotary position embedding and causal
// scaled-dot-product attention with its backward primitives.
//
// Layouts (see update_kernels.h): activations are rank-2 [B*S, H*d]
// row-major with heads interleaved along the row; the probability matrix
// P[B,H,S,S] is flattened [B*H*S, S]. Every kernel decomposes over B*H*S
// (b, h, row) units — a pure function of the problem shape — and each unit
// writes only its own d-segment or its own P row, so any thread count
// computes identical bits (kernel_policy.h). Row reductions accumulate in
// double, matching the normalization and loss families.
// =============================================================================

namespace seeml::update_rt::kernels {

namespace up = seeml::update;

namespace {

/// Decomposes a flat (b, h, i) unit index against S.
struct Unit {
  size_t b, h, i;
};
inline Unit UnitOf(size_t u, size_t H, size_t S) {
  return {u / (H * S), (u / S) % H, u % S};
}

}  // namespace

void RopeFwd(const float* x, float* y, size_t B, size_t S, size_t H, size_t d,
             float base) {
  const size_t D = H * d;
  up::ParallelFor(B * S * H, RowGrain(d, kGrainMath),
                  [&](size_t u0, size_t u1, size_t) {
    for (size_t u = u0; u < u1; ++u) {
      const size_t s = (u / H) % S;
      const float* xr = x + (u / H) * D + (u % H) * d;
      float* yr = y + (u / H) * D + (u % H) * d;
      for (size_t c = 0; c + 1 < d; c += 2) {
        // angle(s, c) = s * base^(-c/d) over the interleaved pair (c, c+1).
        const float theta =
            static_cast<float>(s) *
            std::pow(base, -static_cast<float>(c) / static_cast<float>(d));
        const float cs = std::cos(theta), sn = std::sin(theta);
        const float a = xr[c], b2 = xr[c + 1];
        yr[c] = a * cs - b2 * sn;
        yr[c + 1] = a * sn + b2 * cs;
      }
    }
  });
}

void RopeBwd(const float* dy, float* dx, size_t B, size_t S, size_t H,
             size_t d, float base) {
  // The forward is an orthogonal per-pair rotation; its VJP is the rotation
  // by the negated angle (the transpose).
  const size_t D = H * d;
  up::ParallelFor(B * S * H, RowGrain(d, kGrainMath),
                  [&](size_t u0, size_t u1, size_t) {
    for (size_t u = u0; u < u1; ++u) {
      const size_t s = (u / H) % S;
      const float* gr = dy + (u / H) * D + (u % H) * d;
      float* dr = dx + (u / H) * D + (u % H) * d;
      for (size_t c = 0; c + 1 < d; c += 2) {
        const float theta =
            static_cast<float>(s) *
            std::pow(base, -static_cast<float>(c) / static_cast<float>(d));
        const float cs = std::cos(theta), sn = std::sin(theta);
        const float a = gr[c], b2 = gr[c + 1];
        dr[c] = a * cs + b2 * sn;
        dr[c + 1] = -a * sn + b2 * cs;
      }
    }
  });
}

void AttnFwd(const float* q, const float* k, const float* v, float* o,
             float* probs, size_t B, size_t S, size_t H, size_t d) {
  const size_t D = H * d;
  const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
  // One unit = one query row (b, h, i): writes P row (b,h,i,*) and the o
  // segment (b, i, h*d .. h*d+d) — both exclusively owned by this unit.
  up::ParallelFor(B * H * S, RowGrain(S * (d + 4), kGrainMath),
                  [&](size_t u0, size_t u1, size_t) {
    for (size_t u = u0; u < u1; ++u) {
      const auto [b, h, i] = UnitOf(u, H, S);
      const float* qi = q + (b * S + i) * D + h * d;
      float* prow = probs + u * S;
      // Scores for the causal prefix j <= i, then a stable row softmax.
      float row_max = -INFINITY;
      for (size_t j = 0; j <= i; ++j) {
        const float* kj = k + (b * S + j) * D + h * d;
        double dot = 0.0;
        for (size_t c = 0; c < d; ++c)
          dot += static_cast<double>(qi[c]) * kj[c];
        const float s = static_cast<float>(dot) * inv_sqrt_d;
        prow[j] = s;
        if (s > row_max) row_max = s;
      }
      double denom = 0.0;
      for (size_t j = 0; j <= i; ++j) {
        const float e = std::exp(prow[j] - row_max);
        prow[j] = e;
        denom += e;
      }
      const float inv_denom = 1.0f / static_cast<float>(denom);
      for (size_t j = 0; j <= i; ++j) prow[j] *= inv_denom;
      for (size_t j = i + 1; j < S; ++j) prow[j] = 0.0f;  // causal mask
      // O row segment: o(b, i, h, c) = sum_j P(i, j) * v(b, j, h, c).
      float* oi = o + (b * S + i) * D + h * d;
      for (size_t c = 0; c < d; ++c) oi[c] = 0.0f;
      for (size_t j = 0; j <= i; ++j) {
        const float p = prow[j];
        const float* vj = v + (b * S + j) * D + h * d;
        for (size_t c = 0; c < d; ++c) oi[c] += p * vj[c];
      }
    }
  });
}

void AttnDP(const float* dout, const float* v, float* dp, size_t B, size_t S,
            size_t H, size_t d) {
  const size_t D = H * d;
  up::ParallelFor(B * H * S, RowGrain(S * d, kGrainMath),
                  [&](size_t u0, size_t u1, size_t) {
    for (size_t u = u0; u < u1; ++u) {
      const auto [b, h, i] = UnitOf(u, H, S);
      const float* doi = dout + (b * S + i) * D + h * d;
      float* dprow = dp + u * S;
      for (size_t j = 0; j < S; ++j) {
        const float* vj = v + (b * S + j) * D + h * d;
        double dot = 0.0;
        for (size_t c = 0; c < d; ++c)
          dot += static_cast<double>(doi[c]) * vj[c];
        dprow[j] = static_cast<float>(dot);
      }
    }
  });
}

void AttnDV(const float* probs, const float* dout, float* dv, size_t B,
            size_t S, size_t H, size_t d) {
  const size_t D = H * d;
  // One unit = one value row (b, h, j): dv(b, j, h, c) = sum_i P(i, j) dO(i, c).
  up::ParallelFor(B * H * S, RowGrain(S * d, kGrainMath),
                  [&](size_t u0, size_t u1, size_t) {
    for (size_t u = u0; u < u1; ++u) {
      const auto [b, h, j] = UnitOf(u, H, S);
      float* dvj = dv + (b * S + j) * D + h * d;
      for (size_t c = 0; c < d; ++c) dvj[c] = 0.0f;
      // Causality: P(i, j) == 0 for i < j — start at the first live row.
      for (size_t i = j; i < S; ++i) {
        const float p = probs[((b * H + h) * S + i) * S + j];
        const float* doi = dout + (b * S + i) * D + h * d;
        for (size_t c = 0; c < d; ++c) dvj[c] += p * doi[c];
      }
    }
  });
}

void SoftmaxRowsBwd(const float* probs, const float* dp, float* ds,
                    size_t rows, size_t cols) {
  up::ParallelFor(rows, RowGrain(cols, kGrainMath), [&](size_t r0, size_t r1,
                                                        size_t) {
    for (size_t r = r0; r < r1; ++r) {
      const float* pr = probs + r * cols;
      const float* dpr = dp + r * cols;
      float* dsr = ds + r * cols;
      double dot = 0.0;
      for (size_t c = 0; c < cols; ++c)
        dot += static_cast<double>(dpr[c]) * pr[c];
      const float sum = static_cast<float>(dot);
      // Masked entries carry p == 0, so their dS is exactly 0.
      for (size_t c = 0; c < cols; ++c) dsr[c] = pr[c] * (dpr[c] - sum);
    }
  });
}

void AttnDQ(const float* ds, const float* k, float* dq, size_t B, size_t S,
            size_t H, size_t d) {
  const size_t D = H * d;
  const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
  up::ParallelFor(B * H * S, RowGrain(S * d, kGrainMath),
                  [&](size_t u0, size_t u1, size_t) {
    for (size_t u = u0; u < u1; ++u) {
      const auto [b, h, i] = UnitOf(u, H, S);
      const float* dsrow = ds + u * S;
      float* dqi = dq + (b * S + i) * D + h * d;
      for (size_t c = 0; c < d; ++c) dqi[c] = 0.0f;
      // Causality: dS(i, j) == 0 for j > i.
      for (size_t j = 0; j <= i; ++j) {
        const float g = dsrow[j] * inv_sqrt_d;
        const float* kj = k + (b * S + j) * D + h * d;
        for (size_t c = 0; c < d; ++c) dqi[c] += g * kj[c];
      }
    }
  });
}

void AttnDK(const float* ds, const float* q, float* dk, size_t B, size_t S,
            size_t H, size_t d) {
  const size_t D = H * d;
  const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
  // One unit = one key row (b, h, j): dk(j) = sum_{i >= j} dS(i, j) q(i) / sqrt(d).
  up::ParallelFor(B * H * S, RowGrain(S * d, kGrainMath),
                  [&](size_t u0, size_t u1, size_t) {
    for (size_t u = u0; u < u1; ++u) {
      const auto [b, h, j] = UnitOf(u, H, S);
      float* dkj = dk + (b * S + j) * D + h * d;
      for (size_t c = 0; c < d; ++c) dkj[c] = 0.0f;
      for (size_t i = j; i < S; ++i) {
        const float g = ds[((b * H + h) * S + i) * S + j] * inv_sqrt_d;
        const float* qi = q + (b * S + i) * D + h * d;
        for (size_t c = 0; c < d; ++c) dkj[c] += g * qi[c];
      }
    }
  });
}

}  // namespace seeml::update_rt::kernels
