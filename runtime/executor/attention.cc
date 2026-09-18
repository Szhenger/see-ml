#include <cmath>
#include <vector>

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
// computes identical bits (kernel_policy.h). The dot-product reductions
// (scores, softmax denominator, dP, the softmax-backward rowsum)
// accumulate in double, matching the normalization and loss families; the
// O/dV/dQ/dK accumulations are f32 like the GEMM family — O is a convex
// combination and well-conditioned.
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

namespace {

// The rotation shared by RopeFwd (sign +1) and RopeBwd (sign -1, the
// transpose). angle(s, c) = s * base^(-c/d) over the interleaved pair
// (c, c+1). The per-pair frequency base^(-c/d) is a geometric sequence in
// c, so one pow per call plus a multiplicative recurrence replaces d/2 pow
// calls per unit. The recurrence is a pure function of (c, d) — never of
// chunk or thread — so thread-count invariance holds by construction, and
// both directions evaluate the identical expression, keeping the adjoint
// exact.
//
// With a table (plan v12) the cos/sin come from RopeTable's [S, d/2, 2]
// image instead of being recomputed by every (b, h) unit. RopeTable runs
// this same recurrence through the same libm calls, so the two paths feed
// the rotation identical floats: the table removes ~(1 - 1/(B*H)) of the
// transcendental work and no bits.
template <bool kBackward>
void Rotate(const float* x, float* y, size_t B, size_t S, size_t H, size_t d,
            float base, const float* table) {
  const size_t D = H * d;
  const size_t half = d / 2;
  const float step = std::pow(base, -2.0f / static_cast<float>(d));
  up::ParallelFor(B * S * H, RowGrain(d, table ? kGrainCheap : kGrainMath),
                  [&](size_t u0, size_t u1, size_t) {
    for (size_t u = u0; u < u1; ++u) {
      const size_t s = (u / H) % S;
      const float* xr = x + (u / H) * D + (u % H) * d;
      float* yr = y + (u / H) * D + (u % H) * d;
      const float* row = table ? table + s * half * 2 : nullptr;
      float freq = 1.0f;
      for (size_t c = 0; c + 1 < d; c += 2) {
        float cs, sn;
        if (row) {
          cs = row[c];  // pair c/2 lives at [c, c + 1]
          sn = row[c + 1];
        } else {
          const float theta = static_cast<float>(s) * freq;
          cs = std::cos(theta);
          sn = std::sin(theta);
          freq *= step;
        }
        const float a = xr[c], b2 = xr[c + 1];
        if constexpr (kBackward) {
          yr[c] = a * cs + b2 * sn;
          yr[c + 1] = -a * sn + b2 * cs;
        } else {
          yr[c] = a * cs - b2 * sn;
          yr[c + 1] = a * sn + b2 * cs;
        }
      }
    }
  });
}

}  // namespace

void RopeFwd(const float* x, float* y, size_t B, size_t S, size_t H, size_t d,
             float base, const float* table) {
  Rotate<false>(x, y, B, S, H, d, base, table);
}

void RopeBwd(const float* dy, float* dx, size_t B, size_t S, size_t H,
             size_t d, float base, const float* table) {
  // The forward is an orthogonal per-pair rotation; its VJP is the rotation
  // by the negated angle (the transpose).
  Rotate<true>(dy, dx, B, S, H, d, base, table);
}

void RopeTable(float* table, size_t S, size_t d, float base) {
  // One row per position, each running Rotate's recurrence from freq = 1:
  // rows are independent, so the table parallelizes over s with no effect
  // on any value.
  const size_t half = d / 2;
  const float step = std::pow(base, -2.0f / static_cast<float>(d));
  up::ParallelFor(S, RowGrain(half, kGrainMath),
                  [&](size_t s0, size_t s1, size_t) {
    for (size_t s = s0; s < s1; ++s) {
      float* row = table + s * half * 2;
      float freq = 1.0f;
      for (size_t c = 0; c < half; ++c) {
        const float theta = static_cast<float>(s) * freq;
        row[2 * c] = std::cos(theta);
        row[2 * c + 1] = std::sin(theta);
        freq *= step;
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
      for (size_t j = 0; j <= i; ++j) {
        const float* vj = v + (b * S + j) * D + h * d;
        double dot = 0.0;
        for (size_t c = 0; c < d; ++c)
          dot += static_cast<double>(doi[c]) * vj[c];
        dprow[j] = static_cast<float>(dot);
      }
      // Masked positions: SoftmaxRowsBwd multiplies these by P == 0, so any
      // FINITE value gives dS == 0 — a zero store skips half the S^2 d work
      // bitwise-neutrally (adding +/-0 to the double rowsum never changes
      // its bits, and dQ/dK never read the masked dS entries). The store
      // itself must remain: stale arena bytes could be NaN, and 0 * NaN
      // would poison the rowsum.
      for (size_t j = i + 1; j < S; ++j) dprow[j] = 0.0f;
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


// =============================================================================
// The tiled family (plan v15, E11 / #94): the same attention with no S x S
// matrix anywhere. The cached family keeps P = softmax(mask(Q K^T/sqrt d))
// alive from the forward to the backward of every layer, and the backward
// materializes dP and dS at the same size: at S = 2048, H = 8 that is 134
// MB per matrix per layer — the sequence-length ceiling on a device. Here
// the forward keeps one probability ROW at a time in a scratch buffer and
// stores four floats per row instead of S: the row max m, the inverse
// softmax denominator 1/l, and (written by the dQ pass) the softmax-
// backward rowsum delta. Every backward pass recomputes the probabilities
// it needs from Q, K and those two numbers.
//
// THE BITS. Recomputed is not approximated: p(i, j) is evaluated as
// exp(s - m) * (1/l) with s the same double-accumulated dot product the
// cached forward scored, so every recomputed probability is the float the
// cached path stored; dp(i, j), delta(i), ds(i, j) and the dQ/dK/dV
// accumulations use the cached kernels' expressions in the cached kernels'
// orders. The tiled family therefore computes the SAME BITS as the cached
// family — the compiler may pick either by memory alone — at about twice
// the arithmetic (Q K^T is recomputed by each backward pass, dP by two of
// them) for O(B·H·S) memory instead of O(B·H·S^2). Units are the cached
// kernels' units (a query row for the forward and dQ, a key row for dK and
// dV), each writing only what it owns, so any thread count computes the
// same floats.
// =============================================================================

namespace {

/// The per-row scratch of the tiled kernels: `width` floats per ParallelFor
/// chunk (S of probabilities, S more of dP where a pass needs both), sized
/// once per kernel call for the chunk count the (n, grain) pair yields —
/// the kernels allocate nothing per row, and a chunk's slice is its own.
struct ChunkScratch {
  std::vector<float> buf;
  size_t width;
  ChunkScratch(size_t n, size_t grain, size_t width)
      : buf(up::ParallelChunkCount(n, grain) * width), width(width) {}
  float* at(size_t chunk) { return buf.data() + chunk * width; }
};

/// scores(i, j) for j <= i into `prow`, and the row max — exactly the
/// cached forward's first pass.
inline float ScoreRow(const float* qi, const float* k, size_t b, size_t S,
                      size_t D, size_t h, size_t d, size_t i,
                      float inv_sqrt_d, float* prow) {
  float row_max = -INFINITY;
  for (size_t j = 0; j <= i; ++j) {
    const float* kj = k + (b * S + j) * D + h * d;
    double dot = 0.0;
    for (size_t c = 0; c < d; ++c) dot += static_cast<double>(qi[c]) * kj[c];
    const float s = static_cast<float>(dot) * inv_sqrt_d;
    prow[j] = s;
    if (s > row_max) row_max = s;
  }
  return row_max;
}

/// p(i, j) for j <= i from the stored (m, 1/l): the floats the cached
/// forward wrote into its P row.
inline void ProbRow(const float* qi, const float* k, size_t b, size_t S,
                    size_t D, size_t h, size_t d, size_t i, float inv_sqrt_d,
                    float m, float inv_denom, float* prow) {
  for (size_t j = 0; j <= i; ++j) {
    const float* kj = k + (b * S + j) * D + h * d;
    double dot = 0.0;
    for (size_t c = 0; c < d; ++c) dot += static_cast<double>(qi[c]) * kj[c];
    const float s = static_cast<float>(dot) * inv_sqrt_d;
    const float e = std::exp(s - m);
    prow[j] = e * inv_denom;
  }
}

/// One probability p(i, j): ProbRow's expression for a single (i, j).
inline float Prob(const float* qi, const float* kj, size_t d, float inv_sqrt_d,
                  float m, float inv_denom) {
  double dot = 0.0;
  for (size_t c = 0; c < d; ++c) dot += static_cast<double>(qi[c]) * kj[c];
  const float s = static_cast<float>(dot) * inv_sqrt_d;
  const float e = std::exp(s - m);
  return e * inv_denom;
}

/// dp(i, j) = dO(i) . v(j), AttnDP's expression.
inline float DProb(const float* doi, const float* vj, size_t d) {
  double dot = 0.0;
  for (size_t c = 0; c < d; ++c) dot += static_cast<double>(doi[c]) * vj[c];
  return static_cast<float>(dot);
}

}  // namespace

void AttnFwdTiled(const float* q, const float* k, const float* v, float* o,
                  float* stats, size_t B, size_t S, size_t H, size_t d) {
  const size_t D = H * d;
  const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
  const size_t grain = RowGrain(S * (d + 4), kGrainMath);
  ChunkScratch scratch(B * H * S, grain, S);
  up::ParallelFor(B * H * S, grain,
                  [&](size_t u0, size_t u1, size_t chunk) {
    float* prow = scratch.at(chunk);
    for (size_t u = u0; u < u1; ++u) {
      const auto [b, h, i] = UnitOf(u, H, S);
      const float* qi = q + (b * S + i) * D + h * d;
      const float row_max = ScoreRow(qi, k, b, S, D, h, d, i, inv_sqrt_d, prow);
      double denom = 0.0;
      for (size_t j = 0; j <= i; ++j) {
        const float e = std::exp(prow[j] - row_max);
        prow[j] = e;
        denom += e;
      }
      const float inv_denom = 1.0f / static_cast<float>(denom);
      for (size_t j = 0; j <= i; ++j) prow[j] *= inv_denom;
      float* oi = o + (b * S + i) * D + h * d;
      for (size_t c = 0; c < d; ++c) oi[c] = 0.0f;
      for (size_t j = 0; j <= i; ++j) {
        const float p = prow[j];
        const float* vj = v + (b * S + j) * D + h * d;
        for (size_t c = 0; c < d; ++c) oi[c] += p * vj[c];
      }
      float* st = stats + u * up::kAttnStatsWidth;
      st[0] = row_max;
      st[1] = inv_denom;
      st[2] = 0.0f;  // delta: the dQ pass's
      st[3] = 0.0f;
    }
  });
}

void AttnDQTiled(const float* q, const float* k, const float* v,
                 const float* dout, float* stats, float* dq, size_t B,
                 size_t S, size_t H, size_t d) {
  const size_t D = H * d;
  const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
  // Per query row: p and dP recomputed, delta = rowsum(dP * P) in double
  // (SoftmaxRowsBwd's), dS = P * (dP - delta), dQ = dS K / sqrt(d) in
  // AttnDQ's order. delta is stored for the dK pass.
  const size_t grain = RowGrain(S * 3 * d, kGrainMath);
  ChunkScratch scratch(B * H * S, grain, 2 * S);
  up::ParallelFor(B * H * S, grain,
                  [&](size_t u0, size_t u1, size_t chunk) {
    float* prow = scratch.at(chunk);
    float* dprow = prow + S;
    for (size_t u = u0; u < u1; ++u) {
      const auto [b, h, i] = UnitOf(u, H, S);
      const float* qi = q + (b * S + i) * D + h * d;
      const float* doi = dout + (b * S + i) * D + h * d;
      float* st = stats + u * up::kAttnStatsWidth;
      ProbRow(qi, k, b, S, D, h, d, i, inv_sqrt_d, st[0], st[1], prow);
      double dot = 0.0;
      for (size_t j = 0; j <= i; ++j) {
        const float* vj = v + (b * S + j) * D + h * d;
        dprow[j] = DProb(doi, vj, d);
        dot += static_cast<double>(dprow[j]) * prow[j];
      }
      const float delta = static_cast<float>(dot);
      st[2] = delta;
      float* dqi = dq + (b * S + i) * D + h * d;
      for (size_t c = 0; c < d; ++c) dqi[c] = 0.0f;
      for (size_t j = 0; j <= i; ++j) {
        const float ds = prow[j] * (dprow[j] - delta);
        const float g = ds * inv_sqrt_d;
        const float* kj = k + (b * S + j) * D + h * d;
        for (size_t c = 0; c < d; ++c) dqi[c] += g * kj[c];
      }
    }
  });
}

void AttnDKTiled(const float* q, const float* k, const float* v,
                 const float* dout, const float* stats, float* dk, size_t B,
                 size_t S, size_t H, size_t d) {
  const size_t D = H * d;
  const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
  // Per key row j: dk(j) = sum_{i >= j} dS(i, j) q(i) / sqrt(d), with
  // dS(i, j) recomputed from (m, 1/l, delta) of row i — AttnDK's order.
  up::ParallelFor(B * H * S, RowGrain(S * 3 * d, kGrainMath),
                  [&](size_t u0, size_t u1, size_t) {
    for (size_t u = u0; u < u1; ++u) {
      const auto [b, h, j] = UnitOf(u, H, S);
      const float* kj = k + (b * S + j) * D + h * d;
      const float* vj = v + (b * S + j) * D + h * d;
      float* dkj = dk + (b * S + j) * D + h * d;
      for (size_t c = 0; c < d; ++c) dkj[c] = 0.0f;
      for (size_t i = j; i < S; ++i) {
        const float* qi = q + (b * S + i) * D + h * d;
        const float* doi = dout + (b * S + i) * D + h * d;
        const float* st = stats + ((b * H + h) * S + i) * up::kAttnStatsWidth;
        const float p = Prob(qi, kj, d, inv_sqrt_d, st[0], st[1]);
        const float ds = p * (DProb(doi, vj, d) - st[2]);
        const float g = ds * inv_sqrt_d;
        for (size_t c = 0; c < d; ++c) dkj[c] += g * qi[c];
      }
    }
  });
}

void AttnDVTiled(const float* q, const float* k, const float* dout,
                 const float* stats, float* dv, size_t B, size_t S, size_t H,
                 size_t d) {
  const size_t D = H * d;
  const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
  // Per value row j: dv(j) = sum_{i >= j} P(i, j) dO(i) — AttnDV's order.
  up::ParallelFor(B * H * S, RowGrain(S * 2 * d, kGrainMath),
                  [&](size_t u0, size_t u1, size_t) {
    for (size_t u = u0; u < u1; ++u) {
      const auto [b, h, j] = UnitOf(u, H, S);
      const float* kj = k + (b * S + j) * D + h * d;
      float* dvj = dv + (b * S + j) * D + h * d;
      for (size_t c = 0; c < d; ++c) dvj[c] = 0.0f;
      for (size_t i = j; i < S; ++i) {
        const float* qi = q + (b * S + i) * D + h * d;
        const float* doi = dout + (b * S + i) * D + h * d;
        const float* st = stats + ((b * H + h) * S + i) * up::kAttnStatsWidth;
        const float p = Prob(qi, kj, d, inv_sqrt_d, st[0], st[1]);
        for (size_t c = 0; c < d; ++c) dvj[c] += p * doi[c];
      }
    }
  });
}

}  // namespace seeml::update_rt::kernels
