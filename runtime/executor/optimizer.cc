#include <cmath>

#include "runtime/executor/kernel_policy.h"
#include "runtime/executor/update_kernels.h"
#include "source/parallel/parallel_for.h"

// =============================================================================
// Optimizer family: gradient conditioning and the in-place parameter steps.
// The clip's norm reduces with one partial per chunk, combined in chunk
// order, so clipping decisions are bitwise-deterministic at any thread count.
// =============================================================================

namespace seeml::update_rt::kernels {

namespace up = seeml::update;

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

}  // namespace seeml::update_rt::kernels
