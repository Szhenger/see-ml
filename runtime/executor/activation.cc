#include <cmath>

#include "runtime/executor/kernel_policy.h"
#include "runtime/executor/update_kernels.h"
#include "source/parallel/parallel_for.h"

// =============================================================================
// Activation family: the ReLU / GELU / SiLU forward-backward pairs. Each
// backward kernel differentiates exactly its forward's expression, keeping
// the pair consistent for gradient verification.
// =============================================================================

namespace seeml::update_rt::kernels {

namespace up = seeml::update;

// The forward bodies evaluate the shared expressions of kernel_policy.h
// (ReluExpr / GeluExpr / SiluExpr) — the same inline functions the fused
// GEMM epilogues apply — so a fused and an unfused program compute
// identical bits by construction.

void ReluFwd(const float* x, float* out, size_t n) {
  up::ParallelFor(n, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) out[i] = ReluExpr(x[i]);
  });
}

void ReluBwd(const float* dy, const float* x, float* dx, size_t n) {
  up::ParallelFor(n, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;
  });
}

void GeluFwd(const float* x, float* out, size_t n) {
  up::ParallelFor(n, kGrainMath, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) out[i] = GeluExpr(x[i]);
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
    for (size_t i = b; i < e; ++i) out[i] = SiluExpr(x[i]);
  });
}

void SiluBwd(const float* dy, const float* x, float* dx, size_t n) {
  up::ParallelFor(n, kGrainMath, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) {
      const float s = SigmoidExpr(x[i]);
      dx[i] = dy[i] * (s * (1.0f + x[i] * (1.0f - s)));
    }
  });
}

}  // namespace seeml::update_rt::kernels
