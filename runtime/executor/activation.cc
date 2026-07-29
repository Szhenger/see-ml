#include <cmath>

#include "runtime/executor/kernel_policy.h"
#include "runtime/executor/update_kernels.h"
#include "source/parallel_for.h"

// =============================================================================
// Activation family: the ReLU / GELU / SiLU forward-backward pairs. Each
// backward kernel differentiates exactly its forward's expression, keeping
// the pair consistent for gradient verification.
// =============================================================================

namespace seeml::update_rt::kernels {

namespace up = seeml::update;

namespace {

// gelu(x) = 0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³))) — the tanh
// approximation.
constexpr float kGeluC = 0.7978845608028654f;  // √(2/π)
constexpr float kGeluA = 0.044715f;

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

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

}  // namespace seeml::update_rt::kernels
