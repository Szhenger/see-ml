#include <cmath>

#include "runtime/executor/kernel_policy.h"
#include "runtime/executor/update_kernels.h"
#include "source/parallel/parallel_for.h"

// =============================================================================
// Normalization family: LayerNorm over the last dim, forward caching
// per-row mean and reciprocal stddev for the backward kernel. Row sums
// accumulate in double so the cached statistics are stable at any width.
// =============================================================================

namespace seeml::update_rt::kernels {

namespace up = seeml::update;

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

void RmsNormFwd(const float* x, const float* gamma, float* y, float* rstd,
                size_t rows, size_t cols) {
  constexpr float kEps = 1e-5f;
  up::ParallelFor(rows, RowGrain(cols, kGrainMath), [&](size_t r0, size_t r1,
                                                        size_t) {
    for (size_t r = r0; r < r1; ++r) {
      const float* xr = x + r * cols;
      double ss = 0.0;
      for (size_t c = 0; c < cols; ++c)
        ss += static_cast<double>(xr[c]) * xr[c];
      const float rs = 1.0f / std::sqrt(
          static_cast<float>(ss / static_cast<double>(cols)) + kEps);
      rstd[r] = rs;
      float* yr = y + r * cols;
      for (size_t c = 0; c < cols; ++c) yr[c] = xr[c] * rs * gamma[c];
    }
  });
}

void RmsNormBwd(const float* dy, const float* x, const float* gamma,
                const float* rstd, float* dx, size_t rows, size_t cols) {
  // With r = rstd and g = dy·γ:  dx = r · (g - x · r² · mean(g·x))
  const float inv_d = 1.0f / static_cast<float>(cols);
  up::ParallelFor(rows, RowGrain(cols, kGrainMath), [&](size_t r0, size_t r1,
                                                        size_t) {
    for (size_t r = r0; r < r1; ++r) {
      const float* dyr = dy + r * cols;
      const float* xr = x + r * cols;
      float* dxr = dx + r * cols;
      const float rs = rstd[r];
      double sum_gx = 0.0;
      for (size_t c = 0; c < cols; ++c)
        sum_gx += static_cast<double>(dyr[c] * gamma[c]) * xr[c];
      const float mgx = static_cast<float>(sum_gx) * inv_d;
      for (size_t c = 0; c < cols; ++c)
        dxr[c] = rs * (dyr[c] * gamma[c] - xr[c] * rs * rs * mgx);
    }
  });
}

}  // namespace seeml::update_rt::kernels
