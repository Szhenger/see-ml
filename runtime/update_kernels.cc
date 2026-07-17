#include "runtime/update_kernels.h"

#include <cmath>
#include <cstring>

namespace seeml::update_rt::kernels {

namespace {

// Cache-blocking tile sizes for the GEMM family. The K/N tiles keep the
// working set (one A panel + one B panel + one C panel) inside L1/L2 for
// typical embedded cache geometries; the row-major inner loops vectorize
// under -O2 without intrinsics, keeping the reference kernels portable.
constexpr size_t kTileK = 64;
constexpr size_t kTileN = 256;

inline size_t MinZ(size_t a, size_t b) { return a < b ? a : b; }

// Shared blocked core: C[M,N] (+)= alpha * A[M,K] @ B[K,N] with B row-major.
// GemmNN/GemmAccNN/GemmTN and the q8 variants all reduce to this loop nest.
template <typename BType>
void BlockedNN(const float* A, const BType* B, float* C, size_t M, size_t N,
               size_t K, float alpha, size_t a_stride, bool a_transposed) {
  for (size_t k0 = 0; k0 < K; k0 += kTileK) {
    const size_t k1 = MinZ(k0 + kTileK, K);
    for (size_t n0 = 0; n0 < N; n0 += kTileN) {
      const size_t n1 = MinZ(n0 + kTileN, N);
      for (size_t m = 0; m < M; ++m) {
        float* c_row = C + m * N;
        for (size_t k = k0; k < k1; ++k) {
          const float a =
              alpha * (a_transposed ? A[k * a_stride + m] : A[m * a_stride + k]);
          const BType* b_row = B + k * N;
          for (size_t n = n0; n < n1; ++n)
            c_row[n] += a * static_cast<float>(b_row[n]);
        }
      }
    }
  }
}

}  // namespace

void GemmNN(const float* A, const float* B, float* C, size_t M, size_t N,
            size_t K) {
  std::memset(C, 0, M * N * sizeof(float));
  BlockedNN(A, B, C, M, N, K, 1.0f, K, /*a_transposed=*/false);
}

void GemmNT(const float* A, const float* B, float* C, size_t M, size_t N,
            size_t K) {
  // Both operands stream along K: dot-product form is already cache-friendly;
  // block N so B panels stay resident across the M loop.
  for (size_t n0 = 0; n0 < N; n0 += kTileN) {
    const size_t n1 = MinZ(n0 + kTileN, N);
    for (size_t m = 0; m < M; ++m) {
      const float* a_row = A + m * K;
      for (size_t n = n0; n < n1; ++n) {
        const float* b_row = B + n * K;
        float acc = 0.0f;
        for (size_t k = 0; k < K; ++k) acc += a_row[k] * b_row[k];
        C[m * N + n] = acc;
      }
    }
  }
}

void GemmTN(const float* A, const float* B, float* C, size_t M, size_t N,
            size_t K) {
  std::memset(C, 0, M * N * sizeof(float));
  BlockedNN(A, B, C, M, N, K, 1.0f, M, /*a_transposed=*/true);
}

void GemmAccNN(const float* A, const float* B, float* C, size_t M, size_t N,
               size_t K, float alpha) {
  BlockedNN(A, B, C, M, N, K, alpha, K, /*a_transposed=*/false);
}

void GemmNNQ8(const float* A, const int8_t* B, float* C, size_t M, size_t N,
              size_t K, float scale) {
  std::memset(C, 0, M * N * sizeof(float));
  BlockedNN(A, B, C, M, N, K, scale, K, /*a_transposed=*/false);
}

void GemmNTQ8(const float* A, const int8_t* B, float* C, size_t M, size_t N,
              size_t K, float scale) {
  for (size_t n0 = 0; n0 < N; n0 += kTileN) {
    const size_t n1 = MinZ(n0 + kTileN, N);
    for (size_t m = 0; m < M; ++m) {
      const float* a_row = A + m * K;
      for (size_t n = n0; n < n1; ++n) {
        const int8_t* b_row = B + n * K;
        float acc = 0.0f;
        for (size_t k = 0; k < K; ++k)
          acc += a_row[k] * static_cast<float>(b_row[k]);
        C[m * N + n] = scale * acc;
      }
    }
  }
}


void AddEW(const float* x, const float* y, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = x[i] + y[i];
}

void MulEW(const float* x, const float* y, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = x[i] * y[i];
}

void AddBias(const float* x, const float* b, float* out, size_t rows,
             size_t cols) {
  for (size_t r = 0; r < rows; ++r)
    for (size_t c = 0; c < cols; ++c) out[r * cols + c] = x[r * cols + c] + b[c];
}

void ReluFwd(const float* x, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

void ReluBwd(const float* dy, const float* x, float* dx, size_t n) {
  for (size_t i = 0; i < n; ++i) dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;
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
  for (size_t i = 0; i < n; ++i) {
    const float v = x[i];
    const float t = std::tanh(kGeluC * (v + kGeluA * v * v * v));
    out[i] = 0.5f * v * (1.0f + t);
  }
}

void GeluBwd(const float* dy, const float* x, float* dx, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const float v = x[i];
    const float u = kGeluC * (v + kGeluA * v * v * v);
    const float t = std::tanh(u);
    const float du = kGeluC * (1.0f + 3.0f * kGeluA * v * v);
    // d/dv [0.5 v (1 + tanh(u))] = 0.5 (1 + t) + 0.5 v (1 - t²) u'
    dx[i] = dy[i] * (0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * du);
  }
}

void SiluFwd(const float* x, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = x[i] * Sigmoid(x[i]);
}

void SiluBwd(const float* dy, const float* x, float* dx, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const float s = Sigmoid(x[i]);
    dx[i] = dy[i] * (s * (1.0f + x[i] * (1.0f - s)));
  }
}

void Scale(const float* x, float* out, float alpha, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = alpha * x[i];
}

void ReduceRows(const float* dy, float* db, size_t rows, size_t cols) {
  std::memset(db, 0, cols * sizeof(float));
  for (size_t r = 0; r < rows; ++r)
    for (size_t c = 0; c < cols; ++c) db[c] += dy[r * cols + c];
}

void LayerNormFwd(const float* x, const float* gamma, const float* beta,
                  float* y, float* mean, float* rstd, size_t rows,
                  size_t cols) {
  constexpr float kEps = 1e-5f;
  for (size_t r = 0; r < rows; ++r) {
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
}

void LayerNormBwd(const float* dy, const float* x, const float* gamma,
                  const float* mean, const float* rstd, float* dx, size_t rows,
                  size_t cols) {
  // With x̂ = (x - μ)·rstd and g = dy·γ:
  //   dx = rstd · (g - mean(g) - x̂ · mean(g·x̂))
  const float inv_d = 1.0f / static_cast<float>(cols);
  for (size_t r = 0; r < rows; ++r) {
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
  double total = 0.0;
  for (size_t n = 0; n < N; ++n) {
    RowSoftmax(logits + n * C, probs + n * C, C, 1.0f);
    const float p = probs[n * C + static_cast<size_t>(labels[n])];
    // fmax would silently scrub a NaN probability (fmax(NaN, x) == x) and
    // report a large-but-finite loss while the gradients poison the
    // parameters. Propagate NaN so the engine's finite-loss guard trips;
    // the clamp only rescues genuine underflow.
    const double safe = std::isnan(p) ? static_cast<double>(p)
                                      : static_cast<double>(std::fmax(p, 1e-12f));
    total -= std::log(safe);
  }
  *loss = static_cast<float>(total / static_cast<double>(N));
}

void SoftmaxXEntBwd(const float* probs, const int32_t* labels,
                    const float* seed, float* dlogits, size_t N, size_t C) {
  const float scale = *seed / static_cast<float>(N);
  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c)
      dlogits[n * C + c] = scale * probs[n * C + c];
    dlogits[n * C + static_cast<size_t>(labels[n])] -= scale;
  }
}

void MseFwd(const float* pred, const float* target, float* loss, size_t n) {
  double total = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(pred[i]) - target[i];
    total += d * d;
  }
  *loss = static_cast<float>(total / static_cast<double>(n));
}

void MseBwd(const float* pred, const float* target, const float* seed,
            float* dpred, size_t n) {
  const float scale = 2.0f * *seed / static_cast<float>(n);
  for (size_t i = 0; i < n; ++i) dpred[i] = scale * (pred[i] - target[i]);
}

void KLDistillFwd(const float* s_logits, const float* t_logits, float* loss,
                  float* p_s, float* p_t, size_t N, size_t C, float T) {
  const float inv_T = 1.0f / T;
  double total = 0.0;
  for (size_t n = 0; n < N; ++n) {
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
  *loss = static_cast<float>(total / static_cast<double>(N));
}

void KLDistillBwd(const float* p_s, const float* p_t, const float* seed,
                  float* dlogits, size_t N, size_t C, float T) {
  const float scale = *seed / (static_cast<float>(N) * T);
  for (size_t i = 0; i < N * C; ++i) dlogits[i] = scale * (p_s[i] - p_t[i]);
}

void ClipNorm(float* g, size_t n, float max_norm) {
  double sq = 0.0;
  for (size_t i = 0; i < n; ++i)
    sq += static_cast<double>(g[i]) * static_cast<double>(g[i]);
  const double norm = std::sqrt(sq);
  if (norm <= static_cast<double>(max_norm) || norm == 0.0) return;
  const float s = static_cast<float>(static_cast<double>(max_norm) / norm);
  for (size_t i = 0; i < n; ++i) g[i] *= s;
}

void SgdStep(float* p, const float* g, size_t n, float lr,
             float weight_decay) {
  for (size_t i = 0; i < n; ++i) p[i] -= lr * (g[i] + weight_decay * p[i]);
}

void AdamWStep(float* p, const float* g, float* m, float* v, size_t n,
               float lr, float beta1, float beta2, float eps,
               float weight_decay, uint64_t step) {
  const float bc1 =
      1.0f - std::pow(beta1, static_cast<float>(step));
  const float bc2 =
      1.0f - std::pow(beta2, static_cast<float>(step));
  for (size_t i = 0; i < n; ++i) {
    m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
    v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];
    const float m_hat = m[i] / bc1;
    const float v_hat = v[i] / bc2;
    p[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * p[i]);
  }
}

void Fill(float* dst, float value, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] = value;
}

void Copy(const float* src, float* dst, size_t n) {
  std::memcpy(dst, src, n * sizeof(float));
}

}  // namespace seeml::update_rt::kernels
