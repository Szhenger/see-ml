#include "runtime/update_kernels.h"

#include <cmath>
#include <cstring>

namespace seeml::update_rt::kernels {

void GemmNN(const float* A, const float* B, float* C, size_t M, size_t N,
            size_t K) {
  for (size_t m = 0; m < M; ++m) {
    float* c_row = C + m * N;
    std::memset(c_row, 0, N * sizeof(float));
    const float* a_row = A + m * K;
    for (size_t k = 0; k < K; ++k) {
      const float a = a_row[k];
      const float* b_row = B + k * N;
      for (size_t n = 0; n < N; ++n) c_row[n] += a * b_row[n];
    }
  }
}

void GemmNT(const float* A, const float* B, float* C, size_t M, size_t N,
            size_t K) {
  for (size_t m = 0; m < M; ++m) {
    const float* a_row = A + m * K;
    for (size_t n = 0; n < N; ++n) {
      const float* b_row = B + n * K;
      float acc = 0.0f;
      for (size_t k = 0; k < K; ++k) acc += a_row[k] * b_row[k];
      C[m * N + n] = acc;
    }
  }
}

void GemmTN(const float* A, const float* B, float* C, size_t M, size_t N,
            size_t K) {
  for (size_t m = 0; m < M; ++m) {
    float* c_row = C + m * N;
    std::memset(c_row, 0, N * sizeof(float));
  }
  for (size_t k = 0; k < K; ++k) {
    const float* a_row = A + k * M;
    const float* b_row = B + k * N;
    for (size_t m = 0; m < M; ++m) {
      const float a = a_row[m];
      float* c_row = C + m * N;
      for (size_t n = 0; n < N; ++n) c_row[n] += a * b_row[n];
    }
  }
}

void GemmAccNN(const float* A, const float* B, float* C, size_t M, size_t N,
               size_t K, float alpha) {
  for (size_t m = 0; m < M; ++m) {
    float* c_row = C + m * N;
    const float* a_row = A + m * K;
    for (size_t k = 0; k < K; ++k) {
      const float a = alpha * a_row[k];
      const float* b_row = B + k * N;
      for (size_t n = 0; n < N; ++n) c_row[n] += a * b_row[n];
    }
  }
}

void AddEW(const float* x, const float* y, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = x[i] + y[i];
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

void Scale(const float* x, float* out, float alpha, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = alpha * x[i];
}

void ReduceRows(const float* dy, float* db, size_t rows, size_t cols) {
  std::memset(db, 0, cols * sizeof(float));
  for (size_t r = 0; r < rows; ++r)
    for (size_t c = 0; c < cols; ++c) db[c] += dy[r * cols + c];
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
    total -= std::log(static_cast<double>(std::fmax(p, 1e-12f)));
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
