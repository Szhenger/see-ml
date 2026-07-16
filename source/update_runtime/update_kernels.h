#ifndef SEEML_UPDATE_RUNTIME_UPDATE_KERNELS_H_
#define SEEML_UPDATE_RUNTIME_UPDATE_KERNELS_H_

#include <cstddef>
#include <cstdint>

// =============================================================================
// The training kernel library executed by the UpdateEngine dispatcher.
//
// These are the portable reference implementations; architecture-tuned
// variants (AVX-512 / NEON, mirroring backend/kernels/) are swapped in at
// link time. Every kernel is allocation-free and operates on caller-provided
// arena/rodata pointers — the zero-allocation contract of the update runtime.
// =============================================================================

namespace seeml::update_rt::kernels {

// --- GEMM family: C[M,N] -----------------------------------------------------
void GemmNN(const float* A, const float* B, float* C, size_t M, size_t N,
            size_t K);                       // C = A[M,K] @ B[K,N]
void GemmNT(const float* A, const float* B, float* C, size_t M, size_t N,
            size_t K);                       // C = A[M,K] @ B[N,K]^T
void GemmTN(const float* A, const float* B, float* C, size_t M, size_t N,
            size_t K);                       // C = A[K,M]^T @ B[K,N]
void GemmAccNN(const float* A, const float* B, float* C, size_t M, size_t N,
               size_t K, float alpha);       // C += alpha * A @ B

// --- Elementwise / broadcast -------------------------------------------------
void AddEW(const float* x, const float* y, float* out, size_t n);
void AddBias(const float* x, const float* b, float* out, size_t rows,
             size_t cols);
void ReluFwd(const float* x, float* out, size_t n);
void ReluBwd(const float* dy, const float* x, float* dx, size_t n);
void Scale(const float* x, float* out, float alpha, size_t n);
void ReduceRows(const float* dy, float* db, size_t rows, size_t cols);

// --- Losses ------------------------------------------------------------------
// loss = -(1/N) sum_n log softmax(logits)_n[label_n]; probs cached for bwd.
void SoftmaxXEntFwd(const float* logits, const int32_t* labels, float* loss,
                    float* probs, size_t N, size_t C);
// dlogits = seed * (probs - onehot(labels)) / N
void SoftmaxXEntBwd(const float* probs, const int32_t* labels,
                    const float* seed, float* dlogits, size_t N, size_t C);

// loss = (1/n) sum (pred - target)^2;  dpred = seed * 2 (pred - target) / n
void MseFwd(const float* pred, const float* target, float* loss, size_t n);
void MseBwd(const float* pred, const float* target, const float* seed,
            float* dpred, size_t n);

// Distillation: p = softmax(logits / T).
// loss = (1/N) sum_n KL(p_t_n || p_s_n);  dstudent = seed * (p_s - p_t)/(N*T)
void KLDistillFwd(const float* s_logits, const float* t_logits, float* loss,
                  float* p_s, float* p_t, size_t N, size_t C, float T);
void KLDistillBwd(const float* p_s, const float* p_t, const float* seed,
                  float* dlogits, size_t N, size_t C, float T);

// --- Optimizers (in-place) ---------------------------------------------------
void SgdStep(float* p, const float* g, size_t n, float lr, float weight_decay);
void AdamWStep(float* p, const float* g, float* m, float* v, size_t n,
               float lr, float beta1, float beta2, float eps,
               float weight_decay, uint64_t step);

// --- Utility -------------------------------------------------------------------
void Fill(float* dst, float value, size_t n);
void Copy(const float* src, float* dst, size_t n);

}  // namespace seeml::update_rt::kernels

#endif  // SEEML_UPDATE_RUNTIME_UPDATE_KERNELS_H_
