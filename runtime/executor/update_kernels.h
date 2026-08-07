#ifndef SEEML_RUNTIME_EXECUTOR_UPDATE_KERNELS_H_
#define SEEML_RUNTIME_EXECUTOR_UPDATE_KERNELS_H_

#include <cstddef>
#include <cstdint>

#include "source/plan/instruction.h"  // EpilogueAct: the fused-epilogue ABI

// =============================================================================
// The training kernel library executed by the UpdateEngine dispatcher —
// the executor's façade. The implementations are partitioned per kernel
// family (gemm / elementwise / activation / normalization / loss /
// optimizer), all sharing the decomposition policy of kernel_policy.h.
//
// These are the portable reference implementations; architecture-tuned
// variants (AVX-512 / NEON, informed by compiler/backend/architecture/) are
// swapped in at link time. Every kernel is allocation-free and operates on
// caller-provided arena/rodata pointers — the zero-allocation contract of
// the update runtime.
//
// Aliasing contract: distinct pointer parameters never overlap (the
// compiler's arena allocator does not reuse an operand's slot for a result
// born at the same instruction). The implementations rely on this to
// restrict-qualify their inner loops.
// =============================================================================

namespace seeml::update_rt::kernels {

// --- GEMM family: C[M,N] -----------------------------------------------------
// The forward GEMMs take an optional fused epilogue (plan v5): after a row
// of C is fully accumulated, C[m,n] = act(C[m,n] + bias[n]) is applied in
// the write-back, per-element identical to the standalone kAddBias and
// k<Act>Fwd kernels — one pass over hot C instead of three arena
// round-trips. bias == nullptr skips the bias; act == kNone the activation.
void GemmNN(const float* A, const float* B, float* C, size_t M, size_t N,
            size_t K, const float* bias = nullptr,
            seeml::update::EpilogueAct act =
                seeml::update::EpilogueAct::kNone);  // C = A[M,K] @ B[K,N]
void GemmNT(const float* A, const float* B, float* C, size_t M, size_t N,
            size_t K);                       // C = A[M,K] @ B[N,K]^T
void GemmTN(const float* A, const float* B, float* C, size_t M, size_t N,
            size_t K);                       // C = A[K,M]^T @ B[K,N]
void GemmAccNN(const float* A, const float* B, float* C, size_t M, size_t N,
               size_t K, float alpha);       // C += alpha * A @ B

// --- Quantized GEMM: B is per-tensor symmetric int8, dequantized on the fly.
// GemmNNQ8 takes an activation epilogue only: the instruction's in[3] slot
// carries the dequant scale, so a fused bias has nowhere to ride (ABI note
// in source/plan/instruction.h).
void GemmNNQ8(const float* A, const int8_t* B, float* C, size_t M, size_t N,
              size_t K, float scale,
              seeml::update::EpilogueAct act =
                  seeml::update::EpilogueAct::kNone);
                                            // C = A[M,K] @ (scale*B)[K,N]
void GemmNTQ8(const float* A, const int8_t* B, float* C, size_t M, size_t N,
              size_t K, float scale);       // C = A[M,K] @ (scale*B)[N,K]^T

// --- Elementwise / broadcast -------------------------------------------------
void AddEW(const float* x, const float* y, float* out, size_t n);
void MulEW(const float* x, const float* y, float* out, size_t n);
void AddBias(const float* x, const float* b, float* out, size_t rows,
             size_t cols);
void ReluFwd(const float* x, float* out, size_t n);
void ReluBwd(const float* dy, const float* x, float* dx, size_t n);
void GeluFwd(const float* x, float* out, size_t n);   // tanh approximation
void GeluBwd(const float* dy, const float* x, float* dx, size_t n);
void SiluFwd(const float* x, float* out, size_t n);   // x * sigmoid(x)
void SiluBwd(const float* dy, const float* x, float* dx, size_t n);
void Scale(const float* x, float* out, float alpha, size_t n);
void ReduceRows(const float* dy, float* db, size_t rows, size_t cols);

// --- LayerNorm over the last dim of x[N,D], affine gamma/beta[D] --------------
// Forward caches per-row mean and reciprocal stddev for the backward kernel.
void LayerNormFwd(const float* x, const float* gamma, const float* beta,
                  float* y, float* mean, float* rstd, size_t rows, size_t cols);
void LayerNormBwd(const float* dy, const float* x, const float* gamma,
                  const float* mean, const float* rstd, float* dx, size_t rows,
                  size_t cols);

// --- RMSNorm over the last dim of x[N,D], affine gamma[D] (plan v6) ----------
// y = x * rstd * gamma with rstd = 1/sqrt(mean(x^2) + eps); forward caches
// per-row rstd for the backward kernel. No bias, no mean subtraction.
void RmsNormFwd(const float* x, const float* gamma, float* y, float* rstd,
                size_t rows, size_t cols);
void RmsNormBwd(const float* dy, const float* x, const float* gamma,
                const float* rstd, float* dx, size_t rows, size_t cols);

// --- Transformer family (plan v6) --------------------------------------------
// Activations are rank-2 [B*S, H*d] row-major, heads interleaved along the
// row: element (b, s, h, c) lives at [(b*S + s) * H*d + h*d + c]. The
// attention probability matrix P[B,H,S,S] is flattened [B*H*S, S]:
// P(b,h,i,j) at [((b*H + h)*S + i) * S + j].

// Rotary position embedding on interleaved pairs (2c, 2c+1) within each
// head; angle(s, c) = s * base^(-2c/d), d even. `backward` rotates by the
// negated angle — the exact transpose of the forward rotation.
void RopeFwd(const float* x, float* y, size_t B, size_t S, size_t H, size_t d,
             float base);
void RopeBwd(const float* dy, float* dx, size_t B, size_t S, size_t H,
             size_t d, float base);

// Causal scaled-dot-product attention forward: per (b, h),
//   P = softmax_rows(mask(Q K^T / sqrt(d))),  O = P V
// with the causal mask admitting j <= i. P is written for the backward
// primitives (masked entries are exactly 0).
void AttnFwd(const float* q, const float* k, const float* v, float* o,
             float* probs, size_t B, size_t S, size_t H, size_t d);
// Backward primitives (dispatched in this order by the compiled stream):
void AttnDP(const float* dout, const float* v, float* dp, size_t B, size_t S,
            size_t H, size_t d);             // dP = dO V^T
void AttnDV(const float* probs, const float* dout, float* dv, size_t B,
            size_t S, size_t H, size_t d);   // dV = P^T dO
// Row-softmax backward for any [rows, cols]: dS = P * (dP - rowsum(dP * P)).
void SoftmaxRowsBwd(const float* probs, const float* dp, float* ds,
                    size_t rows, size_t cols);
void AttnDQ(const float* ds, const float* k, float* dq, size_t B, size_t S,
            size_t H, size_t d);             // dQ = (dS K) / sqrt(d)
void AttnDK(const float* ds, const float* q, float* dk, size_t B, size_t S,
            size_t H, size_t d);             // dK = (dS^T Q) / sqrt(d)

// --- Token-native input (plan v7) --------------------------------------------
// Frozen embedding gather: out[t, :] = table[tokens[t], :]. The feeder
// contract proved every token < V before dispatch (exactly as class labels
// are bounded), so the kernel gathers blindly, per the executor doctrine.
void EmbedFwd(const int32_t* tokens, const float* table, float* out,
              size_t rows, size_t dim);

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

// --- Gradient conditioning ---------------------------------------------------
// Per-tensor L2 norm clip in place: g *= min(1, max_norm / ||g||_2).
void ClipNorm(float* g, size_t n, float max_norm);

// --- Optimizers (in-place) ---------------------------------------------------
void SgdStep(float* p, const float* g, size_t n, float lr, float weight_decay);
void AdamWStep(float* p, const float* g, float* m, float* v, size_t n,
               float lr, float beta1, float beta2, float eps,
               float weight_decay, uint64_t step);

// --- Utility -------------------------------------------------------------------
void Fill(float* dst, float value, size_t n);
void Copy(const float* src, float* dst, size_t n);

}  // namespace seeml::update_rt::kernels

#endif  // SEEML_RUNTIME_EXECUTOR_UPDATE_KERNELS_H_
