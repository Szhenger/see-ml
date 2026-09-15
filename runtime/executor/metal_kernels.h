#ifndef SEEML_RUNTIME_EXECUTOR_METAL_KERNELS_H_
#define SEEML_RUNTIME_EXECUTOR_METAL_KERNELS_H_

// =============================================================================
// The Metal kernel library (G1b-4): Metal Shading Language source for every
// opcode the Metal backend executes on the GPU, JIT-compiled once per
// process by metal_backend.mm. Runtime-owned (not compiler-emitted) so the
// CPU and GPU kernel semantics are versioned together with the runtime
// they ship in; the compile-side kernel emitter's job — tile geometry —
// becomes a preamble the backend prepends.
//
// Per-backend determinism: every kernel is a pure function of its
// arguments with a fixed reduction order (serial per thread, or the
// simdgroup MMA order per tile), no atomics — so a dispatch is
// bitwise-reproducible run-to-run on the same device. It is NOT bitwise
// against the CPU library: the GPU contracts FMAs and has no double
// accumulators, so cross-backend comparisons are tolerance-based.
//
// Operands: every kernel takes the arena (buffer 0, writable) and rodata
// (buffer 1, read-only) as raw byte bases plus a KArgs block (buffer 2)
// carrying byte offsets, a per-operand address-space bit, dims and scalars
// — one setBuffer pair per encoder, one setBytes per dispatch, and no
// buffer-offset alignment constraints beyond the element alignment the
// validator already proved.
// =============================================================================

namespace seeml::update_rt {

inline constexpr char kMetalKernelSource[] = R"msl(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct KArgs {
  ulong off[6];
  uint space;
  uint m, n, k;
  uint rows, cols;
  uint B, S, H, D;
  uint flags;
  uint step;
  uint pad0, pad1;
  float f[8];
};

#define KSIG device uchar* ar [[buffer(0)]], device const uchar* ro [[buffer(1)]], \
             constant KArgs& p [[buffer(2)]]
#define RBASE(i) ((p.space & (1u << (i))) ? ro : (device const uchar*)ar)
#define RF(i) ((device const float*)(RBASE(i) + p.off[i]))
#define RQ(i) ((device const char*)(RBASE(i) + p.off[i]))
#define WF(i) ((device float*)(ar + p.off[i]))

// --- Activation expressions (kernel_policy.h) ------------------------------
static inline float sigmoid_expr(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float relu_expr(float x) { return x > 0.0f ? x : 0.0f; }
static inline float gelu_expr(float x) {
  const float t = tanh(0.7978845608028654f * (x + 0.044715f * x * x * x));
  return 0.5f * x * (1.0f + t);
}
static inline float silu_expr(float x) { return x * sigmoid_expr(x); }
static inline float apply_act(float v, uint act) {
  switch (act) {
    case 1: return relu_expr(v);
    case 2: return gelu_expr(v);
    case 3: return silu_expr(v);
    default: return v;
  }
}

// --- GEMM family: 64x64 output tiles, K in panels of 16, 4 simdgroups ------
// Each of the four simdgroups owns a 32x32 quadrant as 4x4 8x8 accumulators
// (simdgroup_float8x8, multiply-accumulated in a fixed K order). The
// staged tiles are zero-filled past the ragged edges; the epilogue runs
// on a threadgroup-staged copy of C so every element's write is guarded.
// AT/BT: operand stored transposed (TN: A is [K,M]; NT: B is [N,K]);
// Q8: B is int8, dequantized as scale * q on the way into the tile;
// BF16: B is bfloat16, widened exactly (its bits in the high half).
template <bool AT, bool BT, bool Q8, bool ACC, bool BF16 = false>
static inline void gemm_tile(device const float* A, device const void* Bv,
                             device float* C, device const float* bias,
                             uint M, uint N, uint K, float alpha, uint act,
                             threadgroup float* smem, uint2 tg, uint tid,
                             uint sgid) {
  const uint m0 = tg.y * 64u, n0 = tg.x * 64u;
  threadgroup float* a_tile = smem;         // 1024 floats
  threadgroup float* b_tile = smem + 1024;  // 1024 floats
  simdgroup_float8x8 acc[4][4];
  for (uint i = 0; i < 4; ++i)
    for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
  const uint sg_r = (sgid >> 1) * 32u, sg_c = (sgid & 1u) * 32u;
  device const float* Bf = (device const float*)Bv;
  device const char* Bq = (device const char*)Bv;
  device const ushort* Bh = (device const ushort*)Bv;
  for (uint k0 = 0; k0 < K; k0 += 16u) {
    for (uint e = tid; e < 1024u; e += 128u) {
      float v = 0.0f;
      if (!AT) {
        const uint r = e >> 4, kk = e & 15u;
        const uint gm = m0 + r, gk = k0 + kk;
        if (gm < M && gk < K) v = A[gm * K + gk];
        a_tile[r * 16u + kk] = v;
      } else {
        const uint kk = e >> 6, r = e & 63u;
        const uint gm = m0 + r, gk = k0 + kk;
        if (gm < M && gk < K) v = A[gk * M + gm];
        a_tile[kk * 64u + r] = v;
      }
    }
    for (uint e = tid; e < 1024u; e += 128u) {
      float v = 0.0f;
      if (!BT) {
        const uint kk = e >> 6, c = e & 63u;
        const uint gn = n0 + c, gk = k0 + kk;
        if (gn < N && gk < K)
          v = Q8     ? (float)Bq[gk * N + gn]
              : BF16 ? as_type<float>((uint)Bh[gk * N + gn] << 16)
                     : Bf[gk * N + gn];
        b_tile[kk * 64u + c] = v;
      } else {
        const uint c = e >> 4, kk = e & 15u;
        const uint gn = n0 + c, gk = k0 + kk;
        if (gn < N && gk < K)
          v = Q8     ? (float)Bq[gn * K + gk]
              : BF16 ? as_type<float>((uint)Bh[gn * K + gk] << 16)
                     : Bf[gn * K + gk];
        b_tile[c * 16u + kk] = v;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 16u; kk += 8u) {
      simdgroup_float8x8 af[4], bf[4];
      for (uint i = 0; i < 4; ++i) {
        if (!AT)
          simdgroup_load(af[i], a_tile + (sg_r + i * 8u) * 16u + kk, 16u);
        else
          simdgroup_load(af[i], a_tile + kk * 64u + sg_r + i * 8u, 64u,
                         ulong2(0, 0), true);
      }
      for (uint j = 0; j < 4; ++j) {
        if (!BT)
          simdgroup_load(bf[j], b_tile + kk * 64u + sg_c + j * 8u, 64u);
        else
          simdgroup_load(bf[j], b_tile + (sg_c + j * 8u) * 16u + kk, 16u,
                         ulong2(0, 0), true);
      }
      for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j)
          simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // Stage C [64][64] over the (now dead) A/B tiles, then a guarded
  // write-back with the epilogue.
  for (uint i = 0; i < 4; ++i)
    for (uint j = 0; j < 4; ++j)
      simdgroup_store(acc[i][j], smem + (sg_r + i * 8u) * 64u + sg_c + j * 8u,
                      64u);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < 4096u; e += 128u) {
    const uint r = e >> 6, c = e & 63u;
    const uint gm = m0 + r, gn = n0 + c;
    if (gm < M && gn < N) {
      float v = smem[e];
      if (ACC) {
        C[gm * N + gn] += alpha * v;
      } else {
        v = alpha * v;
        if (bias) v += bias[gn];
        C[gm * N + gn] = apply_act(v, act);
      }
    }
  }
}

#define GEMM_KERNEL(NAME, AT, BT, Q8, ACC, BIAS_EXPR)                        \
  GEMM_KERNEL_T(NAME, AT, BT, Q8, ACC, false, BIAS_EXPR)
#define GEMM_KERNEL_T(NAME, AT, BT, Q8, ACC, BF16, BIAS_EXPR)                \
  kernel void NAME(KSIG, uint2 tg [[threadgroup_position_in_grid]],           \
                   uint tid [[thread_index_in_threadgroup]],                  \
                   uint sgid [[simdgroup_index_in_threadgroup]]) {            \
    threadgroup float smem[4096];                                             \
    gemm_tile<AT, BT, Q8, ACC, BF16>(RF(0), (device const void*)(RBASE(1) + p.off[1]), \
                               WF(2), BIAS_EXPR, p.m, p.n, p.k, p.f[0],       \
                               p.flags & 7u, smem, tg, tid, sgid);            \
  }

GEMM_KERNEL(k_gemm_nn, false, false, false, false, ((p.flags & 8u) ? RF(3) : (device const float*)0))
GEMM_KERNEL(k_gemm_nt, false, true, false, false, (device const float*)0)
GEMM_KERNEL(k_gemm_tn, true, false, false, false, (device const float*)0)
GEMM_KERNEL(k_gemm_acc, false, false, false, true, (device const float*)0)
GEMM_KERNEL(k_gemm_nn_q8, false, false, true, false, (device const float*)0)
GEMM_KERNEL(k_gemm_nt_q8, false, true, true, false, (device const float*)0)
GEMM_KERNEL_T(k_gemm_nn_bf16, false, false, false, false, true, ((p.flags & 8u) ? RF(3) : (device const float*)0))
GEMM_KERNEL_T(k_gemm_nt_bf16, false, true, false, false, true, (device const float*)0)

// --- Elementwise (one thread per element; p.n = count) ---------------------
kernel void k_add_ew(KSIG, uint g [[thread_position_in_grid]]) {
  if (g < p.n) WF(2)[g] = RF(0)[g] + RF(1)[g];
}
kernel void k_mul_ew(KSIG, uint g [[thread_position_in_grid]]) {
  if (g < p.n) WF(2)[g] = RF(0)[g] * RF(1)[g];
}
kernel void k_add_bias(KSIG, uint g [[thread_position_in_grid]]) {
  if (g < p.rows * p.cols) WF(2)[g] = RF(0)[g] + RF(1)[g % p.cols];
}
kernel void k_relu_fwd(KSIG, uint g [[thread_position_in_grid]]) {
  if (g < p.n) WF(1)[g] = relu_expr(RF(0)[g]);
}
kernel void k_relu_bwd(KSIG, uint g [[thread_position_in_grid]]) {
  if (g < p.n) WF(2)[g] = RF(1)[g] > 0.0f ? RF(0)[g] : 0.0f;
}
kernel void k_gelu_fwd(KSIG, uint g [[thread_position_in_grid]]) {
  if (g < p.n) WF(1)[g] = gelu_expr(RF(0)[g]);
}
kernel void k_gelu_bwd(KSIG, uint g [[thread_position_in_grid]]) {
  if (g >= p.n) return;
  const float v = RF(1)[g];
  const float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
  const float t = tanh(u);
  const float one_minus_t2 = 1.0f - t * t;
  const float tail =
      one_minus_t2 == 0.0f
          ? 0.0f
          : 0.5f * v * one_minus_t2 *
                (0.7978845608028654f * (1.0f + 3.0f * 0.044715f * v * v));
  WF(2)[g] = RF(0)[g] * (0.5f * (1.0f + t) + tail);
}
kernel void k_silu_fwd(KSIG, uint g [[thread_position_in_grid]]) {
  if (g < p.n) WF(1)[g] = silu_expr(RF(0)[g]);
}
kernel void k_silu_bwd(KSIG, uint g [[thread_position_in_grid]]) {
  if (g >= p.n) return;
  const float x = RF(1)[g];
  const float s = sigmoid_expr(x);
  WF(2)[g] = RF(0)[g] * (s * (1.0f + x * (1.0f - s)));
}
kernel void k_scale(KSIG, uint g [[thread_position_in_grid]]) {
  if (g < p.n) WF(1)[g] = p.f[0] * RF(0)[g];
}
kernel void k_fill(KSIG, uint g [[thread_position_in_grid]]) {
  if (g < p.n) WF(0)[g] = p.f[0];
}
kernel void k_copy(KSIG, uint g [[thread_position_in_grid]]) {
  if (g < p.n) WF(1)[g] = RF(0)[g];
}
kernel void k_accumulate(KSIG, uint g [[thread_position_in_grid]]) {
  if (g < p.n) WF(0)[g] += RF(1)[g];
}
// p.f: lr, weight_decay
kernel void k_sgd(KSIG, uint g [[thread_position_in_grid]]) {
  if (g >= p.n) return;
  device float* pp = WF(0);
  pp[g] -= p.f[0] * (RF(1)[g] + p.f[1] * pp[g]);
}
// p.f: lr, beta1, beta2, eps, weight_decay, inv_bc1, inv_bc2
kernel void k_adamw(KSIG, uint g [[thread_position_in_grid]]) {
  if (g >= p.n) return;
  device float* pp = WF(0);
  device float* m = WF(2);
  device float* v = WF(3);
  const float lr = p.f[0], beta1 = p.f[1], beta2 = p.f[2], eps = p.f[3];
  const float wd = p.f[4], inv_bc1 = p.f[5], inv_bc2 = p.f[6];
  const float om_b1 = 1.0f - beta1, om_b2 = 1.0f - beta2;
  const float gi = RF(1)[g];
  const float mi = beta1 * m[g] + om_b1 * gi;
  const float vi = beta2 * v[g] + om_b2 * gi * gi;
  m[g] = mi;
  v[g] = vi;
  const float m_hat = mi * inv_bc1;
  const float v_hat = vi * inv_bc2;
  pp[g] -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * pp[g]);
}

// --- Row / column reductions (one thread per row or column) ----------------
kernel void k_reduce_rows(KSIG, uint c [[thread_position_in_grid]]) {
  if (c >= p.cols) return;
  device const float* dy = RF(0);
  float acc = 0.0f;
  for (uint r = 0; r < p.rows; ++r) acc += dy[r * p.cols + c];
  WF(1)[c] = acc;
}
kernel void k_layernorm_fwd(KSIG, uint r [[thread_position_in_grid]]) {
  if (r >= p.rows) return;
  const uint cols = p.cols;
  device const float* x = RF(0) + r * cols;
  device const float* gamma = RF(1);
  device const float* beta = RF(2);
  device float* y = WF(3) + r * cols;
  float sum = 0.0f;
  for (uint c = 0; c < cols; ++c) sum += x[c];
  const float mu = sum / (float)cols;
  float var = 0.0f;
  for (uint c = 0; c < cols; ++c) {
    const float d = x[c] - mu;
    var += d * d;
  }
  const float rs = 1.0f / sqrt(var / (float)cols + 1e-5f);
  WF(4)[r] = mu;
  WF(5)[r] = rs;
  for (uint c = 0; c < cols; ++c) y[c] = (x[c] - mu) * rs * gamma[c] + beta[c];
}
kernel void k_layernorm_bwd(KSIG, uint r [[thread_position_in_grid]]) {
  if (r >= p.rows) return;
  const uint cols = p.cols;
  device const float* dy = RF(0) + r * cols;
  device const float* x = RF(1) + r * cols;
  device const float* gamma = RF(2);
  device float* dx = WF(3) + r * cols;
  const float mu = RF(4)[r], rs = RF(5)[r];
  const float inv_d = 1.0f / (float)cols;
  float sum_g = 0.0f, sum_gx = 0.0f;
  for (uint c = 0; c < cols; ++c) {
    const float xhat = (x[c] - mu) * rs;
    const float g = dy[c] * gamma[c];
    sum_g += g;
    sum_gx += g * xhat;
  }
  const float mg = sum_g * inv_d, mgx = sum_gx * inv_d;
  for (uint c = 0; c < cols; ++c) {
    const float xhat = (x[c] - mu) * rs;
    const float g = dy[c] * gamma[c];
    dx[c] = rs * (g - mg - xhat * mgx);
  }
}
kernel void k_rmsnorm_fwd(KSIG, uint r [[thread_position_in_grid]]) {
  if (r >= p.rows) return;
  const uint cols = p.cols;
  device const float* x = RF(0) + r * cols;
  device const float* gamma = RF(1);
  device float* y = WF(2) + r * cols;
  float ss = 0.0f;
  for (uint c = 0; c < cols; ++c) ss += x[c] * x[c];
  const float rs = 1.0f / sqrt(ss / (float)cols + 1e-5f);
  WF(3)[r] = rs;
  for (uint c = 0; c < cols; ++c) y[c] = x[c] * rs * gamma[c];
}
kernel void k_rmsnorm_bwd(KSIG, uint r [[thread_position_in_grid]]) {
  if (r >= p.rows) return;
  const uint cols = p.cols;
  device const float* dy = RF(0) + r * cols;
  device const float* x = RF(1) + r * cols;
  device const float* gamma = RF(2);
  device float* dx = WF(3) + r * cols;
  const float rs = RF(4)[r];
  const float inv_d = 1.0f / (float)cols;
  float sum_gx = 0.0f;
  for (uint c = 0; c < cols; ++c) sum_gx += (dy[c] * gamma[c]) * x[c];
  const float mgx = sum_gx * inv_d;
  for (uint c = 0; c < cols; ++c)
    dx[c] = rs * (dy[c] * gamma[c] - x[c] * rs * rs * mgx);
}

// --- ClipNorm: the CPU's chunk geometry, partials combined in chunk order --
// p.n = count, p.k = chunk grain, p.m = chunk count, p.f[0] = max_norm.
// scratch[c] holds chunk c's partial; scratch[256] the resulting scale.
kernel void k_clip_partials(KSIG, device float* scratch [[buffer(3)]],
                            uint c [[thread_position_in_grid]]) {
  if (c >= p.m) return;
  device const float* g = RF(0);
  const uint b = c * p.k;
  const uint e = min(b + p.k, p.n);
  float sq = 0.0f;
  for (uint i = b; i < e; ++i) sq += g[i] * g[i];
  scratch[c] = sq;
}
kernel void k_clip_finish(KSIG, device float* scratch [[buffer(3)]],
                          uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  float sq = 0.0f;
  for (uint c = 0; c < p.m; ++c) sq += scratch[c];
  const float norm = sqrt(sq);
  scratch[256] = (norm <= p.f[0] || norm == 0.0f) ? 1.0f : p.f[0] / norm;
}
kernel void k_clip_apply(KSIG, device float* scratch [[buffer(3)]],
                         uint g [[thread_position_in_grid]]) {
  if (g < p.n) WF(0)[g] *= scratch[256];
}

// --- Transformer family --------------------------------------------------
// One thread per (b, s, h) unit; p.f[0] = base^(-2/d), the recurrence step.
kernel void k_rope_fwd(KSIG, uint u [[thread_position_in_grid]]) {
  const uint B = p.B, S = p.S, H = p.H, d = p.D;
  if (u >= B * S * H) return;
  const uint D = H * d;
  const uint s = (u / H) % S;
  device const float* xr = RF(0) + (u / H) * D + (u % H) * d;
  device float* yr = WF(1) + (u / H) * D + (u % H) * d;
  float freq = 1.0f;
  for (uint c = 0; c + 1 < d; c += 2) {
    const float theta = (float)s * freq;
    const float cs = cos(theta), sn = sin(theta);
    const float a = xr[c], b2 = xr[c + 1];
    yr[c] = a * cs - b2 * sn;
    yr[c + 1] = a * sn + b2 * cs;
    freq *= p.f[0];
  }
}
kernel void k_rope_bwd(KSIG, uint u [[thread_position_in_grid]]) {
  const uint B = p.B, S = p.S, H = p.H, d = p.D;
  if (u >= B * S * H) return;
  const uint D = H * d;
  const uint s = (u / H) % S;
  device const float* gr = RF(0) + (u / H) * D + (u % H) * d;
  device float* dr = WF(1) + (u / H) * D + (u % H) * d;
  float freq = 1.0f;
  for (uint c = 0; c + 1 < d; c += 2) {
    const float theta = (float)s * freq;
    const float cs = cos(theta), sn = sin(theta);
    const float a = gr[c], b2 = gr[c + 1];
    dr[c] = a * cs + b2 * sn;
    dr[c + 1] = -a * sn + b2 * cs;
    freq *= p.f[0];
  }
}
// One thread per (b, h, i) query row: P row + O segment.
kernel void k_attn_fwd(KSIG, uint u [[thread_position_in_grid]]) {
  const uint B = p.B, S = p.S, H = p.H, d = p.D;
  if (u >= B * H * S) return;
  const uint D = H * d;
  const uint b = u / (H * S), h = (u / S) % H, i = u % S;
  device const float* q = RF(0);
  device const float* k = RF(1);
  device const float* v = RF(2);
  device float* o = WF(3);
  device float* prow = WF(4) + u * S;
  device const float* qi = q + (b * S + i) * D + h * d;
  const float inv_sqrt_d = 1.0f / sqrt((float)d);
  float row_max = -INFINITY;
  for (uint j = 0; j <= i; ++j) {
    device const float* kj = k + (b * S + j) * D + h * d;
    float dot = 0.0f;
    for (uint c = 0; c < d; ++c) dot += qi[c] * kj[c];
    const float s = dot * inv_sqrt_d;
    prow[j] = s;
    if (s > row_max) row_max = s;
  }
  float denom = 0.0f;
  for (uint j = 0; j <= i; ++j) {
    const float e = exp(prow[j] - row_max);
    prow[j] = e;
    denom += e;
  }
  const float inv_denom = 1.0f / denom;
  for (uint j = 0; j <= i; ++j) prow[j] *= inv_denom;
  for (uint j = i + 1; j < S; ++j) prow[j] = 0.0f;
  device float* oi = o + (b * S + i) * D + h * d;
  for (uint c = 0; c < d; ++c) oi[c] = 0.0f;
  for (uint j = 0; j <= i; ++j) {
    const float pj = prow[j];
    device const float* vj = v + (b * S + j) * D + h * d;
    for (uint c = 0; c < d; ++c) oi[c] += pj * vj[c];
  }
}
kernel void k_attn_dp(KSIG, uint u [[thread_position_in_grid]]) {
  const uint B = p.B, S = p.S, H = p.H, d = p.D;
  if (u >= B * H * S) return;
  const uint D = H * d;
  const uint b = u / (H * S), h = (u / S) % H, i = u % S;
  device const float* doi = RF(0) + (b * S + i) * D + h * d;
  device const float* v = RF(1);
  device float* dprow = WF(2) + u * S;
  for (uint j = 0; j <= i; ++j) {
    device const float* vj = v + (b * S + j) * D + h * d;
    float dot = 0.0f;
    for (uint c = 0; c < d; ++c) dot += doi[c] * vj[c];
    dprow[j] = dot;
  }
  for (uint j = i + 1; j < S; ++j) dprow[j] = 0.0f;
}
kernel void k_attn_dv(KSIG, uint u [[thread_position_in_grid]]) {
  const uint B = p.B, S = p.S, H = p.H, d = p.D;
  if (u >= B * H * S) return;
  const uint D = H * d;
  const uint b = u / (H * S), h = (u / S) % H, j = u % S;
  device const float* probs = RF(0);
  device const float* dout = RF(1);
  device float* dvj = WF(2) + (b * S + j) * D + h * d;
  for (uint c = 0; c < d; ++c) dvj[c] = 0.0f;
  for (uint i = j; i < S; ++i) {
    const float pij = probs[((b * H + h) * S + i) * S + j];
    device const float* doi = dout + (b * S + i) * D + h * d;
    for (uint c = 0; c < d; ++c) dvj[c] += pij * doi[c];
  }
}
kernel void k_softmax_rows_bwd(KSIG, uint r [[thread_position_in_grid]]) {
  if (r >= p.rows) return;
  const uint cols = p.cols;
  device const float* pr = RF(0) + r * cols;
  device const float* dpr = RF(1) + r * cols;
  device float* dsr = WF(2) + r * cols;
  float dot = 0.0f;
  for (uint c = 0; c < cols; ++c) dot += dpr[c] * pr[c];
  for (uint c = 0; c < cols; ++c) dsr[c] = pr[c] * (dpr[c] - dot);
}
kernel void k_attn_dq(KSIG, uint u [[thread_position_in_grid]]) {
  const uint B = p.B, S = p.S, H = p.H, d = p.D;
  if (u >= B * H * S) return;
  const uint D = H * d;
  const uint b = u / (H * S), h = (u / S) % H, i = u % S;
  device const float* dsrow = RF(0) + u * S;
  device const float* k = RF(1);
  device float* dqi = WF(2) + (b * S + i) * D + h * d;
  const float inv_sqrt_d = 1.0f / sqrt((float)d);
  for (uint c = 0; c < d; ++c) dqi[c] = 0.0f;
  for (uint j = 0; j <= i; ++j) {
    const float g = dsrow[j] * inv_sqrt_d;
    device const float* kj = k + (b * S + j) * D + h * d;
    for (uint c = 0; c < d; ++c) dqi[c] += g * kj[c];
  }
}
kernel void k_attn_dk(KSIG, uint u [[thread_position_in_grid]]) {
  const uint B = p.B, S = p.S, H = p.H, d = p.D;
  if (u >= B * H * S) return;
  const uint D = H * d;
  const uint b = u / (H * S), h = (u / S) % H, j = u % S;
  device const float* ds = RF(0);
  device const float* q = RF(1);
  device float* dkj = WF(2) + (b * S + j) * D + h * d;
  const float inv_sqrt_d = 1.0f / sqrt((float)d);
  for (uint c = 0; c < d; ++c) dkj[c] = 0.0f;
  for (uint i = j; i < S; ++i) {
    const float g = ds[((b * H + h) * S + i) * S + j] * inv_sqrt_d;
    device const float* qi = q + (b * S + i) * D + h * d;
    for (uint c = 0; c < d; ++c) dkj[c] += g * qi[c];
  }
}
)msl";

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_EXECUTOR_METAL_KERNELS_H_
