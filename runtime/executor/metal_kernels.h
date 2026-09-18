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
  // GEMM family only: leading dimensions, batch geometry and the per-
  // operand batch strides (elements): bs[2*i] per outer batch, bs[2*i+1]
  // per inner batch, for operands A, B, C.
  uint lda, ldb, ldc;
  uint batch, batch_h;
  uint pad2, pad3, pad4;
  ulong bs[6];
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

// --- GEMM family --------------------------------------------------------
// Four kernels per variant, chosen by shape on the host. The TILED kernel
// computes 64x64 output tiles with K in panels of 16 and 128 threads = 4
// simdgroups, each owning a 32x32 quadrant as 4x4 simdgroup_float8x8
// accumulators (multiply-accumulated in a fixed K order; every loop over
// them is fully unrolled so they live in registers). Every panel is two
// slabs (A: 64 rows x 16 k, B: 16 k x 64 cols, or the transposed shapes)
// that each thread fetches as two 4-vectors ONE PANEL AHEAD into
// registers and stores to threadgroup memory after the current panel's
// MMAs — global latency overlaps the math. Vector loads need 16-byte
// alignment: the host sets flag bits 4/5 when an operand's byte offset
// and leading dimension allow them; otherwise, and at the ragged edges,
// elements load one at a time (zero past the edge). The skinny kernels
// (K <= 16: one thread per output; N <= 16: one simdgroup per row with a
// fixed shuffle tree; M <= 16: one thread per column) take the shapes a
// 64x64 tile would waste — an adapter's rank-8 factors, a K=8 product.
//
// Every variant is STRIDED and BATCHED: p.lda/ldb/ldc are the leading
// dimensions (elements) and threadgroup/thread z selects a batch whose
// operand offsets are z/batch_h * bs[b] + z%batch_h * bs[h] elements — so
// the attention family runs as GEMMs over the [B*S, H*d] activations
// without any copy. AT/BT: operand stored transposed (TN: A is [K,M]; NT:
// B is [N,K]); Q8: B is int8, dequantized as scale * q; BF16: B is
// bfloat16, widened exactly (its bits in the high half).
static inline float widen(float v) { return v; }
static inline float widen(char v) { return (float)v; }
static inline float widen(ushort v) { return as_type<float>((uint)v << 16); }
static inline float4 widen4(float4 v) { return v; }
static inline float4 widen4(char4 v) { return float4(v); }
static inline float4 widen4(ushort4 v) { return as_type<float4>(uint4(v) << 16); }

// Slab P1: 64 rows x 16 along the contiguous k axis (A when !AT, B when
// BT). Vector v in 0..255: row v>>2, k quad (v&3)*4.
template <typename E>
static inline float4 fetch_p1(device const E* src, uint ld, uint row0,
                              uint rows, uint k0, uint K, uint v, bool vec) {
  const uint r = v >> 2, kq = (v & 3u) * 4u;
  const uint gr = row0 + r, gk = k0 + kq;
  if (gr >= rows) return float4(0.0f);
  if (vec && gk + 4u <= K)
    return widen4(*(device const ::vec<E, 4>*)(src + gr * ld + gk));
  float4 o = float4(0.0f);
  for (uint t = 0; t < 4u; ++t)
    if (gk + t < K) o[t] = widen(src[gr * ld + gk + t]);
  return o;
}
// Slab P2: 16 k rows x 64 along the contiguous column axis (B when !BT,
// A when AT). Vector v in 0..255: k row v>>4, column quad (v&15)*4.
template <typename E>
static inline float4 fetch_p2(device const E* src, uint ld, uint k0, uint K,
                              uint col0, uint cols, uint v, bool vec) {
  const uint kr = v >> 4, cq = (v & 15u) * 4u;
  const uint gk = k0 + kr, gc = col0 + cq;
  if (gk >= K) return float4(0.0f);
  if (vec && gc + 4u <= cols)
    return widen4(*(device const ::vec<E, 4>*)(src + gk * ld + gc));
  float4 o = float4(0.0f);
  for (uint t = 0; t < 4u; ++t)
    if (gc + t < cols) o[t] = widen(src[gk * ld + gc + t]);
  return o;
}

#define LDP1 20u  // [64][16] slab, stride padded to 20 floats
#define LDP2 68u  // [16][64] slab, stride padded to 68 floats
#define GEMM_SMEM 2560u  // two [64][20] slabs (NT) is the largest; also the 32x64 epilogue stage

// Batch offsets (elements) for batch index z.
static inline ulong batch_off(constant KArgs& p, uint z, uint which) {
  return (ulong)(z / p.batch_h) * p.bs[2u * which] +
         (ulong)(z % p.batch_h) * p.bs[2u * which + 1u];
}

// Split-K (p.pad0 = splits > 1, batch == 1 only): threadgroup z owns K
// range [z*kper, min(K, (z+1)*kper)) and writes its raw partial tile to
// the workspace (buffer 4) at z*M*N; k_gemm_splitk_fin then sums the
// partials in split order and applies the epilogue — a fixed order, so
// the result is bitwise-reproducible run-to-run like every kernel here.
template <bool AT, bool BT, typename EB, bool ACC>
static inline void gemm_tile(device const float* A, device const EB* B,
                             device float* C, device const float* bias,
                             uint M, uint N, uint K, uint lda, uint ldb,
                             uint ldc, float alpha, uint act, bool vecA,
                             bool vecB, uint splits, device float* ws,
                             threadgroup float* smem, uint3 tg, uint tid,
                             uint sgid) {
  const uint m0 = tg.y * 64u, n0 = tg.x * 64u;
  uint kbeg = 0, kend = K;
  if (splits > 1u) {
    const uint kper = ((K + splits - 1u) / splits + 15u) & ~15u;
    kbeg = min(K, tg.z * kper);
    kend = min(K, kbeg + kper);
    ws += (ulong)tg.z * (ulong)M * (ulong)N;
  }
  threadgroup float* a_tile = smem;  // [64][LDP1] or [16][LDP2]
  threadgroup float* b_tile = smem + (AT ? 16u * LDP2 : 64u * LDP1);
  simdgroup_float8x8 acc[4][4];
#pragma clang loop unroll(full)
  for (uint i = 0; i < 4; ++i)
#pragma clang loop unroll(full)
    for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
  const uint sg_r = (sgid >> 1) * 32u, sg_c = (sgid & 1u) * 32u;
  const uint v0 = tid, v1 = tid + 128u;
  float4 ra0, ra1, rb0, rb1;
#define GEMM_FETCH(K0)                                                        \
  if (!AT) {                                                                  \
    ra0 = fetch_p1(A, lda, m0, M, (K0), K, v0, vecA);                        \
    ra1 = fetch_p1(A, lda, m0, M, (K0), K, v1, vecA);                        \
  } else {                                                                    \
    ra0 = fetch_p2(A, lda, (K0), K, m0, M, v0, vecA);                        \
    ra1 = fetch_p2(A, lda, (K0), K, m0, M, v1, vecA);                        \
  }                                                                           \
  if (!BT) {                                                                  \
    rb0 = fetch_p2(B, ldb, (K0), K, n0, N, v0, vecB);                        \
    rb1 = fetch_p2(B, ldb, (K0), K, n0, N, v1, vecB);                        \
  } else {                                                                    \
    rb0 = fetch_p1(B, ldb, n0, N, (K0), K, v0, vecB);                        \
    rb1 = fetch_p1(B, ldb, n0, N, (K0), K, v1, vecB);                        \
  }
  GEMM_FETCH(kbeg)
  for (uint k0 = kbeg; k0 < kend; k0 += 16u) {
    if (!AT) {
      *(threadgroup float4*)(a_tile + (v0 >> 2) * LDP1 + (v0 & 3u) * 4u) = ra0;
      *(threadgroup float4*)(a_tile + (v1 >> 2) * LDP1 + (v1 & 3u) * 4u) = ra1;
    } else {
      *(threadgroup float4*)(a_tile + (v0 >> 4) * LDP2 + (v0 & 15u) * 4u) = ra0;
      *(threadgroup float4*)(a_tile + (v1 >> 4) * LDP2 + (v1 & 15u) * 4u) = ra1;
    }
    if (!BT) {
      *(threadgroup float4*)(b_tile + (v0 >> 4) * LDP2 + (v0 & 15u) * 4u) = rb0;
      *(threadgroup float4*)(b_tile + (v1 >> 4) * LDP2 + (v1 & 15u) * 4u) = rb1;
    } else {
      *(threadgroup float4*)(b_tile + (v0 >> 2) * LDP1 + (v0 & 3u) * 4u) = rb0;
      *(threadgroup float4*)(b_tile + (v1 >> 2) * LDP1 + (v1 & 3u) * 4u) = rb1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 + 16u < kend) { GEMM_FETCH(k0 + 16u) }
#pragma clang loop unroll(full)
    for (uint kk = 0; kk < 16u; kk += 8u) {
      simdgroup_float8x8 af[4], bf[4];
#pragma clang loop unroll(full)
      for (uint i = 0; i < 4; ++i) {
        if (!AT)
          simdgroup_load(af[i], a_tile + (sg_r + i * 8u) * LDP1 + kk, LDP1);
        else
          simdgroup_load(af[i], a_tile + kk * LDP2 + sg_r + i * 8u, LDP2,
                         ulong2(0, 0), true);
      }
#pragma clang loop unroll(full)
      for (uint j = 0; j < 4; ++j) {
        if (!BT)
          simdgroup_load(bf[j], b_tile + kk * LDP2 + sg_c + j * 8u, LDP2);
        else
          simdgroup_load(bf[j], b_tile + (sg_c + j * 8u) * LDP1 + kk, LDP1,
                         ulong2(0, 0), true);
      }
#pragma clang loop unroll(full)
      for (uint i = 0; i < 4; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j)
          simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#undef GEMM_FETCH
  // Epilogue: stage each 32-row half of C over the (now dead) slabs, then
  // a guarded write-back with alpha / bias / activation (or accumulate).
  for (uint hf = 0; hf < 2u; ++hf) {
    if ((sgid >> 1) == hf) {
#pragma clang loop unroll(full)
      for (uint i = 0; i < 4; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j)
          simdgroup_store(acc[i][j], smem + (i * 8u) * 64u + sg_c + j * 8u, 64u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 2048u; e += 128u) {
      const uint r = e >> 6, c = e & 63u;
      const uint gm = m0 + hf * 32u + r, gn = n0 + c;
      if (gm < M && gn < N) {
        float v = smem[e];
        if (splits > 1u) {
          ws[gm * N + gn] = v;
        } else if (ACC) {
          C[gm * ldc + gn] += alpha * v;
        } else {
          v = alpha * v;
          if (bias) v += bias[gn];
          C[gm * ldc + gn] = apply_act(v, act);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
// The split-K finish: partials summed in split order, then the epilogue.
// p.pad0 = splits; flags bit 6 = accumulate into C (the GemmAcc form).
kernel void k_gemm_splitk_fin(KSIG, device const float* ws [[buffer(4)]],
                              uint g [[thread_position_in_grid]]) {
  const uint total = p.m * p.n;
  if (g >= total) return;
  float v = 0.0f;
  for (uint z = 0; z < p.pad0; ++z) v += ws[(ulong)z * total + g];
  const uint gm = g / p.n, gn = g % p.n;
  device float* C = WF(2) + gm * p.ldc + gn;
  if (p.flags & 64u) {
    *C += p.f[0] * v;
  } else {
    v = p.f[0] * v;
    if (p.flags & 8u) v += RF(3)[gn];
    *C = apply_act(v, p.flags & 7u);
  }
}

// K <= 16: one thread per output element, serial K.
template <bool AT, bool BT, typename EB, bool ACC>
static inline void gemm_small(device const float* A, device const EB* B,
                              device float* C, device const float* bias,
                              uint M, uint N, uint K, uint lda, uint ldb,
                              uint ldc, float alpha, uint act, uint g) {
  const uint m = g / N, n = g % N;
  float acc = 0.0f;
  for (uint k = 0; k < K; ++k) {
    const float a = AT ? A[k * lda + m] : A[m * lda + k];
    const float b = widen(BT ? B[n * ldb + k] : B[k * ldb + n]);
    acc += a * b;
  }
  if (ACC) {
    C[m * ldc + n] += alpha * acc;
  } else {
    float v = alpha * acc;
    if (bias) v += bias[n];
    C[m * ldc + n] = apply_act(v, act);
  }
}
// N <= 16: one simdgroup per output row, the lanes striding K, the lane
// partials combined by a fixed shuffle tree (deterministic). Every lane
// keeps 16 accumulators in registers: the column index is clamped (never
// branched on) so the unrolled loop stays register-only; columns past N
// compute a harmless duplicate that is never stored.
template <bool AT, bool BT, typename EB, bool ACC>
static inline void gemm_rows(device const float* A, device const EB* B,
                             device float* C, device const float* bias,
                             uint M, uint N, uint K, uint lda, uint ldb,
                             uint ldc, float alpha, uint act, uint m,
                             uint lane) {
  float acc[16];
#pragma clang loop unroll(full)
  for (uint n = 0; n < 16u; ++n) acc[n] = 0.0f;
  for (uint k = lane; k < K; k += 32u) {
    const float a = AT ? A[k * lda + m] : A[m * lda + k];
#pragma clang loop unroll(full)
    for (uint n = 0; n < 16u; ++n) {
      const uint nc = min(n, N - 1u);
      acc[n] += a * widen(BT ? B[nc * ldb + k] : B[k * ldb + nc]);
    }
  }
#pragma clang loop unroll(full)
  for (uint n = 0; n < 16u; ++n) {
    float v = acc[n];
    v += simd_shuffle_down(v, 16u);
    v += simd_shuffle_down(v, 8u);
    v += simd_shuffle_down(v, 4u);
    v += simd_shuffle_down(v, 2u);
    v += simd_shuffle_down(v, 1u);
    acc[n] = v;
  }
  if (lane != 0u) return;
  for (uint n = 0; n < N; ++n) {
    if (ACC) {
      C[m * ldc + n] += alpha * acc[n];
    } else {
      float v = alpha * acc[n];
      if (bias) v += bias[n];
      C[m * ldc + n] = apply_act(v, act);
    }
  }
}
// M <= 16: one thread per output column, serial K, all rows in registers
// (row index clamped, as gemm_rows).
template <bool AT, bool BT, typename EB, bool ACC>
static inline void gemm_cols(device const float* A, device const EB* B,
                             device float* C, device const float* bias,
                             uint M, uint N, uint K, uint lda, uint ldb,
                             uint ldc, float alpha, uint act, uint n) {
  float acc[16];
#pragma clang loop unroll(full)
  for (uint m = 0; m < 16u; ++m) acc[m] = 0.0f;
  for (uint k = 0; k < K; ++k) {
    const float b = widen(BT ? B[n * ldb + k] : B[k * ldb + n]);
#pragma clang loop unroll(full)
    for (uint m = 0; m < 16u; ++m) {
      const uint mc = min(m, M - 1u);
      acc[m] += (AT ? A[k * lda + mc] : A[mc * lda + k]) * b;
    }
  }
  const float bv = bias ? bias[n] : 0.0f;
  for (uint m = 0; m < M; ++m) {
    if (ACC) {
      C[m * ldc + n] += alpha * acc[m];
    } else {
      C[m * ldc + n] = apply_act(alpha * acc[m] + bv, act);
    }
  }
}

// M <= 16, N below the coalescing threshold: one simdgroup per column,
// lanes striding K, the same fixed shuffle tree as gemm_rows.
template <bool AT, bool BT, typename EB, bool ACC>
static inline void gemm_colsg(device const float* A, device const EB* B,
                              device float* C, device const float* bias,
                              uint M, uint N, uint K, uint lda, uint ldb,
                              uint ldc, float alpha, uint act, uint n,
                              uint lane) {
  float acc[16];
#pragma clang loop unroll(full)
  for (uint m = 0; m < 16u; ++m) acc[m] = 0.0f;
  for (uint k = lane; k < K; k += 32u) {
    const float b = widen(BT ? B[n * ldb + k] : B[k * ldb + n]);
#pragma clang loop unroll(full)
    for (uint m = 0; m < 16u; ++m) {
      const uint mc = min(m, M - 1u);
      acc[m] += (AT ? A[k * lda + mc] : A[mc * lda + k]) * b;
    }
  }
#pragma clang loop unroll(full)
  for (uint m = 0; m < 16u; ++m) {
    float v = acc[m];
    v += simd_shuffle_down(v, 16u);
    v += simd_shuffle_down(v, 8u);
    v += simd_shuffle_down(v, 4u);
    v += simd_shuffle_down(v, 2u);
    v += simd_shuffle_down(v, 1u);
    acc[m] = v;
  }
  if (lane != 0u) return;
  const float bv = bias ? bias[n] : 0.0f;
  for (uint m = 0; m < M; ++m) {
    if (ACC) {
      C[m * ldc + n] += alpha * acc[m];
    } else {
      C[m * ldc + n] = apply_act(alpha * acc[m] + bv, act);
    }
  }
}

#define GEMM_OPERANDS(Z, EB, BIAS_EXPR)                                       \
  RF(0) + batch_off(p, (Z), 0u),                                              \
      (device const EB*)(RBASE(1) + p.off[1]) + batch_off(p, (Z), 1u),        \
      WF(2) + batch_off(p, (Z), 2u), BIAS_EXPR, p.m, p.n, p.k, p.lda,         \
      p.ldb, p.ldc, p.f[0], p.flags & 7u
#define GEMM_KERNELS(NAME, AT, BT, EB, ACC, BIAS_EXPR)                        \
  kernel void NAME(KSIG, device float* ws [[buffer(4)]],                      \
                   uint3 tg [[threadgroup_position_in_grid]],                 \
                   uint tid [[thread_index_in_threadgroup]],                  \
                   uint sgid [[simdgroup_index_in_threadgroup]]) {            \
    threadgroup float smem[GEMM_SMEM];                                        \
    const uint z = p.pad0 > 1u ? 0u : tg.z;                                   \
    gemm_tile<AT, BT, EB, ACC>(GEMM_OPERANDS(z, EB, BIAS_EXPR), (p.flags & 16u) != 0u,       \
                               (p.flags & 32u) != 0u, p.pad0, ws, smem, tg,   \
                               tid, sgid);                                    \
  }                                                                           \
  kernel void NAME##_small(KSIG, uint g [[thread_position_in_grid]]) {        \
    const uint per = p.m * p.n, z = g / per;                                  \
    if (z >= p.batch) return;                                                 \
    gemm_small<AT, BT, EB, ACC>(GEMM_OPERANDS(z, EB, BIAS_EXPR), g - z * per);               \
  }                                                                           \
  kernel void NAME##_rows(KSIG, uint g [[thread_position_in_grid]],           \
                          uint lane [[thread_index_in_simdgroup]]) {          \
    const uint row = g >> 5, z = row / p.m;                                   \
    if (z >= p.batch) return;                                                 \
    gemm_rows<AT, BT, EB, ACC>(GEMM_OPERANDS(z, EB, BIAS_EXPR), row - z * p.m, lane);        \
  }                                                                           \
  kernel void NAME##_cols(KSIG, uint g [[thread_position_in_grid]]) {         \
    const uint z = g / p.n;                                                   \
    if (z >= p.batch) return;                                                 \
    gemm_cols<AT, BT, EB, ACC>(GEMM_OPERANDS(z, EB, BIAS_EXPR), g - z * p.n);                \
  }                                                                           \
  kernel void NAME##_colsg(KSIG, uint g [[thread_position_in_grid]],          \
                           uint lane [[thread_index_in_simdgroup]]) {         \
    const uint col = g >> 5, z = col / p.n;                                   \
    if (z >= p.batch) return;                                                 \
    gemm_colsg<AT, BT, EB, ACC>(GEMM_OPERANDS(z, EB, BIAS_EXPR), col - z * p.n, lane);       \
  }

GEMM_KERNELS(k_gemm_nn, false, false, float, false, ((p.flags & 8u) ? RF(3) : (device const float*)0))
GEMM_KERNELS(k_gemm_nt, false, true, float, false, (device const float*)0)
GEMM_KERNELS(k_gemm_tn, true, false, float, false, (device const float*)0)
GEMM_KERNELS(k_gemm_acc, false, false, float, true, (device const float*)0)
GEMM_KERNELS(k_gemm_nn_q8, false, false, char, false, (device const float*)0)
GEMM_KERNELS(k_gemm_nt_q8, false, true, char, false, (device const float*)0)
GEMM_KERNELS(k_gemm_nn_bf16, false, false, ushort, false, ((p.flags & 8u) ? RF(3) : (device const float*)0))
GEMM_KERNELS(k_gemm_nt_bf16, false, true, ushort, false, (device const float*)0)

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
// The fused elementwise chain (plan v13): p.flags = the stage bytes, first
// stage low; p.f[0..1] = the scale immediates; operand slots as on the
// CPU (0 = x, 1..2 = the binary stages' tensors, 3 = out). One thread per
// element runs the stages in order, each in a statement of its own, so a
// stage's product is rounded before the next stage's add can see it — the
// values the k_scale / k_add_ew / k_mul_ew sequence would have stored.
kernel void k_fused_map(KSIG, uint g [[thread_position_in_grid]]) {
  if (g >= p.n) return;
  float run = RF(0)[g];
  for (uint s = 0; s < 4; ++s) {
    const uint stage = (p.flags >> (8u * s)) & 0xFFu;
    const uint kind = stage & 0x0Fu;
    if (kind == 0u) break;
    const uint arg = (stage >> 4u) & 0x3u;
    const bool right = (stage & 0x80u) != 0u;
    float next = run;
    if (kind == 1u) {
      const float y = (arg == 1u ? RF(1) : RF(2))[g];
      next = right ? y + run : run + y;
    } else if (kind == 2u) {
      const float y = (arg == 1u ? RF(1) : RF(2))[g];
      next = right ? y * run : run * y;
    } else if (kind == 3u) {
      next = p.f[arg] * run;
    } else if (kind == 4u) {
      next = relu_expr(run);
    } else if (kind == 5u) {
      next = gelu_expr(run);
    } else if (kind == 6u) {
      next = silu_expr(run);
    }
    run = next;
  }
  WF(3)[g] = run;
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
// The fused-clip steps (plan v12): g scaled by the factor k_clip_finish
// left in scratch[256], never written back; g * s is a
// product in a statement of its own, as the CPU kernels do.
kernel void k_sgd_clip(KSIG, device float* scratch [[buffer(3)]],
                       uint g [[thread_position_in_grid]]) {
  if (g >= p.n) return;
  device float* pp = WF(0);
  const float gi = RF(1)[g] * scratch[256];
  pp[g] -= p.f[0] * (gi + p.f[1] * pp[g]);
}
kernel void k_adamw_clip(KSIG, device float* scratch [[buffer(3)]],
                         uint g [[thread_position_in_grid]]) {
  if (g >= p.n) return;
  device float* pp = WF(0);
  device float* m = WF(2);
  device float* v = WF(3);
  const float lr = p.f[0], beta1 = p.f[1], beta2 = p.f[2], eps = p.f[3];
  const float wd = p.f[4], inv_bc1 = p.f[5], inv_bc2 = p.f[6];
  const float om_b1 = 1.0f - beta1, om_b2 = 1.0f - beta2;
  const float gi = RF(1)[g] * scratch[256];
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
// The normalization family: one simdgroup per row, columns strided by
// lane, every row reduction a fixed shuffle tree (deterministic).
static inline float simd_sum_tree(float v) {
  v += simd_shuffle_xor(v, 16u);
  v += simd_shuffle_xor(v, 8u);
  v += simd_shuffle_xor(v, 4u);
  v += simd_shuffle_xor(v, 2u);
  v += simd_shuffle_xor(v, 1u);
  return v;
}
kernel void k_layernorm_fwd(KSIG, uint g [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
  const uint r = g >> 5;
  if (r >= p.rows) return;
  const uint cols = p.cols;
  device const float* x = RF(0) + (ulong)r * cols;
  device const float* gamma = RF(1);
  device const float* beta = RF(2);
  device float* y = WF(3) + (ulong)r * cols;
  float sum = 0.0f;
  for (uint c = lane; c < cols; c += 32u) sum += x[c];
  const float mu = simd_sum_tree(sum) / (float)cols;
  float var = 0.0f;
  for (uint c = lane; c < cols; c += 32u) {
    const float d = x[c] - mu;
    var += d * d;
  }
  const float rs = 1.0f / sqrt(simd_sum_tree(var) / (float)cols + p.f[0]);
  if (lane == 0u) {
    WF(4)[r] = mu;
    WF(5)[r] = rs;
  }
  for (uint c = lane; c < cols; c += 32u)
    y[c] = (x[c] - mu) * rs * gamma[c] + beta[c];
}
kernel void k_layernorm_bwd(KSIG, uint g [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
  const uint r = g >> 5;
  if (r >= p.rows) return;
  const uint cols = p.cols;
  device const float* dy = RF(0) + (ulong)r * cols;
  device const float* x = RF(1) + (ulong)r * cols;
  device const float* gamma = RF(2);
  device float* dx = WF(3) + (ulong)r * cols;
  const float mu = RF(4)[r], rs = RF(5)[r];
  const float inv_d = 1.0f / (float)cols;
  float sum_g = 0.0f, sum_gx = 0.0f;
  for (uint c = lane; c < cols; c += 32u) {
    const float xhat = (x[c] - mu) * rs;
    const float gg = dy[c] * gamma[c];
    sum_g += gg;
    sum_gx += gg * xhat;
  }
  const float mg = simd_sum_tree(sum_g) * inv_d;
  const float mgx = simd_sum_tree(sum_gx) * inv_d;
  for (uint c = lane; c < cols; c += 32u) {
    const float xhat = (x[c] - mu) * rs;
    const float gg = dy[c] * gamma[c];
    dx[c] = rs * (gg - mg - xhat * mgx);
  }
}
kernel void k_rmsnorm_fwd(KSIG, uint g [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  const uint r = g >> 5;
  if (r >= p.rows) return;
  const uint cols = p.cols;
  device const float* x = RF(0) + (ulong)r * cols;
  device const float* gamma = RF(1);
  device float* y = WF(2) + (ulong)r * cols;
  float ss = 0.0f;
  for (uint c = lane; c < cols; c += 32u) ss += x[c] * x[c];
  const float rs = 1.0f / sqrt(simd_sum_tree(ss) / (float)cols + p.f[0]);
  if (lane == 0u) WF(3)[r] = rs;
  for (uint c = lane; c < cols; c += 32u) y[c] = x[c] * rs * gamma[c];
}
kernel void k_rmsnorm_bwd(KSIG, uint g [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  const uint r = g >> 5;
  if (r >= p.rows) return;
  const uint cols = p.cols;
  device const float* dy = RF(0) + (ulong)r * cols;
  device const float* x = RF(1) + (ulong)r * cols;
  device const float* gamma = RF(2);
  device float* dx = WF(3) + (ulong)r * cols;
  const float rs = RF(4)[r];
  const float inv_d = 1.0f / (float)cols;
  float sum_gx = 0.0f;
  for (uint c = lane; c < cols; c += 32u) sum_gx += (dy[c] * gamma[c]) * x[c];
  const float mgx = simd_sum_tree(sum_gx) * inv_d;
  for (uint c = lane; c < cols; c += 32u)
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
// Causal row softmax over the scores the QK^T GEMM left in the P cache:
// one simdgroup per row (b, h, i), four columns per lane when S = 128,
// max and sum by fixed shuffle trees (deterministic), masked columns
// j > i written as exact zeros — the layout AttnFwd's CPU kernel leaves.
// p.rows = B*H*S, p.cols = S.
kernel void k_attn_softmax(KSIG, uint g [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
  const uint r = g >> 5;
  if (r >= p.rows) return;
  const uint S = p.cols, i = r % S;
  device float* row = WF(0) + (ulong)r * S;
  float m = -INFINITY;
  for (uint j = lane; j <= i; j += 32u) m = max(m, row[j]);
  m = max(m, simd_shuffle_xor(m, 16u));
  m = max(m, simd_shuffle_xor(m, 8u));
  m = max(m, simd_shuffle_xor(m, 4u));
  m = max(m, simd_shuffle_xor(m, 2u));
  m = max(m, simd_shuffle_xor(m, 1u));
  float sum = 0.0f;
  for (uint j = lane; j <= i; j += 32u) {
    const float e = exp(row[j] - m);
    row[j] = e;
    sum += e;
  }
  sum += simd_shuffle_xor(sum, 16u);
  sum += simd_shuffle_xor(sum, 8u);
  sum += simd_shuffle_xor(sum, 4u);
  sum += simd_shuffle_xor(sum, 2u);
  sum += simd_shuffle_xor(sum, 1u);
  const float inv = 1.0f / sum;
  for (uint j = lane; j <= i; j += 32u) row[j] *= inv;
  for (uint j = i + 1u + lane; j < S; j += 32u) row[j] = 0.0f;
}
// dS = P * (dP - rowsum(dP * P)): one simdgroup per row, fixed shuffle
// tree for the rowsum. Masked entries carry p == 0, so their dS is 0.
kernel void k_softmax_rows_bwd(KSIG, uint g [[thread_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
  const uint r = g >> 5;
  if (r >= p.rows) return;
  const uint cols = p.cols;
  device const float* pr = RF(0) + (ulong)r * cols;
  device const float* dpr = RF(1) + (ulong)r * cols;
  device float* dsr = WF(2) + (ulong)r * cols;
  float dot = 0.0f;
  for (uint c = lane; c < cols; c += 32u) dot += dpr[c] * pr[c];
  dot += simd_shuffle_xor(dot, 16u);
  dot += simd_shuffle_xor(dot, 8u);
  dot += simd_shuffle_xor(dot, 4u);
  dot += simd_shuffle_xor(dot, 2u);
  dot += simd_shuffle_xor(dot, 1u);
  for (uint c = lane; c < cols; c += 32u) dsr[c] = pr[c] * (dpr[c] - dot);
}
)msl";

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_EXECUTOR_METAL_KERNELS_H_
