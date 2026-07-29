#ifndef SEEML_SOURCE_PLAN_INSTRUCTION_H_
#define SEEML_SOURCE_PLAN_INSTRUCTION_H_

#include <cstdint>

// =============================================================================
// instruction/ discipline of the plan ABI: the update instruction set —
// tensor references, the opcode vocabulary, and the fixed 64-byte
// instruction word the compiler's lowering emits and the runtime's
// dispatcher executes.
// =============================================================================

namespace seeml::update {

// A tensor reference is a 64-bit word: bit 63 selects the address space
// (0 = mutable arena, 1 = read-only rodata), bits 0..62 are a byte offset.
inline constexpr uint64_t kRodataBit = 1ULL << 63;
inline constexpr uint64_t kNullRef = ~0ULL;

inline constexpr uint64_t MakeArenaRef(uint64_t offset) { return offset; }
inline constexpr uint64_t MakeRodataRef(uint64_t offset) {
  return offset | kRodataBit;
}
inline constexpr bool IsRodataRef(uint64_t ref) { return (ref & kRodataBit) != 0; }
inline constexpr uint64_t RefOffset(uint64_t ref) { return ref & ~kRodataBit; }

enum class OpCode : uint16_t {
  kNop = 0,
  // GEMM family. C[M,N]; refs in in[0..2], dims in out[0..2] = M, N, K.
  kGemmNN = 1,     // C = A[M,K] @ B[K,N]
  kGemmNT = 2,     // C = A[M,K] @ B[N,K]^T   (VJP: dX = dC @ W^T)
  kGemmTN = 3,     // C = A[K,M]^T @ B[K,N]   (VJP: dW = X^T @ dC)
  kGemmAccNN = 4,  // C += alpha * A @ B; alpha f32 bits in in[3] (LoRA merge)
  // Elementwise / broadcast.
  kAddEW = 5,      // out = x + y            in: x, y, out; out[0] = count
  kAddBias = 6,    // out[N,M] = x + b[M]    in: x, b, out; out[0..1] = N, M
  kReluFwd = 7,    // out = max(x, 0)        in: x, out;    out[0] = count
  kReluBwd = 8,    // dx = dy * (x > 0)      in: dy, x, dx; out[0] = count
  kScale = 9,      // out = alpha * x        in: x, out, alpha_bits; out[0] = count
  kReduceRows = 10,  // db[M] = sum_n dY[N,M]  in: dy, db;  out[0..1] = N, M
  // Losses (fwd caches what the bwd kernel needs; scalar loss slot).
  kSoftmaxXEntFwd = 11,  // in: logits, labels(i32), loss, probs; out[0..1]=N,C
  kSoftmaxXEntBwd = 12,  // in: probs, labels, seed, dlogits;     out[0..1]=N,C
  kMseFwd = 13,          // in: pred, target, loss;               out[0]=count
  kMseBwd = 14,          // in: pred, target, seed, dpred;        out[0]=count
  kKLDistillFwd = 15,  // in: s_logits, t_logits, loss, p_s; out[0]=p_t ref,
                       // out[1]=N<<32|C, out[2]=temperature f32 bits
  kKLDistillBwd = 16,  // in: p_s, p_t, seed, dlogits; out[0]=N<<32|C,
                       // out[1]=temperature f32 bits
  // Optimizers (in-place; hyperparameters live in the PlanHeader).
  kSgdStep = 17,    // in: p, g;             out[0] = count
  kAdamWStep = 18,  // in: p, g, m, v;       out[0] = count
  // Utility.
  kFill = 19,  // in: dst, f32 value bits;   out[0] = count
  kCopy = 20,  // in: src, dst;              out[0] = count (floats)
  // Elementwise (v2).
  kMulEW = 21,    // out = x * y             in: x, y, out; out[0] = count
  kGeluFwd = 22,  // out = gelu(x)           in: x, out;    out[0] = count
  kGeluBwd = 23,  // dx = dy * gelu'(x)      in: dy, x, dx; out[0] = count
  kSiluFwd = 24,  // out = x * sigmoid(x)    in: x, out;    out[0] = count
  kSiluBwd = 25,  // dx = dy * silu'(x)      in: dy, x, dx; out[0] = count
  // LayerNorm over the last dim of x[N,D] with affine gamma/beta[D] (v2).
  kLayerNormFwd = 26,  // in: x, gamma, beta, y; out[0]=N<<32|D,
                       // out[1]=mean ref [N], out[2]=rstd ref [N]
  kLayerNormBwd = 27,  // in: dy, x, gamma, dx;  out[0]=mean ref,
                       // out[1]=rstd ref, out[2]=N<<32|D
  // Training utilities (v2).
  kClipNorm = 28,  // g *= min(1, max/||g||)  in: g, max f32 bits; out[0]=count
  // Quantized frozen weights (v2): B is per-tensor symmetric int8 in rodata,
  // dequantized on the fly as scale * q. Layouts mirror the f32 GEMMs.
  kGemmNNQ8 = 29,  // C = A @ dq(B);    in: A, Bq8, C, scale bits; out=M,N,K
  kGemmNTQ8 = 30,  // C = A @ dq(B)^T;  in: A, Bq8, C, scale bits; out=M,N,K
};

#pragma pack(push, 1)

/// One 64-byte instruction: a single L1 cache line, mirroring the design of
/// the inference-side SerializedInstruction in backend/serializer/schema.h.
struct UpdateInstruction {
  uint16_t opcode = 0;
  uint16_t flags = 0;
  uint32_t pad = 0;
  uint64_t in[4] = {kNullRef, kNullRef, kNullRef, kNullRef};
  uint64_t out[3] = {0, 0, 0};
};

#pragma pack(pop)

static_assert(sizeof(UpdateInstruction) == 64,
              "UpdateInstruction must be exactly one cache line.");

}  // namespace seeml::update

#endif  // SEEML_SOURCE_PLAN_INSTRUCTION_H_
