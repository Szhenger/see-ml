#ifndef SEEML_SOURCE_UPDATE_TYPES_H_
#define SEEML_SOURCE_UPDATE_TYPES_H_

#include <bit>
#include <cstdint>
#include <string>
#include <vector>

// =============================================================================
// SeeML shared substrate: compiler configuration and the .seeu binary plan
// schema. Lives in source/ (with smf.h and hash.h) because it is the ABI both
// sides of the product speak: the compiler writes it, the device runtime
// reads it, and neither may depend on the other's internals.
//
// The update compiler consumes:
//   - a source model (the user's on-device model, SMF format),
//   - optionally the open weights of a teacher model (distillation),
//   - a training loss selection,
//   - a LoRA specification (rank, alpha, RNG seed),
// and produces an Update Plan: a fully AOT-compiled training + merge program
// with every tensor bound to a fixed arena offset. The plan is executed by
// the bare-metal UpdateEngine (runtime/) with zero dynamic
// allocations per training step.
// =============================================================================

namespace seeml::update {

// -----------------------------------------------------------------------------
// Compiler configuration
// -----------------------------------------------------------------------------

enum class LossKind : uint8_t {
  kSoftmaxXEnt = 0,  // supervised: cross-entropy over class labels
  kMse = 1,          // supervised: mean squared error over dense targets
  kKLDistill = 2,    // teacher-driven: KL(teacher || student) on logits
  kXEntPlusKL = 3,   // composite: (1-w)*xent + w*kl
};

enum class OptimizerKind : uint8_t { kSgd = 0, kAdamW = 1 };

// Learning-rate schedule applied by the runtime on top of OptimizerSpec::lr.
// Warmup ramps linearly from 0 over `warmup_steps`; cosine then decays to
// lr * min_lr_factor across the plan's default_steps horizon.
enum class LrSchedule : uint32_t { kConstant = 0, kCosineWithWarmup = 1 };

struct LoRASpec {
  int64_t rank = 8;
  float alpha = 16.0f;
  uint64_t seed = 42;
  // Substring filters matched against the frozen weight's tensor name.
  // Empty => every eligible student MatMul is adapted.
  std::vector<std::string> target_filters;
};

struct OptimizerSpec {
  OptimizerKind kind = OptimizerKind::kAdamW;
  float lr = 1e-3f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  float weight_decay = 0.01f;
  // Per-tensor L2 gradient clipping applied before each optimizer step;
  // 0 disables (no clip instructions are emitted).
  float clip_norm = 0.0f;
  LrSchedule lr_schedule = LrSchedule::kConstant;
  uint64_t warmup_steps = 0;
  float min_lr_factor = 0.0f;  // cosine floor as a fraction of lr
};

struct UpdateConfig {
  int64_t batch = 32;
  LossKind loss = LossKind::kSoftmaxXEnt;
  float distill_weight = 0.5f;  // weight on the KL term for kXEntPlusKL
  float temperature = 2.0f;     // distillation softmax temperature
  LoRASpec lora;
  OptimizerSpec optimizer;
  uint64_t default_steps = 1000;
  // Quantize frozen base/teacher weights that feed MatMuls to per-tensor
  // symmetric int8 in the plan's rodata (QLoRA-style). Adapters, gradients,
  // and all activations stay f32; the source .smf on disk is untouched.
  bool quantize_base = false;
  // Test hook: when false, the plan contains forward+backward only (no
  // parameter mutation), enabling finite-difference gradient verification.
  bool emit_optimizer = true;
  // Fail-fast memory gate (ingressor resource analyzer): compilation is
  // rejected up front when the statically proven lower bound of the training
  // footprint exceeds this many bytes. 0 = detect the host's physical memory.
  uint64_t memory_budget_bytes = 0;
};

// -----------------------------------------------------------------------------
// The .seeu Update Plan binary schema
// -----------------------------------------------------------------------------

inline constexpr uint32_t kSeeuMagic = 0x55454553;  // "SEEU" little-endian
// v2: eval program section, plan/model integrity hashes, LR schedule fields,
// int8-quantized rodata opcodes. v1 plans are not accepted by the v2 runtime
// (they lack the integrity contract); recompile the plan.
// v3: source_model_hash is computed with ContentHash64 (the parallel model
// identity hash, source/hash.h) instead of plain Fnv1a64. Older plans are
// rejected by the version gate; recompile the plan.
inline constexpr uint32_t kSeeuVersion = 3;

// The plan is serialized by memcpy of host integers/structs; the documented
// on-disk contract is little-endian. Big-endian hosts need byte-swapping I/O.
static_assert(std::endian::native == std::endian::little,
              ".seeu plan serialization assumes a little-endian host.");

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

/// Master header at byte 0 of a .seeu plan.
struct PlanHeader {
  uint32_t magic = kSeeuMagic;
  uint32_t version = kSeeuVersion;

  // Runtime memory contract (known before the device commits to the update).
  uint64_t arena_size = 0;       // total mutable arena bytes to allocate
  uint64_t persistent_size = 0;  // prefix of the arena that is checkpointed
                                 // (LoRA params + optimizer state)

  // I/O slots inside the arena, filled by the data feeder each step.
  uint64_t input_ref = kNullRef;
  uint64_t input_floats = 0;  // batch * input_dim
  uint64_t label_ref = kNullRef;
  uint64_t label_bytes = 0;   // bytes copied per batch
  uint32_t label_kind = 0;    // 0 = none, 1 = class indices (i32), 2 = dense f32
  uint32_t optimizer_kind = 1;

  uint64_t loss_ref = kNullRef;  // scalar loss slot (read after each step)

  // Program sections (absolute file offsets inside the plan blob).
  uint64_t train_instr_offset = 0;
  uint64_t train_instr_count = 0;
  uint64_t merge_instr_offset = 0;
  uint64_t merge_instr_count = 0;
  uint64_t rodata_offset = 0;  // frozen source (+teacher) weights
  uint64_t rodata_size = 0;
  uint64_t persist_init_offset = 0;  // initial image of the persistent segment
  uint64_t persist_init_size = 0;    // == persistent_size
  uint64_t emit_table_offset = 0;    // EmitEntry[emit_count]
  uint64_t emit_count = 0;

  // Optimizer hyperparameters (single parameter group).
  float lr = 1e-3f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  float weight_decay = 0.01f;
  uint32_t pad2 = 0;

  uint64_t batch = 0;
  uint64_t default_steps = 0;

  // --- v2: evaluation program (forward + loss only, no parameter mutation).
  // Used for held-out validation gating; shares the training arena binding.
  uint64_t eval_instr_offset = 0;
  uint64_t eval_instr_count = 0;

  // --- v2: integrity binding (source/hash.h).
  // ContentHash64 (v3) of the source .smf file this plan was compiled from;
  // CommitToModel refuses to patch a file whose bytes hash differently.
  // 0 = unbound (the model was built in memory, not loaded from a file).
  uint64_t source_model_hash = 0;
  // Hash of the entire plan blob with this field zeroed; verified on load.
  // Also the identity that checkpoints bind to.
  uint64_t plan_hash = 0;

  // --- v2: LR schedule (applied by the runtime on top of `lr`).
  uint32_t lr_schedule = 0;  // LrSchedule
  uint32_t pad3 = 0;
  uint64_t warmup_steps = 0;
  float min_lr_factor = 0.0f;
  // Per-tensor gradient clip threshold baked into the instruction stream;
  // recorded here for introspection (seeu-dump). 0 = no clipping.
  float clip_norm = 0.0f;

  uint64_t reserved[4] = {0, 0, 0, 0};
};

/// Maps a trained weight delta in the arena to the byte range it updates
/// inside the source model file. Commit applies W'[i] = W[i] + delta[i]
/// elementwise over the f32 range — the file's pristine weights are the
/// base, so a quantized plan never bakes its quantization error into the
/// committed model.
struct EmitEntry {
  uint64_t smf_data_offset = 0;  // absolute offset of W inside the source .smf
  uint64_t byte_size = 0;        // f32 byte length of the weight
  uint64_t arena_offset = 0;     // where delta = (α/r)·A@B lives after RunMerge()
};

#pragma pack(pop)

static_assert(sizeof(UpdateInstruction) == 64,
              "UpdateInstruction must be exactly one cache line.");
static_assert(sizeof(EmitEntry) == 24, "EmitEntry layout is part of the ABI.");

}  // namespace seeml::update

#endif  // SEEML_SOURCE_UPDATE_TYPES_H_
