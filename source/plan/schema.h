#ifndef SEEML_SOURCE_PLAN_SCHEMA_H_
#define SEEML_SOURCE_PLAN_SCHEMA_H_

#include <bit>
#include <cstdint>

#include "source/plan/instruction.h"

// =============================================================================
// schema/ discipline of the plan ABI: the .seeu container — identity
// (magic, version), the master PlanHeader that addresses every section, and
// the emit table that maps trained deltas back into the source model file.
// The byte format is documented in docs/formats.md.
// =============================================================================

namespace seeml::update {

inline constexpr uint32_t kSeeuMagic = 0x55454553;  // "SEEU" little-endian
// v2: eval program section, plan/model integrity hashes, LR schedule fields,
// int8-quantized rodata opcodes. v1 plans are not accepted by the v2 runtime
// (they lack the integrity contract); recompile the plan.
// v3: source_model_hash is computed with ContentHash64 (the parallel model
// identity hash, source/identity/hash.h) instead of plain Fnv1a64. Older
// plans are rejected by the version gate; recompile the plan.
// v4: plan_hash is computed with PlanSelfHash (the chunked parallel form of
// the same fold, hash field zeroed) instead of a serial Fnv1a64 pass over
// the whole blob. Older plans are rejected by the version gate; recompile.
// v5: UpdateInstruction::flags carries fused GEMM epilogues (instruction.h).
// A pre-v5 runtime ignores flags entirely and would silently skip the fused
// bias/activation — exactly the misread the version gate exists to reject —
// so plans that may set flags must declare v5. From v5 on the validator
// rejects unknown flag bits, keeping every future flag loud.
// v6: the transformer opcode family (RMSNorm, RoPE, causal attention and
// its backward primitives — instruction.h). Additive: no existing field
// changes meaning, so the readable floor stays. The validator rejects the
// new opcodes in any pre-v6 plan — no pre-v6 compiler emits them, so their
// appearance there is corruption, not a feature.
// v7: token-native input. Two fields carved from reserved (zero = the
// pre-v7 behavior, so the floor stays): input_kind (0 = f32 feature rows,
// 1 = i32 token ids) and seq_len (rows per sequence; the feeder serves
// batch/seq_len token records per step). One new opcode, kEmbedFwd, gated
// exactly like the v6 family.
// v8: the distillation loss carries its scale. The high 32 bits of the
// kKLDistill{Fwd,Bwd} temperature word hold the f32 bits of a loss scale
// the compiler sets to T^2 (Hinton et al.'s convention, so the soft-target
// gradient stays commensurate with a hard-label term as T varies). Additive:
// pre-v8 plans have a zero high word, which the runtime reads as 1.0 — the
// unscaled divergence they were compiled for, bit-for-bit. A pre-v8 runtime
// would silently drop the scale, which is why plans that carry it declare
// v8 (and are rejected by older runtimes as newer than they can prove).
// v9: gradient accumulation (roadmap 2a). Three fields carved from zeros:
// grad_accum_steps (pad4; 0 or 1 = the pre-v9 one-batch step), and the
// step program's offset/count (reserved[0..1]). When grad_accum_steps > 1
// the train section holds the GRAD program (forward + backward + one
// kAccumulate per parameter into a persistent accumulator) and the step
// section holds the optimizer program (clip on the accumulator, the step
// on it, kFill it to zero); the runtime runs G grad executions per
// optimizer step. One new opcode, kAccumulate, gated like the v6 family.
// Additive: a v9 plan with G = 1 has an empty step section and the very
// same monolithic train program a v8 compiler emitted.
// v10: bf16 frozen weights (roadmap 2c). Two opcodes, kGemmNNBF16 and
// kGemmNTBF16, whose B operand is bfloat16 rodata widened exactly to f32
// inside the kernel — half the bytes of f32 with f32's exponent range and
// 8 bits of mantissa (the q8 path stays for 4x). Compute is unchanged f32.
// Additive: no field changes; the opcodes are gated like the v6 family.
// v11: the CPU GEMM tile geometry is a plan property. Two u32 fields carved
// from the header's two pad words: gemm_tile_k (pad2) and gemm_tile_n
// (pad3). The compiler decides them — from a measured, host-keyed
// kernel-policy table (tool/autotune.py) or the kernel defaults — and the
// CPU backend runs its blocked GEMM cores with them; zero selects the
// runtime's compiled-in default (the pre-v11 behavior, so the floor
// stays). Bits never depend on the geometry (kernel_policy.h); the
// validator proves the K tile a multiple of the 4-wide unroll, the only
// contract the kernels have. A pre-v11 runtime would ignore the fields and
// run its defaults — a throughput difference, never a misread — but the
// version gate rejects v11 plans there regardless, as always for a newer
// format.
// v12: the bitwise-safe kernel batch (E3). One opcode, kRopeTable, plus an
// optional third operand on kRopeFwd/kRopeBwd naming its result; and an
// optional clip threshold in out[1] of kSgdStep/kAdamWStep that folds the
// preceding kClipNorm into the step. Additive, in instruction words that
// were kNullRef / zero in every earlier plan, so the header is unchanged
// and the floor stays; the validator rejects all three below v12. Results
// are bit-identical to the v11 programs they replace.
inline constexpr uint32_t kSeeuVersion = 12;

// The version that introduced kRopeTable, the RoPE table operand and the
// fused clip word on the optimizer steps.
inline constexpr uint32_t kSeeuKernelBatchVersion = 12;

// The version that made the GEMM tile fields meaningful: below it they are
// pad words and must be zero.
inline constexpr uint32_t kSeeuGemmTilesVersion = 11;

// The version that introduced the bf16 GEMM opcodes.
inline constexpr uint32_t kSeeuBf16Version = 10;

// The version that introduced gradient accumulation and kAccumulate.
inline constexpr uint32_t kSeeuGradAccumVersion = 9;

// Section alignment. Every section starts 64-byte aligned (one cache line);
// rodata additionally starts on a 16 KiB boundary and the blob is padded to
// a 16 KiB multiple, so a page-aligned embedded plan (the .incbin stub)
// lets a GPU backend wrap the frozen weights as a zero-copy shared buffer.
// Alignment is a layout property the header's explicit offsets already
// describe — an older runtime reads such a plan unchanged — so it needs no
// version bump.
inline constexpr uint64_t kSeeuRodataAlignment = 16384;

// The version that introduced the distillation loss scale: plans below it
// must carry a zero high word on the KL temperature operand, and are
// validated to.
inline constexpr uint32_t kSeeuKlScaleVersion = 8;

// The version that introduced the transformer opcodes: plans below it must
// not carry them, and are validated to.
inline constexpr uint32_t kSeeuTransformerVersion = 6;

// The version that introduced token-native input and kEmbedFwd.
inline constexpr uint32_t kSeeuTokenVersion = 7;

// The version that introduced instruction flags: plans below it must carry
// flags == 0 on every instruction, and are validated to.
inline constexpr uint32_t kSeeuFlagsVersion = 5;

// Version negotiation policy. The runtime accepts every version in
// [kSeeuOldestReadable, kSeeuVersion], not just the version it was built
// at — a fleet's deployed runtimes must not be stranded by every format
// bump. The two constants move under different rules:
//   - An ADDITIVE change — new fields carved out of `reserved`, with zero
//     meaning "feature absent" — bumps kSeeuVersion only. Older plans keep
//     loading; their zeroed fields select the pre-change behavior.
//   - A SEMANTIC break — a field changes meaning or layout, as v3 did to
//     source_model_hash — raises kSeeuOldestReadable to the breaking
//     version, because misreading an old plan is worse than rejecting it.
// v1..v3 are below the floor: v1 lacks the integrity contract entirely, a
// v2 source_model_hash would mis-verify under v3's hash, and a v3 plan_hash
// (serial Fnv1a64) can never match the v4 PlanSelfHash the loader verifies
// unconditionally — admitting v3 would misreport every genuine v3 plan as
// corrupt instead of unsupported. Newer plans than the runtime are always
// rejected — forward compatibility cannot be proven from an unknown format.
inline constexpr uint32_t kSeeuOldestReadable = 4;
static_assert(kSeeuOldestReadable <= kSeeuVersion,
              "the readable floor cannot exceed the current version");

// The plan is serialized by memcpy of host integers/structs; the documented
// on-disk contract is little-endian. Big-endian hosts need byte-swapping I/O.
static_assert(std::endian::native == std::endian::little,
              ".seeu plan serialization assumes a little-endian host.");

#pragma pack(push, 1)

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
  // --- v11: the K tile of the CPU blocked GEMM (0 = the runtime default).
  uint32_t gemm_tile_k = 0;

  uint64_t batch = 0;
  uint64_t default_steps = 0;

  // --- v2: evaluation program (forward + loss only, no parameter mutation).
  // Used for held-out validation gating; shares the training arena binding.
  uint64_t eval_instr_offset = 0;
  uint64_t eval_instr_count = 0;

  // --- v2: integrity binding (source/identity/hash.h).
  // ContentHash64 (v3) of the source .smf file this plan was compiled from;
  // CommitToModel refuses to patch a file whose bytes hash differently.
  // 0 = unbound (the model was built in memory, not loaded from a file).
  uint64_t source_model_hash = 0;
  // Hash of the entire plan blob with this field zeroed; verified on load.
  // Also the identity that checkpoints bind to.
  uint64_t plan_hash = 0;

  // --- v2: LR schedule (applied by the runtime on top of `lr`).
  uint32_t lr_schedule = 0;  // LrSchedule
  // --- v11: the N tile of the CPU blocked GEMM (0 = the runtime default).
  uint32_t gemm_tile_n = 0;
  uint64_t warmup_steps = 0;
  float min_lr_factor = 0.0f;
  // Per-tensor gradient clip threshold baked into the instruction stream;
  // recorded here for introspection (seeu-dump). 0 = no clipping.
  float clip_norm = 0.0f;

  // --- v7: token-native input (both zero = pre-v7 f32-feature behavior).
  // input_kind 1 means the input slot holds i32 token ids; seq_len is the
  // rows-per-sequence the feeder contract and the attention geometry agree
  // on (nonzero exactly when input_kind == 1 or the model is sequential).
  uint32_t input_kind = 0;
  // --- v9: micro-steps per optimizer step (0 or 1 = none; then the step
  // section below is empty and the train section is one whole step).
  uint32_t grad_accum_steps = 0;
  uint64_t seq_len = 0;

  // --- v9: the optimizer (step) program, present iff grad_accum_steps > 1.
  uint64_t step_instr_offset = 0;
  uint64_t step_instr_count = 0;
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

// The header layout is the compiler↔runtime ABI. Additive version bumps
// carve new fields out of `reserved` or pad words and MUST NOT change this
// size — v7/v9 carved reserved[4] into input_kind / grad_accum_steps /
// seq_len / step_instr_offset+count and v11 carved the two pad words into
// the GEMM tiles; each is byte-exact only if this holds. As of v11 no spare
// word remains: the next additive field needs a v12 with a larger header
// and a raised readable floor. Compiler-enforce it rather than argue it.
static_assert(sizeof(PlanHeader) == 280, "PlanHeader layout is part of the ABI.");
static_assert(sizeof(EmitEntry) == 24, "EmitEntry layout is part of the ABI.");

}  // namespace seeml::update

#endif  // SEEML_SOURCE_PLAN_SCHEMA_H_
