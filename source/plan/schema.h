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
inline constexpr uint32_t kSeeuVersion = 5;

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
// v1 and v2 are below the floor: v1 lacks the integrity contract entirely,
// and a v2 source_model_hash would mis-verify under v3's hash. Newer plans
// than the runtime are always rejected — forward compatibility cannot be
// proven from an unknown format.
inline constexpr uint32_t kSeeuOldestReadable = 3;
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
  uint32_t pad2 = 0;

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

static_assert(sizeof(EmitEntry) == 24, "EmitEntry layout is part of the ABI.");

}  // namespace seeml::update

#endif  // SEEML_SOURCE_PLAN_SCHEMA_H_
