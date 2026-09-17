#ifndef SEEML_RUNTIME_CUSTODIAN_CHECKPOINT_H_
#define SEEML_RUNTIME_CUSTODIAN_CHECKPOINT_H_

#include <cstdint>
#include <expected>
#include <string>

// =============================================================================
// SEKP — the SeeML checkpoint container.
//
// A checkpoint is the arena's persistent segment (LoRA adapters + optimizer
// moments) plus the training step, hash-bound to the exact plan that laid the
// segment out. Layout (little-endian):
//   u32 magic "SEKP"; u32 version
//   u64 plan_hash        must equal the plan's PlanHeader::plan_hash
//   u64 step             1-indexed AdamW timestep at save
//   u64 persistent_size  byte length of the payload
//   u64 payload_hash     ContentHash64 of the payload
//   u64 horizon_steps    v4: the run's LR-schedule horizon (see below)
//   payload              the arena's persistent segment
//
// The horizon (v4, E8 #91) is the step count the interrupted run was
// annealing over. A resume with no explicit step count trains exactly the
// remainder to it, on the same schedule — so the resumed run's learning
// rates, and therefore its bits, equal the uninterrupted run's. A v3
// checkpoint has no horizon (read as 0: the plan's compiled default).
//
// Saves are durable (fsync + atomic rename via durable_io); loads verify
// magic, version, plan binding, size, and payload hash before a single byte
// reaches the arena.
// =============================================================================

namespace seeml::update_rt {

/// Durably writes a checkpoint: header + `persistent` payload, gather-written
/// straight from the arena.
[[nodiscard]] std::expected<void, std::string> SaveCheckpointFile(
    const std::string& path, uint64_t plan_hash, uint64_t step,
    const uint8_t* persistent, uint64_t persistent_size,
    uint64_t horizon_steps = 0);

/// Loads and fully verifies a checkpoint bound to `plan_hash`, copying its
/// payload (which must be exactly `persistent_size` bytes) into `dst`.
/// Returns the saved training step, and the run horizon through
/// `horizon_steps` when given (0 for a v3 file). `dst` is untouched on any
/// error.
[[nodiscard]] std::expected<uint64_t, std::string> LoadCheckpointFile(
    const std::string& path, uint64_t plan_hash, uint64_t persistent_size,
    uint8_t* dst, uint64_t* horizon_steps = nullptr);

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_CUSTODIAN_CHECKPOINT_H_
