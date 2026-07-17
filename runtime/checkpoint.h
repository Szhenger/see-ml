#ifndef SEEML_RUNTIME_CHECKPOINT_H_
#define SEEML_RUNTIME_CHECKPOINT_H_

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
//   u64 payload_hash     Fnv1a64 of the payload
//   payload              the arena's persistent segment
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
    const uint8_t* persistent, uint64_t persistent_size);

/// Loads and fully verifies a checkpoint bound to `plan_hash`, copying its
/// payload (which must be exactly `persistent_size` bytes) into `dst`.
/// Returns the saved training step. `dst` is untouched on any error.
[[nodiscard]] std::expected<uint64_t, std::string> LoadCheckpointFile(
    const std::string& path, uint64_t plan_hash, uint64_t persistent_size,
    uint8_t* dst);

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_CHECKPOINT_H_
