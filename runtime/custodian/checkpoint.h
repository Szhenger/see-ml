#ifndef SEEML_RUNTIME_CUSTODIAN_CHECKPOINT_H_
#define SEEML_RUNTIME_CUSTODIAN_CHECKPOINT_H_

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

// =============================================================================
// SEKP — the SeeML checkpoint container.
//
// A checkpoint is the arena's persistent segment (LoRA adapters + optimizer
// moments) plus the training step, hash-bound to the exact plan that laid the
// segment out. Layout (little-endian):
//   u32 magic "SEKP"; u32 version
//   u64 plan_hash        must equal the plan's PlanHeader::plan_hash
//   u64 step             1-indexed AdamW timestep at save
//   u64 persistent_size  byte length of each payload
//   u64 payload_hash     ContentHash64 of the payload
//   u64 horizon_steps    v4: the run's LR-schedule horizon (see below)
//   v5 tail              64 bytes: the run binding, the source model's
//                        validation score, the best state's step / score,
//                        the best payload's hash, the patience counter
//   payload              the arena's persistent segment
//   best payload         v5, optional: the best evaluated segment
//
// The horizon (v4, E8 #91) is the step count the interrupted run was
// annealing over. A resume with no explicit step count trains exactly the
// remainder to it, on the same schedule — so the resumed run's learning
// rates, and therefore its bits, equal the uninterrupted run's. A v3
// checkpoint has no horizon (read as 0: the plan's compiled default).
//
// The v5 tail (E9, #92) makes a resume honest in two more ways. The run
// binding — the shuffle stream's origin and the sizes of the two halves of
// the train/validation split — lets a resume refuse a different seed or
// split, which would otherwise silently move the boundary and train on the
// first run's validation rows. The stored validation score of the SOURCE
// model lets the gate keep comparing the whole update against it: without
// it a resumed run scored the resumed adapter as its "before" and rejected a
// segment that merely held steady. And the best payload is the state the
// update will commit (the best evaluated one, not the last), so it must
// survive an interruption exactly as the last one does.
//
// Saves are durable (fsync + atomic rename via durable_io); loads verify
// magic, version, plan binding, size, and payload hashes before a single
// byte reaches the arena.
// =============================================================================

namespace seeml::update_rt {

/// Everything a checkpoint carries besides the segment itself.
struct CheckpointRecord {
  uint64_t step = 0;
  uint64_t horizon_steps = 0;
  // v5 — absent (all zero / false) from a v3/v4 file.
  bool has_binding = false;
  uint64_t shuffle_origin = 0;
  uint64_t train_samples = 0;
  uint64_t val_samples = 0;
  bool has_val_initial = false;
  bool has_accuracy = false;
  float val_initial_loss = 0.0f;
  float val_initial_accuracy = 0.0f;
  bool has_best = false;  // best_* are live (a tracked run)
  uint64_t best_step = 0;
  float best_loss = 0.0f;
  float best_accuracy = 0.0f;
  uint32_t stale_evals = 0;
  // The best payload (persistent_size bytes) when the best state differs
  // from the main payload; empty when it is the main payload, or untracked.
  std::vector<uint8_t> best_payload;
};

/// Durably writes a checkpoint: header + `persistent` payload, gather-written
/// straight from the arena, plus `record`'s best payload when it has one.
[[nodiscard]] std::expected<void, std::string> SaveCheckpointRecord(
    const std::string& path, uint64_t plan_hash, const CheckpointRecord& record,
    const uint8_t* persistent, uint64_t persistent_size);

/// Loads and fully verifies a checkpoint bound to `plan_hash`, copying its
/// main payload (which must be exactly `persistent_size` bytes) into `dst`
/// and returning everything else it carries. `dst` is untouched on any
/// error.
[[nodiscard]] std::expected<CheckpointRecord, std::string> LoadCheckpointRecord(
    const std::string& path, uint64_t plan_hash, uint64_t persistent_size,
    uint8_t* dst);

/// Reads only the run binding of a checkpoint (no payload is touched or
/// verified): what a driver needs to refuse a mismatched seed or split
/// before anything else happens. A file that is not a checkpoint is an
/// error; a v3/v4 file yields has_binding == false.
[[nodiscard]] std::expected<CheckpointRecord, std::string>
PeekCheckpointRecord(const std::string& path);

/// The v3/v4-shaped calls: a step, a horizon, one payload.
[[nodiscard]] std::expected<void, std::string> SaveCheckpointFile(
    const std::string& path, uint64_t plan_hash, uint64_t step,
    const uint8_t* persistent, uint64_t persistent_size,
    uint64_t horizon_steps = 0);
[[nodiscard]] std::expected<uint64_t, std::string> LoadCheckpointFile(
    const std::string& path, uint64_t plan_hash, uint64_t persistent_size,
    uint8_t* dst, uint64_t* horizon_steps = nullptr);

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_CUSTODIAN_CHECKPOINT_H_
