#ifndef SEEML_RUNTIME_CUSTODIAN_CHECKPOINT_FORMAT_H_
#define SEEML_RUNTIME_CUSTODIAN_CHECKPOINT_FORMAT_H_

#include <cstdint>

// =============================================================================
// The SEKP byte layout, in a header of its own (P6, #86) so the ABI manifest
// can publish it: until now these lived in checkpoint.cc's anonymous
// namespace, where no machine could check a second reader against them.
// The format is documented in checkpoint.h and docs/formats.md.
// =============================================================================

namespace seeml::update_rt {

inline constexpr uint32_t kCkptMagic = 0x504B4553;  // "SEKP"
// v3: payload_hash uses ContentHash64 — the deterministic parallel identity
// hash — instead of serial byte-at-a-time Fnv1a64. The persistent segment
// (parameters + AdamW moments) is the largest thing the custodian hashes,
// every checkpoint_every steps; the buffer is immutable for the duration of
// the call, so the chunked hash is race-free. v2 checkpoints are rejected
// by the version gate (resume restarts from the plan's initial state).
// v4: the header grows a trailing u64, the run's LR-schedule horizon (E8).
// Additive — v3 files are still read, their horizon 0 (the plan default).
// v5 (E9, #92): a second tail binds the run the segment belongs to — the
// shuffle stream and the train/validation split — records the source
// model's validation score so a resumed run's gate still measures the
// whole update against it, and carries the BEST evaluated state (its step,
// loss and a second persistent-segment payload) so the state that will be
// committed survives an interruption. v3/v4 files still load: no binding
// (nothing to refuse), no stored score, no best state.
inline constexpr uint32_t kCkptVersion = 5;
inline constexpr uint32_t kCkptOldestReadable = 3;

#pragma pack(push, 1)
struct CkptHeader {
  uint32_t magic = kCkptMagic;
  uint32_t version = kCkptVersion;
  uint64_t plan_hash = 0;
  uint64_t step = 0;
  uint64_t persistent_size = 0;
  uint64_t payload_hash = 0;
};
/// What v4 appends to the v3 header.
struct CkptHeaderV4Tail {
  uint64_t horizon_steps = 0;
};
/// What v5 appends after the v4 tail. Floats travel as their f32 bits.
/// The best payload (persistent_size bytes, hashed like the main one)
/// follows the main payload when kCkptHasBestPayload is set.
inline constexpr uint32_t kCkptHasValInitial = 1u << 0;
inline constexpr uint32_t kCkptHasAccuracy = 1u << 1;
inline constexpr uint32_t kCkptHasBestPayload = 1u << 2;
inline constexpr uint32_t kCkptHasBest = 1u << 3;  // best_* fields are live
struct CkptHeaderV5Tail {
  uint64_t shuffle_origin = 0;   // Dataset::shuffle_origin(); 0 = unshuffled
  uint64_t train_samples = 0;    // the training set's size after the split
  uint64_t val_samples = 0;      // the validation set's size; 0 = no split
  uint64_t best_step = 0;        // step of the best payload (0 = the start)
  uint64_t best_payload_hash = 0;
  uint32_t flags = 0;
  uint32_t val_initial_loss_bits = 0;
  uint32_t val_initial_accuracy_bits = 0;
  uint32_t best_loss_bits = 0;
  uint32_t best_accuracy_bits = 0;
  uint32_t stale_evals = 0;      // evaluations since the best, for patience
};
#pragma pack(pop)

static_assert(sizeof(CkptHeader) == 40, "the v3 SEKP header is 40 bytes");
static_assert(sizeof(CkptHeaderV4Tail) == 8, "v4 appends one u64");
static_assert(sizeof(CkptHeaderV5Tail) == 64, "v5 appends 64 bytes");

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_CUSTODIAN_CHECKPOINT_FORMAT_H_
