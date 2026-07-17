#ifndef SEEML_SOURCE_HASH_H_
#define SEEML_SOURCE_HASH_H_

#include <cstddef>
#include <cstdint>

// =============================================================================
// Fnv1a64 — the integrity primitive shared by the compiler and the runtime.
//
// Every artifact boundary in the update pipeline is hash-bound with it:
//   - PlanHeader::plan_hash        detects a corrupted / truncated .seeu blob
//   - PlanHeader::source_model_hash binds a plan to the exact .smf file whose
//                                   byte offsets its emit table patches
//   - checkpoint plan_hash + payload_hash bind a checkpoint to its plan and
//                                   detect flash corruption on resume
//
// FNV-1a is not cryptographic; the threat model is corruption and accidental
// mismatch (wrong file, stale artifact), not an adversary forging updates.
// Signature verification of shipped plans belongs one layer up, in the
// device's update transport.
// =============================================================================

namespace seeml::update {

inline constexpr uint64_t kFnvOffsetBasis = 0xcbf29ce484222325ULL;
inline constexpr uint64_t kFnvPrime = 0x100000001b3ULL;

/// Incremental form: feed successive spans, threading the running state.
constexpr uint64_t Fnv1a64(const uint8_t* data, size_t size,
                           uint64_t state = kFnvOffsetBasis) {
  for (size_t i = 0; i < size; ++i) {
    state ^= data[i];
    state *= kFnvPrime;
  }
  return state;
}

}  // namespace seeml::update

#endif  // SEEML_SOURCE_HASH_H_
