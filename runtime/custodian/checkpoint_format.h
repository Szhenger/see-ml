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
inline constexpr uint32_t kCkptVersion = 4;
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
#pragma pack(pop)

static_assert(sizeof(CkptHeader) == 40, "the v3 SEKP header is 40 bytes");
static_assert(sizeof(CkptHeaderV4Tail) == 8, "v4 appends one u64");

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_CUSTODIAN_CHECKPOINT_FORMAT_H_
