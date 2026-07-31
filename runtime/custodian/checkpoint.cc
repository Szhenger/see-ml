#include "runtime/custodian/checkpoint.h"

#include <cstring>
#include <fstream>
#include <vector>

#include "runtime/custodian/durable_io.h"
#include "runtime/diagnostics/persisting/error.h"
#include "source/identity/hash.h"

namespace seeml::update_rt {

namespace up = seeml::update;

namespace {

inline constexpr uint32_t kCkptMagic = 0x504B4553;  // "SEKP"
// v3: payload_hash uses ContentHash64 — the deterministic parallel identity
// hash — instead of serial byte-at-a-time Fnv1a64. The persistent segment
// (parameters + AdamW moments) is the largest thing the custodian hashes,
// every checkpoint_every steps; the buffer is immutable for the duration of
// the call, so the chunked hash is race-free. v2 checkpoints are rejected
// by the version gate (resume restarts from the plan's initial state).
inline constexpr uint32_t kCkptVersion = 3;

#pragma pack(push, 1)
struct CkptHeader {
  uint32_t magic = kCkptMagic;
  uint32_t version = kCkptVersion;
  uint64_t plan_hash = 0;
  uint64_t step = 0;
  uint64_t persistent_size = 0;
  uint64_t payload_hash = 0;
};
#pragma pack(pop)

}  // namespace

std::expected<void, std::string> SaveCheckpointFile(
    const std::string& path, uint64_t plan_hash, uint64_t step,
    const uint8_t* persistent, uint64_t persistent_size) {
  CkptHeader h;
  h.plan_hash = plan_hash;
  h.step = step;
  h.persistent_size = persistent_size;
  h.payload_hash = up::ContentHash64(persistent, persistent_size);

  // Durable: a checkpoint that can vanish in a power cut is not a checkpoint.
  // Gather-write header + persistent segment straight from the arena — no
  // concatenated staging blob, so periodic checkpointing costs no extra
  // allocation or copy of the (potentially large) optimizer state.
  return WriteFileDurable(
      path, {ByteSpan{reinterpret_cast<const uint8_t*>(&h), sizeof(h)},
             ByteSpan{persistent, persistent_size}});
}

std::expected<uint64_t, std::string> LoadCheckpointFile(
    const std::string& path, uint64_t plan_hash, uint64_t persistent_size,
    uint8_t* dst) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return diag::persisting::Error(diag::persisting::kCheckpoint, "no checkpoint at '" + path + "'");
  CkptHeader h;
  f.read(reinterpret_cast<char*>(&h), sizeof(h));
  if (!f || h.magic != kCkptMagic || h.version != kCkptVersion)
    return diag::persisting::Error(diag::persisting::kCheckpoint, "not a v3 checkpoint: '" + path + "'");
  // Binding: a checkpoint carries adapter and optimizer state laid out by
  // one specific plan. Resuming it under any other plan is silent corruption.
  if (h.plan_hash != plan_hash)
    return diag::persisting::Error(
        diag::persisting::kCheckpoint, "checkpoint belongs to a different plan");
  if (h.persistent_size != persistent_size)
    return diag::persisting::Error(diag::persisting::kCheckpoint, "checkpoint incompatible with plan");
  std::vector<uint8_t> payload(h.persistent_size);
  f.read(reinterpret_cast<char*>(payload.data()),
         static_cast<std::streamsize>(payload.size()));
  if (!f) return diag::persisting::Error(diag::persisting::kCheckpoint, "truncated checkpoint");
  if (up::ContentHash64(payload.data(), payload.size()) != h.payload_hash)
    return diag::persisting::Error(diag::persisting::kCheckpoint, "checkpoint payload is corrupt");
  std::memcpy(dst, payload.data(), payload.size());
  return h.step;
}

}  // namespace seeml::update_rt
