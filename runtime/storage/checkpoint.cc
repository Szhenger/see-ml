#include "runtime/storage/checkpoint.h"

#include <bit>
#include <cstring>
#include <fstream>
#include <span>
#include <vector>

#include "runtime/storage/checkpoint_format.h"
#include "runtime/storage/durable_io.h"
#include "runtime/diagnostics/persisting/error.h"
#include "source/identity/hash.h"

namespace seeml::update_rt {

namespace up = seeml::update;

namespace {

std::unexpected<std::string> Bad(const std::string& what) {
  return diag::persisting::Error(diag::persisting::kCheckpoint, what);
}

/// Reads the three header parts; `tails` are zero for what the file's
/// version does not carry.
std::expected<void, std::string> ReadHeaders(std::ifstream& f,
                                             const std::string& path,
                                             CkptHeader* h,
                                             CkptHeaderV4Tail* v4,
                                             CkptHeaderV5Tail* v5) {
  f.read(reinterpret_cast<char*>(h), sizeof(*h));
  if (!f || h->magic != kCkptMagic || h->version < kCkptOldestReadable ||
      h->version > kCkptVersion)
    return Bad("not a v3..v5 checkpoint: '" + path + "'");
  if (h->version >= 4) {
    f.read(reinterpret_cast<char*>(v4), sizeof(*v4));
    if (!f) return Bad("truncated checkpoint");
  }
  if (h->version >= 5) {
    f.read(reinterpret_cast<char*>(v5), sizeof(*v5));
    if (!f) return Bad("truncated checkpoint");
  }
  return {};
}

}  // namespace

std::expected<void, std::string> SaveCheckpointRecord(
    const std::string& path, uint64_t plan_hash, const CheckpointRecord& record,
    const uint8_t* persistent, uint64_t persistent_size) {
  CkptHeader h;
  CkptHeaderV4Tail v4;
  CkptHeaderV5Tail v5;
  h.plan_hash = plan_hash;
  h.step = record.step;
  h.persistent_size = persistent_size;
  h.payload_hash = up::ContentHash64(persistent, persistent_size);
  v4.horizon_steps = record.horizon_steps;
  v5.shuffle_origin = record.shuffle_origin;
  v5.train_samples = record.train_samples;
  v5.val_samples = record.val_samples;
  v5.best_step = record.best_step;
  v5.stale_evals = record.stale_evals;
  v5.val_initial_loss_bits = std::bit_cast<uint32_t>(record.val_initial_loss);
  v5.val_initial_accuracy_bits =
      std::bit_cast<uint32_t>(record.val_initial_accuracy);
  v5.best_loss_bits = std::bit_cast<uint32_t>(record.best_loss);
  v5.best_accuracy_bits = std::bit_cast<uint32_t>(record.best_accuracy);
  if (record.has_val_initial) v5.flags |= kCkptHasValInitial;
  if (record.has_accuracy) v5.flags |= kCkptHasAccuracy;
  if (record.has_best) v5.flags |= kCkptHasBest;
  const bool best = record.has_best && !record.best_payload.empty();
  if (best) {
    if (record.best_payload.size() != persistent_size)
      return Bad("best payload is not the persistent segment's size");
    v5.flags |= kCkptHasBestPayload;
    v5.best_payload_hash =
        up::ContentHash64(record.best_payload.data(), persistent_size);
  }

  // Durable: a checkpoint that can vanish in a power cut is not a checkpoint.
  // Gather-write the header parts and the segment(s) straight from where
  // they live — no concatenated staging blob, so periodic checkpointing
  // costs no extra allocation or copy of the (potentially large) state.
  std::vector<ByteSpan> parts{
      ByteSpan{reinterpret_cast<const uint8_t*>(&h), sizeof(h)},
      ByteSpan{reinterpret_cast<const uint8_t*>(&v4), sizeof(v4)},
      ByteSpan{reinterpret_cast<const uint8_t*>(&v5), sizeof(v5)},
      ByteSpan{persistent, persistent_size}};
  if (best) parts.push_back(ByteSpan{record.best_payload.data(), persistent_size});
  return WriteFileDurable(path, std::span<const ByteSpan>(parts));
}

std::expected<CheckpointRecord, std::string> LoadCheckpointRecord(
    const std::string& path, uint64_t plan_hash, uint64_t persistent_size,
    uint8_t* dst) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return Bad("no checkpoint at '" + path + "'");
  CkptHeader h;
  CkptHeaderV4Tail v4;
  CkptHeaderV5Tail v5;
  if (auto r = ReadHeaders(f, path, &h, &v4, &v5); !r) return std::unexpected(r.error());
  // Binding: a checkpoint carries adapter and optimizer state laid out by
  // one specific plan. Resuming it under any other plan is silent corruption.
  if (h.plan_hash != plan_hash) return Bad("checkpoint belongs to a different plan");
  if (h.persistent_size != persistent_size)
    return Bad("checkpoint incompatible with plan");
  std::vector<uint8_t> payload(h.persistent_size);
  f.read(reinterpret_cast<char*>(payload.data()),
         static_cast<std::streamsize>(payload.size()));
  if (!f) return Bad("truncated checkpoint");
  if (up::ContentHash64(payload.data(), payload.size()) != h.payload_hash)
    return Bad("checkpoint payload is corrupt");

  CheckpointRecord record;
  record.step = h.step;
  record.horizon_steps = v4.horizon_steps;
  if (h.version >= 5) {
    record.has_binding = true;
    record.shuffle_origin = v5.shuffle_origin;
    record.train_samples = v5.train_samples;
    record.val_samples = v5.val_samples;
    record.has_val_initial = (v5.flags & kCkptHasValInitial) != 0;
    record.has_accuracy = (v5.flags & kCkptHasAccuracy) != 0;
    record.val_initial_loss = std::bit_cast<float>(v5.val_initial_loss_bits);
    record.val_initial_accuracy =
        std::bit_cast<float>(v5.val_initial_accuracy_bits);
    record.best_step = v5.best_step;
    record.best_loss = std::bit_cast<float>(v5.best_loss_bits);
    record.best_accuracy = std::bit_cast<float>(v5.best_accuracy_bits);
    record.stale_evals = v5.stale_evals;
    record.has_best = (v5.flags & kCkptHasBest) != 0;
    if (v5.flags & kCkptHasBestPayload) {
      record.best_payload.resize(h.persistent_size);
      f.read(reinterpret_cast<char*>(record.best_payload.data()),
             static_cast<std::streamsize>(record.best_payload.size()));
      if (!f) return Bad("truncated checkpoint (best payload)");
      if (up::ContentHash64(record.best_payload.data(),
                            record.best_payload.size()) != v5.best_payload_hash)
        return Bad("checkpoint best payload is corrupt");
    }
  }
  // Nothing reaches the arena until every check above has passed.
  std::memcpy(dst, payload.data(), payload.size());
  return record;
}

std::expected<CheckpointRecord, std::string> PeekCheckpointRecord(
    const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return Bad("no checkpoint at '" + path + "'");
  CkptHeader h;
  CkptHeaderV4Tail v4;
  CkptHeaderV5Tail v5;
  if (auto r = ReadHeaders(f, path, &h, &v4, &v5); !r) return std::unexpected(r.error());
  CheckpointRecord record;
  record.step = h.step;
  record.horizon_steps = v4.horizon_steps;
  if (h.version >= 5) {
    record.has_binding = true;
    record.shuffle_origin = v5.shuffle_origin;
    record.train_samples = v5.train_samples;
    record.val_samples = v5.val_samples;
  }
  return record;
}

std::expected<void, std::string> SaveCheckpointFile(
    const std::string& path, uint64_t plan_hash, uint64_t step,
    const uint8_t* persistent, uint64_t persistent_size,
    uint64_t horizon_steps) {
  CheckpointRecord record;
  record.step = step;
  record.horizon_steps = horizon_steps;
  return SaveCheckpointRecord(path, plan_hash, record, persistent,
                              persistent_size);
}

std::expected<uint64_t, std::string> LoadCheckpointFile(
    const std::string& path, uint64_t plan_hash, uint64_t persistent_size,
    uint8_t* dst, uint64_t* horizon_steps) {
  auto record = LoadCheckpointRecord(path, plan_hash, persistent_size, dst);
  if (!record) return std::unexpected(record.error());
  if (horizon_steps) *horizon_steps = record->horizon_steps;
  return record->step;
}

}  // namespace seeml::update_rt
