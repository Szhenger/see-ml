#ifndef SEEML_SOURCE_IDENTITY_HASH_H_
#define SEEML_SOURCE_IDENTITY_HASH_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "source/parallel/parallel_for.h"

// =============================================================================
// The integrity primitives shared by the compiler and the runtime.
//
// Every artifact boundary in the update pipeline is hash-bound:
//   - PlanHeader::plan_hash        (PlanSelfHash) detects a corrupted /
//                                   truncated .seeu blob; the hash field
//                                   itself is treated as zero
//   - PlanHeader::source_model_hash (ContentHash64) binds a plan to the exact
//                                   .smf file whose byte offsets its emit
//                                   table patches
//   - checkpoint plan_hash + payload_hash (ContentHash64) bind a checkpoint
//                                   to its plan and detect flash corruption
//                                   on resume
//
// Neither hash is cryptographic; the threat model is corruption and
// accidental mismatch (wrong file, stale artifact), not an adversary forging
// updates. Signature verification of shipped plans belongs one layer up, in
// the device's update transport.
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

// -----------------------------------------------------------------------------
// ContentHash64 — the model-identity hash, engineered for scan throughput.
//
// FNV-1a's recurrence (state ^= byte; state *= prime) is one serial dependency
// chain: each step waits on the previous multiply, so a superscalar core
// retires ~1 byte per multiply latency no matter how wide it is. Model files
// are the largest artifact the ingressor scans, so their identity hash uses
// two levels of parallelism instead:
//
//   ILP     StripedFnv1a64 runs 8 interleaved FNV-1a lanes (byte i feeds lane
//           i % 8). The 8 chains are independent, so the core pipelines them
//           and retires ~8x the bytes per cycle.
//   threads ContentHash64 chunks the buffer over ParallelFor and folds the
//           per-chunk striped digests in chunk order. Chunk geometry is a
//           pure function of the size (never of the worker count), so the
//           digest is bitwise-identical on a 1-core and an 8-core host.
//
// The digest is a different value than plain Fnv1a64 over the same bytes; it
// is a new identity contract (plan format v3), used consistently by the model
// reader, the model writer, and the engine's commit-time identity check.
// -----------------------------------------------------------------------------

inline constexpr size_t kContentHashLanes = 8;
inline constexpr size_t kContentHashChunk = 1u << 20;  // 1 MiB per chunk

/// Folds a 64-bit word into a running FNV-1a state, low byte first.
constexpr uint64_t FnvMixWord(uint64_t state, uint64_t word) {
  for (int b = 0; b < 8; ++b) {
    state ^= (word >> (8 * b)) & 0xff;
    state *= kFnvPrime;
  }
  return state;
}

/// 8-lane interleaved FNV-1a over one contiguous span (the ILP kernel).
constexpr uint64_t StripedFnv1a64(const uint8_t* data, size_t size) {
  uint64_t lane[kContentHashLanes] = {};
  for (size_t l = 0; l < kContentHashLanes; ++l)
    lane[l] = kFnvOffsetBasis ^ (0x9E3779B97F4A7C15ULL * (l + 1));

  size_t i = 0;
  for (; i + kContentHashLanes <= size; i += kContentHashLanes)
    for (size_t l = 0; l < kContentHashLanes; ++l) {
      lane[l] ^= data[i + l];
      lane[l] *= kFnvPrime;
    }
  for (size_t l = 0; i < size; ++i, ++l) {
    lane[l] ^= data[i];
    lane[l] *= kFnvPrime;
  }

  uint64_t h = kFnvOffsetBasis;
  for (size_t l = 0; l < kContentHashLanes; ++l) h = FnvMixWord(h, lane[l]);
  return FnvMixWord(h, size);
}

/// Deterministic parallel content hash of a whole buffer (see banner above):
/// the in-order fold of the per-chunk striped digests, then the total size.
inline uint64_t ContentHash64(const uint8_t* data, size_t size) {
  const size_t chunks = ParallelChunkCount(size, kContentHashChunk);
  uint64_t partial[kMaxParallelChunks] = {};

  if (chunks <= 1) {
    partial[0] = StripedFnv1a64(data, size);
  } else {
    ParallelFor(size, kContentHashChunk,
                [&](size_t begin, size_t end, size_t chunk_index) {
                  partial[chunk_index] =
                      StripedFnv1a64(data + begin, end - begin);
                });
  }

  uint64_t h = kFnvOffsetBasis;
  for (size_t c = 0; c < (chunks ? chunks : 1); ++c)
    h = FnvMixWord(h, partial[c]);
  return FnvMixWord(h, size);
}

/// Deterministic parallel self-hash of a plan blob: ContentHash64's chunked
/// striped fold, with the 8-byte hash field at `hash_offset` treated as
/// zero, so the seal can live inside the sealed bytes. One canonical
/// function for both sides of the contract — the compiler writing
/// PlanHeader::plan_hash and the engine verifying it — replacing the serial
/// byte-at-a-time Fnv1a64 pass over what is mostly megabytes of frozen
/// weights (plan format v4).
inline uint64_t PlanSelfHash(const uint8_t* plan, size_t size,
                             size_t hash_offset) {
  const size_t chunks = ParallelChunkCount(size, kContentHashChunk);
  uint64_t partial[kMaxParallelChunks] = {};

  auto hash_chunk = [&](size_t begin, size_t end, size_t chunk_index) {
    if (hash_offset >= end || hash_offset + sizeof(uint64_t) <= begin) {
      partial[chunk_index] = StripedFnv1a64(plan + begin, end - begin);
      return;
    }
    // The rare chunk overlapping the hash field hashes a patched copy; the
    // field is 8 bytes in a chunk of up to 1 MiB, so this stays off every
    // other chunk's path.
    std::vector<uint8_t> patched(plan + begin, plan + end);
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
      const size_t at = hash_offset + i;
      if (at >= begin && at < end) patched[at - begin] = 0;
    }
    partial[chunk_index] = StripedFnv1a64(patched.data(), patched.size());
  };

  if (chunks <= 1) {
    hash_chunk(0, size, 0);
  } else {
    ParallelFor(size, kContentHashChunk, hash_chunk);
  }

  uint64_t h = kFnvOffsetBasis;
  for (size_t c = 0; c < (chunks ? chunks : 1); ++c)
    h = FnvMixWord(h, partial[c]);
  return FnvMixWord(h, size);
}

}  // namespace seeml::update

#endif  // SEEML_SOURCE_IDENTITY_HASH_H_
