#ifndef SEEML_SOURCE_HASH_H_
#define SEEML_SOURCE_HASH_H_

#include <cstddef>
#include <cstdint>

#include "source/parallel_for.h"

// =============================================================================
// The integrity primitives shared by the compiler and the runtime.
//
// Every artifact boundary in the update pipeline is hash-bound:
//   - PlanHeader::plan_hash        (Fnv1a64) detects a corrupted / truncated
//                                   .seeu blob; hashed incrementally in spans
//   - PlanHeader::source_model_hash (ContentHash64) binds a plan to the exact
//                                   .smf file whose byte offsets its emit
//                                   table patches
//   - checkpoint plan_hash + payload_hash (Fnv1a64) bind a checkpoint to its
//                                   plan and detect flash corruption on resume
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

}  // namespace seeml::update

#endif  // SEEML_SOURCE_HASH_H_
