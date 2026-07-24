#ifndef SEEML_SOURCE_PARALLEL_FOR_H_
#define SEEML_SOURCE_PARALLEL_FOR_H_

#include <cstddef>
#include <type_traits>

// =============================================================================
// ParallelFor — the deterministic data-parallel substrate shared by the
// compiler and the runtime. Lives in source/ (with hash.h and update_types.h)
// because both sides of the product exploit it and neither may depend on the
// other's internals.
//
// ParallelFor(n, grain, body) splits [0, n) into fixed-size chunks and runs
// body(begin, end, chunk_index) for each, on a lazily-created persistent
// worker pool (the calling thread participates). The design axiom is
// determinism: chunk boundaries depend only on (n, grain) — never on the
// worker count or on scheduling — so
//   - disjoint per-chunk writes produce bitwise-identical memory for every
//     thread count, and
//   - a reduction that stores one partial per chunk_index and combines the
//     partials in chunk order is bitwise-deterministic too.
// The same plan therefore computes the same update on a 1-core and an 8-core
// device; "how many threads" is a pure throughput knob, not a numerics knob.
//
// Thread count resolution, in priority order:
//   1. SetParallelThreadCount(n)      — tests / embedders with a core budget
//   2. SEEML_THREADS environment var  — 1 = fully serial, no thread is ever
//                                       created (bare-metal deployments)
//   3. std::thread::hardware_concurrency()
//
// Chunks are pulled by an atomic counter (dynamic load balancing); pool
// workers that call ParallelFor recursively execute inline, so a body may
// safely reach code that itself parallelizes.
//
// Concurrency contract: ParallelFor is safe to call from any number of
// threads at once. In-flight loops are serialized in arrival order — each
// gets the whole pool — so a loop's result (and, for chunk-order reductions,
// its bitwise value) never depends on what other threads submit. External
// concurrency is therefore a latency overlap tool (e.g. one model's file I/O
// against another's hashing), not a way to multiply throughput of two
// CPU-bound loops.
// =============================================================================

namespace seeml::update {

/// Hard ceiling on the number of chunks a single ParallelFor call produces.
/// Reduction call sites can size fixed (stack) partial arrays with it — no
/// per-call allocation, honoring the runtime's closed-world execution
/// contract.
inline constexpr size_t kMaxParallelChunks = 256;

/// The effective chunk size for a call: at least `grain` (never split work
/// finer than the caller's overhead estimate), raised so the chunk count
/// stays within kMaxParallelChunks. Pure function of (n, grain).
constexpr size_t ParallelChunkGrain(size_t n, size_t grain) {
  if (grain == 0) grain = 1;
  const size_t cap = (n + kMaxParallelChunks - 1) / kMaxParallelChunks;
  return grain < cap ? cap : grain;
}

/// Number of chunks ParallelFor(n, grain, ...) executes; the valid
/// chunk_index range for partial-reduction arrays.
constexpr size_t ParallelChunkCount(size_t n, size_t grain) {
  if (n == 0) return 0;
  const size_t g = ParallelChunkGrain(n, grain);
  return (n + g - 1) / g;
}

/// The resolved worker-pool width (>= 1). 1 means every ParallelFor runs
/// inline on the calling thread.
size_t ParallelThreadCount();

/// Overrides the pool width; 0 restores automatic resolution. Joins any
/// existing workers first. Must not race a running ParallelFor — this is an
/// initialization / test knob, not a per-step control.
void SetParallelThreadCount(size_t count);

namespace internal_par {

using ChunkFn = void (*)(void* ctx, size_t begin, size_t end,
                         size_t chunk_index);

void ParallelForRaw(size_t n, size_t grain, ChunkFn fn, void* ctx);

}  // namespace internal_par

/// body: void(size_t begin, size_t end, size_t chunk_index). Blocks until
/// every chunk has executed. body must only write state that is disjoint
/// per chunk (or per chunk_index slot, for reductions).
template <typename Body>
void ParallelFor(size_t n, size_t grain, Body&& body) {
  using Decayed = std::remove_reference_t<Body>;
  internal_par::ParallelForRaw(
      n, grain,
      [](void* ctx, size_t begin, size_t end, size_t chunk_index) {
        (*static_cast<Decayed*>(ctx))(begin, end, chunk_index);
      },
      const_cast<std::remove_const_t<Decayed>*>(&body));
}

}  // namespace seeml::update

#endif  // SEEML_SOURCE_PARALLEL_FOR_H_
