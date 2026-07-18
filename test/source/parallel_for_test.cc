// =============================================================================
// ParallelFor substrate tests: the correctness axioms the parallel kernels
// are built on. Coverage (every index exactly once, disjoint chunks), the
// determinism contract (chunk geometry and ordered partial reductions are
// pure functions of the shape — never of the thread count), the chunk-count
// cap that lets reductions use fixed stack arrays, and pool hygiene
// (serial mode, nested calls, back-to-back reuse).
// =============================================================================

#include <cstdint>
#include <cstring>
#include <vector>

#include "source/parallel_for.h"
#include "test/framework/seetest.h"

namespace {

using seeml::update::kMaxParallelChunks;
using seeml::update::ParallelChunkCount;
using seeml::update::ParallelChunkGrain;
using seeml::update::ParallelFor;
using seeml::update::ParallelThreadCount;
using seeml::update::SetParallelThreadCount;

/// Pins the pool width for one test; restores automatic resolution after.
struct ScopedThreads {
  explicit ScopedThreads(size_t n) { SetParallelThreadCount(n); }
  ~ScopedThreads() { SetParallelThreadCount(0); }
};

TEST(ParallelFor, CoversEveryIndexExactlyOnce) {
  ScopedThreads threads(4);
  const size_t n = 100'000;
  std::vector<uint8_t> hits(n, 0);
  // Chunks are disjoint, so unsynchronized per-index writes are race-free —
  // exactly the property the kernels rely on.
  ParallelFor(n, 7, [&](size_t begin, size_t end, size_t) {
    for (size_t i = begin; i < end; ++i) ++hits[i];
  });
  size_t covered = 0;
  for (uint8_t h : hits) covered += h;
  EXPECT_EQ(covered, n);  // every index hit...
  bool all_once = true;
  for (uint8_t h : hits) all_once = all_once && h == 1;
  EXPECT_TRUE(all_once);  // ...exactly once
}

TEST(ParallelFor, ChunkGeometryIgnoresThreadCount) {
  const size_t n = 12'345, grain = 100;
  const size_t chunks = ParallelChunkCount(n, grain);
  ASSERT_GT(chunks, 1u);

  auto record = [&](size_t thread_count) {
    ScopedThreads threads(thread_count);
    std::vector<std::pair<size_t, size_t>> bounds(chunks);
    ParallelFor(n, grain, [&](size_t begin, size_t end, size_t chunk) {
      bounds[chunk] = {begin, end};
    });
    return bounds;
  };
  const auto serial = record(1);
  const auto wide = record(5);
  EXPECT_TRUE(serial == wide);
  // The geometry tiles [0, n) in order with no gaps.
  size_t expect_begin = 0;
  for (const auto& [begin, end] : serial) {
    EXPECT_EQ(begin, expect_begin);
    EXPECT_GT(end, begin);
    expect_begin = end;
  }
  EXPECT_EQ(expect_begin, n);
}

TEST(ParallelFor, ChunkCountNeverExceedsTheCap) {
  EXPECT_LE(ParallelChunkCount(1'000'000'000, 1), kMaxParallelChunks);
  EXPECT_LE(ParallelChunkCount(SIZE_MAX / 2, 1), kMaxParallelChunks);
  // Small problems honor the caller's grain exactly.
  EXPECT_EQ(ParallelChunkCount(1000, 100), 10u);
  EXPECT_EQ(ParallelChunkGrain(1000, 100), 100u);
  EXPECT_EQ(ParallelChunkCount(0, 100), 0u);
  EXPECT_EQ(ParallelChunkCount(1, 100), 1u);
}

TEST(ParallelFor, OrderedPartialReductionIsBitwiseThreadCountInvariant) {
  const size_t n = 200'000, grain = 1000;
  std::vector<float> values(n);
  uint64_t state = 0x9E3779B97F4A7C15ULL;
  for (size_t i = 0; i < n; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    values[i] = static_cast<float>(static_cast<int64_t>(state >> 40)) * 1e-6f;
  }

  auto reduce = [&](size_t thread_count) {
    ScopedThreads threads(thread_count);
    double partials[kMaxParallelChunks] = {};
    ParallelFor(n, grain, [&](size_t begin, size_t end, size_t chunk) {
      double sum = 0.0;
      for (size_t i = begin; i < end; ++i) sum += values[i];
      partials[chunk] = sum;
    });
    double total = 0.0;
    const size_t chunks = ParallelChunkCount(n, grain);
    for (size_t c = 0; c < chunks; ++c) total += partials[c];
    return total;
  };
  const double serial = reduce(1);
  const double wide = reduce(8);
  EXPECT_EQ(std::memcmp(&serial, &wide, sizeof(double)), 0);  // bitwise
}

TEST(ParallelFor, ZeroIterationsInvokeNothing) {
  ScopedThreads threads(4);
  bool called = false;
  ParallelFor(0, 16, [&](size_t, size_t, size_t) { called = true; });
  EXPECT_FALSE(called);
}

TEST(ParallelFor, SerialModeNeverSpawnsAndStillCovers) {
  ScopedThreads threads(1);
  EXPECT_EQ(ParallelThreadCount(), 1u);
  size_t sum = 0;
  // With one thread every chunk runs inline on this thread, so plain
  // accumulation is safe.
  ParallelFor(1000, 10, [&](size_t begin, size_t end, size_t) {
    for (size_t i = begin; i < end; ++i) sum += i;
  });
  EXPECT_EQ(sum, 1000u * 999u / 2);
}

TEST(ParallelFor, NestedCallsNeitherDeadlockNorDoubleExecute) {
  ScopedThreads threads(4);
  const size_t outer = 8, inner = 1000, igrain = 10;
  std::vector<uint64_t> totals(outer, 0);
  ParallelFor(outer, 1, [&](size_t b, size_t e, size_t) {
    for (size_t o = b; o < e; ++o) {
      // Pool workers re-entering ParallelFor must not deadlock on their own
      // pool (they run the nested loop inline); the calling thread's nested
      // loops may fan out, so the inner body uses the per-chunk-slot
      // discipline like any reduction.
      uint64_t partials[kMaxParallelChunks] = {};
      ParallelFor(inner, igrain, [&](size_t ib, size_t ie, size_t chunk) {
        uint64_t s = 0;
        for (size_t i = ib; i < ie; ++i) s += i + o;
        partials[chunk] = s;
      });
      uint64_t total = 0;
      for (size_t c = 0; c < ParallelChunkCount(inner, igrain); ++c)
        total += partials[c];
      totals[o] = total;
    }
  });
  for (size_t o = 0; o < outer; ++o)
    EXPECT_EQ(totals[o], inner * (inner - 1) / 2 + o * inner);
}

TEST(ParallelFor, BackToBackCallsReuseThePool) {
  ScopedThreads threads(4);
  // Many small jobs in a row: exercises the park/wake protocol under TSan
  // far more than one large loop would.
  for (size_t round = 0; round < 200; ++round) {
    const size_t n = 512 + round;
    std::vector<uint32_t> out(n, 0);
    ParallelFor(n, 32, [&](size_t begin, size_t end, size_t) {
      for (size_t i = begin; i < end; ++i) out[i] = static_cast<uint32_t>(i);
    });
    EXPECT_EQ(out[n - 1], static_cast<uint32_t>(n - 1));
  }
}

}  // namespace
