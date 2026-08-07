// =============================================================================
// ContentHash64 tests: the parallel model-identity hash must be a pure
// function of the bytes — invariant to worker count — and sensitive to every
// byte and to the length, including the striped tail and chunk boundaries.
// =============================================================================

#include <cstdint>
#include <vector>

#include "source/identity/hash.h"
#include "source/parallel/parallel_for.h"
#include "test/framework/seetest.h"

namespace {

using namespace seeml::update;

std::vector<uint8_t> DeterministicBytes(size_t n, uint64_t seed) {
  std::vector<uint8_t> v(n);
  uint64_t s = seed * 0x9E3779B97F4A7C15ULL + 1;
  for (size_t i = 0; i < n; ++i) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    v[i] = static_cast<uint8_t>(s >> 56);
  }
  return v;
}

// The serial definition of ContentHash64: fold the per-chunk striped digests
// in chunk order, then the size. The production function must equal this for
// every worker count.
uint64_t SerialReference(const uint8_t* data, size_t size) {
  const size_t grain = ParallelChunkGrain(size, kContentHashChunk);
  uint64_t h = kFnvOffsetBasis;
  if (size == 0) {
    h = FnvMixWord(h, StripedFnv1a64(data, 0));
  } else {
    for (size_t begin = 0; begin < size; begin += grain) {
      const size_t end = begin + grain < size ? begin + grain : size;
      h = FnvMixWord(h, StripedFnv1a64(data + begin, end - begin));
    }
  }
  return FnvMixWord(h, size);
}

TEST(ContentHash, MatchesSerialDefinitionAcrossThreadCounts) {
  // 2.5 MiB spans several 1 MiB chunks with a partial tail chunk.
  auto bytes = DeterministicBytes((5u << 20) / 2, /*seed=*/7);
  const uint64_t reference = SerialReference(bytes.data(), bytes.size());

  for (size_t threads : {size_t{1}, size_t{4}}) {
    SetParallelThreadCount(threads);
    EXPECT_EQ(ContentHash64(bytes.data(), bytes.size()), reference);
  }
  SetParallelThreadCount(0);  // restore automatic resolution
}

TEST(ContentHash, EveryBytePositionMatters) {
  auto bytes = DeterministicBytes(3 * kContentHashChunk / 2, /*seed=*/11);
  const uint64_t base = ContentHash64(bytes.data(), bytes.size());

  // First byte, a mid-chunk byte, the chunk boundary, and the very last
  // byte (the striped tail).
  for (size_t pos : {size_t{0}, bytes.size() / 3, kContentHashChunk,
                     bytes.size() - 1}) {
    bytes[pos] ^= 0x01;
    EXPECT_NE(ContentHash64(bytes.data(), bytes.size()), base);
    bytes[pos] ^= 0x01;
  }
  EXPECT_EQ(ContentHash64(bytes.data(), bytes.size()), base);
}

TEST(ContentHash, LengthMatters) {
  auto bytes = DeterministicBytes(1024, /*seed=*/3);
  EXPECT_NE(ContentHash64(bytes.data(), 1024),
            ContentHash64(bytes.data(), 1023));
}

TEST(ContentHash, SmallAndUnalignedSizes) {
  // Exercise every striped-tail length around one lane group, plus empty.
  auto bytes = DeterministicBytes(32, /*seed=*/5);
  uint64_t prev = ContentHash64(bytes.data(), 0);
  for (size_t n = 1; n <= 17; ++n) {
    const uint64_t h = ContentHash64(bytes.data(), n);
    EXPECT_NE(h, prev);
    prev = h;
  }
}

TEST(PlanSelfHash, EqualsContentHashOfTheZeroedBlobAtEveryOffsetClass) {
  // The defining identity: PlanSelfHash(blob, at) must equal ContentHash64
  // of a copy with the 8-byte field at `at` zeroed — including when the
  // field STRADDLES a chunk boundary, the one code path where the patched
  // copy must be split across two chunks' hashes. A multi-chunk blob makes
  // that path real (single-chunk blobs never split the field).
  const size_t size = 3 * seeml::update::kContentHashChunk + 123;
  std::vector<uint8_t> blob(size);
  for (size_t i = 0; i < size; ++i)
    blob[i] = static_cast<uint8_t>(i * 2654435761u >> 13);
  const size_t grain =
      seeml::update::ParallelChunkGrain(size, seeml::update::kContentHashChunk);
  ASSERT_TRUE(grain < size);  // genuinely multi-chunk

  const size_t offsets[] = {0,
                            5,
                            grain - 7,  // straddles the first boundary
                            grain - 1,  // straddles by a single byte
                            grain,      // exactly at the boundary
                            2 * grain - 3,
                            size - sizeof(uint64_t)};
  for (const size_t at : offsets) {
    std::vector<uint8_t> zeroed = blob;
    std::memset(zeroed.data() + at, 0, sizeof(uint64_t));
    EXPECT_EQ(seeml::update::PlanSelfHash(blob.data(), size, at),
              seeml::update::ContentHash64(zeroed.data(), size));
  }
  // And the seal is insensitive to what the field currently holds — the
  // property Initialize relies on when verifying a sealed plan.
  std::vector<uint8_t> resealed = blob;
  std::memset(resealed.data() + grain - 4, 0xAB, sizeof(uint64_t));
  EXPECT_EQ(seeml::update::PlanSelfHash(blob.data(), size, grain - 4),
            seeml::update::PlanSelfHash(resealed.data(), size, grain - 4));
}

TEST(ContentHash, DistinctContentDistinctDigest) {
  auto a = DeterministicBytes(4096, /*seed=*/1);
  auto b = DeterministicBytes(4096, /*seed=*/2);
  EXPECT_NE(ContentHash64(a.data(), a.size()),
            ContentHash64(b.data(), b.size()));
}

}  // namespace
