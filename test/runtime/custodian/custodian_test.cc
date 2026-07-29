// =============================================================================
// custodian/ unit tests: the durable write path (gather writes, atomic
// replacement, whole-file reads) and the SEKP checkpoint container (round
// trip, plan binding, and the corruption/truncation rejections that must
// fire before a byte reaches the arena).
// =============================================================================

#include <cstdint>
#include <fstream>
#include <numeric>
#include <vector>

#include "runtime/custodian/checkpoint.h"
#include "runtime/custodian/durable_io.h"
#include "runtime/engine/contract.h"
#include "test/framework/seetest.h"
#include "test/support/scoped_temp_dir.h"

namespace {

using namespace seeml::update_rt;
using seeml::testing::ScopedTempDir;

TEST(DurableIo, RoundTripsSingleAndGatherWrites) {
  ScopedTempDir dir;
  const std::vector<uint8_t> payload{1, 2, 3, 4, 5};
  ASSERT_OK(WriteFileDurable(dir.File("one.bin"), payload.data(),
                             payload.size()));
  ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> back,
                       ReadFileBytes(dir.File("one.bin")));
  EXPECT_TRUE(back == payload);

  // Gather form: header + payload written as one durable concatenation.
  const std::vector<uint8_t> head{9, 9};
  ASSERT_OK(WriteFileDurable(
      dir.File("two.bin"),
      {ByteSpan{head.data(), head.size()},
       ByteSpan{payload.data(), payload.size()}}));
  ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> both,
                       ReadFileBytes(dir.File("two.bin")));
  const std::vector<uint8_t> want{9, 9, 1, 2, 3, 4, 5};
  EXPECT_TRUE(both == want);
}

TEST(DurableIo, ReplacesExistingFilesAtomically) {
  ScopedTempDir dir;
  const std::vector<uint8_t> old_bytes(128, 0xAA);
  const std::vector<uint8_t> new_bytes{7};
  ASSERT_OK(WriteFileDurable(dir.File("f.bin"), old_bytes.data(),
                             old_bytes.size()));
  ASSERT_OK(WriteFileDurable(dir.File("f.bin"), new_bytes.data(),
                             new_bytes.size()));
  ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> back,
                       ReadFileBytes(dir.File("f.bin")));
  EXPECT_TRUE(back == new_bytes);  // fully replaced, not appended or torn
}

TEST(DurableIo, ReadReportsMissingFilesAsWellFormedDiagnostics) {
  ScopedTempDir dir;
  const auto r = ReadFileBytes(dir.File("absent.bin"));
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(WellFormedDiagnostic(r.error()));
}

// =============================================================================
// Checkpoints
// =============================================================================

constexpr uint64_t kPlanHash = 0x5EEA11ULL;
constexpr uint64_t kBytes = 256;

std::vector<uint8_t> Segment() {
  std::vector<uint8_t> seg(kBytes);
  std::iota(seg.begin(), seg.end(), 0);
  return seg;
}

TEST(Checkpoint, RoundTripsTheSegmentAndStep) {
  ScopedTempDir dir;
  const std::vector<uint8_t> seg = Segment();
  ASSERT_OK(SaveCheckpointFile(dir.File("c.ckpt"), kPlanHash, /*step=*/7,
                               seg.data(), kBytes));

  std::vector<uint8_t> restored(kBytes, 0xFF);
  ASSERT_OK_AND_ASSIGN(uint64_t step,
                       LoadCheckpointFile(dir.File("c.ckpt"), kPlanHash,
                                          kBytes, restored.data()));
  EXPECT_EQ(step, 7u);
  EXPECT_TRUE(restored == seg);
}

TEST(Checkpoint, RejectsForeignPlansWithoutTouchingTheArena) {
  ScopedTempDir dir;
  const std::vector<uint8_t> seg = Segment();
  ASSERT_OK(SaveCheckpointFile(dir.File("c.ckpt"), kPlanHash, 3, seg.data(),
                               kBytes));

  std::vector<uint8_t> dst(kBytes, 0xFF);
  const auto r =
      LoadCheckpointFile(dir.File("c.ckpt"), kPlanHash + 1, kBytes,
                         dst.data());
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(WellFormedDiagnostic(r.error()));
  for (uint8_t b : dst) EXPECT_EQ(b, 0xFF);  // dst untouched on failure

  // A layout mismatch (wrong persistent size) is equally foreign.
  EXPECT_ERROR(LoadCheckpointFile(dir.File("c.ckpt"), kPlanHash, kBytes / 2,
                                  dst.data()));
}

TEST(Checkpoint, RejectsBitFlippedAndTruncatedFiles) {
  ScopedTempDir dir;
  const std::vector<uint8_t> seg = Segment();
  ASSERT_OK(SaveCheckpointFile(dir.File("c.ckpt"), kPlanHash, 3, seg.data(),
                               kBytes));
  ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> raw,
                       ReadFileBytes(dir.File("c.ckpt")));

  std::vector<uint8_t> dst(kBytes);

  // Flip one payload byte.
  std::vector<uint8_t> flipped = raw;
  flipped.back() ^= 0x01;
  {
    std::ofstream f(dir.File("flipped.ckpt"), std::ios::binary);
    f.write(reinterpret_cast<const char*>(flipped.data()),
            static_cast<std::streamsize>(flipped.size()));
  }
  EXPECT_ERROR(LoadCheckpointFile(dir.File("flipped.ckpt"), kPlanHash,
                                  kBytes, dst.data()));

  // Cut the payload short.
  {
    std::ofstream f(dir.File("short.ckpt"), std::ios::binary);
    f.write(reinterpret_cast<const char*>(raw.data()),
            static_cast<std::streamsize>(raw.size() - 16));
  }
  EXPECT_ERROR(LoadCheckpointFile(dir.File("short.ckpt"), kPlanHash, kBytes,
                                  dst.data()));
}

}  // namespace
