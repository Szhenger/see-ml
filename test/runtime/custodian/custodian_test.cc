// =============================================================================
// custodian/ unit tests: the durable write path (gather writes, atomic
// replacement, whole-file reads) and the SEKP checkpoint container (round
// trip, plan binding, and the corruption/truncation rejections that must
// fire before a byte reaches the arena).
// =============================================================================

#include <cstdint>
#include <fstream>
#include <numeric>
#include <thread>
#include <vector>

#include "runtime/custodian/checkpoint.h"
#include "runtime/custodian/durable_io.h"
#include "source/identity/hash.h"
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

TEST(DurableIo, StreamingFileHashMatchesContentHash64) {
  ScopedTempDir dir;
  // Cross the parallel hash's 1 MiB chunk boundary so the streamed fold
  // reproduces the multi-chunk geometry, not just the single-chunk case.
  std::vector<uint8_t> bytes((1u << 20) * 2 + 12345);
  for (size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = static_cast<uint8_t>((i * 131) ^ (i >> 7));
  const std::string path = dir.File("blob.bin");
  ASSERT_OK(WriteFileDurable(path, bytes.data(), bytes.size()));

  ASSERT_OK_AND_ASSIGN(uint64_t streamed, HashFileContent(path));
  EXPECT_EQ(streamed, seeml::update::ContentHash64(bytes.data(), bytes.size()));

  // Empty file: identical to hashing an empty buffer.
  const std::string empty = dir.File("empty.bin");
  ASSERT_OK(WriteFileDurable(empty, nullptr, 0));
  ASSERT_OK_AND_ASSIGN(uint64_t empty_hash, HashFileContent(empty));
  EXPECT_EQ(empty_hash, seeml::update::ContentHash64(nullptr, 0));
}

TEST(DurableIo, ConcurrentWritersToOnePathYieldOneCompleteFile) {
  // Bounded nondeterminism: two writers racing on the same destination may
  // land in either order, but the surviving file must be ONE writer's
  // complete payload — never an interleaving. (Per-writer-unique sidecar
  // names are what rule out the shared-inode mix.)
  ScopedTempDir dir;
  const std::string path = dir.File("contended.bin");
  const std::vector<uint8_t> a(256 * 1024, 0xAA);
  const std::vector<uint8_t> b(256 * 1024, 0xBB);

  for (int round = 0; round < 8; ++round) {
    std::thread ta([&] {
      auto r = WriteFileDurable(path, {{a.data(), a.size()}});
      (void)r;
    });
    std::thread tb([&] {
      auto r = WriteFileDurable(path, {{b.data(), b.size()}});
      (void)r;
    });
    ta.join();
    tb.join();
    ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> got, ReadFileBytes(path));
    EXPECT_TRUE(got == a || got == b);
  }
}

TEST(DurableIo, CommitLockIsExclusivePerTarget) {
  ScopedTempDir dir;
  const std::string target = dir.File("model.smf");
  auto first = CommitLock::Acquire(target);
  ASSERT_TRUE(first.has_value());
  // Second committer to the same target is refused with a diagnostic.
  auto second = CommitLock::Acquire(target);
  ASSERT_FALSE(second.has_value());
  EXPECT_STR_CONTAINS(second.error(), "another update is committing");
  EXPECT_TRUE(WellFormedDiagnostic(second.error()));
  // A different target is independent.
  EXPECT_TRUE(CommitLock::Acquire(dir.File("other.smf")).has_value());
  // Releasing the first lock frees the target.
  first = CommitLock::Acquire(dir.File("third.smf"));
  EXPECT_TRUE(CommitLock::Acquire(target).has_value());
}

TEST(DurableIo, DurableFileEditPatchesACopyAndDiscardsOnAbort) {
  ScopedTempDir dir;
  const std::string src = dir.File("src.bin");
  const std::string dst = dir.File("dst.bin");
  std::vector<uint8_t> bytes(256);
  std::iota(bytes.begin(), bytes.end(), 0);
  ASSERT_OK(WriteFileDurable(src, bytes.data(), bytes.size()));

  {
    // Abort path: destruction without Commit leaves no destination and no
    // sidecar behind.
    ASSERT_OK_AND_ASSIGN(auto edit, DurableFileEdit::Begin(src, dst));
    EXPECT_EQ(edit.size(), bytes.size());
    const uint8_t patch[4] = {0xAA, 0xBB, 0xCC, 0xDD};
    ASSERT_OK(edit.WriteAt(16, patch, sizeof(patch)));
  }
  EXPECT_FALSE(std::ifstream(dst).good());
  EXPECT_FALSE(std::ifstream(dst + ".tmp").good());

  {
    ASSERT_OK_AND_ASSIGN(auto edit, DurableFileEdit::Begin(src, dst));
    // Bounds are enforced on the sidecar's size.
    uint8_t two[2] = {0, 0};
    EXPECT_ERROR(edit.ReadAt(bytes.size(), two, 1));
    EXPECT_ERROR(edit.WriteAt(bytes.size() - 1, two, 2));
    const uint8_t patch[4] = {0xAA, 0xBB, 0xCC, 0xDD};
    ASSERT_OK(edit.WriteAt(16, patch, sizeof(patch)));
    ASSERT_OK(edit.Commit());
  }
  ASSERT_OK_AND_ASSIGN(auto committed, ReadFileBytes(dst));
  std::vector<uint8_t> expect = bytes;
  expect[16] = 0xAA; expect[17] = 0xBB; expect[18] = 0xCC; expect[19] = 0xDD;
  EXPECT_TRUE(committed == expect);
  // The source is untouched.
  ASSERT_OK_AND_ASSIGN(auto src_bytes, ReadFileBytes(src));
  EXPECT_TRUE(src_bytes == bytes);
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
