// =============================================================================
// custodian/ unit tests: the durable write path (gather writes, atomic
// replacement, whole-file reads) and the SEKP checkpoint container (round
// trip, plan binding, and the corruption/truncation rejections that must
// fire before a byte reaches the arena).
// =============================================================================

#include <algorithm>
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

TEST(Checkpoint, TheV5RecordRoundTripsWithItsBestPayload) {
  // SEKP v5 (E9, #92): the run binding, the source model's score and the
  // best state travel with the segment; the best payload is hashed and
  // verified like the main one, and a peek reads the binding alone.
  ScopedTempDir dir;
  const std::vector<uint8_t> seg = Segment();
  std::vector<uint8_t> best = seg;
  for (auto& b : best) b ^= 0x5A;
  CheckpointRecord record;
  record.step = 12;
  record.horizon_steps = 30;
  record.shuffle_origin = 0x1234;
  record.train_samples = 100;
  record.val_samples = 11;
  record.has_val_initial = true;
  record.has_accuracy = true;
  record.val_initial_loss = 0.75f;
  record.val_initial_accuracy = 0.5f;
  record.has_best = true;
  record.best_step = 9;
  record.best_loss = 0.25f;
  record.best_accuracy = 0.875f;
  record.stale_evals = 3;
  record.best_payload = best;
  ASSERT_OK(SaveCheckpointRecord(dir.File("v5.ckpt"), kPlanHash, record,
                                 seg.data(), kBytes));

  std::vector<uint8_t> dst(kBytes, 0xFF);
  ASSERT_OK_AND_ASSIGN(
      CheckpointRecord got,
      LoadCheckpointRecord(dir.File("v5.ckpt"), kPlanHash, kBytes, dst.data()));
  EXPECT_TRUE(dst == seg);
  EXPECT_EQ(got.step, 12u);
  EXPECT_EQ(got.horizon_steps, 30u);
  EXPECT_TRUE(got.has_binding);
  EXPECT_EQ(got.shuffle_origin, 0x1234u);
  EXPECT_EQ(got.train_samples, 100u);
  EXPECT_EQ(got.val_samples, 11u);
  EXPECT_TRUE(got.has_val_initial);
  EXPECT_TRUE(got.has_accuracy);
  EXPECT_EQ(got.val_initial_loss, 0.75f);
  EXPECT_EQ(got.val_initial_accuracy, 0.5f);
  EXPECT_TRUE(got.has_best);
  EXPECT_EQ(got.best_step, 9u);
  EXPECT_EQ(got.best_loss, 0.25f);
  EXPECT_EQ(got.best_accuracy, 0.875f);
  EXPECT_EQ(got.stale_evals, 3u);
  EXPECT_TRUE(got.best_payload == best);

  ASSERT_OK_AND_ASSIGN(CheckpointRecord peek,
                       PeekCheckpointRecord(dir.File("v5.ckpt")));
  EXPECT_TRUE(peek.has_binding);
  EXPECT_EQ(peek.shuffle_origin, 0x1234u);
  EXPECT_EQ(peek.train_samples, 100u);
  EXPECT_EQ(peek.val_samples, 11u);
  EXPECT_EQ(peek.step, 12u);

  // A flipped byte in the BEST payload is corruption too — and it is
  // caught before the main payload reaches the arena.
  ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> raw,
                       ReadFileBytes(dir.File("v5.ckpt")));
  EXPECT_EQ(raw.size(), 40u + 8 + 64 + 2 * kBytes);
  raw.back() ^= 0x01;
  {
    std::ofstream f(dir.File("bad-best.ckpt"), std::ios::binary);
    f.write(reinterpret_cast<const char*>(raw.data()),
            static_cast<std::streamsize>(raw.size()));
  }
  std::fill(dst.begin(), dst.end(), 0xFF);
  EXPECT_ERROR(LoadCheckpointRecord(dir.File("bad-best.ckpt"), kPlanHash,
                                    kBytes, dst.data()));
  EXPECT_TRUE(std::all_of(dst.begin(), dst.end(),
                          [](uint8_t b) { return b == 0xFF; }));

  // Without a best payload the file is header + one segment, and the
  // v3/v4-shaped call reads it (step and horizon) as before.
  record.best_payload.clear();
  ASSERT_OK(SaveCheckpointRecord(dir.File("lean.ckpt"), kPlanHash, record,
                                 seg.data(), kBytes));
  ASSERT_OK_AND_ASSIGN(raw, ReadFileBytes(dir.File("lean.ckpt")));
  EXPECT_EQ(raw.size(), 40u + 8 + 64 + kBytes);
  uint64_t horizon = 0;
  ASSERT_OK_AND_ASSIGN(uint64_t step,
                       LoadCheckpointFile(dir.File("lean.ckpt"), kPlanHash,
                                          kBytes, dst.data(), &horizon));
  EXPECT_EQ(step, 12u);
  EXPECT_EQ(horizon, 30u);
  EXPECT_TRUE(dst == seg);
}

}  // namespace
