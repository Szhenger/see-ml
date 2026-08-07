#ifndef SEEML_RUNTIME_CUSTODIAN_DURABLE_IO_H_
#define SEEML_RUNTIME_CUSTODIAN_DURABLE_IO_H_

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <initializer_list>
#include <string>
#include <vector>

// =============================================================================
// Durable file primitives shared by the update runtime's persistence paths
// (model commit, checkpoints) and its bulk loaders (plans, model files).
//
// WriteFileDurable is the runtime's only way to put bytes on disk that must
// survive a power cut: fsync'd sidecar, atomic rename, best-effort directory
// fsync. Write-tmp + rename alone is atomic but NOT durable — after a power
// cut the rename may survive while the data does not, leaving a truncated
// file on exactly the class of device this runtime targets.
// =============================================================================

namespace seeml::update_rt {

/// One contiguous piece of a gather write.
struct ByteSpan {
  const uint8_t* data;
  size_t size;
};

/// Durably replaces `path` with the concatenation of `parts`. The gather form
/// lets callers with a header + payload (checkpoints) avoid staging a
/// concatenated blob.
[[nodiscard]] std::expected<void, std::string> WriteFileDurable(
    const std::string& path, std::initializer_list<ByteSpan> parts);

[[nodiscard]] std::expected<void, std::string> WriteFileDurable(
    const std::string& path, const uint8_t* data, size_t size);

/// Whole-file read: stat once, size the vector once, one bulk transfer.
[[nodiscard]] std::expected<std::vector<uint8_t>, std::string> ReadFileBytes(
    const std::string& path);

/// Streaming ContentHash64 of a file: bitwise-identical to
/// ContentHash64(bytes, size) over the file's full contents, computed one
/// deterministic chunk at a time (the chunk geometry is a pure function of
/// the size — source/identity/hash.h), so identity checks never require the
/// whole file resident in memory.
[[nodiscard]] std::expected<uint64_t, std::string> HashFileContent(
    const std::string& path);

/// Exclusive advisory lock over a commit target. Two updates committing to
/// the same path would otherwise race at the atomic rename — last writer
/// wins, silently. Acquire turns that race into an explicit error. The lock
/// is an OS-level flock on a `.lock` sidecar, not the sidecar's existence:
/// it dies with the process, so a crash never wedges future commits, and
/// the sidecar file itself is deliberately left in place (unlinking a live
/// lock file races a third committer into a second lock).
class CommitLock {
 public:
  [[nodiscard]] static std::expected<CommitLock, std::string> Acquire(
      const std::string& target_path);
  CommitLock(CommitLock&& o) noexcept;
  CommitLock& operator=(CommitLock&& o) noexcept;
  CommitLock(const CommitLock&) = delete;
  CommitLock& operator=(const CommitLock&) = delete;
  ~CommitLock();

 private:
  CommitLock() = default;
  // POSIX: the lock file's fd (flock dies with it). Windows: the HANDLE of
  // a CreateFile open with dwShareMode = 0 — the share-mode refusal is the
  // exclusion, and CloseHandle (or process death) releases it.
  intptr_t handle_ = -1;
};

/// Copy-on-write durable editor for ranged patches. Begin() streams
/// `source` into a sidecar of `dest` in bounded chunks; ReadAt/WriteAt
/// read-modify-write byte ranges of the sidecar; Commit() fsyncs it and
/// atomically renames it over `dest` (plus best-effort directory fsync).
/// Destruction without Commit() removes the sidecar — the destination is
/// either its old self or the fully patched file, never a mixture, and at
/// no point is the whole file resident in memory.
class DurableFileEdit {
 public:
  [[nodiscard]] static std::expected<DurableFileEdit, std::string> Begin(
      const std::string& source, const std::string& dest);
  DurableFileEdit(DurableFileEdit&& o) noexcept;
  DurableFileEdit& operator=(DurableFileEdit&& o) noexcept;
  DurableFileEdit(const DurableFileEdit&) = delete;
  DurableFileEdit& operator=(const DurableFileEdit&) = delete;
  ~DurableFileEdit();

  uint64_t size() const { return size_; }
  /// The sidecar's path while the edit is open — hash the copy you are
  /// about to patch, not the source it came from (no check-to-patch gap).
  const std::string& sidecar_path() const { return tmp_; }

  [[nodiscard]] std::expected<void, std::string> ReadAt(uint64_t offset,
                                                        uint8_t* buf,
                                                        size_t n);
  [[nodiscard]] std::expected<void, std::string> WriteAt(uint64_t offset,
                                                         const uint8_t* buf,
                                                         size_t n);
  [[nodiscard]] std::expected<void, std::string> Commit();

 private:
  DurableFileEdit() = default;
  void CloseAndDiscard();

  std::FILE* f_ = nullptr;
  std::string tmp_;
  std::string dest_;
  uint64_t size_ = 0;
  bool committed_ = false;
};

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_CUSTODIAN_DURABLE_IO_H_
