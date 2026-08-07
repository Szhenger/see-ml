#include "runtime/custodian/durable_io.h"

#include "runtime/diagnostics/persisting/error.h"
#include "source/identity/hash.h"
#include "source/parallel/parallel_for.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#else
#include <io.h>
#include <windows.h>
#endif

namespace seeml::update_rt {

namespace up = seeml::update;

namespace {

/// Per-writer-unique sidecar name in the destination's directory (rename
/// atomicity requires same-filesystem). A fixed ".tmp" would let two
/// concurrent writers to one path interleave on a shared inode and rename
/// a mixed-content file into place; pid + a process-local counter keeps
/// every writer on its own sidecar. Crash leftovers are inert — nothing
/// ever reads a sidecar it did not just create.
std::string UniqueTmpPath(const std::string& path) {
  static std::atomic<uint64_t> counter{0};
#ifndef _WIN32
  const unsigned long pid = static_cast<unsigned long>(::getpid());
#else
  const unsigned long pid = static_cast<unsigned long>(::GetCurrentProcessId());
#endif
  return path + ".tmp." + std::to_string(pid) + "." +
         std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
}

}  // namespace

std::expected<void, std::string> WriteFileDurable(
    const std::string& path, std::initializer_list<ByteSpan> parts) {
  const std::string tmp = UniqueTmpPath(path);
#ifndef _WIN32
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0)
    return diag::persisting::Error(diag::persisting::kDurableIo, "cannot write '" + tmp + "'");
  for (const ByteSpan& part : parts) {
    size_t written = 0;
    while (written < part.size) {
      const ssize_t n = ::write(fd, part.data + written, part.size - written);
      if (n < 0) {
        // A signal landing mid-write (checkpoints are large and SIGINT-era
        // stop flags are common) is a retry, not a failed update.
        if (errno == EINTR) continue;
        ::close(fd);
        return diag::persisting::Error(diag::persisting::kDurableIo, "short write to '" + tmp + "'");
      }
      written += static_cast<size_t>(n);
    }
  }
  if (::fsync(fd) != 0) {
    ::close(fd);
    return diag::persisting::Error(diag::persisting::kDurableIo, "fsync of '" + tmp + "' failed");
  }
  ::close(fd);
  if (std::rename(tmp.c_str(), path.c_str()) != 0)
    return diag::persisting::Error(diag::persisting::kDurableIo, "atomic rename to '" + path +
                           "' failed");
  // Persist the rename itself.
  const size_t slash = path.find_last_of('/');
  const std::string dir = slash == std::string::npos
                              ? std::string(".")
                              : path.substr(0, slash == 0 ? 1 : slash);
  const int dfd = ::open(dir.c_str(), O_RDONLY);
  if (dfd >= 0) {
    ::fsync(dfd);  // best effort: some filesystems reject directory fsync
    ::close(dfd);
  }
  return {};
#else
  // Raw Win32, mirroring the POSIX branch's guarantees: every byte's write
  // status is checked (no ofstream buffer whose final flush nobody sees),
  // FlushFileBuffers is the fsync the durability contract demands, and
  // MoveFileEx with MOVEFILE_REPLACE_EXISTING installs over an existing
  // file — CRT rename() refuses that, which broke every checkpoint after
  // the first.
  HANDLE h = ::CreateFileA(tmp.c_str(), GENERIC_WRITE, 0, nullptr,
                           CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "cannot write '" + tmp + "'");
  for (const ByteSpan& part : parts) {
    size_t written = 0;
    while (written < part.size) {
      const DWORD want = static_cast<DWORD>(
          std::min<size_t>(part.size - written, 1u << 30));
      DWORD got = 0;
      if (!::WriteFile(h, part.data + written, want, &got, nullptr) ||
          got == 0) {
        ::CloseHandle(h);
        return diag::persisting::Error(diag::persisting::kDurableIo,
                                       "short write to '" + tmp + "'");
      }
      written += got;
    }
  }
  if (!::FlushFileBuffers(h)) {
    ::CloseHandle(h);
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "fsync of '" + tmp + "' failed");
  }
  ::CloseHandle(h);
  if (!::MoveFileExA(tmp.c_str(), path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "atomic rename to '" + path + "' failed");
  return {};
#endif
}

std::expected<void, std::string> WriteFileDurable(const std::string& path,
                                                  const uint8_t* data,
                                                  size_t size) {
  return WriteFileDurable(path, {ByteSpan{data, size}});
}

std::expected<std::vector<uint8_t>, std::string> ReadFileBytes(
    const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return diag::persisting::Error(diag::persisting::kDurableIo, "cannot open '" + path + "'");
  f.seekg(0, std::ios::end);
  const std::streamoff end = f.tellg();
  if (end < 0)
    return diag::persisting::Error(diag::persisting::kDurableIo, "cannot stat '" + path + "'");
  f.seekg(0);
  std::vector<uint8_t> bytes(static_cast<size_t>(end));
  if (!bytes.empty() &&
      !f.read(reinterpret_cast<char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size())))
    return diag::persisting::Error(diag::persisting::kDurableIo, "cannot read '" + path + "'");
  return bytes;
}

std::expected<uint64_t, std::string> HashFileContent(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "cannot open '" + path + "'");
  f.seekg(0, std::ios::end);
  const std::streamoff end = f.tellg();
  if (end < 0)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "cannot stat '" + path + "'");
  f.seekg(0);
  const size_t size = static_cast<size_t>(end);

  // Mirror ContentHash64's structure exactly (source/identity/hash.h): the
  // same chunk geometry — a pure function of the size — the same per-chunk
  // striped digests folded in chunk order, the same final size mix. The
  // only difference is that each chunk is read from disk on demand, so the
  // peak resident set is one chunk, not one file.
  const size_t grain = up::ParallelChunkGrain(size, up::kContentHashChunk);
  const size_t chunks = up::ParallelChunkCount(size, up::kContentHashChunk);
  std::vector<uint8_t> buf(chunks <= 1 ? size : grain);

  uint64_t h = up::kFnvOffsetBasis;
  const size_t folds = chunks ? chunks : 1;
  for (size_t c = 0; c < folds; ++c) {
    const size_t begin = c * grain;
    const size_t n = chunks <= 1 ? size : std::min(grain, size - begin);
    if (n != 0 &&
        !f.read(reinterpret_cast<char*>(buf.data()),
                static_cast<std::streamsize>(n)))
      return diag::persisting::Error(diag::persisting::kDurableIo,
                                     "cannot read '" + path + "'");
    h = up::FnvMixWord(h, up::StripedFnv1a64(buf.data(), n));
  }
  return up::FnvMixWord(h, size);
}

// --- CommitLock --------------------------------------------------------------

namespace {

/// Releases whichever handle kind this platform stores in CommitLock.
void ReleaseLockHandle(intptr_t handle) {
#ifndef _WIN32
  if (handle >= 0) ::close(static_cast<int>(handle));  // drops the flock
#else
  if (handle != -1) ::CloseHandle(reinterpret_cast<HANDLE>(handle));
#endif
}

}  // namespace

std::expected<CommitLock, std::string> CommitLock::Acquire(
    const std::string& target_path) {
  CommitLock lock;
  const std::string lock_path = target_path + ".lock";
#ifndef _WIN32
  const int fd = ::open(lock_path.c_str(), O_RDWR | O_CREAT, 0644);
  if (fd < 0)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "cannot open lock '" + lock_path + "'");
  if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
    ::close(fd);
    return diag::persisting::Error(
        diag::persisting::kDurableIo,
        "another update is committing to '" + target_path +
            "' — refusing to race it");
  }
  lock.handle_ = fd;
#else
  // dwShareMode = 0 is the exclusion: a second CreateFile on the live
  // handle fails with a sharing violation, and process death releases it —
  // the same crash-safe, no-stale-lock contract as flock.
  HANDLE h = ::CreateFileA(lock_path.c_str(), GENERIC_WRITE, /*share=*/0,
                           nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL,
                           nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    if (::GetLastError() == ERROR_SHARING_VIOLATION)
      return diag::persisting::Error(
          diag::persisting::kDurableIo,
          "another update is committing to '" + target_path +
              "' — refusing to race it");
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "cannot open lock '" + lock_path + "'");
  }
  lock.handle_ = reinterpret_cast<intptr_t>(h);
#endif
  return lock;
}

CommitLock::CommitLock(CommitLock&& o) noexcept : handle_(o.handle_) {
  o.handle_ = -1;
}

CommitLock& CommitLock::operator=(CommitLock&& o) noexcept {
  if (this != &o) {
    ReleaseLockHandle(handle_);
    handle_ = o.handle_;
    o.handle_ = -1;
  }
  return *this;
}

CommitLock::~CommitLock() { ReleaseLockHandle(handle_); }

// --- DurableFileEdit ---------------------------------------------------------

std::expected<DurableFileEdit, std::string> DurableFileEdit::Begin(
    const std::string& source, const std::string& dest) {
  DurableFileEdit edit;
  edit.tmp_ = dest + ".tmp";
  edit.dest_ = dest;

  {
    std::ifstream in(source, std::ios::binary);
    if (!in)
      return diag::persisting::Error(diag::persisting::kDurableIo,
                                     "cannot open '" + source + "'");
    std::ofstream out(edit.tmp_, std::ios::binary | std::ios::trunc);
    if (!out)
      return diag::persisting::Error(diag::persisting::kDurableIo,
                                     "cannot write '" + edit.tmp_ + "'");
    std::vector<char> buf(1u << 20);
    while (in) {
      in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
      const std::streamsize got = in.gcount();
      if (got <= 0) break;
      out.write(buf.data(), got);
      if (!out)
        return diag::persisting::Error(diag::persisting::kDurableIo,
                                       "short write to '" + edit.tmp_ + "'");
      edit.size_ += static_cast<uint64_t>(got);
    }
    if (in.bad())
      return diag::persisting::Error(diag::persisting::kDurableIo,
                                     "cannot read '" + source + "'");
    // The final sub-buffer chunk may still sit in the filebuf; the
    // destructor's flush swallows failure, which would leave a silently
    // truncated sidecar that passes every size_-based bounds check (size_
    // was counted from the SOURCE reads). Flush and close explicitly —
    // close sets failbit when the flush fails.
    out.flush();
    out.close();
    if (!out)
      return diag::persisting::Error(diag::persisting::kDurableIo,
                                     "cannot flush '" + edit.tmp_ + "'");
  }

  edit.f_ = std::fopen(edit.tmp_.c_str(), "r+b");
  if (!edit.f_)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "cannot reopen '" + edit.tmp_ + "'");
  return edit;
}

DurableFileEdit::DurableFileEdit(DurableFileEdit&& o) noexcept
    : f_(o.f_),
      tmp_(std::move(o.tmp_)),
      dest_(std::move(o.dest_)),
      size_(o.size_),
      committed_(o.committed_) {
  o.f_ = nullptr;
  o.committed_ = true;  // the moved-from shell owns nothing to discard
}

DurableFileEdit& DurableFileEdit::operator=(DurableFileEdit&& o) noexcept {
  if (this != &o) {
    CloseAndDiscard();
    f_ = o.f_;
    tmp_ = std::move(o.tmp_);
    dest_ = std::move(o.dest_);
    size_ = o.size_;
    committed_ = o.committed_;
    o.f_ = nullptr;
    o.committed_ = true;
  }
  return *this;
}

DurableFileEdit::~DurableFileEdit() { CloseAndDiscard(); }

void DurableFileEdit::CloseAndDiscard() {
  if (f_) {
    std::fclose(f_);
    f_ = nullptr;
  }
  if (!committed_ && !tmp_.empty()) std::remove(tmp_.c_str());
}

namespace {
bool SeekTo(std::FILE* f, uint64_t offset) {
#ifndef _WIN32
  return ::fseeko(f, static_cast<off_t>(offset), SEEK_SET) == 0;
#else
  // CRT fseek takes a 32-bit long on Windows: a commit target with weight
  // ranges past 2 GiB would wrap the offset and patch the wrong bytes (the
  // bounds check upstream runs on the untruncated value, so nothing trips).
  return ::_fseeki64(f, static_cast<long long>(offset), SEEK_SET) == 0;
#endif
}
}  // namespace

std::expected<void, std::string> DurableFileEdit::ReadAt(uint64_t offset,
                                                         uint8_t* buf,
                                                         size_t n) {
  if (!f_ || offset > size_ || n > size_ - offset)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "edit read outside '" + tmp_ + "'");
  if (!SeekTo(f_, offset) || std::fread(buf, 1, n, f_) != n)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "cannot read '" + tmp_ + "'");
  return {};
}

std::expected<void, std::string> DurableFileEdit::WriteAt(uint64_t offset,
                                                          const uint8_t* buf,
                                                          size_t n) {
  if (!f_ || offset > size_ || n > size_ - offset)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "edit write outside '" + tmp_ + "'");
  if (!SeekTo(f_, offset) || std::fwrite(buf, 1, n, f_) != n)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "short write to '" + tmp_ + "'");
  return {};
}

std::expected<void, std::string> DurableFileEdit::Commit() {
  if (!f_)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "commit of a closed edit");
  if (std::fflush(f_) != 0)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "cannot flush '" + tmp_ + "'");
  // Same durability discipline as WriteFileDurable on both platforms: data
  // fsync (FlushFileBuffers) before the rename, and a rename that can
  // replace an existing destination — CRT rename cannot on Windows, which
  // would fail every repeat commit (the exact defect WriteFileDurable's
  // Win32 branch documents).
#ifndef _WIN32
  if (::fsync(::fileno(f_)) != 0)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "fsync of '" + tmp_ + "' failed");
  std::fclose(f_);
  f_ = nullptr;
  if (std::rename(tmp_.c_str(), dest_.c_str()) != 0)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "atomic rename to '" + dest_ + "' failed");
#else
  if (!::FlushFileBuffers(
          reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(f_)))))
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "fsync of '" + tmp_ + "' failed");
  std::fclose(f_);
  f_ = nullptr;
  if (!::MoveFileExA(tmp_.c_str(), dest_.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "atomic rename to '" + dest_ + "' failed");
#endif
  committed_ = true;
#ifndef _WIN32
  const size_t slash = dest_.find_last_of('/');
  const std::string dir = slash == std::string::npos
                              ? std::string(".")
                              : dest_.substr(0, slash == 0 ? 1 : slash);
  const int dfd = ::open(dir.c_str(), O_RDONLY);
  if (dfd >= 0) {
    ::fsync(dfd);  // best effort: some filesystems reject directory fsync
    ::close(dfd);
  }
#endif
  return {};
}

}  // namespace seeml::update_rt
