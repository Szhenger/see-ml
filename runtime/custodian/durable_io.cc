#include "runtime/custodian/durable_io.h"

#include "runtime/diagnostics/persisting/error.h"

#include <algorithm>
#include <cstdio>
#include <fstream>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#else
#include <windows.h>
#endif

namespace seeml::update_rt {

std::expected<void, std::string> WriteFileDurable(
    const std::string& path, std::initializer_list<ByteSpan> parts) {
  const std::string tmp = path + ".tmp";
#ifndef _WIN32
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0)
    return diag::persisting::Error(diag::persisting::kDurableIo, "cannot write '" + tmp + "'");
  for (const ByteSpan& part : parts) {
    size_t written = 0;
    while (written < part.size) {
      const ssize_t n = ::write(fd, part.data + written, part.size - written);
      if (n < 0) {
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

}  // namespace seeml::update_rt
