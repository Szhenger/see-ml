#include "runtime/durable_io.h"

#include <cstdio>
#include <fstream>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace seeml::update_rt {

std::expected<void, std::string> WriteFileDurable(
    const std::string& path, std::initializer_list<ByteSpan> parts) {
  const std::string tmp = path + ".tmp";
#ifndef _WIN32
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0)
    return std::unexpected("UpdateEngine: cannot write '" + tmp + "'");
  for (const ByteSpan& part : parts) {
    size_t written = 0;
    while (written < part.size) {
      const ssize_t n = ::write(fd, part.data + written, part.size - written);
      if (n < 0) {
        ::close(fd);
        return std::unexpected("UpdateEngine: short write to '" + tmp + "'");
      }
      written += static_cast<size_t>(n);
    }
  }
  if (::fsync(fd) != 0) {
    ::close(fd);
    return std::unexpected("UpdateEngine: fsync of '" + tmp + "' failed");
  }
  ::close(fd);
  if (std::rename(tmp.c_str(), path.c_str()) != 0)
    return std::unexpected("UpdateEngine: atomic rename to '" + path +
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
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out)
      return std::unexpected("UpdateEngine: cannot write '" + tmp + "'");
    for (const ByteSpan& part : parts) {
      out.write(reinterpret_cast<const char*>(part.data),
                static_cast<std::streamsize>(part.size));
      if (!out)
        return std::unexpected("UpdateEngine: short write to '" + tmp + "'");
    }
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0)
    return std::unexpected("UpdateEngine: atomic rename to '" + path +
                           "' failed");
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
  if (!f) return std::unexpected("UpdateEngine: cannot open '" + path + "'");
  f.seekg(0, std::ios::end);
  const std::streamoff end = f.tellg();
  if (end < 0)
    return std::unexpected("UpdateEngine: cannot stat '" + path + "'");
  f.seekg(0);
  std::vector<uint8_t> bytes(static_cast<size_t>(end));
  if (!bytes.empty() &&
      !f.read(reinterpret_cast<char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size())))
    return std::unexpected("UpdateEngine: cannot read '" + path + "'");
  return bytes;
}

}  // namespace seeml::update_rt
