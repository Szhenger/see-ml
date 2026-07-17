#ifndef SEEML_RUNTIME_DURABLE_IO_H_
#define SEEML_RUNTIME_DURABLE_IO_H_

#include <cstddef>
#include <cstdint>
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

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_DURABLE_IO_H_
