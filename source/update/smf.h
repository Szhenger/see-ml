#ifndef SEEML_UPDATE_SMF_H_
#define SEEML_UPDATE_SMF_H_

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

// =============================================================================
// SMF — SeeML Model Format.
//
// A minimal, dependency-free binary container for the feed-forward models the
// update compiler operates on. It plays the role ONNX ingestion plays on the
// inference side, without requiring protobuf at build time; the companion
// exporter (tools/export_model.py) converts PyTorch modules into SMF.
//
// Layout (little-endian):
//   u32 magic "SMF1"    u32 version
//   u32 num_tensors     u32 num_ops
//   str input_name      str output_name          (str = u16 length + bytes)
//   tensors[num_tensors]:
//     str name; u8 rank; u8 flags (bit0 = constant);
//     i64 dims[rank]; u64 data_offset (absolute, 0 if not constant); u64 byte_size
//   ops[num_ops] (topologically ordered):
//     u8 kind; str name; u8 num_inputs; str inputs[num_inputs]; str output
//   data section: each constant tensor's f32 blob at its 64-aligned data_offset
//
// The absolute data_offset of every weight is preserved through compilation —
// it is how the runtime's EmitTable patches the updated weights back into a
// copy of the model file during atomic commit.
// =============================================================================

namespace seeml::update {

inline constexpr uint32_t kSmfMagic = 0x31464D53;  // "SMF1" little-endian
inline constexpr uint32_t kSmfVersion = 1;

enum class SmfOpKind : uint8_t { kMatMul = 0, kAddBias = 1, kRelu = 2 };

struct SmfTensor {
  std::string name;
  std::vector<int64_t> dims;
  bool is_const = false;
  uint64_t data_offset = 0;  // absolute file offset of the f32 blob
  uint64_t byte_size = 0;
  std::vector<uint8_t> data;  // populated for constant tensors on load
};

struct SmfOp {
  SmfOpKind kind;
  std::string name;
  std::vector<std::string> inputs;
  std::string output;
};

struct SmfModel {
  std::string input_name;
  std::string output_name;
  std::vector<SmfTensor> tensors;
  std::vector<SmfOp> ops;  // topologically ordered

  const SmfTensor* FindTensor(std::string_view name) const;
};

[[nodiscard]] std::expected<SmfModel, std::string> LoadSmf(
    const std::string& path);

/// Serializes a model. Constant tensors must carry their data in
/// SmfTensor::data; data_offset/byte_size are (re)computed during the write.
[[nodiscard]] std::expected<void, std::string> SaveSmf(const std::string& path,
                                                       SmfModel& model);

}  // namespace seeml::update

#endif  // SEEML_UPDATE_SMF_H_
