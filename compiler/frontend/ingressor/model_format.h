#ifndef SEEML_COMPILER_FRONTEND_INGRESSOR_MODEL_FORMAT_H_
#define SEEML_COMPILER_FRONTEND_INGRESSOR_MODEL_FORMAT_H_

#include <bit>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// =============================================================================
// SMF — SeeML Model Format: the on-disk contract and its in-memory form.
//
// A minimal, dependency-free binary container for the feed-forward models the
// update compiler operates on. It plays the role ONNX ingestion plays on the
// inference side, without requiring protobuf at build time; the companion
// exporter (tool/export_model.py) converts PyTorch modules into SMF.
//
// This header defines the format constants and the parsed model structures
// only; deserialization lives in model_reader.h, serialization in model_writer.h.
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
// v2 extends the op-kind vocabulary (Gelu/Silu/Mul/LayerNorm); the byte
// layout is unchanged. Readers accept v1 files; the writer emits v2.
inline constexpr uint32_t kSmfVersion = 2;
inline constexpr uint32_t kSmfMinVersion = 1;

// SMF is read/written by memcpy of host integers; the documented on-disk
// contract is little-endian. Big-endian hosts need byte-swapping I/O.
static_assert(std::endian::native == std::endian::little,
              "SMF serialization assumes a little-endian host.");

enum class SmfOpKind : uint8_t {
  kMatMul = 0,
  kAddBias = 1,
  kRelu = 2,
  // v2 vocabulary.
  kGelu = 3,       // tanh-approximation GELU
  kSilu = 4,       // x * sigmoid(x)
  kMul = 5,        // elementwise product of two same-shape activations
  kLayerNorm = 6,  // inputs: {x, gamma, beta}; normalizes the last dim
};
inline constexpr uint8_t kSmfOpKindMax = 6;

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

/// Immutable once loaded: every consumer (parser, resource analyzer, rodata
/// packing) takes it by const reference, so one loaded model may be shared
/// across threads freely.
struct SmfModel {
  std::string input_name;
  std::string output_name;
  std::vector<SmfTensor> tensors;
  std::vector<SmfOp> ops;  // topologically ordered

  // ContentHash64 of the file bytes this model was loaded from
  // (source/hash.h). 0 for models constructed in memory. Compiled into the
  // plan so the runtime refuses to patch a model file the plan was not
  // built against.
  uint64_t content_hash = 0;

  const SmfTensor* FindTensor(std::string_view name) const;
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_FRONTEND_INGRESSOR_MODEL_FORMAT_H_
