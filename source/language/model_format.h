#ifndef SEEML_SOURCE_LANGUAGE_MODEL_FORMAT_H_
#define SEEML_SOURCE_LANGUAGE_MODEL_FORMAT_H_

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
// only — the source-language contract itself, which is why it lives in
// source/ (with hash.h and update_types.h) rather than in the frontend: the
// frontend abstracts the *process* of ingesting SMF, not the language.
// Deserialization lives in compiler/frontend/ingressor/model_reader.h,
// serialization in model_writer.h.
//
// Layout (little-endian):
//   u32 magic "SMF1"    u32 version
//   u32 num_tensors     u32 num_ops
//   str input_name      str output_name          (str = u16 length + bytes)
//   u64 seq_len                                  (v3+ only; 0 = non-sequential)
//   tensors[num_tensors]:
//     str name; u8 rank; u8 flags (bit0 = constant);
//     i64 dims[rank]; u64 data_offset (absolute, 0 if not constant); u64 byte_size
//   ops[num_ops] (topologically ordered):
//     u8 kind; str name; u8 num_inputs; str inputs[num_inputs]; str output;
//     u32 attr0                                  (v3+ only; per-kind meaning)
//     u32 attr1                                  (v5+ only; per-kind meaning)
//   data section: each constant tensor's f32 blob at its 64-aligned data_offset
//
// The absolute data_offset of every weight is preserved through compilation —
// it is how the runtime's EmitTable patches the updated weights back into a
// copy of the model file during atomic commit.
// =============================================================================

namespace seeml::update {

inline constexpr uint32_t kSmfMagic = 0x31464D53;  // "SMF1" little-endian
// v2 extends the op-kind vocabulary (Gelu/Silu/Mul/LayerNorm); the byte
// layout is unchanged. v3 adds the transformer vocabulary
// (Add/RmsNorm/Rope/Attention), a model-level u64 seq_len after the
// output name, and a per-op u32 attr0 word after the output (heads for
// Rope/Attention, 0 elsewhere). v4 adds kEmbedding — with it, a model may
// declare a rank-1 dynamic ({-1}) non-const input, meaning i32 token ids
// consumed exclusively by embedding ops; the byte layout is v3's. v5 adds
// a second per-op u32 attr1 after attr0: for kRope it carries the IEEE-754
// bits of the rotary base θ (0 = the classic 10000); 0 elsewhere. Without
// it a Llama-3 (θ=500k) or Qwen (θ=1M) port compiled cleanly to the wrong
// rotation — "unsupported" silently became "wrong".
// Readers accept v1..v5; the writer emits v5.
inline constexpr uint32_t kSmfVersion = 5;
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
  // v3 vocabulary: the transformer family. Rows are sequence positions,
  // grouped into sequences of the model-level seq_len; heads ride attr0.
  kAdd = 7,        // elementwise sum of two same-shape activations (residual)
  kRmsNorm = 8,    // inputs: {x, gamma}; RMS-normalizes the last dim
  kRope = 9,       // rotary position embedding; attr0 = num_heads
  kAttention = 10, // inputs: {q, k, v}; causal SDPA; attr0 = num_heads
  // v4 vocabulary.
  kEmbedding = 11, // inputs: {tokens (the i32 graph input), table [V, D]}
};
inline constexpr uint8_t kSmfOpKindMax = 11;
// A file carrying a kind newer than its own version is corrupt, not
// forward-compatible: v1 ended at kRelu, v2 at kLayerNorm, v3 at
// kAttention.
inline constexpr uint8_t kSmfOpKindMaxV1 = 2;
inline constexpr uint8_t kSmfOpKindMaxV2 = 6;
inline constexpr uint8_t kSmfOpKindMaxV3 = 10;

struct SmfTensor {
  std::string name;
  std::vector<int64_t> dims;
  bool is_const = false;
  uint64_t data_offset = 0;  // absolute file offset of the f32 blob
  uint64_t byte_size = 0;
  // Populated for constant tensors on load. The `{}` matters: designated
  // initializers all over the tests omit this field, and GCC's -Wextra
  // flags an omitted field as missing unless it has a default initializer.
  std::vector<uint8_t> data{};
};

struct SmfOp {
  SmfOpKind kind;
  std::string name;
  std::vector<std::string> inputs;
  std::string output;
  // v3: per-kind scalar attribute — num_heads for kRope/kAttention, 0
  // elsewhere (and for every op of a pre-v3 file).
  uint32_t attr0 = 0;
  // v5: second per-kind scalar — for kRope the f32 bits of the rotary base
  // θ (0 = kSmfDefaultRopeBase); 0 elsewhere (and for every pre-v5 file).
  uint32_t attr1 = 0;
};

/// The rotary base a kRope op means when its attr1 is zero (pre-v5 files,
/// and v5 writers that leave the default).
inline constexpr float kSmfDefaultRopeBase = 10000.0f;

/// Immutable once loaded: every consumer (parser, resource analyzer, rodata
/// packing) takes it by const reference, so one loaded model may be shared
/// across threads freely.
struct SmfModel {
  std::string input_name;
  std::string output_name;
  // v3: rows-per-sequence for the transformer ops (causal masking and RoPE
  // positions). 0 = non-sequential model; required nonzero by the parser
  // whenever a kRope/kAttention op is present.
  uint64_t seq_len = 0;
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

#endif  // SEEML_SOURCE_LANGUAGE_MODEL_FORMAT_H_
