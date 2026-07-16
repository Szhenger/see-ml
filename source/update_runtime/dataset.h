#ifndef SEEML_UPDATE_RUNTIME_DATASET_H_
#define SEEML_UPDATE_RUNTIME_DATASET_H_

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

// =============================================================================
// SDS — SeeML Dataset format: fixed-shape samples for the AOT training loop.
//
// Layout (little-endian):
//   u32 magic "SDS1"; u32 version
//   u64 num_samples; u64 input_dim
//   u32 label_kind (0 = none, 1 = class index i32, 2 = dense f32); u32 pad
//   u64 label_dim (dense labels only)
//   records[num_samples]: f32 input[input_dim], then the label
//     (i32 for class labels, f32[label_dim] for dense, nothing for kind 0)
//
// Batches are served sequentially with wraparound — deterministic and
// allocation-free, matching the closed-world execution contract.
// =============================================================================

namespace seeml::update_rt {

inline constexpr uint32_t kSdsMagic = 0x31534453;  // "SDS1" little-endian

class Dataset {
 public:
  [[nodiscard]] static std::expected<Dataset, std::string> LoadFromFile(
      const std::string& path);

  /// In-memory construction (tests / embedded corpora). `labels` is raw label
  /// bytes laid out per-sample exactly as in the file format.
  [[nodiscard]] static std::expected<Dataset, std::string> FromMemory(
      std::vector<float> inputs, std::vector<uint8_t> labels,
      uint64_t num_samples, uint64_t input_dim, uint32_t label_kind,
      uint64_t label_dim);

  uint64_t num_samples() const { return num_samples_; }
  uint64_t input_dim() const { return input_dim_; }
  uint32_t label_kind() const { return label_kind_; }
  uint64_t label_bytes_per_sample() const;

  /// Copies the next `batch` samples (with wraparound) into the plan's I/O
  /// slots. `label_slot` may be null when the plan takes no labels
  /// (pure distillation).
  void FillBatch(uint64_t batch, float* input_slot, uint8_t* label_slot);

  /// Serializes to the .sds file format.
  [[nodiscard]] std::expected<void, std::string> SaveToFile(
      const std::string& path) const;

 private:
  Dataset() = default;

  std::vector<float> inputs_;   // num_samples * input_dim
  std::vector<uint8_t> labels_; // num_samples * label_bytes_per_sample
  uint64_t num_samples_ = 0;
  uint64_t input_dim_ = 0;
  uint32_t label_kind_ = 0;
  uint64_t label_dim_ = 0;
  uint64_t cursor_ = 0;
};

}  // namespace seeml::update_rt

#endif  // SEEML_UPDATE_RUNTIME_DATASET_H_
