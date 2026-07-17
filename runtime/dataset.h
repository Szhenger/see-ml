#ifndef SEEML_RUNTIME_DATASET_H_
#define SEEML_RUNTIME_DATASET_H_

#include <bit>
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
// Batches are served sequentially with wraparound, or — with EnableShuffle —
// through a seeded permutation that is re-drawn every epoch. Both modes are
// deterministic and allocation-free per batch, matching the closed-world
// execution contract (the permutation buffer is allocated once, up front).
// =============================================================================

namespace seeml::update_rt {

inline constexpr uint32_t kSdsMagic = 0x31534453;  // "SDS1" little-endian

// SDS is read/written by memcpy of host integers; the documented on-disk
// contract is little-endian. Big-endian hosts need byte-swapping I/O.
static_assert(std::endian::native == std::endian::little,
              "SDS serialization assumes a little-endian host.");

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

  /// Checks every class-index label lies in [0, num_classes) — the training
  /// kernels index rows of size num_classes with these values. No-op for
  /// non-class label kinds or num_classes == 0.
  [[nodiscard]] std::expected<void, std::string> ValidateClassLabels(
      uint64_t num_classes) const;

  /// Copies the next `batch` samples (with wraparound) into the plan's I/O
  /// slots. `label_slot` may be null when the plan takes no labels
  /// (pure distillation).
  void FillBatch(uint64_t batch, float* input_slot, uint8_t* label_slot);

  /// Switches batch serving to a seeded random permutation, re-shuffled at
  /// every epoch boundary. Deterministic for a given (seed, epoch) pair.
  void EnableShuffle(uint64_t seed);

  /// Splits off the LAST `fraction` of samples (before any shuffling) as a
  /// held-out validation set, removing them from this dataset. Deterministic:
  /// the same file and fraction always produce the same split. At least one
  /// sample stays on each side.
  [[nodiscard]] std::expected<Dataset, std::string> SplitValidation(
      double fraction);

  /// Serializes to the .sds file format.
  [[nodiscard]] std::expected<void, std::string> SaveToFile(
      const std::string& path) const;

 private:
  Dataset() = default;

  void Reshuffle();

  std::vector<float> inputs_;   // num_samples * input_dim
  std::vector<uint8_t> labels_; // num_samples * label_bytes_per_sample
  uint64_t num_samples_ = 0;
  uint64_t input_dim_ = 0;
  uint32_t label_kind_ = 0;
  uint64_t label_dim_ = 0;
  uint64_t cursor_ = 0;

  // Shuffled serving order; empty when shuffling is disabled.
  std::vector<uint64_t> order_;
  uint64_t shuffle_state_ = 0;  // splitmix64 state; 0 = shuffling off
};

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_DATASET_H_
