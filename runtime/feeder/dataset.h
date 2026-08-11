#ifndef SEEML_RUNTIME_FEEDER_DATASET_H_
#define SEEML_RUNTIME_FEEDER_DATASET_H_

#include <bit>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

// =============================================================================
// SDS — SeeML Dataset format: fixed-shape samples for the AOT training loop.
//
// Layout (little-endian):
//   u32 magic "SDS1"; u32 version (1 or 2)
//   u64 num_samples; u64 input_dim
//   u32 label_kind (0 = none, 1 = class index i32, 2 = dense f32)
//   u32 input_kind (v2: 0 = f32 feature rows, 1 = i32 token records;
//                   the word was header padding in v1, always 0)
//   u64 label_dim (dense labels only)
//   records[num_samples]:
//     input_kind 0:  f32 input[input_dim], then the label (i32 class,
//                    f32[label_dim] dense, nothing for kind 0)
//     input_kind 1:  i32 tokens[input_dim + 1] and NO stored label — the
//                    record is one sequence; inputs are tokens[0..S) and
//                    the class labels are the shifted view tokens[1..S],
//                    derived at serving time (label_kind must be 1)
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

  /// Token-corpus construction (input_kind 1): `tokens` holds num_records
  /// sequences of (seq + 1) non-negative i32 token ids each. Inputs are a
  /// record's tokens[0..seq); the next-token class labels are the shifted
  /// view tokens[1..seq] — derived, never stored.
  [[nodiscard]] static std::expected<Dataset, std::string> FromTokens(
      std::vector<int32_t> tokens, uint64_t num_records, uint64_t seq);

  uint64_t num_samples() const { return num_samples_; }
  uint64_t input_dim() const { return input_dim_; }
  uint32_t label_kind() const { return label_kind_; }
  uint32_t input_kind() const { return input_kind_; }
  uint64_t label_bytes_per_sample() const;

  /// Checks every class-index label lies in [0, num_classes) — the training
  /// kernels index rows of size num_classes with these values. No-op for
  /// non-class label kinds or num_classes == 0.
  [[nodiscard]] std::expected<void, std::string> ValidateClassLabels(
      uint64_t num_classes) const;

  /// Copies the next `batch` samples (with wraparound) into the plan's I/O
  /// slots. `label_slot` may be null when the plan takes no labels
  /// (pure distillation). For token corpora, `batch` counts token ROWS
  /// (the plan's batch): it must be a whole number of records
  /// (batch % input_dim == 0, the feeder contract's job), the input slot
  /// receives i32 token ids, and the label slot receives the derived
  /// next-token ids.
  void FillBatch(uint64_t batch, float* input_slot, uint8_t* label_slot);

  /// Restarts batch serving from the beginning of the current order (the
  /// current permutation, if shuffling). Evaluation passes rewind first so
  /// every pass consumes the identical sample multiset regardless of where
  /// earlier passes left the cursor.
  void Rewind() { cursor_ = 0; }

  /// Switches batch serving to a seeded random permutation, re-shuffled at
  /// every epoch boundary. Deterministic for a given (seed, epoch) pair.
  void EnableShuffle(uint64_t seed);

  /// Opaque serving-position snapshot: the cursor plus the permutation
  /// epoch it indexes. Trivially cheap — no permutation is copied.
  struct ServingPos {
    uint64_t cursor = 0;
    uint64_t epoch = 0;
  };
  ServingPos SaveServingPos() const { return {cursor_, epoch_}; }

  /// Restores a snapshot taken earlier on this dataset. The batch feeder
  /// un-consumes its staged-but-unserved batch at teardown with this, so
  /// the cursor always reads "exactly the batches the engine consumed" at
  /// any thread count — pipelining must never be observable in the serving
  /// sequence. When the epoch moved past the snapshot, the permutation is
  /// replayed deterministically from the shuffle seed.
  void RestoreServingPos(ServingPos pos);

  /// Advances the serving position as though `rows` batch rows had already
  /// been served — the checkpoint-resume replay (the engine passes
  /// restored_step × batch). The row-to-cursor exchange rate is the
  /// dataset's own: token corpora serve one *record* per seq_len rows,
  /// float corpora one sample per row. Token plans compile batch as a
  /// whole number of sequences, so `rows` must divide into whole records;
  /// a remainder is an error, never a silent truncation.
  [[nodiscard]] std::expected<void, std::string> SkipServed(uint64_t rows);

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

  std::vector<float> inputs_;   // num_samples * input_dim (input_kind 0)
  std::vector<uint8_t> labels_; // num_samples * label_bytes_per_sample
  // Token corpora (input_kind 1): num_samples records of (input_dim + 1)
  // ids each. A separate store — punning i32 bits through float storage
  // would risk NaN-signaling mutation on copies.
  std::vector<int32_t> tokens_;
  uint64_t num_samples_ = 0;
  uint64_t input_dim_ = 0;
  uint32_t label_kind_ = 0;
  uint32_t input_kind_ = 0;
  uint64_t label_dim_ = 0;
  uint64_t cursor_ = 0;

  // Shuffled serving order; empty when shuffling is disabled.
  std::vector<uint64_t> order_;
  uint64_t shuffle_state_ = 0;  // splitmix64 state; 0 = shuffling off
  uint64_t shuffle_origin_ = 0;  // state EnableShuffle started from (replay)
  uint64_t epoch_ = 0;           // permutations drawn beyond the first
};

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_FEEDER_DATASET_H_
