#ifndef SEEML_RUNTIME_UPDATE_ENGINE_H_
#define SEEML_RUNTIME_UPDATE_ENGINE_H_

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "compiler/backend/update_types.h"
#include "runtime/dataset.h"

// =============================================================================
// UpdateEngine — the bare-metal virtual machine that executes a compiled
// .seeu Update Plan on-device.
//
// Lifecycle (the ML analog of an OS software update):
//   Load       mmap-equivalent ingest of the plan; one aligned arena
//              allocation — the *only* allocation of the whole update.
//   Train      N steps of {feed batch → execute instruction stream}; each
//              step is forward + backward + optimizer, fully pre-compiled.
//              Interruptible: the persistent segment checkpoints atomically.
//   RunMerge   executes the merge program: W' = W + (α/r)·A@B.
//   Commit     patches the merged weights into a copy of the source model
//              file and atomically renames it into place. The source model
//              is never modified — a failed update leaves the device intact.
// =============================================================================

namespace seeml::update_rt {

struct TrainOptions {
  std::string checkpoint_path;   // empty = no checkpointing
  uint64_t checkpoint_every = 0; // steps between checkpoints (0 = off)
  uint64_t log_every = 100;      // steps between loss log lines (0 = quiet)
  bool resume = false;           // load checkpoint_path before training
};

struct TrainReport {
  uint64_t steps = 0;
  float initial_avg_loss = 0.0f;  // mean loss over the first window
  float final_avg_loss = 0.0f;    // mean loss over the last window
  bool improved() const { return final_avg_loss < initial_avg_loss; }
};

class UpdateEngine {
 public:
  UpdateEngine() = default;
  ~UpdateEngine();

  UpdateEngine(const UpdateEngine&) = delete;
  UpdateEngine& operator=(const UpdateEngine&) = delete;

  /// Loads a plan the caller keeps alive (embedded object-file byte arrays).
  [[nodiscard]] std::expected<void, std::string> LoadFromMemory(
      const uint8_t* plan, size_t size);
  [[nodiscard]] std::expected<void, std::string> LoadFromFile(
      const std::string& path);

  [[nodiscard]] std::expected<TrainReport, std::string> Train(
      Dataset& data, uint64_t steps, const TrainOptions& options = {});

  /// Executes the merge program (adapters folded into weight copies).
  [[nodiscard]] std::expected<void, std::string> RunMerge();

  /// Applies the emit table: copies the source model file, patches every
  /// merged weight's byte range, and atomically renames into `out_path`.
  [[nodiscard]] std::expected<void, std::string> CommitToModel(
      const std::string& source_model_path, const std::string& out_path) const;

  [[nodiscard]] std::expected<void, std::string> SaveCheckpoint(
      const std::string& path) const;
  [[nodiscard]] std::expected<void, std::string> LoadCheckpoint(
      const std::string& path);

  // --- Introspection / test hooks -------------------------------------------
  const seeml::update::PlanHeader& header() const { return header_; }
  float LossValue() const;
  uint8_t* arena() { return arena_; }
  uint64_t step() const { return step_; }
  void SetStep(uint64_t s) { step_ = s; }

  /// Executes the training instruction stream once against whatever is in the
  /// I/O slots right now (no data feeding). Used for gradient verification.
  void ExecuteTrainOnce();

 private:
  [[nodiscard]] std::expected<void, std::string> Initialize();
  void Execute(const std::vector<seeml::update::UpdateInstruction>& program);

  const float* ReadPtr(uint64_t ref) const;
  float* WritePtr(uint64_t ref);

  seeml::update::PlanHeader header_{};
  std::vector<uint8_t> owned_plan_;       // file-loaded plans
  const uint8_t* plan_ = nullptr;         // borrowed or owned_plan_.data()
  size_t plan_size_ = 0;

  std::vector<seeml::update::UpdateInstruction> train_program_;
  std::vector<seeml::update::UpdateInstruction> merge_program_;
  std::vector<seeml::update::EmitEntry> emit_table_;

  uint8_t* arena_ = nullptr;              // single aligned allocation
  const uint8_t* rodata_ = nullptr;       // points into the plan blob
  uint64_t step_ = 0;                     // 1-indexed AdamW timestep
  uint64_t num_classes_ = 0;              // softmax width, 0 = no class loss
  bool merged_ = false;
};

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_UPDATE_ENGINE_H_
