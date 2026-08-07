#ifndef SEEML_RUNTIME_ENGINE_UPDATE_ENGINE_H_
#define SEEML_RUNTIME_ENGINE_UPDATE_ENGINE_H_

#include <cstdint>
#include <expected>
#include <functional>
#include <string>
#include <vector>

#include "source/plan/update_types.h"
#include "runtime/feeder/dataset.h"

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
//
// The runtime is partitioned by role; the engine only abstracts the update
// process, verifying at every boundary that each subsystem was used
// correctly (contract.h):
//   feeder/       SDS corpus decode and pipelined batch staging — gated by
//                 VerifyFeederContract
//   executor/     the kernel families the dispatcher executes — gated by
//                 VerifyExecutorContract (via validator/)
//   validator/    load-time bounds proof of every instruction operand
//   custodian/    durable state: checkpoints and the atomic commit path
//   diagnostics/  every error crossing the engine's Train boundary must be
//                 attributable to a registered unit (WellFormedDiagnostic)
// =============================================================================

namespace seeml::update_rt {

struct TrainOptions {
  std::string checkpoint_path;   // empty = no checkpointing
  uint64_t checkpoint_every = 0; // steps between checkpoints (0 = off)
  uint64_t log_every = 100;      // steps between loss log lines (0 = quiet)
  bool resume = false;           // load checkpoint_path before training

  // Held-out validation set: evaluated with the plan's eval program (forward
  // + loss, no parameter mutation) before and after training. When present,
  // TrainReport::improved() gates on validation loss — the honest regression
  // gate — instead of the training-loss windows.
  Dataset* validation = nullptr;

  // Cooperative cancellation, polled once per step. A long-running update on
  // a device must be interruptible; combined with checkpointing the update
  // resumes where it stopped.
  std::function<bool()> should_stop;

  // Record the per-step training loss into TrainReport::loss_curve
  // (one float per step, reserved up front).
  bool record_loss_curve = false;
};

/// One evaluation pass's results. Loss is the mean over compiled-batch
/// chunks (the final partial batch wraps, per the fixed-shape contract);
/// accuracy — available when the plan trains against class labels — is
/// exact: argmax-vs-label over every real sample once, with the wrapped
/// duplicates of the final batch excluded from the count.
struct EvalMetrics {
  float loss = 0.0f;
  float accuracy = 0.0f;
  bool has_accuracy = false;
};

struct TrainReport {
  uint64_t steps = 0;             // steps actually executed
  bool stopped_early = false;     // should_stop() interrupted the run
  float initial_avg_loss = 0.0f;  // mean training loss over the first window
  float final_avg_loss = 0.0f;    // mean training loss over the last window

  bool has_validation = false;
  float val_initial_loss = 0.0f;  // eval-program loss before training
  float val_final_loss = 0.0f;    // eval-program loss after training

  // Task-level metric next to the loss the gate compares: exact held-out
  // argmax accuracy before/after, for plans trained on class labels. The
  // gate itself stays loss-driven — accuracy is reported so callers (and
  // the generated driver) can see what the update did to task quality.
  bool has_val_accuracy = false;
  float val_initial_accuracy = 0.0f;
  float val_final_accuracy = 0.0f;

  std::vector<float> loss_curve;  // per-step loss (record_loss_curve)

  /// The regression gate: validation loss when a held-out set was supplied,
  /// training-window trend otherwise.
  bool improved() const {
    return has_validation ? val_final_loss < val_initial_loss
                          : final_avg_loss < initial_avg_loss;
  }
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

  /// Runs the evaluation program (forward + loss, no parameter mutation)
  /// over `data`, one full pass in compiled-batch chunks, and returns the
  /// mean loss. The training state is untouched.
  [[nodiscard]] std::expected<float, std::string> Evaluate(Dataset& data);

  /// Evaluate(), plus exact argmax accuracy for class-label plans (the
  /// probabilities the eval program's softmax already materializes are
  /// compared against the staged labels; wrapped duplicate samples in the
  /// final batch are excluded, so every sample counts exactly once).
  [[nodiscard]] std::expected<EvalMetrics, std::string> EvaluateMetrics(
      Dataset& data);

  /// Executes the merge program (materializes each adapter's weight delta).
  [[nodiscard]] std::expected<void, std::string> RunMerge();

  /// Applies the emit table: copies the source model file — verifying it
  /// hashes to the plan's source_model_hash first — adds every adapter delta
  /// to its f32 weight range, and durably (fsync + atomic rename) writes
  /// `out_path`.
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
  /// The scheduled learning rate at the current step — a pure function of
  /// (header_, step_). Public so the schedule's boundary behavior (warmup
  /// edges, horizon clamp, floor) is directly testable; it scales every
  /// optimizer step.
  float EffectiveLr() const;

  /// Executes the training instruction stream once against whatever is in the
  /// I/O slots right now (no data feeding). Used for gradient verification.
  void ExecuteTrainOnce();

 private:
  /// Validates the candidate plan and commits engine state only if every
  /// contract passes: a rejected re-Load leaves the previous plan loaded
  /// and fully usable.
  [[nodiscard]] std::expected<void, std::string> Initialize(
      const uint8_t* plan, size_t plan_size);
  /// The training loop itself; Train wraps it with the diagnostics contract.
  [[nodiscard]] std::expected<TrainReport, std::string> TrainImpl(
      Dataset& data, uint64_t steps, const TrainOptions& options);
  [[nodiscard]] std::expected<void, std::string> ValidateDataset(
      Dataset& data) const;
  void Execute(const std::vector<seeml::update::UpdateInstruction>& program);

  const float* ReadPtr(uint64_t ref) const;
  const int8_t* ReadPtrQ8(uint64_t ref) const;
  float* WritePtr(uint64_t ref);

  seeml::update::PlanHeader header_{};
  std::vector<uint8_t> owned_plan_;       // file-loaded plans
  const uint8_t* plan_ = nullptr;         // borrowed or owned_plan_.data()
  size_t plan_size_ = 0;

  std::vector<seeml::update::UpdateInstruction> train_program_;
  std::vector<seeml::update::UpdateInstruction> merge_program_;
  std::vector<seeml::update::UpdateInstruction> eval_program_;
  std::vector<seeml::update::EmitEntry> emit_table_;

  uint8_t* arena_ = nullptr;              // single aligned allocation
  const uint8_t* rodata_ = nullptr;       // points into the plan blob
  uint64_t step_ = 0;                     // 1-indexed AdamW timestep
  uint64_t num_classes_ = 0;              // softmax width, 0 = no class loss
  bool merged_ = false;

  // Where the eval program materializes its softmax probabilities — the
  // basis of the accuracy metric. kNullRef when the eval program carries no
  // class-label softmax (MSE / pure-distillation plans).
  uint64_t eval_probs_ref_ = seeml::update::kNullRef;
  uint64_t eval_softmax_rows_ = 0;
  uint64_t eval_softmax_cols_ = 0;
};

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_ENGINE_UPDATE_ENGINE_H_
