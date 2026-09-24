#ifndef SEEML_RUNTIME_DISPATCHER_UPDATE_ENGINE_H_
#define SEEML_RUNTIME_DISPATCHER_UPDATE_ENGINE_H_

#include <cstdint>
#include <expected>
#include <functional>
#include <string>
#include <vector>

#include "source/plan/update_types.h"
#include "runtime/executor/backend.h"
#include "runtime/pipeline/dataset.h"

// =============================================================================
// UpdateEngine — the bare-metal virtual machine that executes a compiled
// .seeu Update Plan on-device.
//
// Lifecycle (the ML analog of an OS software update):
//   Load       the plan is read whole into one heap buffer (LoadFromFile:
//              a buffered read, not a mapping) or borrowed from the binary
//              image (LoadFromMemory: the package's .incbin blob, which a
//              GPU backend may wrap zero-copy); then one aligned arena
//              allocation. A loaded update therefore holds the plan plus
//              the arena — the figure the final memory gate and the field
//              reports' RSS / (arena + plan) ratio both count.
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
//   pipeline/     SDS corpus decode and pipelined batch staging — gated by
//                 VerifyFeederContract
//   executor/     the kernel families the dispatcher executes — gated by
//                 VerifyExecutorContract (via verifier/) — behind the
//                 ExecutorBackend seam: the CPU library is the reference,
//                 a GPU backend (Metal) is opt-in per SelectBackend
//   verifier/     load-time bounds proof of every instruction operand
//   storage/      durable state: checkpoints and the atomic commit path
//   diagnostics/  every error crossing the engine's Train boundary must be
//                 attributable to a registered unit (WellFormedDiagnostic)
// =============================================================================

namespace seeml::update_rt {

struct TrainOptions {
  std::string checkpoint_path;   // empty = no checkpointing
  uint64_t checkpoint_every = 0; // steps between checkpoints (0 = off)
  uint64_t log_every = 100;      // steps between loss log lines (0 = quiet)
  bool resume = false;           // load checkpoint_path before training

  // The LR schedule's horizon, in optimizer steps (E8, #91). 0 = derive it
  // from the run, which is what every ordinary caller wants:
  //   a fresh run of N steps anneals over N;
  //   a resumed run with no explicit step count trains the REMAINDER to
  //     the horizon its checkpoint recorded, on that same schedule — so an
  //     interrupted-and-resumed update is bit-identical to an
  //     uninterrupted one;
  //   a resumed run given an explicit step count anneals over
  //     (resumed step + that count): more steps were asked for, so the
  //     schedule is stretched to cover them rather than left to idle at
  //     the floor.
  // Set it to train the first k steps of a longer schedule deliberately.
  // Until E8 the horizon was the plan's compile-time default_steps whatever
  // the run did: a shorter run never annealed, a longer one sat at the
  // floor (LR 0 by the old default) for the excess, and --resume with no
  // step count ran a whole extra budget entirely past the horizon.
  uint64_t horizon_steps = 0;

  // Held-out validation set: evaluated with the plan's eval program (forward
  // + loss, no parameter mutation) before and after training. When present,
  // TrainReport::improved() gates on validation loss — the honest regression
  // gate — instead of the training-loss windows.
  Dataset* validation = nullptr;

  // Periodic evaluation and the best state (E9, #92). With a validation set
  // and eval_every > 0 the run evaluates every eval_every optimizer steps
  // (kEvalEveryAuto: max(1, steps / 10)) and keeps a copy of the persistent
  // segment at the best validation loss seen — the source model's own score
  // being the bar to beat. When training ends, THAT state is what the arena
  // holds, so merge and commit ship the best evaluated update rather than
  // the last one: on a small corpus the endpoint is often past the minimum,
  // and a state that is still better than step 0 used to be committed while
  // a better one was thrown away. The extra evaluation never touches a
  // training bit (the eval program writes transients only), and the copy is
  // one more persistent_size allocation, made at Train.
  // patience > 0 stops the run after that many consecutive evaluations
  // without a new best. eval_every == 0 is the pre-E9 loop: evaluate
  // before and after, commit the last state, bit for bit.
  static constexpr uint64_t kEvalEveryAuto = ~0ULL;
  uint64_t eval_every = 0;
  uint64_t patience = 0;

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

/// Accumulated wall-clock split of the training instruction stream, per
/// phase: forward (stream start through the loss forward), backward, and
/// clip + optimizer. Populated only when the runtime is built with
/// -DSEEML_STEP_TIMING (the benchmark-harness build; see
/// docs/benchmarks.md) — in every other build all fields stay zero and no
/// clock is read inside the step, so the shipped runtime is untouched.
/// The fields are unconditional so the header is identical under both
/// builds (no ODR hazard between differently-configured objects).
struct StepTimings {
  uint64_t steps = 0;         // optimizer steps accumulated (G grad
                              // executions each under accumulation)
  double fwd_seconds = 0.0;
  double bwd_seconds = 0.0;
  double opt_seconds = 0.0;
};

struct TrainReport {
  uint64_t steps = 0;             // steps actually executed
  bool stopped_early = false;     // should_stop() interrupted the run
  float initial_avg_loss = 0.0f;  // mean training loss over the first window
  float final_avg_loss = 0.0f;    // mean training loss over the last window

  bool has_validation = false;
  float val_initial_loss = 0.0f;  // eval-program loss before training —
                                  // the SOURCE model's, also across resumes
  float val_final_loss = 0.0f;    // eval-program loss of the state the run
                                  // leaves in the arena (the best one when
                                  // tracked, else the last)

  // The best-state bookkeeping (E9). When best tracking ran, best_step is
  // the step whose state the arena holds and val_last_* score the state the
  // run actually ended on; they equal val_final_* whenever the endpoint was
  // the best. eval_every / patience are the values the run resolved.
  bool best_tracked = false;
  uint64_t best_step = 0;
  float val_last_loss = 0.0f;
  float val_last_accuracy = 0.0f;
  uint64_t evaluations = 0;        // periodic evaluations run (not the two
                                   // bracketing ones)
  bool stopped_by_patience = false;
  uint64_t eval_every = 0;
  uint64_t patience = 0;

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

  /// The gate with a margin (#67): the loss must fall by at least
  /// `min_fraction` of its initial value. Zero is exactly improved() —
  /// today's strict comparison — so a calibration-only drift of 1e-7 still
  /// passes at the default and is refused at any positive margin.
  bool ImprovedBy(float min_fraction) const {
    if (!improved()) return false;
    if (!(min_fraction > 0.0f)) return true;
    const float before = has_validation ? val_initial_loss : initial_avg_loss;
    const float after = has_validation ? val_final_loss : final_avg_loss;
    return after <= before * (1.0f - min_fraction);
  }

  /// The accuracy gate (#67): held-out argmax accuracy must not have
  /// dropped. Vacuously true when the plan reports no accuracy (MSE,
  /// distillation, no validation split) — callers that require it must
  /// check has_val_accuracy first and treat its absence as a usage error.
  bool AccuracyHeld() const {
    return !has_val_accuracy || val_final_accuracy >= val_initial_accuracy;
  }
};

class UpdateEngine {
 public:
  UpdateEngine();
  ~UpdateEngine();

  UpdateEngine(const UpdateEngine&) = delete;
  UpdateEngine& operator=(const UpdateEngine&) = delete;

  /// Chooses the executor backend (default: cpu). kAuto resolves to Metal
  /// when a device exists and to the CPU otherwise (backend_note() says
  /// which and why); kMetal is a hard error where unavailable. May be
  /// called before or after a Load — a loaded plan is re-bound, and a
  /// backend that cannot bind it is refused, leaving the current one.
  [[nodiscard]] std::expected<void, std::string> SelectBackend(
      BackendKind requested);
  /// The resolved backend: its name ("cpu" | "metal"), its device label,
  /// and the fallback note from an `auto` resolution (empty if none).
  BackendKind backend_kind() const { return backend_kind_; }
  const char* backend_name() const;
  std::string backend_device() const;
  const std::string& backend_note() const { return backend_note_; }
  /// The CPU GEMM tile geometry the loaded plan's header selects (v11;
  /// the runtime defaults for zero fields or a pre-v11 plan). A
  /// throughput knob the compiler decided; no result bit depends on it.
  kernels::GemmTiles gemm_tiles() const;

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

  /// Checks that `source_model_path` hashes to the plan's source_model_hash,
  /// without training or writing anything. CommitToModel re-verifies the
  /// copy it patches (no TOCTOU window); calling this right after load
  /// merely fails fast — a mismatched model costs one file scan instead of
  /// a full training run. Plans with no hash binding (hash 0) always pass.
  /// Binds the source model file for the eval program (plan v17, E12 #95).
  /// A plan compiled with --quantize-base / --bf16-base from a model FILE
  /// scores what ships: its eval program reads the student's frozen weights
  /// as the f32 the commit patches, straight from this file, instead of the
  /// plan's narrow copies. The file is verified against source_model_hash,
  /// checked to cover every extent the eval program reads, and mapped
  /// read-only for the engine's lifetime (or the next plan load). Until it
  /// is bound such a plan refuses to evaluate. A plan that does not need it
  /// (scores_shipped() == false) accepts the call as VerifySourceModel.
  [[nodiscard]] std::expected<void, std::string> BindSourceModel(
      const std::string& source_model_path);
  /// The eval program reads the source model file (v17): its losses are
  /// the shipped function's, not the in-plan proxy's.
  bool scores_shipped() const { return source_required_ > 0; }

  [[nodiscard]] std::expected<void, std::string> VerifySourceModel(
      const std::string& source_model_path) const;

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
  /// The horizon EffectiveLr anneals over: the current run's (set by
  /// Train, restored by LoadCheckpoint), or the plan's default_steps when
  /// no run has set one. SetHorizon is the test hook beside SetStep.
  uint64_t horizon() const {
    return horizon_ ? horizon_ : header_.default_steps;
  }
  void SetHorizon(uint64_t steps) { horizon_ = steps; }
  /// The scheduled learning rate at the current step — a pure function of
  /// (header_, step_). Public so the schedule's boundary behavior (warmup
  /// edges, horizon clamp, floor) is directly testable; it scales every
  /// optimizer step.
  float EffectiveLr() const;

  /// Executes the training instruction stream once against whatever is in the
  /// I/O slots right now (no data feeding). Used for gradient verification.
  /// Under gradient accumulation this is ONE grad execution: it folds the
  /// slots' micro-batch into the accumulators and never steps.
  void ExecuteTrainOnce();
  /// Micro-batches accumulated per optimizer step (1 = the classic step).
  uint64_t grad_accum_steps() const { return grad_accum_; }

  /// The step-latency accumulator (zeros unless built with
  /// -DSEEML_STEP_TIMING; reset by every successful Load).
  StepTimings step_timings() const { return timings_; }
  void ResetStepTimings() { timings_ = {}; }

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
  /// Dispatches [begin, end) of `program` through the backend and flushes
  /// it, so the arena is coherent when the range returns. A backend failure
  /// (a GPU command buffer that did not complete) is an executor error.
  [[nodiscard]] std::expected<void, std::string> Execute(
      const std::vector<seeml::update::UpdateInstruction>& program);
  [[nodiscard]] std::expected<void, std::string> ExecuteRange(
      const std::vector<seeml::update::UpdateInstruction>& program,
      size_t begin, size_t end);
  /// Execute(train_program_), phase-timed under SEEML_STEP_TIMING.
  [[nodiscard]] std::expected<void, std::string> ExecuteTrainProgram();
  /// Execute(step_program_) (gradient accumulation), timed as optimizer.
  [[nodiscard]] std::expected<void, std::string> ExecuteStepProgram();

  const float* ReadPtr(uint64_t ref) const;
  float* WritePtr(uint64_t ref);

  seeml::update::PlanHeader header_{};
  // Step-latency instrumentation (SEEML_STEP_TIMING): phase boundaries of
  // the training stream, scanned once at load; zeros/unused otherwise.
  // Unconditional members keep the class layout build-independent.
  StepTimings timings_{};
  [[maybe_unused]] size_t train_bwd_begin_ = 0;  // first instr after loss fwd
  [[maybe_unused]] size_t train_opt_begin_ = 0;  // first clip/optimizer instr
  std::vector<uint8_t> owned_plan_;       // file-loaded plans
  const uint8_t* plan_ = nullptr;         // borrowed or owned_plan_.data()
  size_t plan_size_ = 0;

  std::vector<seeml::update::UpdateInstruction> train_program_;
  std::vector<seeml::update::UpdateInstruction> merge_program_;
  std::vector<seeml::update::UpdateInstruction> eval_program_;
  // Gradient accumulation (v9): the optimizer program, run once per
  // grad_accum_ executions of train_program_; empty when grad_accum_ == 1.
  std::vector<seeml::update::UpdateInstruction> step_program_;
  uint64_t grad_accum_ = 1;
  std::vector<seeml::update::EmitEntry> emit_table_;

  uint8_t* arena_ = nullptr;              // single aligned allocation
  size_t arena_bytes_ = 0;                // its rounded-up length
  const uint8_t* rodata_ = nullptr;       // points into the plan blob
  // The executor behind the seam (backend.h); constructed as the CPU
  // reference, replaced by SelectBackend. Declared after the address
  // spaces it binds; the destructor still releases it explicitly first.
  std::unique_ptr<ExecutorBackend> backend_;
  BackendKind backend_kind_ = BackendKind::kCpu;
  std::string backend_note_;
  // The bound source model (v17): a read-only mmap of the file, and the
  // byte extent the eval program's source refs need from it (0 = none).
  struct SourceMapping {
    const uint8_t* data = nullptr;
    uint64_t bytes = 0;
    ~SourceMapping();
    void Release();
  } source_;
  uint64_t source_required_ = 0;
  uint64_t step_ = 0;                     // 1-indexed AdamW timestep
  uint64_t horizon_ = 0;                  // run LR horizon; 0 = plan default

  // The run's identity and the best state (E9, #92), carried into every
  // checkpoint (SEKP v5) and restored from one. `binding_` is set from the
  // datasets at Train; a resume whose datasets disagree with the
  // checkpoint's is refused. best_state_ is empty until a tracked run
  // snapshots the persistent segment.
  struct RunBinding {
    bool bound = false;
    uint64_t shuffle_origin = 0;
    uint64_t train_samples = 0;
    uint64_t val_samples = 0;
  } binding_;
  bool has_val_initial_ = false;          // the source model's score
  bool val_initial_has_accuracy_ = false;
  float val_initial_loss_ = 0.0f;
  float val_initial_accuracy_ = 0.0f;
  bool has_best_ = false;
  uint64_t best_step_ = 0;
  float best_loss_ = 0.0f;
  float best_accuracy_ = 0.0f;
  uint32_t stale_evals_ = 0;
  std::vector<uint8_t> best_state_;
  void SnapshotBest(uint64_t step, float loss, float accuracy);
  uint64_t num_classes_ = 0;              // softmax width, 0 = no class loss
  uint64_t vocab_bound_ = 0;              // narrowest embedding table, 0 = none
  bool merged_ = false;

  // Where the eval program materializes its softmax probabilities — the
  // basis of the accuracy metric. kNullRef when the eval program carries no
  // class-label softmax (MSE / pure-distillation plans).
  // How the eval loss excludes a final batch's wrapped duplicates (G13):
  // exactly for a lone cross-entropy (a = probs, b = labels, width =
  // classes) or MSE (a = pred, b = target, width = elements per row),
  // by real-row weight for a composite loss.
  struct EvalTail {
    enum Kind { kWeighted, kXent, kMse } kind = kWeighted;
    uint64_t a = seeml::update::kNullRef;
    uint64_t b = seeml::update::kNullRef;
    uint64_t width = 0;
  } eval_tail_;
  double RealRowLoss(uint64_t real, const uint8_t* label_slot) const;
  uint64_t eval_probs_ref_ = seeml::update::kNullRef;
  uint64_t eval_softmax_rows_ = 0;
  uint64_t eval_softmax_cols_ = 0;
};

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_DISPATCHER_UPDATE_ENGINE_H_
