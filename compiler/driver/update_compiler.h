#ifndef SEEML_COMPILER_DRIVER_UPDATE_COMPILER_H_
#define SEEML_COMPILER_DRIVER_UPDATE_COMPILER_H_

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "compiler/analysis/updater/pass_manager.h"  // PassTiming
#include "source/language/model_format.h"
#include "source/plan/update_types.h"

// =============================================================================
// UpdateCompiler — the AOT driver that turns
//     (source model, [teacher model], loss, LoRA spec, optimizer spec)
// into a self-contained .seeu Update Plan:
//
//   SMF ingest ──▶ forward SIR (+ frozen teacher subgraph)
//              ──▶ loss grafting
//              ──▶ LoRA grafting                  (update_passes)
//              ──▶ trainable-set reverse autodiff (update_passes)
//              ──▶ optimizer synthesis            (update_passes)
//              ──▶ merge program                  (update_passes)
//              ──▶ segmented arena binding: RODATA | PERSISTENT | IO | TRANSIENT
//              ──▶ instruction lowering + plan assembly
//
// Every byte the runtime will touch is bound here, at compile time.
//
// The driver owns the compilation *process*, nothing else: it sequences the
// subsystems and verifies at every boundary that each was used correctly
// (contract.h):
//   frontend/     ingest feasibility (ingressor) and SMF graph -> forward
//                 SIR (parser) — gated by VerifyFrontendContract
//   analysis/     the pass-managed structural and calculus phases (updater/
//                 algebra/calculus) plus quantization review (reviewer) —
//                 gated by VerifyAnalysisContract
//   backend/      arena binding, instruction lowering, plan assembly, and
//                 native packaging (trainer; the kernel policy the tuner's
//                 table resolved rides in the config) — gated by
//                 VerifyGeneratedPlan
//   diagnostics/  every error crossing the driver's boundary must be
//                 attributable to a registered unit (WellFormedDiagnostic)
// =============================================================================

namespace seeml::update {

/// Debug/verification hooks exposed alongside the plan (used by the test
/// suite for finite-difference gradient checks and merge validation).
struct ParamDebugInfo {
  std::string id;
  uint64_t param_ref = kNullRef;  // arena ref of the parameter
  uint64_t grad_ref = kNullRef;   // arena ref of its gradient
  uint64_t acc_ref = kNullRef;    // its accumulator (grad_accum_steps > 1)
  uint64_t count = 0;             // element count
};

struct AdapterDebugInfo {
  std::string weight_name;
  uint64_t weight_rodata_ref = kNullRef;
  uint64_t a_ref = kNullRef;
  uint64_t b_ref = kNullRef;
  uint64_t delta_ref = kNullRef;  // Δ = (α/r)·A@B after RunMerge()
  int64_t k = 0, m = 0, r = 0;
  float scale = 1.0f;
  // Per-tensor int8 scale when the frozen weight was quantized into rodata;
  // 0 when the weight is stored as f32 or bf16.
  float quant_scale = 0.0f;
  // The frozen weight is stored as bfloat16 rodata (roadmap 2c).
  bool bf16 = false;
};

struct CompiledUpdate {
  std::vector<uint8_t> plan;  // the .seeu blob
  std::string sir_dump;       // human-readable training program; empty
                              // unless UpdateConfig::dump_sir
  // Wall time of every pass and driver phase, in order (E5, #84).
  std::vector<PassTiming> pass_timings;
  std::vector<ParamDebugInfo> params;
  std::vector<AdapterDebugInfo> adapters;
  uint64_t arena_size = 0;
  uint64_t persistent_size = 0;
  uint64_t train_instruction_count = 0;
  uint64_t merge_instruction_count = 0;
  uint64_t eval_instruction_count = 0;
  uint64_t step_instruction_count = 0;  // 0 unless grad_accum_steps > 1
  uint32_t grad_accum_steps = 1;
  uint64_t rodata_size = 0;
  uint32_t gemm_tile_k = 0;  // the header's CPU GEMM tiles (0 = default)
  uint32_t gemm_tile_n = 0;
  // The attention decision (E11): whether the plan runs the tiled family,
  // and the probability-cache bytes the cached family would have held.
  bool attention_tiled = false;
  uint64_t probs_cache_bytes = 0;
  // E12: the eval program reads the student's frozen weights from the
  // source model file (the shipped f32), not the plan's narrow copies.
  bool scores_shipped = false;
  // Instructions of the train and step programs carrying kFlagRelaxed
  // (plan v18): 0 under Precision::kF32.
  uint64_t relaxed_gemms = 0;
};

class UpdateCompiler {
 public:
  explicit UpdateCompiler(UpdateConfig config) : config_(std::move(config)) {}

  /// `teacher` may be null; it is required for kKLDistill / kXEntPlusKL.
  /// The teacher must share the source model's input dimensionality.
  /// Any error returned is a well-formed diagnostic ("<unit>: <message>"
  /// for a unit registered in diagnostics/) — the driver enforces this at
  /// its boundary.
  [[nodiscard]] std::expected<CompiledUpdate, std::string> Compile(
      const SmfModel& source, const SmfModel* teacher = nullptr);

  /// The consuming form, for callers done with the models (the CLI): the
  /// same plan, byte for byte, but each frozen weight's payload — in
  /// `source` and, when given, `*teacher` — is released as soon as plan
  /// assembly has written it, so the weights are resident about once
  /// rather than twice while the blob is built (E2, #81). On return the
  /// models keep their metadata and content hashes and no payload that was
  /// packed; on error they are unspecified-but-valid.
  [[nodiscard]] std::expected<CompiledUpdate, std::string> Compile(
      SmfModel&& source, SmfModel* teacher = nullptr);

 private:
  /// The pipeline itself; Compile wraps it with the diagnostics contract.
  [[nodiscard]] std::expected<CompiledUpdate, std::string> CompileImpl(
      const SmfModel& source, const SmfModel* teacher, bool consume);

  UpdateConfig config_;
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_BACKEND_UPDATE_COMPILER_H_
