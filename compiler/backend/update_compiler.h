#ifndef SEEML_COMPILER_BACKEND_UPDATE_COMPILER_H_
#define SEEML_COMPILER_BACKEND_UPDATE_COMPILER_H_

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "source/model_format.h"
#include "source/update_types.h"

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
// The backend is partitioned by role; this driver only abstracts the process:
//   trainer/       code generation for the training program — arena binding,
//                  instruction lowering, the native C++ host package
//                  (native_emitter), and GPU kernel source (kernel_emitter)
//   architecture/  host ISA/microarchitecture analysis (SIMD width, cache
//                  and core geometry) and the GEMM tiling hints it derives
//                  for the trainer
//   tuner/         reinforcement tuning: a UCB1 bandit and the GEMM-tiling
//                  autotuner that spends a measurement budget to refine the
//                  architecture hint on the real machine
// =============================================================================

namespace seeml::update {

/// Debug/verification hooks exposed alongside the plan (used by the test
/// suite for finite-difference gradient checks and merge validation).
struct ParamDebugInfo {
  std::string id;
  uint64_t param_ref = kNullRef;  // arena ref of the parameter
  uint64_t grad_ref = kNullRef;   // arena ref of its gradient
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
  // 0 when the weight is stored as f32.
  float quant_scale = 0.0f;
};

struct CompiledUpdate {
  std::vector<uint8_t> plan;  // the .seeu blob
  std::string sir_dump;       // human-readable training program
  std::vector<ParamDebugInfo> params;
  std::vector<AdapterDebugInfo> adapters;
  uint64_t arena_size = 0;
  uint64_t persistent_size = 0;
  uint64_t train_instruction_count = 0;
  uint64_t merge_instruction_count = 0;
  uint64_t eval_instruction_count = 0;
  uint64_t rodata_size = 0;
};

class UpdateCompiler {
 public:
  explicit UpdateCompiler(UpdateConfig config) : config_(std::move(config)) {}

  /// `teacher` may be null; it is required for kKLDistill / kXEntPlusKL.
  /// The teacher must share the source model's input dimensionality.
  [[nodiscard]] std::expected<CompiledUpdate, std::string> Compile(
      const SmfModel& source, const SmfModel* teacher = nullptr);

 private:
  UpdateConfig config_;
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_BACKEND_UPDATE_COMPILER_H_
