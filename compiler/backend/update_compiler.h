#ifndef SEEML_COMPILER_BACKEND_UPDATE_COMPILER_H_
#define SEEML_COMPILER_BACKEND_UPDATE_COMPILER_H_

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "source/smf.h"
#include "compiler/backend/update_types.h"

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
