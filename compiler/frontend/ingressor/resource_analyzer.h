#ifndef SEEML_COMPILER_FRONTEND_INGRESSOR_RESOURCE_ANALYZER_H_
#define SEEML_COMPILER_FRONTEND_INGRESSOR_RESOURCE_ANALYZER_H_

#include <cstdint>
#include <expected>
#include <string>

#include "compiler/frontend/ingressor/model_format.h"

// =============================================================================
// Resource analyzer — fail-fast static feasibility analysis for local
// training. Before any SIR is built, the ingested model is walked to compute
// a conservative LOWER BOUND on the bytes training must keep resident:
//
//   weight_bytes      one copy of every frozen constant tensor (rodata)
//   activation_bytes  every forward activation at the compiled batch size —
//                     all of them are cached for the backward pass
//
// The bound deliberately excludes gradients, optimizer state, and transient
// buffers (they depend on the LoRA/optimizer configuration): everything it
// counts is certainly required, so when the bound alone exceeds the local
// memory budget the compilation is provably infeasible and is rejected
// before any work is done. A model that passes may still be tight — this
// gate only guarantees rejections are correct, never that admissions are.
// =============================================================================

namespace seeml::update {

struct TrainingFootprint {
  uint64_t weight_bytes = 0;      // frozen const tensors, one resident copy
  uint64_t activation_bytes = 0;  // forward activations at the given batch

  /// Saturating sum of the components.
  uint64_t total_bytes() const;

  /// Member-wise saturating accumulation (student + teacher subgraphs).
  TrainingFootprint& operator+=(const TrainingFootprint& o);
};

/// Statically analyzes `model`'s training footprint at `batch`. Activation
/// widths are propagated through the op list with the same shape rules the
/// parser applies; anything unresolvable contributes zero, keeping the
/// estimate a lower bound.
TrainingFootprint EstimateTrainingFootprint(const SmfModel& model,
                                            int64_t batch);

/// Physical memory of this host in bytes; 0 when undetectable.
uint64_t DetectLocalMemoryBytes();

/// Fail-fast gate: errors when `footprint` certainly exceeds the budget.
/// `budget_bytes` of 0 means "detect the host's physical memory"; if
/// detection fails the check passes (infeasibility cannot be proven).
[[nodiscard]] std::expected<void, std::string> CheckTrainableLocally(
    const TrainingFootprint& footprint, uint64_t budget_bytes);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_FRONTEND_INGRESSOR_RESOURCE_ANALYZER_H_
