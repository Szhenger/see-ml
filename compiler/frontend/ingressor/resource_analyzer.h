#ifndef SEEML_COMPILER_FRONTEND_INGRESSOR_RESOURCE_ANALYZER_H_
#define SEEML_COMPILER_FRONTEND_INGRESSOR_RESOURCE_ANALYZER_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>

#include "source/language/model_format.h"

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
  // The merged LoRA deltas (E2, #81 — audit finding #8): one full-size f32
  // image of every adapted weight, pinned in the arena above the training
  // high-water mark so RunMerge can hand commit W + delta. With a
  // quantized base they are 4x the weights they patch — the largest term
  // the early gate used to leave to the final one.
  uint64_t delta_bytes = 0;
  // The attention probability caches (E11, #94): P [B*H*S, S] f32 per
  // attention op, alive from forward to backward, summed over the ops.
  // Counted apart from the activations because the compiler may choose
  // not to keep them at all — the tiled family holds kAttnStatsWidth
  // floats per query row instead — and the choice is made on this number.
  uint64_t probs_cache_bytes = 0;
  uint64_t attention_stats_bytes = 0;  // what the tiled family holds instead

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

/// f32 bytes of the delta segment LoRA will need for `model`: every constant
/// tensor the grafter would adapt — consumed only as the weight (second)
/// operand of two-input MatMuls, and named by one of `target_filters`
/// (empty = all). The same eligibility rule as LoraGrafter, read off the
/// SMF op list; a tensor it cannot classify contributes zero, keeping the
/// estimate a lower bound.
uint64_t EstimateLoraDeltaBytes(const SmfModel& model,
                                std::span<const std::string> target_filters);

/// The frozen-forward variant, for teacher models under distillation: no
/// backward pass consumes the activations, so they die at their single
/// reader and the arena binder reuses their slots — the honest lower bound
/// charges the single widest live term instead of the sum (which over-counts
/// a deep teacher by more than an order of magnitude and could refuse a
/// provably feasible compile).
TrainingFootprint EstimateFrozenForwardFootprint(const SmfModel& model,
                                                 int64_t batch);

/// Physical memory of this host in bytes; 0 when undetectable.
uint64_t DetectLocalMemoryBytes();

/// Fail-fast gate: errors when `footprint` certainly exceeds the budget.
/// `budget_bytes` of 0 means "detect the host's physical memory"; if
/// detection fails the check passes (infeasibility cannot be proven).
[[nodiscard]] std::expected<void, std::string> CheckTrainableLocally(
    const TrainingFootprint& footprint, uint64_t budget_bytes);

/// Final gate, run after code generation has bound every byte: errors when
/// what the runtime will actually keep resident — the plan blob (rodata and
/// the persistent image live inside it) plus the arena it allocates —
/// exceeds the budget. The early estimate is a lower bound that knows
/// nothing of gradients, optimizer state, or transients; this check is
/// exact for the compiled artifact, so admissions become trustworthy too
/// (the corpus and the source model at commit remain extra).
[[nodiscard]] std::expected<void, std::string> CheckPlanFitsLocally(
    uint64_t arena_bytes, uint64_t plan_bytes, uint64_t budget_bytes);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_FRONTEND_INGRESSOR_RESOURCE_ANALYZER_H_
