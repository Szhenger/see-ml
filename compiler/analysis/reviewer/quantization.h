#ifndef SEEML_COMPILER_ANALYSIS_REVIEWER_QUANTIZATION_H_
#define SEEML_COMPILER_ANALYSIS_REVIEWER_QUANTIZATION_H_

#include <cstddef>
#include <unordered_map>
#include <unordered_set>

#include "compiler/frontend/parser/graph_build.h"
#include "compiler/frontend/representation/sir.h"

// =============================================================================
// Reviewer — preprocessing of the ingested local model that configures the
// backend: the review runs before arena binding and its verdicts steer how
// the backend lays out and lowers the program. Today's review is
// quantization selection: which frozen weights can safely pack as int8
// rodata (4x smaller), and at what per-tensor scale. The backend consumes
// the verdict twice — rodata packing quantizes exactly the selected
// weights, and instruction lowering emits the int8-aware kernels for them.
// =============================================================================

namespace seeml::update {

/// Elements per chunk when sweeping a frozen weight blob. Weight tensors
/// reach tens of megabytes; the sweeps are elementwise over disjoint ranges,
/// so they parallelize with bit-identical results. Shared with the backend's
/// int8 pack so both sides of the decision sweep with the same geometry.
inline constexpr size_t kWeightSweepGrain = 65536;

/// Reviews every frozen sc_mem.weight and selects those safe to quantize:
/// weights consumed exclusively as the weight operand of matmul kernels
/// (never as an activation). Returns value -> per-tensor symmetric int8
/// scale (max_abs / 127). The max-abs scan parallelizes per tensor with a
/// deterministic chunk reduction.
std::unordered_map<const seeml::sir::Value*, float> SelectQuantizedWeights(
    seeml::sir::Block& block, const GraphBuild& build);

/// The bf16 review (roadmap 2c): the same eligibility — weights consumed
/// only as the weight operand of matmul kernels — with no scale to derive
/// (bf16 is a rounding of the f32 bits, exact on widening). Every eligible
/// weight is selected: there is no range condition, bf16 keeps f32's
/// exponent.
std::unordered_set<const seeml::sir::Value*> SelectBf16Weights(
    seeml::sir::Block& block, const GraphBuild& build);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_REVIEWER_QUANTIZATION_H_
