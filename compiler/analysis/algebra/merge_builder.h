#ifndef SEEML_COMPILER_ANALYSIS_ALGEBRA_MERGE_BUILDER_H_
#define SEEML_COMPILER_ANALYSIS_ALGEBRA_MERGE_BUILDER_H_

#include <expected>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "compiler/analysis/algebra/lora_grafter.h"
#include "compiler/frontend/representation/sir.h"

// =============================================================================
// MergeBuilder — builds the separate merge program that materializes each
// adapter's weight delta, Δ = (α/r)·A@B, as one fused kernel per adapter
// (sc_low.gemm_acc: GEMM + scaled accumulate in a single op). Commit adds Δ
// to the pristine f32 weights inside the model file itself, so the merge
// never needs the frozen base (which may live in rodata as quantized int8).
// =============================================================================

namespace seeml::update {

/// The merge program lives in its own block. Its A/B operands alias storage
/// owned by the training block (the persistent segment), so the builder
/// returns an alias map that lowering uses to resolve those mirrors to the
/// training program's already-bound offsets.
struct MergeProgram {
  std::unique_ptr<seeml::sir::Block> block;
  // mirror value in the merge block -> original value in the training block
  std::unordered_map<seeml::sir::Value*, seeml::sir::Value*> aliases;
  // delta output value -> the adapter it belongs to (for the emit table)
  std::vector<std::pair<seeml::sir::Value*, const GraftedAdapter*>> outputs;
};

class MergeBuilder {
 public:
  [[nodiscard]] std::expected<MergeProgram, std::string> Run(
      const std::vector<GraftedAdapter>& adapters);
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_ALGEBRA_MERGE_BUILDER_H_
