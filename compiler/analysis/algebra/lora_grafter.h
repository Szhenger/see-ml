#ifndef SEEML_COMPILER_ANALYSIS_ALGEBRA_LORA_GRAFTER_H_
#define SEEML_COMPILER_ANALYSIS_ALGEBRA_LORA_GRAFTER_H_

#include <expected>
#include <string>
#include <vector>

#include "compiler/frontend/representation/sir.h"
#include "source/update_types.h"

// =============================================================================
// LoraGrafter — the low-rank adapter algebra: rewrites C = X@W into
// C' = X@W + (α/r)·(X@A)@B, injecting randomized trainable adapters A, B.
// Part of analysis/algebra/ — the matmul-structure rewrites whose merge
// counterpart (merge_builder.h) materializes Δ = (α/r)·A@B.
// =============================================================================

namespace seeml::update {

/// One grafted LoRA adapter and the frozen weight it updates.
struct GraftedAdapter {
  seeml::sir::Value* frozen_weight = nullptr;  // W  [K, M] (rodata)
  seeml::sir::Value* A = nullptr;              // A  [K, r] (randn init)
  seeml::sir::Value* B = nullptr;              // B  [r, M] (zeros init)
  float scale = 1.0f;                           // α / r
  // Unique per-site naming stem ("w", or "w@1" for the second graft onto a
  // tied weight); every value id derived from this adapter uses it, keeping
  // ids unique across the training AND merge programs (Block::verify
  // enforces uniqueness).
  std::string id_base;
};

class LoraGrafter {
 public:
  explicit LoraGrafter(LoRASpec spec) : spec_(std::move(spec)) {}

  /// Grafts adapters onto every eligible MatMul: the second operand must be a
  /// frozen sc_mem.weight whose name passes the target filters, and the op
  /// must not belong to the teacher subgraph (ids prefixed "t::").
  /// Because B is zero-initialized, step 0 of the update is exactly the
  /// source model — the update starts from the identity.
  [[nodiscard]] std::expected<std::vector<GraftedAdapter>, std::string> Run(
      seeml::sir::Block& block);

 private:
  LoRASpec spec_;
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_ALGEBRA_LORA_GRAFTER_H_
