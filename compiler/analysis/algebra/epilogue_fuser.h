#ifndef SEEML_COMPILER_ANALYSIS_ALGEBRA_EPILOGUE_FUSER_H_
#define SEEML_COMPILER_ANALYSIS_ALGEBRA_EPILOGUE_FUSER_H_

#include <expected>
#include <string>
#include <unordered_set>
#include <vector>

#include "compiler/frontend/representation/sir.h"

// =============================================================================
// GemmEpilogueFuser — the arena-bandwidth fusion algebra: folds
//
//   C = X@W;  Y = C + b;  Z = act(Y)      (act ∈ {relu, gelu, silu})
//
// into one matmul carrying the whole chain as a fused write-back epilogue
// (bias operand + "epilogue_act" attribute, lowered to instruction flags —
// plan v5). An MLP layer's three arena round-trips become one, and the
// orphaned intermediates stop costing transient arena space.
//
// Legality is read off the use-lists, which is why the pass runs AFTER
// autodiff: any intermediate the backward program consumes (a pre-activation
// read by relu_grad, say) carries that consumer as an extra user and the
// chain simply fails to match. What does match is exactly the no-backward
// compute — the frozen teacher subgraph, and the bias step of unadapted
// student layers (whose raw GEMM output no backward op reads). Fusion is
// therefore bitwise-neutral by construction; the runtime applies the
// epilogue with the same per-element expressions as the standalone kernels.
//
// Fuse-then-rebind: the matmul's result VALUE takes over the chain output's
// identity — consumers of Z (or Y) are rewired onto C, and the dead AddBias
// / activation ops are reported to the caller, who must (a) drop them from
// any op-list snapshot it holds (the driver's primal/eval snapshot) and
// (b) let DeadCodeElimination sweep them from the block.
// =============================================================================

namespace seeml::update {

struct EpilogueFusion {
  // Chains folded into a GEMM epilogue this run.
  size_t fused_chains = 0;
  // The orphaned AddBias / activation ops, in program order. Still owned by
  // the block (DCE removes them); any snapshot holding these pointers must
  // drop them before the sweep frees them.
  std::vector<seeml::sir::Operation*> fused_away;
};

class GemmEpilogueFuser {
 public:
  /// Fuses every legal chain in `block`.
  ///
  /// `quantized_weights`: frozen weights selected for int8 rodata. Their
  /// GEMMs lower to the q8 opcode, whose in[3] slot carries the dequant
  /// scale — a fused bias ref has nowhere to ride, so those chains keep the
  /// standalone AddBias (an activation directly on the GEMM still fuses).
  ///
  /// `protected_values`: values read from outside the instruction stream
  /// (the loss slot, parameter gradients). A protected value's identity is
  /// never rewired away, and its producing op is never orphaned.
  [[nodiscard]] std::expected<EpilogueFusion, std::string> Run(
      seeml::sir::Block& block,
      const std::unordered_set<const seeml::sir::Value*>& quantized_weights,
      const std::unordered_set<const seeml::sir::Value*>& protected_values);
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_ALGEBRA_EPILOGUE_FUSER_H_
