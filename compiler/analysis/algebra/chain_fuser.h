#ifndef SEEML_COMPILER_ANALYSIS_ALGEBRA_CHAIN_FUSER_H_
#define SEEML_COMPILER_ANALYSIS_ALGEBRA_CHAIN_FUSER_H_

#include <cstddef>
#include <expected>
#include <string>
#include <unordered_set>
#include <vector>

#include "compiler/frontend/representation/sir.h"

// =============================================================================
// ElementwiseChainFuser — roadmap Phase 1b (E4, #83): folds a chain of
// same-shape elementwise ops whose intermediates have exactly one reader
//
//   s = alpha * u;  c' = c + s            (every LoRA site, forward)
//   h = silu(g);    y = h + r;  z = beta * y
//
// into ONE op, lowered to kFusedMap: a micro-program of up to four stages
// over a running value, with at most two other tensor operands and two
// scale immediates. Each folded op was a full arena round trip — an
// activation-sized write, then the same bytes read back — for one
// arithmetic operation per element.
//
// The chain's TAIL op is the fused op. It keeps its mnemonic, its result
// value and its place in the program — so the loss slot, a pinned gradient
// or the eval snapshot's membership survive untouched — and gains the
// chain as attributes ("fused_stages", "fused_imm0/1") plus rewired
// operands: [x, other tensors...]. The ops before it are left orphaned for
// the DCE sweep and reported, fuse-then-rebind, exactly as
// GemmEpilogueFuser does.
//
// Legality is read off the use-lists, which is why the pass runs AFTER
// autodiff: an intermediate the backward program reads (a SiLU output
// feeding both the gate product and its own adjoint) has a second user and
// simply ends the chain. Bitwise-neutral by construction: the runtime runs
// every stage as its own loop, with the expression of the standalone
// kernel, in the chain's order, keeping each binary stage's operand order.
// =============================================================================

namespace seeml::update {

struct ChainFusion {
  size_t fused_chains = 0;
  // Ops folded into a tail: orphaned, to be dropped from any op-list
  // snapshot the caller holds and swept by DeadCodeElimination.
  std::vector<seeml::sir::Operation*> fused_away;
};

class ElementwiseChainFuser {
 public:
  /// `protected_values` are read outside the program (the loss, the
  /// parameter gradients): they may END a chain but never be folded into
  /// one.
  [[nodiscard]] std::expected<ChainFusion, std::string> Run(
      seeml::sir::Block& block,
      const std::unordered_set<const seeml::sir::Value*>& protected_values);
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_ALGEBRA_CHAIN_FUSER_H_
