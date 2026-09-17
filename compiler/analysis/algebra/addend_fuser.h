#ifndef SEEML_COMPILER_ANALYSIS_ALGEBRA_ADDEND_FUSER_H_
#define SEEML_COMPILER_ANALYSIS_ALGEBRA_ADDEND_FUSER_H_

#include <cstddef>
#include <expected>
#include <string>
#include <unordered_set>
#include <vector>

#include "compiler/frontend/representation/sir.h"

// =============================================================================
// GemmAddendFuser — E10 (#93): folds a GEMM whose only reader is an
// elementwise add into that add,
//
//   u  = ts @ B;      c' = c + u          (every LoRA site, forward)
//   dl = dt @ A^T;    dx = db + dl        (every LoRA site, backward: the
//                                          fan-out sum autodiff injects)
//
// so the sum is written as the dot products come out of the GEMM core
// (kFlagGemmAddend, plan v14) instead of by a second instruction that reads
// two activation-sized tensors and writes a third. What disappears per fold
// is one full pass over the tensor and the GEMM's transient result.
//
// The ADD is the fused op, as in the other fusers: it keeps its mnemonic,
// its result value and its place in the program, and gains the attribute
// "gemm_addend" ("nn" | "nt" | "tn") with operands rewired to
// [addend, A, B]. The GEMM is left orphaned for the DCE sweep and reported.
//
// Legality, read off the use-lists (so the pass runs after autodiff):
//   - the GEMM result has exactly one reader, this add, and is not read
//     outside the program;
//   - the GEMM is a plain f32 one: no fused epilogue, no int8 / bf16 weight;
//   - the fold moves the GEMM's reads to the add's position, so nothing that
//     mutates storage in place (an optimizer step, an accumulate, a clip)
//     may sit between the two.
// When both operands qualify the GEMM with the smaller inner dimension is
// folded — at a LoRA site that is the rank-r one, and the frozen-weight
// GEMM keeps its own instruction (and its int8 / bf16 forms).
//
// Bitwise-neutral by construction: each element becomes `d + s` over the
// complete dot product s, and float addition commutes, so either operand
// order of the add reads the same sum the two-instruction form stored.
// =============================================================================

namespace seeml::update {

struct AddendFusion {
  size_t fused = 0;
  // GEMMs folded into an add: orphaned, to be dropped from any op-list
  // snapshot the caller holds and swept by DeadCodeElimination.
  std::vector<seeml::sir::Operation*> fused_away;
};

class GemmAddendFuser {
 public:
  /// `narrow_weights`: frozen weights stored as int8 or bf16 (their GEMMs
  /// have no addend form). `protected_values` are read outside the program.
  [[nodiscard]] std::expected<AddendFusion, std::string> Run(
      seeml::sir::Block& block,
      const std::unordered_set<const seeml::sir::Value*>& narrow_weights,
      const std::unordered_set<const seeml::sir::Value*>& protected_values);
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_ALGEBRA_ADDEND_FUSER_H_
