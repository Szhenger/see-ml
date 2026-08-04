#ifndef SEEML_COMPILER_ANALYSIS_UPDATER_CONV_LOWERING_H_
#define SEEML_COMPILER_ANALYSIS_UPDATER_CONV_LOWERING_H_

#include <expected>
#include <string>

#include "compiler/frontend/representation/sir.h"

// =============================================================================
// ConvLowering — structural rewrite of convolutions into im2col-GEMM form.
// Each sc_high.conv2d over NCHW input becomes:
//
//   cols = sc_low.im2col(x)               [N*OH*OW, Cin*KH*KW]
//   wmat = sc_low.filter_matrix(filter)   [Cin*KH*KW, Cout]  (OIHW repack)
//   prod = sc_high.matmul(cols, wmat)     [N*OH*OW, Cout]
//   (+ sc_high.add_bias when the conv carries a bias — Cout is the last dim)
//   y    = sc_low.col2im(prod)            [N, Cout, OH, OW]
//
// A block with no convolutions is untouched. group != 1 and dilation != 1
// are unsupported and error (the im2col patch geometry does not model them).
//
// Scope: a structural SIR rewrite only — the staked-out lowering for a
// conv-capable frontend, exercised today at the SIR level by the updater
// suite. The rest of the update pipeline cannot yet train through it: the
// sc_low.im2col / filter_matrix / col2im primitives have no autodiff VJP
// rules and no UpdateInstruction lowering, and LoRA grafting targets only
// matmuls whose weight is a raw sc_mem.weight (a repacked filter_matrix is
// skipped). Nothing is miscompiled by this gap — SMF defines no conv op, so
// no driver compilation reaches the rewrite, and a conv graph introduced by
// other means is rejected downstream (no VJP rule / cannot lower), never
// silently dropped.
// =============================================================================

namespace seeml::update {

class ConvLowering {
 public:
  [[nodiscard]] std::expected<void, std::string> Run(seeml::sir::Block& block);
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_UPDATER_CONV_LOWERING_H_
