#ifndef SEEML_COMPILER_ANALYSIS_UPDATER_CONV_LOWERING_H_
#define SEEML_COMPILER_ANALYSIS_UPDATER_CONV_LOWERING_H_

#include <expected>
#include <string>

#include "compiler/frontend/representation/sir.h"

// =============================================================================
// ConvLowering — structural lowering of convolutions to the GEMM form the
// rest of the pipeline (LoRA grafting, autodiff, the matmul kernels) is
// built around. Each sc_high.conv2d over NCHW input becomes:
//
//   cols = sc_low.im2col(x)               [N*OH*OW, Cin*KH*KW]
//   wmat = sc_low.filter_matrix(filter)   [Cin*KH*KW, Cout]  (OIHW repack)
//   prod = sc_high.matmul(cols, wmat)     [N*OH*OW, Cout]
//   (+ sc_high.add_bias when the conv carries a bias — Cout is the last dim)
//   y    = sc_low.col2im(prod)            [N, Cout, OH, OW]
//
// A block with no convolutions is untouched. group != 1 and dilation != 1
// are unsupported and error (the im2col patch geometry does not model them).
// =============================================================================

namespace seeml::update {

class ConvLowering {
 public:
  [[nodiscard]] std::expected<void, std::string> Run(seeml::sir::Block& block);
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_UPDATER_CONV_LOWERING_H_
