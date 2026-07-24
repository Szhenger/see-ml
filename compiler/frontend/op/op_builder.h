#ifndef SEEML_COMPILER_FRONTEND_OP_BUILDER_H_
#define SEEML_COMPILER_FRONTEND_OP_BUILDER_H_

#include <memory>
#include <vector>

#include "compiler/frontend/sir.h"

// =============================================================================
// OpBuilder — factories for well-formed sc_high.* / sc_low.* operations.
// Each factory wires its operands (registering uses), records the op's
// attributes, and infers the result shape from the operand shapes —
// per-dimension, with kDynamic for anything not statically known. A factory
// never fabricates a scalar for a shape it cannot compute.
//
// Factories check structure (non-null operands, geometry attribute arity —
// asserted), not operand agreement: inner-dimension and width consistency
// belong to sema and Block::verify at the layers above.
// =============================================================================

namespace seeml::sir {

struct OpBuilder {
    /// NCHW convolution. `filter` is [Cout, Cin/group, KH, KW]; `pads` is
    /// {top, left, bottom, right}. Result: [N, Cout, OH, OW].
    static std::unique_ptr<Operation> conv2d(
        Value* input, Value* filter, Value* bias,
        std::vector<int64_t> strides,
        std::vector<int64_t> pads = {0, 0, 0, 0},
        std::vector<int64_t> dilations = {1, 1},
        int64_t group = 1
    );

    static std::unique_ptr<Operation> batchNorm(
        Value* input, Value* scale, Value* bias,
        Value* running_mean, Value* running_var,
        float epsilon = 1e-5f
    );

    /// Result: [rows of op(A), cols of op(B)] under the transpose flags.
    static std::unique_ptr<Operation> gemm(
        Value* A, Value* B, Value* bias = nullptr,
        bool trans_a = false, bool trans_b = false
    );

    static std::unique_ptr<Operation> relu(Value* input);

    /// Patch-matrix extraction over NCHW input.
    /// Result: [N * OH * OW, C * KH * KW].
    static std::unique_ptr<Operation> im2col(
        Value* input,
        std::vector<int64_t> kernel_shape,
        std::vector<int64_t> strides,
        std::vector<int64_t> pads
    );
};

}  // namespace seeml::sir

#endif  // SEEML_COMPILER_FRONTEND_OP_BUILDER_H_
