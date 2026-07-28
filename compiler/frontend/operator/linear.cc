// Linear-algebra family: sc_high.gemm.

#include <cassert>

#include "compiler/frontend/operator/op_builder.h"

namespace seeml::sir {

std::unique_ptr<Operation> OpBuilder::gemm(
    Value* A, Value* B, Value* bias, bool trans_a, bool trans_b) {
    assert(A && "gemm: null A");
    assert(B && "gemm: null B");
    assert(A->shape().rank() == 2 && B->shape().rank() == 2 &&
           "gemm: operands must be rank-2");

    const auto& a = A->shape().dims;
    const auto& b = B->shape().dims;
    Shape out{a[trans_a ? 1 : 0], b[trans_b ? 0 : 1]};

    auto op = std::make_unique<Operation>("sc_high.gemm");
    op->addOperand(A);
    op->addOperand(B);
    if (bias) op->addOperand(bias);

    op->setAttribute("trans_a", static_cast<int64_t>(trans_a));
    op->setAttribute("trans_b", static_cast<int64_t>(trans_b));
    op->addResult("", A->dtype(), std::move(out));
    return op;
}

}  // namespace seeml::sir
