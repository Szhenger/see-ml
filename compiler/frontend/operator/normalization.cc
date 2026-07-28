// Normalization family: sc_high.batch_norm.

#include <cassert>

#include "compiler/frontend/operator/op_builder.h"

namespace seeml::sir {

std::unique_ptr<Operation> OpBuilder::batchNorm(
    Value* input, Value* scale, Value* bias,
    Value* running_mean, Value* running_var, float epsilon) {
    assert(input        && "batchNorm: null input");
    assert(scale        && "batchNorm: null scale");
    assert(bias         && "batchNorm: null bias");
    assert(running_mean && "batchNorm: null running_mean");
    assert(running_var  && "batchNorm: null running_var");

    auto op = std::make_unique<Operation>("sc_high.batch_norm");
    op->addOperand(input);
    op->addOperand(scale);
    op->addOperand(bias);
    op->addOperand(running_mean);
    op->addOperand(running_var);
    op->setAttribute("epsilon", epsilon);

    op->addResult("", input->dtype(), input->shape());
    return op;
}

}  // namespace seeml::sir
