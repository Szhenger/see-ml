// Activation family: sc_high.relu.

#include <cassert>

#include "compiler/frontend/operator/op_builder.h"

namespace seeml::sir {

std::unique_ptr<Operation> OpBuilder::relu(Value* input) {
    assert(input && "relu: null input");
    auto op = std::make_unique<Operation>("sc_high.relu");
    op->addOperand(input);
    op->addResult("", input->dtype(), input->shape());
    return op;
}

}  // namespace seeml::sir
