#include "compiler/frontend/op_builder.h"

#include <cassert>

namespace seeml::sir {

namespace {

/// One spatial output extent: kDynamic propagates, and a window that does
/// not fit yields kDynamic rather than a negative extent.
int64_t spatialDim(int64_t in, int64_t kernel, int64_t pad_begin,
                   int64_t pad_end, int64_t stride, int64_t dilation) {
    if (in == Shape::kDynamic || kernel == Shape::kDynamic || stride <= 0)
        return Shape::kDynamic;
    const int64_t effective = (kernel - 1) * dilation + 1;
    const int64_t span = in + pad_begin + pad_end - effective;
    return span < 0 ? Shape::kDynamic : span / stride + 1;
}

int64_t mulOrDynamic(int64_t a, int64_t b) {
    if (a == Shape::kDynamic || b == Shape::kDynamic) return Shape::kDynamic;
    return a * b;
}

}  // namespace

std::unique_ptr<Operation> OpBuilder::conv2d(
    Value* input, Value* filter, Value* bias,
    std::vector<int64_t> strides, std::vector<int64_t> pads,
    std::vector<int64_t> dilations, int64_t group) {
    assert(input  && "conv2d: null input");
    assert(filter && "conv2d: null filter");
    assert(input->shape().rank() == 4 && "conv2d: input must be NCHW");
    assert(filter->shape().rank() == 4 && "conv2d: filter must be OIHW");
    assert(strides.size() == 2 && pads.size() == 4 && dilations.size() == 2 &&
           "conv2d: geometry is {sh,sw} / {t,l,b,r} / {dh,dw}");

    const auto& in = input->shape().dims;
    const auto& f = filter->shape().dims;
    Shape out{in[0], f[0],
              spatialDim(in[2], f[2], pads[0], pads[2], strides[0],
                         dilations[0]),
              spatialDim(in[3], f[3], pads[1], pads[3], strides[1],
                         dilations[1])};

    auto op = std::make_unique<Operation>("sc_high.conv2d");
    op->addOperand(input);
    op->addOperand(filter);
    if (bias) op->addOperand(bias);

    op->setAttribute("strides",   std::move(strides));
    op->setAttribute("pads",      std::move(pads));
    op->setAttribute("dilations", std::move(dilations));
    op->setAttribute("group",     group);

    op->addResult("", input->dtype(), std::move(out));
    return op;
}

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

std::unique_ptr<Operation> OpBuilder::relu(Value* input) {
    assert(input && "relu: null input");
    auto op = std::make_unique<Operation>("sc_high.relu");
    op->addOperand(input);
    op->addResult("", input->dtype(), input->shape());
    return op;
}

std::unique_ptr<Operation> OpBuilder::im2col(
    Value* input, std::vector<int64_t> kernel_shape,
    std::vector<int64_t> strides, std::vector<int64_t> pads) {
    assert(input && "im2col: null input");
    assert(input->shape().rank() == 4 && "im2col: input must be NCHW");
    assert(kernel_shape.size() == 2 && strides.size() == 2 &&
           pads.size() == 4 &&
           "im2col: geometry is {kh,kw} / {sh,sw} / {t,l,b,r}");

    const auto& in = input->shape().dims;
    const int64_t oh = spatialDim(in[2], kernel_shape[0], pads[0], pads[2],
                                  strides[0], 1);
    const int64_t ow = spatialDim(in[3], kernel_shape[1], pads[1], pads[3],
                                  strides[1], 1);
    Shape out{mulOrDynamic(in[0], mulOrDynamic(oh, ow)),
              mulOrDynamic(in[1],
                           mulOrDynamic(kernel_shape[0], kernel_shape[1]))};

    auto op = std::make_unique<Operation>("sc_low.im2col");
    op->addOperand(input);
    op->setAttribute("kernel_shape", std::move(kernel_shape));
    op->setAttribute("strides",      std::move(strides));
    op->setAttribute("pads",         std::move(pads));
    op->addResult("", input->dtype(), std::move(out));
    return op;
}

}  // namespace seeml::sir
