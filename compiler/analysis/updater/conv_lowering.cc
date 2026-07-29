#include "compiler/analysis/updater/conv_lowering.h"

#include <memory>
#include <vector>

#include "compiler/diagnostics/passing/error.h"
#include "compiler/frontend/operator/op_builder.h"

namespace seeml::update {

namespace sir = seeml::sir;
namespace passing = seeml::diag::passing;

std::expected<void, std::string> ConvLowering::Run(sir::Block& block) {
  // Snapshot the targets first: lowering mutates the op list.
  std::vector<sir::Operation*> targets;
  block.walk([&](sir::Operation* op) {
    if (op->mnemonic() == "sc_high.conv2d") targets.push_back(op);
  });
  if (targets.empty()) return {};

  for (sir::Operation* conv : targets) {
    sir::Value* x = conv->operand(0);                              // NCHW
    sir::Value* filter = conv->operand(1);                         // OIHW
    sir::Value* bias =
        conv->numOperands() > 2 ? conv->operand(2) : nullptr;      // [Cout]
    sir::Value* y = conv->result(0);                               // NCHW
    const std::string base(y->id());

    const int64_t group = conv->getAttrAs<int64_t>("group").value_or(1);
    const auto dilations = conv->getAttrAs<std::vector<int64_t>>("dilations")
                               .value_or(std::vector<int64_t>{1, 1});
    if (group != 1 || dilations != std::vector<int64_t>{1, 1})
      return passing::LoweringError(base,
                                    "uses group/dilation, which the "
                                    "im2col-GEMM form does not model");
    auto strides = conv->getAttrAs<std::vector<int64_t>>("strides");
    auto pads = conv->getAttrAs<std::vector<int64_t>>("pads");
    if (!strides || !pads)
      return passing::LoweringError(base,
                                    "lacks stride/pad geometry attributes");

    const auto& fd = filter->shape().dims;  // [Cout, Cin, KH, KW]
    const int64_t cout = fd.at(0);

    // Record Y's consumers before building the replacement subgraph, so the
    // rewire below cannot capture the new ops.
    std::vector<sir::Operation*> consumers(y->users().begin(),
                                           y->users().end());

    auto cols_op =
        sir::OpBuilder::im2col(x, {fd.at(2), fd.at(3)}, *strides, *pads);
    sir::Value* cols = cols_op->result(0);

    auto wmat_op = std::make_unique<sir::Operation>("sc_low.filter_matrix");
    wmat_op->addOperand(filter);
    sir::Value* wmat =
        wmat_op->addResult(base + ".wmat", filter->dtype(),
                           sir::Shape{cols->shape().dims.at(1), cout});

    auto prod_op = std::make_unique<sir::Operation>("sc_high.matmul");
    prod_op->addOperand(cols);
    prod_op->addOperand(wmat);
    sir::Value* tail =
        prod_op->addResult(base + ".p2d", x->dtype(),
                           sir::Shape{cols->shape().dims.at(0), cout});

    std::vector<std::unique_ptr<sir::Operation>> new_ops;
    new_ops.push_back(std::move(cols_op));
    new_ops.push_back(std::move(wmat_op));
    new_ops.push_back(std::move(prod_op));

    if (bias) {
      auto bias_op = std::make_unique<sir::Operation>("sc_high.add_bias");
      bias_op->addOperand(tail);
      bias_op->addOperand(bias);
      tail = bias_op->addResult(base + ".p2d_b", tail->dtype(), tail->shape());
      new_ops.push_back(std::move(bias_op));
    }

    auto out_op = std::make_unique<sir::Operation>("sc_low.col2im");
    out_op->addOperand(tail);
    sir::Value* y_nchw =
        out_op->addResult(base + ".nchw", y->dtype(), y->shape());
    new_ops.push_back(std::move(out_op));

    block.insertOpsAfter(conv, std::move(new_ops));
    for (sir::Operation* consumer : consumers)
      for (size_t i = 0; i < consumer->numOperands(); ++i)
        if (consumer->operand(i) == y) consumer->setOperand(i, y_nchw);
    block.removeOp(conv);
  }

  seeml::diag::Note(passing::kConvLowering,
                    "lowered " + std::to_string(targets.size()) +
                        " convolution(s) to im2col-GEMM form");
  return {};
}

}  // namespace seeml::update
