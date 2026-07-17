#include "compiler/frontend/forward_builder.h"

#include <unordered_map>

namespace seeml::update {

namespace sir = seecpp::sir;

std::expected<sir::Value*, std::string> BuildForward(
    sir::Block& block, const SmfModel& model, const std::string& prefix,
    sir::Value* input, int64_t batch, GraphBuild& build) {
  std::unordered_map<std::string, sir::Value*> values;
  values[model.input_name] = input;

  auto materialize_weight =
      [&](const std::string& name) -> std::expected<sir::Value*, std::string> {
    const SmfTensor* t = model.FindTensor(name);
    if (!t || !t->is_const)
      return std::unexpected("UpdateCompiler: '" + name +
                             "' is not a constant tensor in the model");
    sir::Operation* op = block.appendOp("sc_mem.weight");
    op->setAttribute("smf_offset", static_cast<int64_t>(t->data_offset));
    sir::Value* v = op->addResult(prefix + name, sir::DataType::F32,
                                  sir::Shape(t->dims));
    build.weight_sources[v] = t;
    return v;
  };

  auto resolve =
      [&](const std::string& name) -> std::expected<sir::Value*, std::string> {
    if (auto it = values.find(name); it != values.end()) return it->second;
    // Cache the materialized weight so a multiply-referenced (tied) tensor
    // resolves to a single SIR value: one adapter, one rodata copy, and one
    // emit-table entry instead of several entries patching the same range.
    auto w = materialize_weight(name);
    if (w) values[name] = *w;
    return w;
  };

  for (const SmfOp& op : model.ops) {
    switch (op.kind) {
      case SmfOpKind::kMatMul: {
        if (op.inputs.size() != 2)
          return std::unexpected("UpdateCompiler: MatMul '" + op.name +
                                 "' needs 2 inputs");
        auto x = resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        auto w = resolve(op.inputs[1]);
        if (!w) return std::unexpected(w.error());
        if ((*x)->shape().dims.size() != 2 || (*w)->shape().dims.size() != 2)
          return std::unexpected("UpdateCompiler: MatMul '" + op.name +
                                 "' operands must be rank-2");
        const int64_t k_x = (*x)->shape().dims.at(1);
        const int64_t k_w = (*w)->shape().dims.at(0);
        if (k_x != k_w)
          return std::unexpected("UpdateCompiler: MatMul '" + op.name +
                                 "' inner dimensions disagree (" +
                                 std::to_string(k_x) + " vs " +
                                 std::to_string(k_w) + ")");
        sir::Operation* mm = block.appendOp("sc_high.matmul");
        mm->addOperand(*x);
        mm->addOperand(*w);
        values[op.output] = mm->addResult(
            prefix + op.output, sir::DataType::F32,
            sir::Shape{batch, (*w)->shape().dims.at(1)});
        break;
      }
      case SmfOpKind::kAddBias: {
        if (op.inputs.size() != 2)
          return std::unexpected("UpdateCompiler: AddBias '" + op.name +
                                 "' needs 2 inputs");
        auto x = resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        auto b = resolve(op.inputs[1]);
        if (!b) return std::unexpected(b.error());
        if ((*x)->shape().dims.empty() || (*b)->shape().dims.size() != 1 ||
            (*b)->shape().dims[0] != (*x)->shape().dims.back())
          return std::unexpected("UpdateCompiler: AddBias '" + op.name +
                                 "' bias width does not match its input");
        sir::Operation* ab = block.appendOp("sc_high.add_bias");
        ab->addOperand(*x);
        ab->addOperand(*b);
        values[op.output] = ab->addResult(prefix + op.output,
                                          sir::DataType::F32, (*x)->shape());
        break;
      }
      case SmfOpKind::kRelu:
      case SmfOpKind::kGelu:
      case SmfOpKind::kSilu: {
        const char* mnemonic = op.kind == SmfOpKind::kRelu ? "sc_high.relu"
                               : op.kind == SmfOpKind::kGelu
                                   ? "sc_high.gelu"
                                   : "sc_high.silu";
        if (op.inputs.size() != 1)
          return std::unexpected("UpdateCompiler: '" + op.name +
                                 "' needs 1 input");
        auto x = resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        sir::Operation* r = block.appendOp(mnemonic);
        r->addOperand(*x);
        values[op.output] = r->addResult(prefix + op.output,
                                         sir::DataType::F32, (*x)->shape());
        break;
      }
      case SmfOpKind::kMul: {
        if (op.inputs.size() != 2)
          return std::unexpected("UpdateCompiler: Mul '" + op.name +
                                 "' needs 2 inputs");
        auto x = resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        auto y = resolve(op.inputs[1]);
        if (!y) return std::unexpected(y.error());
        if ((*x)->shape() != (*y)->shape())
          return std::unexpected("UpdateCompiler: Mul '" + op.name +
                                 "' operand shapes disagree");
        sir::Operation* mul = block.appendOp("sc_high.mul");
        mul->addOperand(*x);
        mul->addOperand(*y);
        values[op.output] = mul->addResult(prefix + op.output,
                                           sir::DataType::F32, (*x)->shape());
        break;
      }
      case SmfOpKind::kLayerNorm: {
        if (op.inputs.size() != 3)
          return std::unexpected("UpdateCompiler: LayerNorm '" + op.name +
                                 "' needs 3 inputs (x, gamma, beta)");
        auto x = resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        auto gamma = resolve(op.inputs[1]);
        if (!gamma) return std::unexpected(gamma.error());
        auto beta = resolve(op.inputs[2]);
        if (!beta) return std::unexpected(beta.error());
        if ((*x)->shape().dims.size() != 2)
          return std::unexpected("UpdateCompiler: LayerNorm '" + op.name +
                                 "' input must be rank-2");
        const int64_t d = (*x)->shape().dims.back();
        for (const sir::Value* affine : {*gamma, *beta})
          if (affine->shape().dims.size() != 1 ||
              affine->shape().dims[0] != d)
            return std::unexpected("UpdateCompiler: LayerNorm '" + op.name +
                                   "' gamma/beta width does not match input");
        const int64_t rows = (*x)->shape().dims.at(0);
        sir::Operation* ln = block.appendOp("sc_high.layer_norm");
        ln->addOperand(*x);
        ln->addOperand(*gamma);
        ln->addOperand(*beta);
        values[op.output] = ln->addResult(prefix + op.output,
                                          sir::DataType::F32, (*x)->shape());
        // Row statistics cached for the backward kernel.
        ln->addResult(prefix + op.output + ".mean", sir::DataType::F32,
                      sir::Shape{rows});
        ln->addResult(prefix + op.output + ".rstd", sir::DataType::F32,
                      sir::Shape{rows});
        break;
      }
    }
  }

  auto out = values.find(model.output_name);
  if (out == values.end())
    return std::unexpected("UpdateCompiler: model output '" +
                           model.output_name + "' was never produced");
  return out->second;
}

}  // namespace seeml::update
