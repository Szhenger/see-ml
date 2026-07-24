#include "compiler/frontend/parser/parser.h"

#include "compiler/frontend/parser/sema.h"
#include "compiler/frontend/parser/value_resolver.h"

namespace seeml::update {

namespace sir = seeml::sir;

std::expected<sir::Value*, std::string> BuildForward(
    sir::Block& block, const SmfModel& model, const std::string& prefix,
    sir::Value* input, int64_t batch, GraphBuild& build) {
  if (batch < 1)
    return std::unexpected("UpdateCompiler: batch must be at least 1, got " +
                           std::to_string(batch));
  // Whole-graph checks first: after this, every op introduces a fresh name,
  // names resolve in topological order, and the model output is known to be
  // producible — the parse loop can fail only on per-op grounds.
  if (auto graph_ok = sema::CheckGraph(model); !graph_ok)
    return std::unexpected(graph_ok.error());

  ValueResolver resolver(block, model, prefix, build);
  resolver.Bind(model.input_name, input);

  for (const SmfOp& op : model.ops) {
    switch (op.kind) {
      case SmfOpKind::kMatMul: {
        if (op.inputs.size() != 2)
          return std::unexpected("UpdateCompiler: MatMul '" + op.name +
                                 "' needs 2 inputs");
        auto x = resolver.Resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        auto w = resolver.Resolve(op.inputs[1]);
        if (!w) return std::unexpected(w.error());
        if (auto ok = sema::CheckMatMul(op, **x, **w); !ok)
          return std::unexpected(ok.error());
        sir::Operation* mm = block.appendOp("sc_high.matmul");
        mm->addOperand(*x);
        mm->addOperand(*w);
        resolver.Bind(op.output,
                      mm->addResult(prefix + op.output, sir::DataType::F32,
                                    sir::Shape{batch,
                                               (*w)->shape().dims.at(1)}));
        break;
      }
      case SmfOpKind::kAddBias: {
        if (op.inputs.size() != 2)
          return std::unexpected("UpdateCompiler: AddBias '" + op.name +
                                 "' needs 2 inputs");
        auto x = resolver.Resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        auto b = resolver.Resolve(op.inputs[1]);
        if (!b) return std::unexpected(b.error());
        if (auto ok = sema::CheckAddBias(op, **x, **b); !ok)
          return std::unexpected(ok.error());
        sir::Operation* ab = block.appendOp("sc_high.add_bias");
        ab->addOperand(*x);
        ab->addOperand(*b);
        resolver.Bind(op.output,
                      ab->addResult(prefix + op.output, sir::DataType::F32,
                                    (*x)->shape()));
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
        auto x = resolver.Resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        sir::Operation* r = block.appendOp(mnemonic);
        r->addOperand(*x);
        resolver.Bind(op.output,
                      r->addResult(prefix + op.output, sir::DataType::F32,
                                   (*x)->shape()));
        break;
      }
      case SmfOpKind::kMul: {
        if (op.inputs.size() != 2)
          return std::unexpected("UpdateCompiler: Mul '" + op.name +
                                 "' needs 2 inputs");
        auto x = resolver.Resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        auto y = resolver.Resolve(op.inputs[1]);
        if (!y) return std::unexpected(y.error());
        if (auto ok = sema::CheckMul(op, **x, **y); !ok)
          return std::unexpected(ok.error());
        sir::Operation* mul = block.appendOp("sc_high.mul");
        mul->addOperand(*x);
        mul->addOperand(*y);
        resolver.Bind(op.output,
                      mul->addResult(prefix + op.output, sir::DataType::F32,
                                     (*x)->shape()));
        break;
      }
      case SmfOpKind::kLayerNorm: {
        if (op.inputs.size() != 3)
          return std::unexpected("UpdateCompiler: LayerNorm '" + op.name +
                                 "' needs 3 inputs (x, gamma, beta)");
        auto x = resolver.Resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        auto gamma = resolver.Resolve(op.inputs[1]);
        if (!gamma) return std::unexpected(gamma.error());
        auto beta = resolver.Resolve(op.inputs[2]);
        if (!beta) return std::unexpected(beta.error());
        if (auto ok = sema::CheckLayerNorm(op, **x, **gamma, **beta); !ok)
          return std::unexpected(ok.error());
        const int64_t rows = (*x)->shape().dims.at(0);
        sir::Operation* ln = block.appendOp("sc_high.layer_norm");
        ln->addOperand(*x);
        ln->addOperand(*gamma);
        ln->addOperand(*beta);
        resolver.Bind(op.output,
                      ln->addResult(prefix + op.output, sir::DataType::F32,
                                    (*x)->shape()));
        // Row statistics cached for the backward kernel.
        ln->addResult(prefix + op.output + ".mean", sir::DataType::F32,
                      sir::Shape{rows});
        ln->addResult(prefix + op.output + ".rstd", sir::DataType::F32,
                      sir::Shape{rows});
        break;
      }
    }
  }

  // CheckGraph proved the output is an op's output, and every op binds its
  // output on success — this lookup cannot fail; it is kept as a defensive
  // invariant, not a reachable error path.
  sir::Value* out = resolver.Lookup(model.output_name);
  if (!out)
    return std::unexpected("UpdateCompiler: model output '" +
                           model.output_name + "' was never produced");
  return out;
}

}  // namespace seeml::update
