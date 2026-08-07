#include "compiler/frontend/parser/parser.h"

#include "compiler/diagnostics/parsing/error.h"
#include "compiler/frontend/parser/sema.h"
#include "compiler/frontend/parser/value_resolver.h"

namespace seeml::update {

namespace sir = seeml::sir;
namespace parsing = seeml::diag::parsing;

std::expected<sir::Value*, std::string> BuildForward(
    sir::Block& block, const SmfModel& model, const std::string& prefix,
    sir::Value* input, int64_t batch, GraphBuild& build) {
  if (batch < 1)
    return parsing::Error("batch must be at least 1, got " +
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
          return parsing::OpError("MatMul", op.name, "needs 2 inputs");
        auto x = resolver.Resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        auto w = resolver.Resolve(op.inputs[1]);
        if (!w) return std::unexpected(w.error());
        if (auto ok = sema::CheckMatMul(op, **x, **w); !ok)
          return std::unexpected(ok.error());
        sir::Operation* mm = block.appendOp("sc_high.matmul");
        mm->addOperand(*x);
        mm->addOperand(*w);
        // Rows come from the actual LHS, not the config batch: x may be a
        // materialized constant with its own row count, and lowering sizes
        // the GEMM M dimension from this declared shape.
        resolver.Bind(op.output,
                      mm->addResult(prefix + op.output, sir::DataType::F32,
                                    sir::Shape{(*x)->shape().dims.at(0),
                                               (*w)->shape().dims.at(1)}));
        break;
      }
      case SmfOpKind::kAddBias: {
        if (op.inputs.size() != 2)
          return parsing::OpError("AddBias", op.name, "needs 2 inputs");
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
          return parsing::Error("'" + op.name + "' needs 1 input");
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
          return parsing::OpError("Mul", op.name, "needs 2 inputs");
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
      case SmfOpKind::kEmbedding: {
        if (op.inputs.size() != 2)
          return parsing::OpError("Embedding", op.name,
                                  "needs 2 inputs (tokens, table)");
        auto tokens = resolver.Resolve(op.inputs[0]);
        if (!tokens) return std::unexpected(tokens.error());
        auto table = resolver.Resolve(op.inputs[1]);
        if (!table) return std::unexpected(table.error());
        if (auto ok = sema::CheckEmbedding(op, **tokens, **table); !ok)
          return std::unexpected(ok.error());
        sir::Operation* emb = block.appendOp("sc_high.embedding");
        emb->addOperand(*tokens);
        emb->addOperand(*table);
        resolver.Bind(
            op.output,
            emb->addResult(prefix + op.output, sir::DataType::F32,
                           sir::Shape{(*tokens)->shape().dims.at(0),
                                      (*table)->shape().dims.at(1)}));
        break;
      }
      case SmfOpKind::kAdd: {
        if (op.inputs.size() != 2)
          return parsing::OpError("Add", op.name, "needs 2 inputs");
        auto x = resolver.Resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        auto y = resolver.Resolve(op.inputs[1]);
        if (!y) return std::unexpected(y.error());
        if (auto ok = sema::CheckAdd(op, **x, **y); !ok)
          return std::unexpected(ok.error());
        sir::Operation* add = block.appendOp("sc_high.add");
        add->addOperand(*x);
        add->addOperand(*y);
        resolver.Bind(op.output,
                      add->addResult(prefix + op.output, sir::DataType::F32,
                                     (*x)->shape()));
        break;
      }
      case SmfOpKind::kRmsNorm: {
        if (op.inputs.size() != 2)
          return parsing::OpError("RmsNorm", op.name,
                                  "needs 2 inputs (x, gamma)");
        auto x = resolver.Resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        auto gamma = resolver.Resolve(op.inputs[1]);
        if (!gamma) return std::unexpected(gamma.error());
        if (auto ok = sema::CheckRmsNorm(op, **x, **gamma); !ok)
          return std::unexpected(ok.error());
        sir::Operation* rn = block.appendOp("sc_high.rms_norm");
        rn->addOperand(*x);
        rn->addOperand(*gamma);
        resolver.Bind(op.output,
                      rn->addResult(prefix + op.output, sir::DataType::F32,
                                    (*x)->shape()));
        // Row statistic cached for the backward kernel.
        rn->addResult(prefix + op.output + ".rstd", sir::DataType::F32,
                      sir::Shape{(*x)->shape().dims.at(0)});
        break;
      }
      case SmfOpKind::kRope: {
        if (op.inputs.size() != 1)
          return parsing::OpError("Rope", op.name, "needs 1 input");
        auto x = resolver.Resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        if (auto ok = sema::CheckRope(op, **x, op.attr0, model.seq_len); !ok)
          return std::unexpected(ok.error());
        sir::Operation* r = block.appendOp("sc_high.rope");
        r->setAttribute("heads", static_cast<int64_t>(op.attr0));
        r->setAttribute("seq", static_cast<int64_t>(model.seq_len));
        r->addOperand(*x);
        resolver.Bind(op.output,
                      r->addResult(prefix + op.output, sir::DataType::F32,
                                   (*x)->shape()));
        break;
      }
      case SmfOpKind::kAttention: {
        if (op.inputs.size() != 3)
          return parsing::OpError("Attention", op.name,
                                  "needs 3 inputs (q, k, v)");
        auto q = resolver.Resolve(op.inputs[0]);
        if (!q) return std::unexpected(q.error());
        auto k = resolver.Resolve(op.inputs[1]);
        if (!k) return std::unexpected(k.error());
        auto v = resolver.Resolve(op.inputs[2]);
        if (!v) return std::unexpected(v.error());
        if (auto ok =
                sema::CheckAttention(op, **q, **k, **v, op.attr0, model.seq_len);
            !ok)
          return std::unexpected(ok.error());
        const int64_t rows = (*q)->shape().dims.at(0);
        const auto heads = static_cast<int64_t>(op.attr0);
        const auto seq = static_cast<int64_t>(model.seq_len);
        sir::Operation* at = block.appendOp("sc_high.attention");
        at->setAttribute("heads", heads);
        at->setAttribute("seq", seq);
        at->addOperand(*q);
        at->addOperand(*k);
        at->addOperand(*v);
        resolver.Bind(op.output,
                      at->addResult(prefix + op.output, sir::DataType::F32,
                                    (*q)->shape()));
        // Probability matrix P[B,H,S,S] flattened [B*H*S, S] = [rows*H, S],
        // cached for the backward primitives.
        at->addResult(prefix + op.output + ".probs", sir::DataType::F32,
                      sir::Shape{rows * heads, seq});
        break;
      }
      case SmfOpKind::kLayerNorm: {
        if (op.inputs.size() != 3)
          return parsing::OpError("LayerNorm", op.name,
                                  "needs 3 inputs (x, gamma, beta)");
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
    return parsing::Error("model output '" + model.output_name +
                          "' was never produced");
  // Loss grafting consumes [batch, classes] logits, and the driver reads
  // dims[1] unchecked; a rank-1 output (reachable via an activation of a
  // rank-1 constant) must be a diagnostic, not an out_of_range crash.
  if (out->shape().dims.size() != 2)
    return parsing::Error("model output '" + model.output_name +
                          "' must be rank-2 [batch, classes], got rank " +
                          std::to_string(out->shape().dims.size()));
  return out;
}

}  // namespace seeml::update
