#include "compiler/frontend/parser/sema.h"

#include <string_view>
#include <unordered_set>

#include "compiler/diagnostics/parsing/error.h"

namespace seeml::update::sema {

namespace sir = seeml::sir;
namespace parsing = seeml::diag::parsing;

std::expected<void, std::string> CheckGraph(const SmfModel& model) {
  // Every name an op may produce, for distinguishing "consumed before
  // produced" (a topological-order violation) from "does not exist" (left to
  // the resolver, which knows tensor constness).
  std::unordered_set<std::string_view> all_outputs;
  all_outputs.reserve(model.ops.size());
  for (const SmfOp& op : model.ops) all_outputs.insert(op.output);

  // Names bound so far, in op order: tensors and the input up front, then
  // each op's output as it appears.
  std::unordered_set<std::string_view> bound;
  bound.reserve(model.tensors.size() + model.ops.size() + 1);
  for (const SmfTensor& t : model.tensors) bound.insert(t.name);
  bound.insert(model.input_name);

  // The graph input is I/O by definition. A constant tensor carrying the
  // input's name would be silently shadowed by the batch input (the resolver
  // binds the input first and never materializes the weight), compiling to
  // wrong math instead of an error — reject the contradiction here.
  for (const SmfTensor& t : model.tensors)
    if (t.is_const && t.name == model.input_name)
      return parsing::Error("model input '" + model.input_name +
                            "' is declared as a constant tensor");

  for (const SmfOp& op : model.ops) {
    for (const std::string& in : op.inputs)
      if (!bound.contains(in) && all_outputs.contains(in))
        return parsing::OpError(
            "op", op.name,
            "consumes '" + in +
                "' before it is produced (the op list must be topologically "
                "ordered)");
    if (!bound.insert(op.output).second)
      return parsing::OpError("op", op.name,
                              "output '" + op.output +
                                  "' redefines an existing value");
  }

  // Some ops derive companion value ids from their output (LayerNorm's
  // "<output>.mean"/".rstd", RMSNorm's ".rstd", Attention's ".probs"); a
  // model name equal to one of them would collide in the SIR value table
  // far from either offender, so reject it here.
  for (const SmfOp& op : model.ops) {
    const char* suffixes[2] = {nullptr, nullptr};
    switch (op.kind) {
      case SmfOpKind::kLayerNorm:
        suffixes[0] = ".mean";
        suffixes[1] = ".rstd";
        break;
      case SmfOpKind::kRmsNorm:   suffixes[0] = ".rstd"; break;
      case SmfOpKind::kAttention: suffixes[0] = ".probs"; break;
      default: continue;
    }
    for (const char* suffix : suffixes) {
      if (suffix == nullptr) continue;
      const std::string derived = op.output + suffix;
      if (bound.contains(derived))
        return parsing::OpError(
            "op", op.name,
            "output '" + op.output + "' derives companion id '" + derived +
                "', which collides with an existing name");
    }
  }

  if (!all_outputs.contains(model.output_name))
    return parsing::Error("model output '" + model.output_name +
                          "' was never produced by an operation");
  return {};
}

std::expected<void, std::string> CheckMatMul(const SmfOp& op,
                                             const sir::Value& x,
                                             const sir::Value& w) {
  if (x.shape().dims.size() != 2 || w.shape().dims.size() != 2)
    return parsing::OpError("MatMul", op.name, "operands must be rank-2");
  const int64_t k_x = x.shape().dims.at(1);
  const int64_t k_w = w.shape().dims.at(0);
  if (k_x != k_w)
    return parsing::OpError("MatMul", op.name,
                            "inner dimensions disagree (" +
                                std::to_string(k_x) + " vs " +
                                std::to_string(k_w) + ")");
  return {};
}

std::expected<void, std::string> CheckAddBias(const SmfOp& op,
                                              const sir::Value& x,
                                              const sir::Value& b) {
  // Rank-2 x is a lowering requirement, not a preference: sc_high.add_bias
  // lowering reads dims 0/1 as rows/cols, so a rank-1 input aborts the
  // compiler and a rank-3 input compiles to wrong math.
  if (x.shape().dims.size() != 2)
    return parsing::OpError("AddBias", op.name, "input must be rank-2");
  if (b.shape().dims.size() != 1 ||
      b.shape().dims[0] != x.shape().dims.back())
    return parsing::OpError("AddBias", op.name,
                            "bias width does not match its input");
  return {};
}

std::expected<void, std::string> CheckMul(const SmfOp& op, const sir::Value& x,
                                          const sir::Value& y) {
  if (x.shape() != y.shape())
    return parsing::OpError("Mul", op.name, "operand shapes disagree");
  return {};
}

std::expected<void, std::string> CheckLayerNorm(const SmfOp& op,
                                                const sir::Value& x,
                                                const sir::Value& gamma,
                                                const sir::Value& beta) {
  if (x.shape().dims.size() != 2)
    return parsing::OpError("LayerNorm", op.name, "input must be rank-2");
  const int64_t d = x.shape().dims.back();
  for (const sir::Value* affine : {&gamma, &beta})
    if (affine->shape().dims.size() != 1 || affine->shape().dims[0] != d)
      return parsing::OpError("LayerNorm", op.name,
                              "gamma/beta width does not match input");
  return {};
}

std::expected<void, std::string> CheckAdd(const SmfOp& op, const sir::Value& x,
                                          const sir::Value& y) {
  if (x.shape() != y.shape())
    return parsing::OpError("Add", op.name, "operand shapes disagree");
  return {};
}

std::expected<void, std::string> CheckRmsNorm(const SmfOp& op,
                                              const sir::Value& x,
                                              const sir::Value& gamma) {
  if (x.shape().dims.size() != 2)
    return parsing::OpError("RmsNorm", op.name, "input must be rank-2");
  if (gamma.shape().dims.size() != 1 ||
      gamma.shape().dims[0] != x.shape().dims.back())
    return parsing::OpError("RmsNorm", op.name,
                            "gamma width does not match input");
  return {};
}

namespace {

/// Shared sequence-geometry gate for the ops that interpret rows as
/// positions: rows split into sequences of seq_len, and heads divide the
/// width. `even_head` additionally requires an even head width (RoPE pairs).
std::expected<void, std::string> CheckSeqGeometry(const char* unit,
                                                  const SmfOp& op,
                                                  const sir::Value& x,
                                                  uint32_t heads,
                                                  uint64_t seq_len,
                                                  bool even_head) {
  if (x.shape().dims.size() != 2)
    return parsing::OpError(unit, op.name, "input must be rank-2");
  if (heads == 0)
    return parsing::OpError(unit, op.name,
                            "num_heads (attr0) must be positive");
  if (seq_len == 0)
    return parsing::OpError(unit, op.name,
                            "model declares no seq_len, which this op needs");
  // seq_len is untrusted u64 from the file; a value past INT64_MAX would
  // wrap negative through the int64 casts below (rows % -1 == 0 accepts
  // everything) and reach the parser as a negative shape dim.
  if (seq_len > static_cast<uint64_t>(INT64_MAX))
    return parsing::OpError(unit, op.name,
                            "seq_len does not fit a signed 64-bit dimension");
  const int64_t rows = x.shape().dims.at(0);
  const int64_t width = x.shape().dims.at(1);
  if (rows % static_cast<int64_t>(seq_len) != 0)
    return parsing::OpError(
        unit, op.name,
        "batch rows (" + std::to_string(rows) +
            ") are not a whole number of sequences of seq_len " +
            std::to_string(seq_len));
  if (width % static_cast<int64_t>(heads) != 0)
    return parsing::OpError(unit, op.name,
                            "width " + std::to_string(width) +
                                " is not divisible by num_heads " +
                                std::to_string(heads));
  if (even_head && (width / static_cast<int64_t>(heads)) % 2 != 0)
    return parsing::OpError(unit, op.name,
                            "head width must be even for the rotation pairs");
  return {};
}

}  // namespace

std::expected<void, std::string> CheckRope(const SmfOp& op, const sir::Value& x,
                                           uint32_t heads, uint64_t seq_len) {
  return CheckSeqGeometry("Rope", op, x, heads, seq_len, /*even_head=*/true);
}

std::expected<void, std::string> CheckAttention(const SmfOp& op,
                                                const sir::Value& q,
                                                const sir::Value& k,
                                                const sir::Value& v,
                                                uint32_t heads,
                                                uint64_t seq_len) {
  if (q.shape() != k.shape() || q.shape() != v.shape())
    return parsing::OpError("Attention", op.name, "q/k/v shapes disagree");
  return CheckSeqGeometry("Attention", op, q, heads, seq_len,
                          /*even_head=*/false);
}

}  // namespace seeml::update::sema
