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

  // LayerNorm results derive companion value ids ("<output>.mean" /
  // "<output>.rstd"); a model name equal to one of them would collide in the
  // SIR value table far from either offender, so reject it here.
  for (const SmfOp& op : model.ops) {
    if (op.kind != SmfOpKind::kLayerNorm) continue;
    for (const char* suffix : {".mean", ".rstd"}) {
      const std::string derived = op.output + suffix;
      if (bound.contains(derived))
        return parsing::OpError(
            "LayerNorm", op.name,
            "output '" + op.output + "' derives statistic id '" + derived +
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

}  // namespace seeml::update::sema
