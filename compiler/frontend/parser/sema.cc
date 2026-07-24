#include "compiler/frontend/parser/sema.h"

#include <string_view>
#include <unordered_set>

namespace seeml::update::sema {

namespace sir = seeml::sir;

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

  for (const SmfOp& op : model.ops) {
    for (const std::string& in : op.inputs)
      if (!bound.contains(in) && all_outputs.contains(in))
        return std::unexpected(
            "UpdateCompiler: op '" + op.name + "' consumes '" + in +
            "' before it is produced (the op list must be topologically "
            "ordered)");
    if (!bound.insert(op.output).second)
      return std::unexpected("UpdateCompiler: op '" + op.name +
                             "' output '" + op.output +
                             "' redefines an existing value");
  }

  if (!all_outputs.contains(model.output_name))
    return std::unexpected("UpdateCompiler: model output '" +
                           model.output_name +
                           "' was never produced by an operation");
  return {};
}

std::expected<void, std::string> CheckMatMul(const SmfOp& op,
                                             const sir::Value& x,
                                             const sir::Value& w) {
  if (x.shape().dims.size() != 2 || w.shape().dims.size() != 2)
    return std::unexpected("UpdateCompiler: MatMul '" + op.name +
                           "' operands must be rank-2");
  const int64_t k_x = x.shape().dims.at(1);
  const int64_t k_w = w.shape().dims.at(0);
  if (k_x != k_w)
    return std::unexpected("UpdateCompiler: MatMul '" + op.name +
                           "' inner dimensions disagree (" +
                           std::to_string(k_x) + " vs " + std::to_string(k_w) +
                           ")");
  return {};
}

std::expected<void, std::string> CheckAddBias(const SmfOp& op,
                                              const sir::Value& x,
                                              const sir::Value& b) {
  if (x.shape().dims.empty() || b.shape().dims.size() != 1 ||
      b.shape().dims[0] != x.shape().dims.back())
    return std::unexpected("UpdateCompiler: AddBias '" + op.name +
                           "' bias width does not match its input");
  return {};
}

std::expected<void, std::string> CheckMul(const SmfOp& op, const sir::Value& x,
                                          const sir::Value& y) {
  if (x.shape() != y.shape())
    return std::unexpected("UpdateCompiler: Mul '" + op.name +
                           "' operand shapes disagree");
  return {};
}

std::expected<void, std::string> CheckLayerNorm(const SmfOp& op,
                                                const sir::Value& x,
                                                const sir::Value& gamma,
                                                const sir::Value& beta) {
  if (x.shape().dims.size() != 2)
    return std::unexpected("UpdateCompiler: LayerNorm '" + op.name +
                           "' input must be rank-2");
  const int64_t d = x.shape().dims.back();
  for (const sir::Value* affine : {&gamma, &beta})
    if (affine->shape().dims.size() != 1 || affine->shape().dims[0] != d)
      return std::unexpected("UpdateCompiler: LayerNorm '" + op.name +
                             "' gamma/beta width does not match input");
  return {};
}

}  // namespace seeml::update::sema
