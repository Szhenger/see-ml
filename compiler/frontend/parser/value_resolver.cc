#include "compiler/frontend/parser/value_resolver.h"

namespace seeml::update {

namespace sir = seeml::sir;

ValueResolver::ValueResolver(sir::Block& block, const SmfModel& model,
                             std::string prefix, GraphBuild& build)
    : block_(block), prefix_(std::move(prefix)), build_(build) {
  values_.reserve(model.tensors.size() + model.ops.size());
  tensor_index_.reserve(model.tensors.size());
  for (const SmfTensor& t : model.tensors) tensor_index_[t.name] = &t;
}

void ValueResolver::Bind(std::string_view name, sir::Value* v) {
  values_[name] = v;
}

sir::Value* ValueResolver::Lookup(std::string_view name) const {
  auto it = values_.find(name);
  return it == values_.end() ? nullptr : it->second;
}

std::expected<sir::Value*, std::string> ValueResolver::Resolve(
    std::string_view name) {
  if (auto it = values_.find(name); it != values_.end()) return it->second;
  auto w = MaterializeWeight(name);
  if (w) values_[name] = *w;
  return w;
}

std::expected<sir::Value*, std::string> ValueResolver::MaterializeWeight(
    std::string_view name) {
  auto found = tensor_index_.find(name);
  const SmfTensor* t = found == tensor_index_.end() ? nullptr : found->second;
  if (!t || !t->is_const)
    return std::unexpected("UpdateCompiler: '" + std::string(name) +
                           "' is not a constant tensor in the model");
  sir::Operation* op = block_.appendOp("sc_mem.weight");
  op->setAttribute("smf_offset", static_cast<int64_t>(t->data_offset));
  sir::Value* v = op->addResult(prefix_ + std::string(name),
                                sir::DataType::F32, sir::Shape(t->dims));
  build_.weight_sources[v] = t;
  return v;
}

}  // namespace seeml::update
