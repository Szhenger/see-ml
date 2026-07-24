#ifndef SEEML_COMPILER_FRONTEND_PARSER_VALUE_RESOLVER_H_
#define SEEML_COMPILER_FRONTEND_PARSER_VALUE_RESOLVER_H_

#include <expected>
#include <string>
#include <string_view>
#include <unordered_map>

#include "compiler/frontend/ingressor/model_format.h"
#include "compiler/frontend/parser/graph_build.h"
#include "compiler/frontend/sir.h"

// =============================================================================
// ValueResolver — name binding for the parser: resolves SMF tensor names to
// SIR values, materializing frozen sc_mem.weight ops on first reference to a
// constant tensor. Owns the parse-local symbol state (name -> value map and
// the tensor-table index).
//
// Lifetime contract: every name passed to Bind/Resolve/Lookup must outlive
// the resolver — both maps key on string_views of the caller's storage. The
// parser satisfies this for free: every name it binds or resolves is owned
// by the (const, outliving) SmfModel, so the symbol tables never copy a
// string.
// =============================================================================

namespace seeml::update {

class ValueResolver {
 public:
  /// `model`, `block`, and `build` must outlive the resolver. `prefix`
  /// namespaces the value ids of materialized weights.
  ValueResolver(seeml::sir::Block& block, const SmfModel& model,
                std::string prefix, GraphBuild& build);

  /// Binds `name` to a parsed value (the graph input or an op output).
  void Bind(std::string_view name, seeml::sir::Value* v);

  /// Returns the value bound to `name`, or materializes it as a frozen
  /// sc_mem.weight if `name` is a constant tensor of the model. A
  /// multiply-referenced (tied) tensor resolves to a single SIR value: one
  /// adapter, one rodata copy, and one emit-table entry instead of several
  /// entries patching the same range.
  [[nodiscard]] std::expected<seeml::sir::Value*, std::string> Resolve(
      std::string_view name);

  /// Returns the value bound to `name`, or nullptr if it was never bound.
  seeml::sir::Value* Lookup(std::string_view name) const;

 private:
  [[nodiscard]] std::expected<seeml::sir::Value*, std::string>
  MaterializeWeight(std::string_view name);

  seeml::sir::Block& block_;
  std::string prefix_;
  GraphBuild& build_;
  std::unordered_map<std::string_view, seeml::sir::Value*> values_;
  // Index over the model's tensor table: every op input resolves through this
  // map instead of a linear FindTensor scan, turning the import from
  // O(ops * tensors) into O(ops + tensors). Keys view the model's own
  // (const, stable) tensor names — no copies.
  std::unordered_map<std::string_view, const SmfTensor*> tensor_index_;
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_FRONTEND_PARSER_VALUE_RESOLVER_H_
