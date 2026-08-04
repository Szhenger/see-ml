#ifndef SEEML_COMPILER_FRONTEND_PARSER_GRAPH_BUILD_H_
#define SEEML_COMPILER_FRONTEND_PARSER_GRAPH_BUILD_H_

#include <unordered_map>

#include "source/language/model_format.h"
#include "compiler/frontend/representation/sir.h"

namespace seeml::update {

/// Import state threaded through the student and teacher builds and consumed
/// downstream by arena binding (rodata packing) and the emit table.
struct GraphBuild {
  seeml::sir::Value* input = nullptr;   // shared batch input block argument
  // Network output (logits). Valid only until the first rewriting pass:
  // LoRA grafting redirects consumers to an adapted value and conv lowering
  // may destroy the value outright, and neither updates this field. The
  // driver nulls it once the loss is grafted so a stale read fails loudly.
  seeml::sir::Value* output = nullptr;
  // Frozen weight values -> their SMF tensor (for rodata packing + commit).
  std::unordered_map<const seeml::sir::Value*, const SmfTensor*>
      weight_sources;
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_FRONTEND_PARSER_GRAPH_BUILD_H_
