#ifndef SEEML_COMPILER_FRONTEND_PARSER_GRAPH_BUILD_H_
#define SEEML_COMPILER_FRONTEND_PARSER_GRAPH_BUILD_H_

#include <unordered_map>

#include "compiler/frontend/ingressor/model_format.h"
#include "compiler/frontend/sir.h"

namespace seeml::update {

/// Import state threaded through the student and teacher builds and consumed
/// downstream by arena binding (rodata packing) and the emit table.
struct GraphBuild {
  seeml::sir::Value* input = nullptr;   // shared batch input block argument
  seeml::sir::Value* output = nullptr;  // network output (logits)
  // Frozen weight values -> their SMF tensor (for rodata packing + commit).
  std::unordered_map<const seeml::sir::Value*, const SmfTensor*>
      weight_sources;
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_FRONTEND_PARSER_GRAPH_BUILD_H_
