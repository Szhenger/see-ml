#ifndef SEEML_COMPILER_FRONTEND_PARSER_PARSER_H_
#define SEEML_COMPILER_FRONTEND_PARSER_PARSER_H_

#include <cstdint>
#include <expected>
#include <string>

#include "compiler/frontend/ingressor/model_format.h"
#include "compiler/frontend/parser/graph_build.h"
#include "compiler/frontend/sir.h"

// =============================================================================
// Parser — the driver of the update compiler's front end: walks a
// topologically ordered SMF graph and builds sc_high.* forward SIR, resolving
// names through ValueResolver and checking shapes through sema. Called once
// for the student and, under a distillation loss, again for the frozen
// teacher subgraph (value ids prefixed "t::") sharing the same batch input.
// =============================================================================

namespace seeml::update {

/// Appends `model`'s forward graph to `block`, resolving tensors to SIR
/// values (materializing frozen sc_mem.weight ops on demand) and returning
/// the model's output value. `prefix` namespaces the emitted value ids.
///
/// Thread-compatible: concurrent calls are safe when each has its own
/// (block, build, input) — `model` may be shared, it is only read. Two calls
/// may NOT share an input Value even across distinct blocks: emitting an op
/// that consumes it writes the value's use-list (see the SIR threading
/// model). The student + teacher builds share one input and one block, so
/// they run sequentially by design.
[[nodiscard]] std::expected<seeml::sir::Value*, std::string> BuildForward(
    seeml::sir::Block& block, const SmfModel& model, const std::string& prefix,
    seeml::sir::Value* input, int64_t batch, GraphBuild& build);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_FRONTEND_PARSER_PARSER_H_
