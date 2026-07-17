#ifndef SEEML_COMPILER_FRONTEND_FORWARD_BUILDER_H_
#define SEEML_COMPILER_FRONTEND_FORWARD_BUILDER_H_

#include <cstdint>
#include <expected>
#include <string>
#include <unordered_map>

#include "compiler/frontend/sir.h"
#include "source/smf.h"

// =============================================================================
// ForwardBuilder — the front end of the update compiler: imports a
// topologically ordered SMF graph into sc_high.* forward SIR, performing
// semantic analysis (rank / inner-dimension / bias-width checks) along the
// way. Called once for the student and, under a distillation loss, again for
// the frozen teacher subgraph (value ids prefixed "t::") sharing the same
// batch input.
// =============================================================================

namespace seeml::update {

/// Import state threaded through the student and teacher builds and consumed
/// downstream by arena binding (rodata packing) and the emit table.
struct GraphBuild {
  seecpp::sir::Value* input = nullptr;   // shared batch input block argument
  seecpp::sir::Value* output = nullptr;  // network output (logits)
  // Frozen weight values -> their SMF tensor (for rodata packing + commit).
  std::unordered_map<const seecpp::sir::Value*, const SmfTensor*>
      weight_sources;
};

/// Appends `model`'s forward graph to `block`, resolving tensors to SIR
/// values (materializing frozen sc_mem.weight ops on demand) and returning
/// the model's output value. `prefix` namespaces the emitted value ids.
[[nodiscard]] std::expected<seecpp::sir::Value*, std::string> BuildForward(
    seecpp::sir::Block& block, const SmfModel& model, const std::string& prefix,
    seecpp::sir::Value* input, int64_t batch, GraphBuild& build);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_FRONTEND_FORWARD_BUILDER_H_
