#ifndef SEEML_COMPILER_ANALYSIS_UPDATER_DCE_H_
#define SEEML_COMPILER_ANALYSIS_UPDATER_DCE_H_

#include <expected>
#include <string>
#include <unordered_set>

#include "compiler/frontend/representation/sir.h"

// =============================================================================
// Dead-code elimination — the cleanup discipline of the optimization phase.
//
// The driver constructs its programs minimally, so on today's pipelines this
// pass removes nothing; it exists to keep that a *verified invariant* rather
// than an accident, and to be the seam where rewriting passes (fusion,
// simplification) can leave dead ops behind and trust the phase to sweep
// them. An op survives when any of its results is rooted (read outside the
// program: the loss slot, parameter gradients, every primal-snapshot value
// the eval program lowers), when any result still has users, or when the op
// is effectful — storage declarations (`sc_mem.*`, which the arena binder
// and graph build hold pointers into) and the in-place optimizer family,
// whose work is a side effect on the parameter buffers, not a result.
// =============================================================================

namespace seeml::update {

class DeadCodeElimination {
 public:
  /// Removes every non-effectful op none of whose results is rooted or
  /// used, sweeping backward so a dead consumer frees its dead producers in
  /// the same run. Returns the number of ops removed.
  std::expected<size_t, std::string> Run(
      sir::Block& block, const std::unordered_set<const sir::Value*>& roots);
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_UPDATER_DCE_H_
