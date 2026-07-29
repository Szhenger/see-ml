#include "compiler/analysis/updater/pass_manager.h"

#include "compiler/diagnostics/passing/error.h"

namespace seeml::update {

namespace passing = seeml::diag::passing;

std::expected<void, std::string> PassManager::Run(seeml::sir::Block& block) {
  for (const Pass& pass : passes_) {
    // A pass's own error is propagated verbatim; only corruption discovered
    // by the post-pass verify is attributed here.
    if (auto r = pass.run(block); !r) return std::unexpected(r.error());
    if (auto v = block.verify(); !v)
      return passing::InvariantsViolated(pass.name, v.error());
    passing::PassNote(pass.name, block.numOps());
  }
  return {};
}

}  // namespace seeml::update
