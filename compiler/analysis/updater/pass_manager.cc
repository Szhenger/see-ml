#include "compiler/analysis/updater/pass_manager.h"

#include <chrono>

#include "compiler/diagnostics/passing/error.h"

namespace seeml::update {

namespace passing = seeml::diag::passing;

std::expected<void, std::string> PassManager::Run(seeml::sir::Block& block) {
  using Clock = std::chrono::steady_clock;
  for (const Pass& pass : passes_) {
    const auto t0 = Clock::now();
    // A pass's own error is propagated verbatim; only corruption discovered
    // by the post-pass verify is attributed here.
    if (auto r = pass.run(block); !r) return std::unexpected(r.error());
    if (auto v = block.verify(); !v)
      return passing::InvariantsViolated(pass.name, v.error());
    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    timings_.push_back({pass.name, block.numOps(), ms});
    passing::PassNote(pass.name, block.numOps(), ms);
  }
  return {};
}

}  // namespace seeml::update
