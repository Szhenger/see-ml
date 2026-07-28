#include "compiler/analysis/updater/pass_manager.h"

#include "compiler/diagnostics/logger.h"

namespace seeml::update {

using seecpp::utility::Logger;

std::expected<void, std::string> PassManager::Run(seeml::sir::Block& block) {
  for (const Pass& pass : passes_) {
    if (auto r = pass.run(block); !r) return std::unexpected(r.error());
    if (auto v = block.verify(); !v)
      return std::unexpected("PassManager: SIR invariants violated after "
                             "pass '" + pass.name + "': " + v.error());
    Logger::Info("PassManager: pass '" + pass.name + "' ok (" +
                 std::to_string(block.numOps()) + " ops)");
  }
  return {};
}

}  // namespace seeml::update
