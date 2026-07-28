#ifndef SEEML_COMPILER_ANALYSIS_UPDATER_PASS_MANAGER_H_
#define SEEML_COMPILER_ANALYSIS_UPDATER_PASS_MANAGER_H_

#include <expected>
#include <functional>
#include <string>
#include <vector>

#include "compiler/frontend/representation/sir.h"

// =============================================================================
// PassManager — the driver of the SIR-to-SIR pipeline. Runs registered
// passes in order and re-runs Block::verify (the structural invariant gate)
// after every pass, so a corrupted use-list or broken SSA order is reported
// against the pass that introduced it — not discovered three passes later.
// Passes with results (adapters, gradient maps) register as capturing
// closures that store into caller-owned state.
// =============================================================================

namespace seeml::update {

/// One named SIR-to-SIR pass: mutates the block or reports why it cannot.
struct Pass {
  std::string name;
  std::function<std::expected<void, std::string>(seeml::sir::Block&)> run;
};

class PassManager {
 public:
  void Add(std::string name,
           std::function<std::expected<void, std::string>(seeml::sir::Block&)>
               run) {
    passes_.push_back({std::move(name), std::move(run)});
  }

  /// Runs the registered passes in order. A pass error is returned verbatim
  /// (passes identify themselves in their messages); a verification failure
  /// is returned naming the offending pass.
  [[nodiscard]] std::expected<void, std::string> Run(seeml::sir::Block& block);

 private:
  std::vector<Pass> passes_;
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_UPDATER_PASS_MANAGER_H_
