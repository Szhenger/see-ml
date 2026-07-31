#ifndef SEEML_COMPILER_BACKEND_TUNER_AUTOTUNER_H_
#define SEEML_COMPILER_BACKEND_TUNER_AUTOTUNER_H_

#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include "compiler/backend/architecture/host_arch.h"

// =============================================================================
// GEMM-tiling autotuner — the tuner's application of the bandit to the
// trainer: the architecture analysis proposes a tiling, this module builds a
// deterministic candidate neighborhood around it and spends a fixed budget
// of measurements, allocated by UCB1, to find the configuration the real
// machine actually runs fastest. The measurement itself is injected
// (a benchmark closure returning higher-is-better reward, e.g. GFLOP/s of a
// training-shaped GEMM), which keeps this module deterministic and testable
// while the caller decides what "fast" means.
// =============================================================================

namespace seeml::update {

/// Higher is better (e.g. measured throughput of one training step or one
/// representative GEMM at the candidate tiling).
using TilingBenchmarkFn = std::function<double(const GemmTiling&)>;

/// The candidate neighborhood around the architecture hint: the hint itself
/// plus each dimension halved and doubled independently, deduplicated,
/// every dimension kept a positive multiple of the SIMD width. Order is
/// deterministic (hint first).
std::vector<GemmTiling> TilingCandidates(const HostArchInfo& arch);

struct AutotuneResult {
  GemmTiling best;                  // highest observed mean reward
  size_t best_arm = 0;              // index into the candidate list
  std::vector<double> mean_reward;  // per candidate, post-run
  std::vector<uint64_t> pulls;      // per candidate, post-run
};

/// Spends `trials` measurements over `candidates` under UCB1 and returns
/// the winner. `trials` is clamped up to candidates.size() so every arm is
/// measured at least once. An empty candidate list yields the conservative
/// default tiling (SuggestGemmTiling of an all-defaults HostArchInfo) with
/// no measurements recorded — `best` is always a valid geometry.
AutotuneResult AutotuneGemmTiling(std::span<const GemmTiling> candidates,
                                  const TilingBenchmarkFn& measure,
                                  size_t trials);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_BACKEND_TUNER_AUTOTUNER_H_
