#include "compiler/backend/tuner/autotuner.h"

#include <algorithm>

#include "compiler/backend/tuner/bandit.h"
#include "compiler/diagnostics/architecting/error.h"

namespace seeml::update {

namespace architecting = seeml::diag::architecting;

namespace {

/// Keeps a halved/doubled dimension a positive multiple of the SIMD width.
size_t ClampDim(size_t v, size_t simd) {
  return std::max(simd, v - v % simd);
}

}  // namespace

std::vector<GemmTiling> TilingCandidates(const HostArchInfo& arch) {
  const size_t simd = std::max<size_t>(arch.simd_width_f32, 4);
  const GemmTiling hint = SuggestGemmTiling(arch);

  std::vector<GemmTiling> candidates{hint};
  auto add = [&](GemmTiling t) {
    t.mc = ClampDim(t.mc, simd);
    t.kc = ClampDim(t.kc, simd);
    t.nc = ClampDim(t.nc, simd);
    if (std::find(candidates.begin(), candidates.end(), t) ==
        candidates.end())
      candidates.push_back(t);
  };

  for (const double factor : {0.5, 2.0}) {
    add({static_cast<size_t>(hint.mc * factor), hint.kc, hint.nc});
    add({hint.mc, static_cast<size_t>(hint.kc * factor), hint.nc});
    add({hint.mc, hint.kc, static_cast<size_t>(hint.nc * factor)});
  }
  return candidates;
}

AutotuneResult AutotuneGemmTiling(std::span<const GemmTiling> candidates,
                                  const TilingBenchmarkFn& measure,
                                  size_t trials) {
  AutotuneResult result;
  if (candidates.empty()) {
    architecting::DetectionFallback(
        architecting::kAutotuner,
        "no tiling candidates to measure; falling back to the conservative "
        "default tiling");
    // A default-constructed result would carry the all-zero tiling, which
    // ValidateGemmTiling rejects and a blocked GEMM cannot execute. The
    // all-defaults hint is the documented always-valid fallback geometry.
    result.best = SuggestGemmTiling(HostArchInfo{});
    return result;
  }

  Ucb1Bandit bandit(candidates.size());
  const size_t budget = std::max(trials, candidates.size());
  if (budget > trials)
    architecting::DetectionFallback(
        architecting::kAutotuner,
        "trial budget raised to " + std::to_string(budget) +
            " so every candidate is measured at least once");
  for (size_t t = 0; t < budget; ++t) {
    const size_t arm = bandit.Select();
    bandit.Update(arm, measure(candidates[arm]));
  }

  result.best_arm = bandit.BestArm();
  result.best = candidates[result.best_arm];
  result.mean_reward.reserve(candidates.size());
  result.pulls.reserve(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    result.mean_reward.push_back(bandit.meanReward(i));
    result.pulls.push_back(bandit.pulls(i));
  }
  return result;
}

}  // namespace seeml::update
