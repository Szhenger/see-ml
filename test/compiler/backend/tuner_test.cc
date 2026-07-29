// =============================================================================
// Backend partition tests: architecture/ (host detection sanity, tiling
// derivation), tuner/ (UCB1 bandit behavior, tiling autotune convergence),
// and the trainer/ GPU kernel emitter the two configure.
// =============================================================================

#include <string>
#include <vector>

#include "compiler/backend/architecture/host_arch.h"
#include "compiler/backend/trainer/kernel_emitter.h"
#include "compiler/backend/tuner/autotuner.h"
#include "compiler/backend/tuner/bandit.h"
#include "test/framework/seetest.h"

namespace {

using namespace seeml::update;

// =============================================================================
// architecture/
// =============================================================================

TEST(HostArch, DetectionYieldsUsableGeometry) {
  const HostArchInfo arch = DetectHostArch();
  EXPECT_TRUE(arch.isa == "arm64" || arch.isa == "x86_64" ||
              arch.isa == "unknown");
  EXPECT_GE(arch.simd_width_f32, 4u);
  EXPECT_GE(arch.physical_cores, 1u);
  EXPECT_GE(arch.cache_line_bytes, 32u);
}

TEST(HostArch, TilingIsPureAndSimdAligned) {
  HostArchInfo arch;
  arch.simd_width_f32 = 8;
  arch.l1d_bytes = 64u << 10;
  arch.l2_bytes = 1u << 20;

  const GemmTiling a = SuggestGemmTiling(arch);
  const GemmTiling b = SuggestGemmTiling(arch);
  EXPECT_TRUE(a == b);  // pure function of the host description

  EXPECT_GT(a.mc, 0u);
  EXPECT_GT(a.kc, 0u);
  EXPECT_EQ(a.nc, 4u * 8u);
  EXPECT_EQ(a.mc % 8, 0u);
  EXPECT_EQ(a.kc % 8, 0u);

  // kc x nc panel of B must fit in half of L1, mc x kc panel of A in half
  // of L2 — the contract the heuristic documents.
  EXPECT_LE(a.kc * a.nc * sizeof(float), arch.l1d_bytes / 2);
  EXPECT_LE(a.mc * a.kc * sizeof(float), arch.l2_bytes / 2);
}

TEST(HostArch, UnknownCachesFallBackToUsableTiling) {
  HostArchInfo arch;  // no cache info at all
  const GemmTiling t = SuggestGemmTiling(arch);
  EXPECT_GT(t.mc, 0u);
  EXPECT_GT(t.kc, 0u);
  EXPECT_GT(t.nc, 0u);
}

// =============================================================================
// tuner/ — Ucb1Bandit
// =============================================================================

TEST(Ucb1Bandit, PullsEveryArmOnceBeforeExploiting) {
  Ucb1Bandit bandit(3);
  for (size_t expected = 0; expected < 3; ++expected) {
    const size_t arm = bandit.Select();
    EXPECT_EQ(arm, expected);
    bandit.Update(arm, 1.0);
  }
  EXPECT_EQ(bandit.totalPulls(), 3u);
}

TEST(Ucb1Bandit, ConvergesOnTheBestArm) {
  // Deterministic reward model: arm 1 dominates.
  const std::vector<double> reward = {0.2, 0.9, 0.4};
  Ucb1Bandit bandit(reward.size());
  for (size_t t = 0; t < 200; ++t) {
    const size_t arm = bandit.Select();
    bandit.Update(arm, reward[arm]);
  }
  EXPECT_EQ(bandit.BestArm(), 1u);
  // Exploitation: the winning arm absorbed most of the budget, but UCB1's
  // confidence bonus kept every arm sampled.
  EXPECT_GT(bandit.pulls(1), 100u);
  EXPECT_GT(bandit.pulls(0), 1u);
  EXPECT_GT(bandit.pulls(2), 1u);
  EXPECT_NEAR(bandit.meanReward(1), 0.9, 1e-9);
}

// =============================================================================
// tuner/ — autotuner
// =============================================================================

TEST(Autotuner, CandidatesAreDeterministicSimdMultiples) {
  HostArchInfo arch;
  arch.simd_width_f32 = 4;
  arch.l1d_bytes = 32u << 10;
  arch.l2_bytes = 512u << 10;

  const std::vector<GemmTiling> a = TilingCandidates(arch);
  const std::vector<GemmTiling> b = TilingCandidates(arch);
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) EXPECT_TRUE(a[i] == b[i]);

  ASSERT_GE(a.size(), 4u);
  EXPECT_TRUE(a[0] == SuggestGemmTiling(arch));  // hint first
  for (const GemmTiling& t : a) {
    EXPECT_EQ(t.mc % 4, 0u);
    EXPECT_EQ(t.kc % 4, 0u);
    EXPECT_EQ(t.nc % 4, 0u);
  }
}

TEST(Autotuner, FindsTheFastestCandidate) {
  const std::vector<GemmTiling> candidates =
      TilingCandidates(DetectHostArch());
  // Synthetic benchmark: reward peaks at the last candidate.
  const GemmTiling target = candidates.back();
  auto measure = [&](const GemmTiling& t) {
    return t == target ? 10.0 : 1.0;
  };

  const AutotuneResult r =
      AutotuneGemmTiling(candidates, measure, candidates.size() * 8);
  EXPECT_TRUE(r.best == target);
  EXPECT_EQ(r.best_arm, candidates.size() - 1);
  ASSERT_EQ(r.pulls.size(), candidates.size());
  for (uint64_t pulls : r.pulls) EXPECT_GT(pulls, 0u);  // full coverage
}

// =============================================================================
// trainer/ — GPU kernel emitter
// =============================================================================

TEST(KernelEmitter, GpuTilingClampsHostTiling) {
  const GpuTiling t = GpuTilingFromHost({.mc = 512, .kc = 96, .nc = 16});
  EXPECT_EQ(t.tile_m, 32u);  // clamped down from 512
  EXPECT_EQ(t.tile_k, 32u);
  EXPECT_EQ(t.tile_n, 16u);

  const GpuTiling tiny = GpuTilingFromHost({.mc = 4, .kc = 4, .nc = 4});
  EXPECT_EQ(tiny.tile_m, 8u);  // clamped up to the SIMD-group minimum
}

TEST(KernelEmitter, EmitsAllTrainingKernelsAtTheRequestedTiling) {
  const std::string src = EmitMetalKernels({.tile_m = 16, .tile_n = 8,
                                            .tile_k = 24});
  for (const char* kernel :
       {"kernel void seeml_matmul(", "kernel void seeml_matmul_nt(",
        "kernel void seeml_matmul_tn(", "kernel void seeml_gemm_acc("})
    EXPECT_NE(src.find(kernel), std::string::npos);

  EXPECT_NE(src.find("threadgroup float a_tile[16][24];"), std::string::npos);
  EXPECT_NE(src.find("threadgroup float b_tile[24][8];"), std::string::npos);
  // The merge kernel accumulates with the adapter scale; the forward stores.
  EXPECT_NE(src.find("C[row * dims.n + col] += dims.alpha * acc;"),
            std::string::npos);
  EXPECT_NE(src.find("C[row * dims.n + col] = acc;"), std::string::npos);
  EXPECT_NE(src.find("#include <metal_stdlib>"), std::string::npos);
}

}  // namespace
