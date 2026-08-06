// =============================================================================
// Metal GEMM dispatch tests (G1a): JIT-compile the kernel emitter's MSL on
// the local device and prove every GEMM-family kernel against the CPU
// reference implementations. Tolerance-based, not bitwise — the GPU
// contracts FMA differently (the per-backend determinism doctrine,
// docs/roadmap.md Project 5). On a Metal-less host (or CI container) the
// suite passes vacuously with a note.
// =============================================================================

#include <cmath>
#include <cstdio>
#include <vector>

#include "test/framework/seetest.h"
#include "test/support/builders.h"

#if defined(__APPLE__)

#include "compiler/backend/trainer/kernel_emitter.h"
#include "runtime/executor/metal_gemm.h"
#include "runtime/executor/update_kernels.h"

namespace {

namespace k = seeml::update_rt::kernels;
using seeml::testing::RandnVector;
using seeml::update::EmitMetalKernels;
using seeml::update::GpuTiling;
using seeml::update_rt::MetalGemmRunner;
using Kind = seeml::update_rt::MetalGemmRunner::Kind;

// Ragged sizes on purpose: none is a multiple of the 16x16x16 tiling, so
// every edge guard in the emitted kernels is exercised.
constexpr size_t kM = 33, kN = 29, kK = 21;

std::unique_ptr<MetalGemmRunner> MakeRunner() {
  const GpuTiling tiling;  // 16 x 16 x 16
  auto runner = MetalGemmRunner::Create(EmitMetalKernels(tiling),
                                        tiling.tile_m, tiling.tile_n);
  if (!runner) {
    ADD_FAILURE("Metal runner creation failed: " + runner.error());
    return nullptr;
  }
  return std::move(*runner);
}

void ExpectClose(const std::vector<float>& got, const std::vector<float>& want) {
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    const float tol =
        2e-4f * (1.0f + std::fabs(want[i]));  // FMA-contraction slack
    EXPECT_NEAR(got[i], want[i], tol);
  }
}

TEST(MetalGemm, EmittedKernelsMatchCpuReference) {
  if (!MetalGemmRunner::Available()) {
    std::puts("  [ note ] no Metal device; GPU dispatch untested here");
    return;
  }
  auto runner = MakeRunner();
  ASSERT_TRUE(runner != nullptr);

  const std::vector<float> a = RandnVector(kM * kK, 201);
  const std::vector<float> b = RandnVector(kK * kN, 202);

  std::vector<float> want(kM * kN), got(kM * kN, -7.0f);
  k::GemmNN(a.data(), b.data(), want.data(), kM, kN, kK);
  ASSERT_OK(runner->Run(Kind::kNN, a.data(), b.data(), got.data(), kM, kN,
                        kK));
  ExpectClose(got, want);

  // NT: B is [N, K] row-major.
  const std::vector<float> bt = RandnVector(kN * kK, 203);
  k::GemmNT(a.data(), bt.data(), want.data(), kM, kN, kK);
  ASSERT_OK(runner->Run(Kind::kNT, a.data(), bt.data(), got.data(), kM, kN,
                        kK));
  ExpectClose(got, want);

  // TN: A is [K, M] row-major.
  const std::vector<float> at = RandnVector(kK * kM, 204);
  k::GemmTN(at.data(), b.data(), want.data(), kM, kN, kK);
  ASSERT_OK(runner->Run(Kind::kTN, at.data(), b.data(), got.data(), kM, kN,
                        kK));
  ExpectClose(got, want);

  // Accumulate: C += alpha * A@B over a nonzero C.
  const std::vector<float> c0 = RandnVector(kM * kN, 205);
  want = c0;
  k::GemmAccNN(a.data(), b.data(), want.data(), kM, kN, kK, 0.5f);
  got = c0;
  ASSERT_OK(runner->Run(Kind::kAccNN, a.data(), b.data(), got.data(), kM, kN,
                        kK, 0.5f));
  ExpectClose(got, want);
}

TEST(MetalGemm, DispatchIsReproducibleRunToRun) {
  if (!MetalGemmRunner::Available()) {
    std::puts("  [ note ] no Metal device; GPU dispatch untested here");
    return;
  }
  auto runner = MakeRunner();
  ASSERT_TRUE(runner != nullptr);
  const std::vector<float> a = RandnVector(kM * kK, 206);
  const std::vector<float> b = RandnVector(kK * kN, 207);
  std::vector<float> first(kM * kN), again(kM * kN);
  ASSERT_OK(runner->Run(Kind::kNN, a.data(), b.data(), first.data(), kM, kN,
                        kK));
  ASSERT_OK(runner->Run(Kind::kNN, a.data(), b.data(), again.data(), kM, kN,
                        kK));
  // Per-backend determinism: same device, same grid, same K order — the
  // bits must not wobble between dispatches.
  for (size_t i = 0; i < first.size(); ++i) EXPECT_EQ(first[i], again[i]);
}

}  // namespace

#else  // !__APPLE__

// Metal does not exist off Apple platforms. The runner treats an empty
// test selection as an error (exit 2), so one trivially-green test keeps
// the uniform cross-host suite list from failing where the platform simply
// has nothing to check.
namespace {
TEST(MetalGemm, UnsupportedPlatformIsVacuouslyGreen) { EXPECT_TRUE(true); }
}  // namespace

#endif
