// =============================================================================
// Reference-kernel tests: every kernel in the update runtime's dispatch table
// checked against hand-computed values or an independent naive formulation.
// These are the numerical ground truth the compiled plans execute on.
// =============================================================================

#include <cmath>
#include <cstdint>
#include <vector>

#include "runtime/update_kernels.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"

namespace {

namespace k = seeml::update_rt::kernels;
using seeml::testing::RandnVector;

// --- GEMM family ---------------------------------------------------------------

TEST(Gemm, NNMatchesHandComputedProduct) {
  // A[2,3] @ B[3,2]
  const std::vector<float> a = {1, 2, 3, 4, 5, 6};
  const std::vector<float> b = {7, 8, 9, 10, 11, 12};
  std::vector<float> c(4, -1.0f);  // pre-poisoned: GemmNN overwrites
  k::GemmNN(a.data(), b.data(), c.data(), 2, 2, 3);
  EXPECT_NEAR(c[0], 58.0f, 1e-6);   // 1*7 + 2*9 + 3*11
  EXPECT_NEAR(c[1], 64.0f, 1e-6);   // 1*8 + 2*10 + 3*12
  EXPECT_NEAR(c[2], 139.0f, 1e-6);  // 4*7 + 5*9 + 6*11
  EXPECT_NEAR(c[3], 154.0f, 1e-6);
}

TEST(Gemm, NTMatchesNNWithExplicitTranspose) {
  const size_t M = 3, N = 4, K = 5;
  const std::vector<float> a = RandnVector(M * K, 1);
  const std::vector<float> b = RandnVector(K * N, 2);  // [K, N]
  std::vector<float> b_t(N * K);                       // [N, K]
  for (size_t kk = 0; kk < K; ++kk)
    for (size_t n = 0; n < N; ++n) b_t[n * K + kk] = b[kk * N + n];

  std::vector<float> want(M * N), got(M * N);
  k::GemmNN(a.data(), b.data(), want.data(), M, N, K);
  k::GemmNT(a.data(), b_t.data(), got.data(), M, N, K);
  for (size_t i = 0; i < M * N; ++i) EXPECT_NEAR(got[i], want[i], 1e-5);
}

TEST(Gemm, TNMatchesNNWithExplicitTranspose) {
  const size_t M = 3, N = 4, K = 5;
  const std::vector<float> a = RandnVector(M * K, 3);    // [M, K] for NN
  const std::vector<float> b = RandnVector(K * N, 4);    // [K, N]
  std::vector<float> a_t(K * M);                         // [K, M]
  for (size_t m = 0; m < M; ++m)
    for (size_t kk = 0; kk < K; ++kk) a_t[kk * M + m] = a[m * K + kk];

  std::vector<float> want(M * N), got(M * N);
  k::GemmNN(a.data(), b.data(), want.data(), M, N, K);
  k::GemmTN(a_t.data(), b.data(), got.data(), M, N, K);
  for (size_t i = 0; i < M * N; ++i) EXPECT_NEAR(got[i], want[i], 1e-5);
}

TEST(Gemm, AccNNAccumulatesScaledProduct) {
  const std::vector<float> a = {1, 0, 0, 1};  // I2
  const std::vector<float> b = {5, 6, 7, 8};
  std::vector<float> c = {100, 200, 300, 400};
  k::GemmAccNN(a.data(), b.data(), c.data(), 2, 2, 2, 0.5f);
  EXPECT_NEAR(c[0], 102.5f, 1e-6);
  EXPECT_NEAR(c[1], 203.0f, 1e-6);
  EXPECT_NEAR(c[2], 303.5f, 1e-6);
  EXPECT_NEAR(c[3], 404.0f, 1e-6);
}

// --- Elementwise / broadcast ------------------------------------------------------

TEST(Elementwise, AddEW) {
  const std::vector<float> x = {1, 2, 3};
  const std::vector<float> y = {10, 20, 30};
  std::vector<float> out(3);
  k::AddEW(x.data(), y.data(), out.data(), 3);
  EXPECT_NEAR(out[0], 11.0f, 1e-6);
  EXPECT_NEAR(out[2], 33.0f, 1e-6);
}

TEST(Elementwise, AddBiasBroadcastsOverRows) {
  const std::vector<float> x = {1, 2, 3, 4, 5, 6};  // [2, 3]
  const std::vector<float> b = {10, 20, 30};
  std::vector<float> out(6);
  k::AddBias(x.data(), b.data(), out.data(), 2, 3);
  EXPECT_NEAR(out[0], 11.0f, 1e-6);
  EXPECT_NEAR(out[2], 33.0f, 1e-6);
  EXPECT_NEAR(out[3], 14.0f, 1e-6);
  EXPECT_NEAR(out[5], 36.0f, 1e-6);
}

TEST(Elementwise, ReluFwdClampsNegatives) {
  const std::vector<float> x = {-2, -0.5f, 0, 0.5f, 2};
  std::vector<float> out(5);
  k::ReluFwd(x.data(), out.data(), 5);
  EXPECT_NEAR(out[0], 0.0f, 0.0);
  EXPECT_NEAR(out[1], 0.0f, 0.0);
  EXPECT_NEAR(out[2], 0.0f, 0.0);
  EXPECT_NEAR(out[3], 0.5f, 0.0);
  EXPECT_NEAR(out[4], 2.0f, 0.0);
}

TEST(Elementwise, ReluBwdGatesOnPrimalInput) {
  const std::vector<float> dy = {1, 1, 1, 1};
  const std::vector<float> x = {-1, 0, 0.5f, 3};
  std::vector<float> dx(4);
  k::ReluBwd(dy.data(), x.data(), dx.data(), 4);
  EXPECT_NEAR(dx[0], 0.0f, 0.0);
  EXPECT_NEAR(dx[1], 0.0f, 0.0);  // subgradient 0 at the kink
  EXPECT_NEAR(dx[2], 1.0f, 0.0);
  EXPECT_NEAR(dx[3], 1.0f, 0.0);
}

TEST(Elementwise, Scale) {
  const std::vector<float> x = {1, -2, 4};
  std::vector<float> out(3);
  k::Scale(x.data(), out.data(), 2.5f, 3);
  EXPECT_NEAR(out[0], 2.5f, 1e-6);
  EXPECT_NEAR(out[1], -5.0f, 1e-6);
  EXPECT_NEAR(out[2], 10.0f, 1e-6);
}

TEST(Elementwise, ReduceRowsSumsColumns) {
  const std::vector<float> dy = {1, 2, 3, 10, 20, 30};  // [2, 3]
  std::vector<float> db(3, -99.0f);                     // overwritten
  k::ReduceRows(dy.data(), db.data(), 2, 3);
  EXPECT_NEAR(db[0], 11.0f, 1e-6);
  EXPECT_NEAR(db[1], 22.0f, 1e-6);
  EXPECT_NEAR(db[2], 33.0f, 1e-6);
}

// --- Softmax cross-entropy -----------------------------------------------------------

TEST(SoftmaxXEnt, UniformLogitsGiveLogC) {
  const size_t N = 2, C = 4;
  const std::vector<float> logits(N * C, 0.0f);
  const std::vector<int32_t> labels = {0, 3};
  std::vector<float> probs(N * C);
  float loss = 0.0f;
  k::SoftmaxXEntFwd(logits.data(), labels.data(), &loss, probs.data(), N, C);
  EXPECT_NEAR(loss, std::log(4.0), 1e-6);
  for (float p : probs) EXPECT_NEAR(p, 0.25f, 1e-6);
}

TEST(SoftmaxXEnt, ForwardIsShiftInvariantAndBoundedBelow) {
  const size_t N = 1, C = 3;
  const std::vector<float> logits = {1.0f, 2.0f, 3.0f};
  const std::vector<float> shifted = {101.0f, 102.0f, 103.0f};
  const std::vector<int32_t> labels = {2};
  std::vector<float> probs(C);
  float loss_a = 0.0f, loss_b = 0.0f;
  k::SoftmaxXEntFwd(logits.data(), labels.data(), &loss_a, probs.data(), N, C);
  k::SoftmaxXEntFwd(shifted.data(), labels.data(), &loss_b, probs.data(), N,
                    C);
  EXPECT_NEAR(loss_a, loss_b, 1e-6);  // numerically stable softmax
  EXPECT_GT(loss_a, 0.0f);
}

TEST(SoftmaxXEnt, BackwardIsScaledProbsMinusOneHot) {
  const size_t N = 2, C = 3;
  const std::vector<float> probs = {0.2f, 0.3f, 0.5f, 0.1f, 0.8f, 0.1f};
  const std::vector<int32_t> labels = {2, 1};
  const float seed = 2.0f;
  std::vector<float> dlogits(N * C);
  k::SoftmaxXEntBwd(probs.data(), labels.data(), &seed, dlogits.data(), N, C);

  // dlogits = seed * (probs - onehot) / N
  EXPECT_NEAR(dlogits[0], 0.2f, 1e-6);           // 2 * 0.2 / 2
  EXPECT_NEAR(dlogits[2], 0.5f - 1.0f, 1e-6);    // label hit
  EXPECT_NEAR(dlogits[4], 0.8f - 1.0f, 1e-6);
  // Each row of dlogits sums to zero (softmax gradient identity).
  EXPECT_NEAR(dlogits[0] + dlogits[1] + dlogits[2], 0.0f, 1e-6);
  EXPECT_NEAR(dlogits[3] + dlogits[4] + dlogits[5], 0.0f, 1e-6);
}

// --- MSE ----------------------------------------------------------------------------

TEST(Mse, ForwardIsMeanSquaredError) {
  const std::vector<float> pred = {1, 2};
  const std::vector<float> target = {0, 0};
  float loss = 0.0f;
  k::MseFwd(pred.data(), target.data(), &loss, 2);
  EXPECT_NEAR(loss, 2.5f, 1e-6);  // (1 + 4) / 2
}

TEST(Mse, BackwardIsScaledResidual) {
  const std::vector<float> pred = {1, 2};
  const std::vector<float> target = {0, 0};
  const float seed = 2.0f;
  std::vector<float> dpred(2);
  k::MseBwd(pred.data(), target.data(), &seed, dpred.data(), 2);
  // dpred = seed * 2 * (pred - target) / n = 2 * (pred - target)
  EXPECT_NEAR(dpred[0], 2.0f, 1e-6);
  EXPECT_NEAR(dpred[1], 4.0f, 1e-6);
}

// --- KL distillation -----------------------------------------------------------------

TEST(KLDistill, IdenticalLogitsGiveZeroLossAndGradient) {
  const size_t N = 2, C = 3;
  const std::vector<float> logits = {1, 2, 3, -1, 0, 1};
  std::vector<float> p_s(N * C), p_t(N * C);
  float loss = -1.0f;
  k::KLDistillFwd(logits.data(), logits.data(), &loss, p_s.data(), p_t.data(),
                  N, C, 2.0f);
  EXPECT_NEAR(loss, 0.0f, 1e-6);

  const float seed = 1.0f;
  std::vector<float> dlogits(N * C);
  k::KLDistillBwd(p_s.data(), p_t.data(), &seed, dlogits.data(), N, C, 2.0f);
  for (float d : dlogits) EXPECT_NEAR(d, 0.0f, 1e-7);
}

TEST(KLDistill, DivergentLogitsGivePositiveLossAndZeroSumRows) {
  const size_t N = 1, C = 3;
  const std::vector<float> s_logits = {2, 0, -1};
  const std::vector<float> t_logits = {0, 1, 0};
  std::vector<float> p_s(C), p_t(C);
  float loss = 0.0f;
  k::KLDistillFwd(s_logits.data(), t_logits.data(), &loss, p_s.data(),
                  p_t.data(), N, C, 1.0f);
  EXPECT_GT(loss, 0.0f);

  // Cached distributions are proper softmaxes.
  EXPECT_NEAR(p_s[0] + p_s[1] + p_s[2], 1.0f, 1e-6);
  EXPECT_NEAR(p_t[0] + p_t[1] + p_t[2], 1.0f, 1e-6);

  const float seed = 1.0f;
  std::vector<float> dlogits(C);
  k::KLDistillBwd(p_s.data(), p_t.data(), &seed, dlogits.data(), N, C, 1.0f);
  // dstudent = (p_s - p_t) / (N*T): rows sum to zero.
  EXPECT_NEAR(dlogits[0] + dlogits[1] + dlogits[2], 0.0f, 1e-6);
  EXPECT_NEAR(dlogits[0], p_s[0] - p_t[0], 1e-6);
}

TEST(KLDistill, TemperatureSoftensDistributions) {
  const size_t N = 1, C = 2;
  const std::vector<float> s_logits = {4, 0};
  const std::vector<float> t_logits = {0, 4};
  std::vector<float> p_s(C), p_t(C);
  float sharp = 0.0f, soft = 0.0f;
  k::KLDistillFwd(s_logits.data(), t_logits.data(), &sharp, p_s.data(),
                  p_t.data(), N, C, 1.0f);
  k::KLDistillFwd(s_logits.data(), t_logits.data(), &soft, p_s.data(),
                  p_t.data(), N, C, 8.0f);
  EXPECT_GT(sharp, soft);  // high temperature flattens the divergence
}

// --- Optimizers -----------------------------------------------------------------------

TEST(Optimizers, SgdStepWithDecoupledWeightDecay) {
  std::vector<float> p = {1.0f, -2.0f};
  const std::vector<float> g = {0.5f, 0.5f};
  k::SgdStep(p.data(), g.data(), 2, /*lr=*/0.1f, /*weight_decay=*/0.2f);
  // p -= lr * (g + wd * p)
  EXPECT_NEAR(p[0], 1.0f - 0.1f * (0.5f + 0.2f * 1.0f), 1e-6);
  EXPECT_NEAR(p[1], -2.0f - 0.1f * (0.5f + 0.2f * -2.0f), 1e-6);
}

TEST(Optimizers, AdamWFirstStepMatchesClosedForm) {
  const float lr = 0.01f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.1f;
  const float g0 = 0.4f, p0 = 2.0f;
  std::vector<float> p = {p0}, m = {0.0f}, v = {0.0f};
  const std::vector<float> g = {g0};
  k::AdamWStep(p.data(), g.data(), m.data(), v.data(), 1, lr, b1, b2, eps, wd,
               /*step=*/1);

  // Step 1 bias correction makes m_hat = g and v_hat = g^2 exactly.
  EXPECT_NEAR(m[0], (1.0f - b1) * g0, 1e-7);
  EXPECT_NEAR(v[0], (1.0f - b2) * g0 * g0, 1e-9);
  const float expected =
      p0 - lr * (g0 / (std::sqrt(g0 * g0) + eps) + wd * p0);
  EXPECT_NEAR(p[0], expected, 1e-6);
}

TEST(Optimizers, AdamWTracksIndependentReference) {
  // Two steps of AdamW against a from-scratch double-precision reference.
  const float lr = 0.05f, b1 = 0.9f, b2 = 0.99f, eps = 1e-8f, wd = 0.01f;
  std::vector<float> p = {1.5f}, m = {0.0f}, v = {0.0f};
  const std::vector<float> grads = {0.3f, -0.2f};

  double rp = 1.5, rm = 0.0, rv = 0.0;
  for (uint64_t step = 1; step <= 2; ++step) {
    const float gf = grads[step - 1];
    k::AdamWStep(p.data(), &gf, m.data(), v.data(), 1, lr, b1, b2, eps, wd,
                 step);

    const double gd = gf;
    rm = b1 * rm + (1.0 - b1) * gd;
    rv = b2 * rv + (1.0 - b2) * gd * gd;
    const double mh = rm / (1.0 - std::pow(b1, static_cast<double>(step)));
    const double vh = rv / (1.0 - std::pow(b2, static_cast<double>(step)));
    rp -= lr * (mh / (std::sqrt(vh) + eps) + wd * rp);
  }
  EXPECT_NEAR(p[0], rp, 1e-5);
}

// --- Utility -------------------------------------------------------------------------

TEST(Utility, FillAndCopy) {
  std::vector<float> dst(4, 0.0f);
  k::Fill(dst.data(), 3.5f, 4);
  for (float x : dst) EXPECT_NEAR(x, 3.5f, 0.0);

  const std::vector<float> src = {1, 2, 3, 4};
  k::Copy(src.data(), dst.data(), 4);
  EXPECT_NEAR(dst[0], 1.0f, 0.0);
  EXPECT_NEAR(dst[3], 4.0f, 0.0);
}

}  // namespace
