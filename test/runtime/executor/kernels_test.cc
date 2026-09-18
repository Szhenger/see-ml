// =============================================================================
// Reference-kernel tests: every kernel in the update runtime's dispatch table
// checked against hand-computed values or an independent naive formulation.
// These are the numerical ground truth the compiled plans execute on.
// =============================================================================

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "runtime/executor/update_kernels.h"
#include "source/parallel/parallel_for.h"
#include "source/plan/bf16.h"
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

TEST(Gemm, EpilogueIsBitwiseIdenticalToUnfusedSequence) {
  // The fused write-back must evaluate exactly the floats the standalone
  // kGemmNN + kAddBias + k<Act>Fwd sequence would — fusion is a bandwidth
  // optimization, never a numerics change. Exact equality, not tolerance.
  using seeml::update::EpilogueAct;
  const size_t M = 5, N = 7, K = 6;
  const std::vector<float> a = RandnVector(M * K, 11);
  const std::vector<float> b = RandnVector(K * N, 12);
  const std::vector<float> bias = RandnVector(N, 13);

  const struct {
    EpilogueAct act;
    void (*fwd)(const float*, float*, size_t);
  } acts[] = {
      {EpilogueAct::kNone, nullptr},
      {EpilogueAct::kRelu, k::ReluFwd},
      {EpilogueAct::kGelu, k::GeluFwd},
      {EpilogueAct::kSilu, k::SiluFwd},
  };
  for (const auto& [act, fwd] : acts) {
    // Distinct buffers per stage: the standalone kernels carry the
    // no-overlap restrict contract the arena binder guarantees.
    std::vector<float> raw(M * N), summed(M * N), want(M * N), got(M * N);
    k::GemmNN(a.data(), b.data(), raw.data(), M, N, K);
    k::AddBias(raw.data(), bias.data(), summed.data(), M, N);
    if (fwd) fwd(summed.data(), want.data(), M * N);
    else want = summed;

    k::GemmNN(a.data(), b.data(), got.data(), M, N, K, bias.data(), act);
    for (size_t i = 0; i < M * N; ++i) EXPECT_EQ(got[i], want[i]);
  }
}

TEST(Gemm, ActOnlyEpilogueSkipsBias) {
  using seeml::update::EpilogueAct;
  const size_t M = 3, N = 5, K = 4;
  const std::vector<float> a = RandnVector(M * K, 21);
  const std::vector<float> b = RandnVector(K * N, 22);
  std::vector<float> raw(M * N), want(M * N), got(M * N);
  k::GemmNN(a.data(), b.data(), raw.data(), M, N, K);
  k::SiluFwd(raw.data(), want.data(), M * N);
  k::GemmNN(a.data(), b.data(), got.data(), M, N, K, nullptr,
            EpilogueAct::kSilu);
  for (size_t i = 0; i < M * N; ++i) EXPECT_EQ(got[i], want[i]);
}

TEST(Gemm, Q8ActEpilogueMatchesUnfusedSequence) {
  using seeml::update::EpilogueAct;
  const size_t M = 4, N = 6, K = 5;
  const std::vector<float> a = RandnVector(M * K, 31);
  std::vector<int8_t> bq(K * N);
  for (size_t i = 0; i < bq.size(); ++i)
    bq[i] = static_cast<int8_t>((static_cast<int>(i * 37) % 255) - 127);
  const float scale = 0.033f;

  std::vector<float> raw(M * N), want(M * N), got(M * N);
  k::GemmNNQ8(a.data(), bq.data(), raw.data(), M, N, K, scale);
  k::ReluFwd(raw.data(), want.data(), M * N);
  k::GemmNNQ8(a.data(), bq.data(), got.data(), M, N, K, scale,
              EpilogueAct::kRelu);
  for (size_t i = 0; i < M * N; ++i) EXPECT_EQ(got[i], want[i]);
}

TEST(Gemm, Bf16VariantsAreBitwiseTheF32GemmsOverTheWidenedMatrix) {
  using seeml::update::Bf16BitsToFloat32;
  using seeml::update::EpilogueAct;
  using seeml::update::Float32ToBf16Bits;
  const size_t M = 37, N = 259, K = 21;  // ragged, tile-crossing
  const std::vector<float> a = RandnVector(M * K, 71);
  const std::vector<float> b = RandnVector(K * N, 72);
  const std::vector<float> bt = RandnVector(N * K, 73);
  const std::vector<float> bias = RandnVector(N, 74);
  std::vector<uint16_t> bh(b.size()), bth(bt.size());
  std::vector<float> bw(b.size()), btw(bt.size());  // the widened matrices
  for (size_t i = 0; i < b.size(); ++i) {
    bh[i] = Float32ToBf16Bits(b[i]);
    bw[i] = Bf16BitsToFloat32(bh[i]);
    EXPECT_NEAR(bw[i], b[i], 4e-3 * std::fabs(b[i]) + 1e-38);  // 8-bit mantissa
  }
  for (size_t i = 0; i < bt.size(); ++i) {
    bth[i] = Float32ToBf16Bits(bt[i]);
    btw[i] = Bf16BitsToFloat32(bth[i]);
  }
  std::vector<float> want(M * N), got(M * N);
  k::GemmNN(a.data(), bw.data(), want.data(), M, N, K, bias.data(),
            EpilogueAct::kGelu);
  k::GemmNNBF16(a.data(), bh.data(), got.data(), M, N, K, bias.data(),
                EpilogueAct::kGelu);
  for (size_t i = 0; i < M * N; ++i) EXPECT_EQ(got[i], want[i]);
  k::GemmNT(a.data(), btw.data(), want.data(), M, N, K);
  k::GemmNTBF16(a.data(), bth.data(), got.data(), M, N, K);
  for (size_t i = 0; i < M * N; ++i) EXPECT_EQ(got[i], want[i]);
  // Round-to-nearest-even and NaN quieting, spot-checked.
  EXPECT_EQ(Float32ToBf16Bits(1.0f), 0x3F80u);
  EXPECT_EQ(Bf16BitsToFloat32(Float32ToBf16Bits(1.00390625f)), 1.0f);  // tie -> even
  EXPECT_EQ(Bf16BitsToFloat32(Float32ToBf16Bits(1.01171875f)), 1.015625f);
  EXPECT_TRUE(std::isnan(Bf16BitsToFloat32(Float32ToBf16Bits(
      std::numeric_limits<float>::quiet_NaN()))));
  // Infinities stay infinite; the largest finite f32 rounds up to +inf
  // (the standard overflow of round-to-nearest-even narrowing), never NaN.
  EXPECT_EQ(Float32ToBf16Bits(std::numeric_limits<float>::infinity()), 0x7F80u);
  EXPECT_EQ(Float32ToBf16Bits(-std::numeric_limits<float>::infinity()), 0xFF80u);
  EXPECT_EQ(Float32ToBf16Bits(std::numeric_limits<float>::max()), 0x7F80u);
  EXPECT_EQ(Float32ToBf16Bits(-0.0f), 0x8000u);
  // Negative values round by magnitude (sign-magnitude representation).
  EXPECT_EQ(Bf16BitsToFloat32(Float32ToBf16Bits(-1.01171875f)), -1.015625f);
  EXPECT_EQ(Bf16BitsToFloat32(Float32ToBf16Bits(-1.00390625f)), -1.0f);
  // A subnormal whose rounding carries into the exponent: just below the
  // smallest normal rounds up to it.
  const float below_min = std::bit_cast<float>(0x007FFFFFu);
  EXPECT_EQ(Float32ToBf16Bits(below_min), 0x0080u);
  EXPECT_EQ(Bf16BitsToFloat32(0x0080u), std::numeric_limits<float>::min());
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
                  N, C, 2.0f, 4.0f);
  EXPECT_NEAR(loss, 0.0f, 1e-6);

  const float seed = 1.0f;
  std::vector<float> dlogits(N * C);
  k::KLDistillBwd(p_s.data(), p_t.data(), &seed, dlogits.data(), N, C, 2.0f,
                  4.0f);
  for (float d : dlogits) EXPECT_NEAR(d, 0.0f, 1e-7);
}

TEST(KLDistill, DivergentLogitsGivePositiveLossAndZeroSumRows) {
  const size_t N = 1, C = 3;
  const std::vector<float> s_logits = {2, 0, -1};
  const std::vector<float> t_logits = {0, 1, 0};
  std::vector<float> p_s(C), p_t(C);
  float loss = 0.0f;
  k::KLDistillFwd(s_logits.data(), t_logits.data(), &loss, p_s.data(),
                  p_t.data(), N, C, 1.0f, 1.0f);
  EXPECT_GT(loss, 0.0f);

  // Cached distributions are proper softmaxes.
  EXPECT_NEAR(p_s[0] + p_s[1] + p_s[2], 1.0f, 1e-6);
  EXPECT_NEAR(p_t[0] + p_t[1] + p_t[2], 1.0f, 1e-6);

  const float seed = 1.0f;
  std::vector<float> dlogits(C);
  k::KLDistillBwd(p_s.data(), p_t.data(), &seed, dlogits.data(), N, C, 1.0f,
                  1.0f);
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
  // Unscaled (loss_scale = 1): the divergence itself flattens with T.
  k::KLDistillFwd(s_logits.data(), t_logits.data(), &sharp, p_s.data(),
                  p_t.data(), N, C, 1.0f, 1.0f);
  k::KLDistillFwd(s_logits.data(), t_logits.data(), &soft, p_s.data(),
                  p_t.data(), N, C, 8.0f, 1.0f);
  EXPECT_GT(sharp, soft);  // high temperature flattens the divergence
}

TEST(KLDistill, LossScaleIsHintonTemperatureSquared) {
  // loss(T, scale) == scale * loss(T, 1) exactly (one f32 multiply), and
  // with scale = T^2 the gradient is T*(p_s - p_t)/N — the convention the
  // compiler emits (#13), commensurate with a hard-label term at any T.
  const size_t N = 2, C = 3;
  const std::vector<float> s_logits = {2, 0, -1, 0.5f, 0.5f, -2};
  const std::vector<float> t_logits = {0, 1, 0, -1, 2, 0};
  const float T = 2.0f;
  std::vector<float> p_s(N * C), p_t(N * C);
  float plain = 0.0f, scaled = 0.0f;
  k::KLDistillFwd(s_logits.data(), t_logits.data(), &plain, p_s.data(),
                  p_t.data(), N, C, T, 1.0f);
  k::KLDistillFwd(s_logits.data(), t_logits.data(), &scaled, p_s.data(),
                  p_t.data(), N, C, T, T * T);
  EXPECT_EQ(scaled, plain * (T * T));

  const float seed = 1.0f;
  std::vector<float> dlogits(N * C);
  k::KLDistillBwd(p_s.data(), p_t.data(), &seed, dlogits.data(), N, C, T,
                  T * T);
  for (size_t i = 0; i < N * C; ++i)
    EXPECT_NEAR(dlogits[i], T * (p_s[i] - p_t[i]) / static_cast<float>(N),
                1e-6);
}

TEST(KLDistill, ScaledGradientMatchesFiniteDifferences) {
  // d loss / d s_logit at T = 2 with the T^2 scale, central differences in
  // double against the f32 kernel pair (G13's ask: the kernel-level FD
  // check the sign/monotonicity tests never gave this loss).
  const size_t N = 2, C = 4;
  std::vector<float> s_logits = {1.5f, -0.5f, 0.25f, 2.0f,
                                 -1.0f, 0.0f, 0.75f, 0.5f};
  const std::vector<float> t_logits = {0.0f, 1.0f, -1.0f, 0.5f,
                                       2.0f, -0.5f, 0.0f, 1.0f};
  const float T = 2.0f, scale = T * T;
  std::vector<float> p_s(N * C), p_t(N * C), dlogits(N * C);
  float loss = 0.0f;
  const float seed = 1.0f;
  k::KLDistillFwd(s_logits.data(), t_logits.data(), &loss, p_s.data(),
                  p_t.data(), N, C, T, scale);
  k::KLDistillBwd(p_s.data(), p_t.data(), &seed, dlogits.data(), N, C, T,
                  scale);
  const float eps = 1e-2f;
  for (size_t i = 0; i < N * C; ++i) {
    const float saved = s_logits[i];
    float plus = 0.0f, minus = 0.0f;
    s_logits[i] = saved + eps;
    k::KLDistillFwd(s_logits.data(), t_logits.data(), &plus, p_s.data(),
                    p_t.data(), N, C, T, scale);
    s_logits[i] = saved - eps;
    k::KLDistillFwd(s_logits.data(), t_logits.data(), &minus, p_s.data(),
                    p_t.data(), N, C, T, scale);
    s_logits[i] = saved;
    const double numeric = (static_cast<double>(plus) - minus) / (2.0 * eps);
    EXPECT_NEAR(dlogits[i], numeric, 2e-4);
  }
}

// --- Optimizers -----------------------------------------------------------------------

TEST(Optimizers, SgdStepWithDecoupledWeightDecay) {
  std::vector<float> p = {1.0f, -2.0f};
  const std::vector<float> g = {0.5f, 0.5f};
  EXPECT_TRUE(k::SgdStep(p.data(), g.data(), 2, /*lr=*/0.1f, /*weight_decay=*/0.2f));
  // p -= lr * (g + wd * p)
  EXPECT_NEAR(p[0], 1.0f - 0.1f * (0.5f + 0.2f * 1.0f), 1e-6);
  EXPECT_NEAR(p[1], -2.0f - 0.1f * (0.5f + 0.2f * -2.0f), 1e-6);
}

TEST(Optimizers, AdamWFirstStepMatchesClosedForm) {
  const float lr = 0.01f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.1f;
  const float g0 = 0.4f, p0 = 2.0f;
  std::vector<float> p = {p0}, m = {0.0f}, v = {0.0f};
  const std::vector<float> g = {g0};
  EXPECT_TRUE(k::AdamWStep(p.data(), g.data(), m.data(), v.data(), 1, lr, b1, b2, eps, wd,
               /*step=*/1));

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
    EXPECT_TRUE(k::AdamWStep(p.data(), &gf, m.data(), v.data(), 1, lr, b1, b2, eps, wd,
                 step));

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

// --- Parallel execution: bitwise determinism across thread counts -------------
//
// The kernels partition work into chunks whose geometry depends only on the
// problem shape, so serial and wide executions must produce identical BITS —
// not merely close values. Shapes below are chosen large enough that the
// wide runs actually fan out (multiple chunks per kernel).

/// Pins the pool width for one test; restores automatic resolution after.
struct ScopedThreads {
  explicit ScopedThreads(size_t n) {
    seeml::update::SetParallelThreadCount(n);
  }
  ~ScopedThreads() { seeml::update::SetParallelThreadCount(0); }
};

template <typename Fn>
std::vector<float> RunAtWidth(size_t threads, size_t out_size, Fn&& fill) {
  ScopedThreads scoped(threads);
  std::vector<float> out(out_size, 0.0f);
  fill(out.data());
  return out;
}

#define EXPECT_BITWISE_EQ_F32(a, b)                                       \
  do {                                                                    \
    const std::vector<float> bitwise_a_ = (a);                            \
    const std::vector<float> bitwise_b_ = (b);                            \
    ASSERT_EQ(bitwise_a_.size(), bitwise_b_.size());                      \
    EXPECT_EQ(std::memcmp(bitwise_a_.data(), bitwise_b_.data(),           \
                          bitwise_a_.size() * sizeof(float)),             \
              0);                                                         \
  } while (0)

TEST(ParallelDeterminism, GemmNtTailPathsAreThreadCountInvariant) {
  // #66's eight-lane NT core: K = 21 runs the 8-lane main loop AND the
  // k % 8 tail, N = 259 runs the 4-column path AND the 1-column path across
  // the kTileN boundary (256), M = 5 gives one ragged row chunk. Both NT
  // variants must be bit-identical at 1 and 8 threads — the lane rule is a
  // pure function of K.
  const size_t M = 5, N = 259, K = 21;
  const std::vector<float> a = RandnVector(M * K, 15);
  const std::vector<float> bt = RandnVector(N * K, 16);
  std::vector<int8_t> q8(N * K);
  for (size_t i = 0; i < q8.size(); ++i)
    q8[i] = static_cast<int8_t>((i * 53 + 7) % 255 - 127);
  auto nt = [&](float* c) { k::GemmNT(a.data(), bt.data(), c, M, N, K); };
  auto ntq8 = [&](float* c) {
    k::GemmNTQ8(a.data(), q8.data(), c, M, N, K, 0.02f);
  };
  EXPECT_BITWISE_EQ_F32(RunAtWidth(1, M * N, nt), RunAtWidth(8, M * N, nt));
  EXPECT_BITWISE_EQ_F32(RunAtWidth(1, M * N, ntq8),
                        RunAtWidth(8, M * N, ntq8));
  // And the values are right: against a double reference with dequant.
  const std::vector<float> got = RunAtWidth(1, M * N, ntq8);
  for (size_t m = 0; m < M; ++m)
    for (size_t n = 0; n < N; ++n) {
      double ref = 0.0;
      for (size_t kk = 0; kk < K; ++kk)
        ref += static_cast<double>(a[m * K + kk]) * q8[n * K + kk];
      EXPECT_NEAR(got[m * N + n], 0.02 * ref, 1e-4 * (1.0 + std::fabs(ref)));
    }
}

TEST(Gemm, Q8NtIsBitwiseTheF32NtOverTheWidenedMatrixTimesScale) {
  // #104: GemmNTQ8 widens each eight-wide block of B before the lane loop
  // (so the int8 form vectorizes like the f32 one). Widening is exact, the
  // lane rule and combine order are shared, and the scale is one final
  // multiply — so the q8 kernel must equal scale * GemmNT over the widened
  // matrix bit for bit, at 1 and 8 threads. K = 21 runs the 8-lane loop and
  // the tail; N = 259 runs the 4-column and 1-column paths across the
  // kTileN boundary; M = 37 gives ragged row chunks.
  const size_t M = 37, N = 259, K = 21;
  const float scale = 0.0123f;
  const std::vector<float> a = RandnVector(M * K, 75);
  std::vector<int8_t> q8(N * K);
  std::vector<float> widened(N * K);
  for (size_t i = 0; i < q8.size(); ++i) {
    q8[i] = static_cast<int8_t>((i * 97 + 13) % 255 - 127);
    widened[i] = static_cast<float>(q8[i]);
  }
  for (const size_t threads : {size_t{1}, size_t{8}}) {
    ScopedThreads scoped(threads);
    std::vector<float> want(M * N), got(M * N);
    k::GemmNT(a.data(), widened.data(), want.data(), M, N, K);
    for (float& w : want) w = scale * w;
    k::GemmNTQ8(a.data(), q8.data(), got.data(), M, N, K, scale);
    EXPECT_EQ(std::memcmp(got.data(), want.data(), M * N * sizeof(float)), 0);
  }
}

TEST(ParallelDeterminism, GemmFamilyIsThreadCountInvariant) {
  const size_t M = 64, N = 96, K = 48;
  const std::vector<float> a = RandnVector(M * K, 11);
  const std::vector<float> b = RandnVector(K * N, 12);
  const std::vector<float> bt = RandnVector(N * K, 13);
  const std::vector<float> at = RandnVector(K * M, 14);
  std::vector<int8_t> q8(K * N);
  for (size_t i = 0; i < q8.size(); ++i)
    q8[i] = static_cast<int8_t>((i * 37 + 11) % 255 - 127);

  auto nn = [&](float* c) { k::GemmNN(a.data(), b.data(), c, M, N, K); };
  auto nt = [&](float* c) { k::GemmNT(a.data(), bt.data(), c, M, N, K); };
  auto tn = [&](float* c) { k::GemmTN(at.data(), b.data(), c, M, N, K); };
  auto acc = [&](float* c) {
    k::Fill(c, 0.25f, M * N);
    k::GemmAccNN(a.data(), b.data(), c, M, N, K, 0.5f);
  };
  auto nnq8 = [&](float* c) {
    k::GemmNNQ8(a.data(), q8.data(), c, M, N, K, 0.01f);
  };
  EXPECT_BITWISE_EQ_F32(RunAtWidth(1, M * N, nn), RunAtWidth(8, M * N, nn));
  EXPECT_BITWISE_EQ_F32(RunAtWidth(1, M * N, nt), RunAtWidth(8, M * N, nt));
  EXPECT_BITWISE_EQ_F32(RunAtWidth(1, M * N, tn), RunAtWidth(8, M * N, tn));
  EXPECT_BITWISE_EQ_F32(RunAtWidth(1, M * N, acc), RunAtWidth(8, M * N, acc));
  EXPECT_BITWISE_EQ_F32(RunAtWidth(1, M * N, nnq8),
                        RunAtWidth(8, M * N, nnq8));
}

TEST(ParallelDeterminism, GemmTilesNeverChangeBits) {
  // The tile geometry (plan v11, tool/autotune.py's knob) picks traversal
  // order and the K panel a pass folds in; with the K tile a multiple of
  // the 4-wide unroll the per-element expression is identical under every
  // geometry — the contract that lets a measured table pick any of them.
  // Ragged M/N/K cross every tile boundary the candidate set has.
  const size_t M = 37, N = 517, K = 131;
  const std::vector<float> a = RandnVector(M * K, 21);
  const std::vector<float> b = RandnVector(K * N, 22);
  const std::vector<float> bt = RandnVector(N * K, 23);
  const std::vector<float> at = RandnVector(K * M, 24);
  const std::vector<float> bias = RandnVector(N, 25);
  std::vector<int8_t> q8(K * N), q8t(N * K);
  std::vector<uint16_t> bh(K * N), bth(N * K);
  for (size_t i = 0; i < q8.size(); ++i) {
    q8[i] = static_cast<int8_t>((i * 37 + 11) % 255 - 127);
    q8t[i] = static_cast<int8_t>((i * 53 + 7) % 255 - 127);
    bh[i] = seeml::update::Float32ToBf16Bits(b[i]);
    bth[i] = seeml::update::Float32ToBf16Bits(bt[i]);
  }
  using seeml::update::EpilogueAct;
  const k::GemmTiles candidates[] = {{64, 256}, {4, 1}, {16, 64}, {128, 4},
                                     {512, 16}, {256, 1024}, {8, 517}};
  auto run = [&](const k::GemmTiles& t) {
    std::vector<float> out;
    auto push = [&](const std::vector<float>& c) {
      out.insert(out.end(), c.begin(), c.end());
    };
    std::vector<float> c(M * N);
    k::GemmNN(a.data(), b.data(), c.data(), M, N, K, bias.data(),
              EpilogueAct::kGelu, t);
    push(c);
    k::GemmNT(a.data(), bt.data(), c.data(), M, N, K, t);
    push(c);
    k::GemmTN(at.data(), b.data(), c.data(), M, N, K, t);
    push(c);
    k::Fill(c.data(), 0.25f, M * N);
    k::GemmAccNN(a.data(), b.data(), c.data(), M, N, K, 0.5f, t);
    push(c);
    k::GemmNNQ8(a.data(), q8.data(), c.data(), M, N, K, 0.01f,
                EpilogueAct::kSilu, t);
    push(c);
    k::GemmNTQ8(a.data(), q8t.data(), c.data(), M, N, K, 0.02f, t);
    push(c);
    k::GemmNNBF16(a.data(), bh.data(), c.data(), M, N, K, bias.data(),
                  EpilogueAct::kRelu, t);
    push(c);
    k::GemmNTBF16(a.data(), bth.data(), c.data(), M, N, K, t);
    push(c);
    return out;
  };
  const std::vector<float> reference = run(k::kDefaultGemmTiles);
  for (const k::GemmTiles& t : candidates) {
    EXPECT_TRUE(k::GemmTilesValid(t));
    EXPECT_BITWISE_EQ_F32(run(t), reference);
    ScopedThreads eight(8);
    EXPECT_BITWISE_EQ_F32(run(t), reference);
  }
  EXPECT_FALSE(k::GemmTilesValid({6, 16}));
  EXPECT_FALSE(k::GemmTilesValid({0, 16}));
  EXPECT_FALSE(k::GemmTilesValid({64, 0}));
}

TEST(ParallelDeterminism, LossReductionsAreThreadCountInvariant) {
  const size_t N = 2048, C = 8;
  const std::vector<float> logits = RandnVector(N * C, 21);
  const std::vector<float> targets = RandnVector(N * C, 22);
  std::vector<int32_t> labels(N);
  for (size_t i = 0; i < N; ++i) labels[i] = static_cast<int32_t>(i % C);

  auto xent = [&](float* out) {
    std::vector<float> probs(N * C);
    k::SoftmaxXEntFwd(logits.data(), labels.data(), out, probs.data(), N, C);
    std::memcpy(out + 1, probs.data(),
                std::min<size_t>(N * C, 63) * sizeof(float));
  };
  auto mse = [&](float* out) {
    k::MseFwd(logits.data(), targets.data(), out, N * C);
  };
  auto kl = [&](float* out) {
    std::vector<float> p_s(N * C), p_t(N * C);
    k::KLDistillFwd(logits.data(), targets.data(), out, p_s.data(),
                    p_t.data(), N, C, 2.0f, 4.0f);
  };
  EXPECT_BITWISE_EQ_F32(RunAtWidth(1, 64, xent), RunAtWidth(8, 64, xent));
  EXPECT_BITWISE_EQ_F32(RunAtWidth(1, 1, mse), RunAtWidth(8, 1, mse));
  EXPECT_BITWISE_EQ_F32(RunAtWidth(1, 1, kl), RunAtWidth(8, 1, kl));
}

TEST(ParallelDeterminism, StatefulKernelsAreThreadCountInvariant) {
  const size_t n = 100'000;
  const std::vector<float> g = RandnVector(n, 31);
  const std::vector<float> p0 = RandnVector(n, 32);
  const std::vector<float> m0 = RandnVector(n, 33, 0.1f);
  std::vector<float> v0(n);
  for (size_t i = 0; i < n; ++i) v0[i] = std::fabs(m0[i]) + 0.01f;

  auto adamw = [&](float* out) {
    std::vector<float> p = p0, m = m0, v = v0;
    EXPECT_TRUE(k::AdamWStep(p.data(), g.data(), m.data(), v.data(), n, 0.01f, 0.9f,
                 0.999f, 1e-8f, 0.01f, 3));
    std::memcpy(out, p.data(), n * sizeof(float));
  };
  auto clip = [&](float* out) {
    std::memcpy(out, g.data(), n * sizeof(float));
    k::ClipNorm(out, n, 1.0f);  // ||g|| >> 1 for n randn values: rescales
  };
  const size_t rows = 64, cols = 2048;
  const std::vector<float> dy = RandnVector(rows * cols, 34);
  auto reduce = [&](float* out) {
    k::ReduceRows(dy.data(), out, rows, cols);
  };
  EXPECT_BITWISE_EQ_F32(RunAtWidth(1, n, adamw), RunAtWidth(8, n, adamw));
  EXPECT_BITWISE_EQ_F32(RunAtWidth(1, n, clip), RunAtWidth(8, n, clip));
  EXPECT_BITWISE_EQ_F32(RunAtWidth(1, 2048, reduce),
                        RunAtWidth(8, 2048, reduce));
}

TEST(ParallelDeterminism, RowAndElementwiseKernelsAreThreadCountInvariant) {
  const size_t rows = 512, cols = 96;
  const size_t n = rows * cols;
  const std::vector<float> x = RandnVector(n, 41);
  const std::vector<float> dy = RandnVector(n, 42);
  const std::vector<float> gamma = RandnVector(cols, 43);
  const std::vector<float> beta = RandnVector(cols, 44);

  auto layernorm = [&](float* out) {
    std::vector<float> mean(rows), rstd(rows), y(n), dx(n);
    k::LayerNormFwd(x.data(), gamma.data(), beta.data(), y.data(),
                    mean.data(), rstd.data(), rows, cols);
    k::LayerNormBwd(dy.data(), x.data(), gamma.data(), mean.data(),
                    rstd.data(), dx.data(), rows, cols);
    std::memcpy(out, y.data(), n * sizeof(float));
    std::memcpy(out + n, dx.data(), n * sizeof(float));
  };
  auto gelu = [&](float* out) {
    k::GeluFwd(x.data(), out, n);
    k::GeluBwd(dy.data(), x.data(), out + n, n);
  };
  auto silu = [&](float* out) {
    k::SiluFwd(x.data(), out, n);
    k::SiluBwd(dy.data(), x.data(), out + n, n);
  };
  EXPECT_BITWISE_EQ_F32(RunAtWidth(1, 2 * n, layernorm),
                        RunAtWidth(8, 2 * n, layernorm));
  EXPECT_BITWISE_EQ_F32(RunAtWidth(1, 2 * n, gelu),
                        RunAtWidth(8, 2 * n, gelu));
  EXPECT_BITWISE_EQ_F32(RunAtWidth(1, 2 * n, silu),
                        RunAtWidth(8, 2 * n, silu));
}

// --- Transformer family (plan v6) ---------------------------------------------

TEST(RmsNorm, ForwardMatchesNaiveFormula) {
  const size_t rows = 3, cols = 5;
  const std::vector<float> x = RandnVector(rows * cols, 60);
  const std::vector<float> gamma = RandnVector(cols, 61);
  std::vector<float> y(rows * cols), rstd(rows);
  k::RmsNormFwd(x.data(), gamma.data(), y.data(), rstd.data(), rows, cols);
  for (size_t r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (size_t c = 0; c < cols; ++c)
      ss += static_cast<double>(x[r * cols + c]) * x[r * cols + c];
    const double rs = 1.0 / std::sqrt(ss / cols + 1e-5);
    EXPECT_NEAR(rstd[r], static_cast<float>(rs), 1e-5);
    for (size_t c = 0; c < cols; ++c)
      EXPECT_NEAR(y[r * cols + c],
                  static_cast<float>(x[r * cols + c] * rs * gamma[c]), 1e-5);
  }
}

TEST(RmsNorm, BackwardMatchesFiniteDifferences) {
  const size_t rows = 2, cols = 4, n = rows * cols;
  const std::vector<float> x = RandnVector(n, 62);
  const std::vector<float> gamma = RandnVector(cols, 63);
  const std::vector<float> dy = RandnVector(n, 64);
  std::vector<float> y(n), rstd(rows), dx(n);
  k::RmsNormFwd(x.data(), gamma.data(), y.data(), rstd.data(), rows, cols);
  k::RmsNormBwd(dy.data(), x.data(), gamma.data(), rstd.data(), dx.data(),
                rows, cols);
  // L = sum(dy * y): dL/dx_i must match the central difference.
  const double eps = 1e-3;
  for (size_t i = 0; i < n; ++i) {
    std::vector<float> xp = x, xm = x;
    xp[i] += static_cast<float>(eps);
    xm[i] -= static_cast<float>(eps);
    std::vector<float> yp(n), ym(n), tmp(rows);
    k::RmsNormFwd(xp.data(), gamma.data(), yp.data(), tmp.data(), rows, cols);
    k::RmsNormFwd(xm.data(), gamma.data(), ym.data(), tmp.data(), rows, cols);
    double lp = 0.0, lm = 0.0;
    for (size_t j = 0; j < n; ++j) {
      lp += static_cast<double>(dy[j]) * yp[j];
      lm += static_cast<double>(dy[j]) * ym[j];
    }
    EXPECT_NEAR(dx[i], static_cast<float>((lp - lm) / (2 * eps)), 2e-3);
  }
}

TEST(Rope, BackwardIsTheExactAdjointOfForward) {
  // For an orthogonal per-pair rotation R: <R x, y> == <x, R^T y> for all
  // x, y — the defining property of the VJP the backward kernel implements.
  const size_t B = 2, S = 3, H = 2, d = 4, n = B * S * H * d;
  const std::vector<float> x = RandnVector(n, 70);
  const std::vector<float> y = RandnVector(n, 71);
  std::vector<float> rx(n), rty(n);
  k::RopeFwd(x.data(), rx.data(), B, S, H, d, 10000.0f);
  k::RopeBwd(y.data(), rty.data(), B, S, H, d, 10000.0f);
  double lhs = 0.0, rhs = 0.0;
  for (size_t i = 0; i < n; ++i) {
    lhs += static_cast<double>(rx[i]) * y[i];
    rhs += static_cast<double>(x[i]) * rty[i];
  }
  EXPECT_NEAR(static_cast<float>(lhs), static_cast<float>(rhs), 1e-4);
  // Position 0 rotates by angle 0: the first row of every sequence is
  // passed through bit-for-bit.
  for (size_t b = 0; b < B; ++b)
    for (size_t c = 0; c < H * d; ++c)
      EXPECT_EQ(rx[b * S * H * d + c], x[b * S * H * d + c]);
}

TEST(Attention, ForwardMatchesNaiveReferenceAndIsCausal) {
  const size_t B = 2, S = 3, H = 2, d = 2, D = H * d;
  const std::vector<float> q = RandnVector(B * S * D, 80);
  const std::vector<float> k2 = RandnVector(B * S * D, 81);
  const std::vector<float> v = RandnVector(B * S * D, 82);
  std::vector<float> o(B * S * D), p(B * H * S * S);
  k::AttnFwd(q.data(), k2.data(), v.data(), o.data(), p.data(), B, S, H, d);

  const double inv_sqrt_d = 1.0 / std::sqrt(static_cast<double>(d));
  for (size_t b = 0; b < B; ++b)
    for (size_t h = 0; h < H; ++h)
      for (size_t i = 0; i < S; ++i) {
        // Softmax over the causal prefix, computed independently.
        std::vector<double> e(S, 0.0);
        double denom = 0.0;
        for (size_t j = 0; j <= i; ++j) {
          double dot = 0.0;
          for (size_t c = 0; c < d; ++c)
            dot += static_cast<double>(q[(b * S + i) * D + h * d + c]) *
                   k2[(b * S + j) * D + h * d + c];
          e[j] = std::exp(dot * inv_sqrt_d);
          denom += e[j];
        }
        for (size_t j = 0; j <= i; ++j)
          EXPECT_NEAR(p[((b * H + h) * S + i) * S + j],
                      static_cast<float>(e[j] / denom), 1e-5);
        // Masked entries must be EXACTLY 0.0f — the backward primitives'
        // causal early-exits (AttnDV from i=j, AttnDQ/DK to j<=i) are only
        // mathematically valid on that bit, not on "small".
        for (size_t j = i + 1; j < S; ++j)
          EXPECT_EQ(p[((b * H + h) * S + i) * S + j], 0.0f);
        for (size_t c = 0; c < d; ++c) {
          double acc = 0.0;
          for (size_t j = 0; j <= i; ++j)
            acc += (e[j] / denom) * v[(b * S + j) * D + h * d + c];
          EXPECT_NEAR(o[(b * S + i) * D + h * d + c],
                      static_cast<float>(acc), 1e-5);
        }
      }
}

TEST(Attention, BackwardPrimitivesMatchFiniteDifferences) {
  // L = sum(w * O): the primitive chain dP -> dS -> {dQ, dK} plus dV must
  // match central differences of L through the full forward.
  const size_t B = 1, S = 3, H = 2, d = 2, D = H * d, n = B * S * D;
  const std::vector<float> q = RandnVector(n, 90);
  const std::vector<float> k2 = RandnVector(n, 91);
  const std::vector<float> v = RandnVector(n, 92);
  const std::vector<float> w = RandnVector(n, 93);  // dL/dO

  const size_t pn = B * H * S * S;
  std::vector<float> o(n), p(pn);
  k::AttnFwd(q.data(), k2.data(), v.data(), o.data(), p.data(), B, S, H, d);
  std::vector<float> dp(pn), ds(pn), dq(n), dk(n), dv(n);
  k::AttnDP(w.data(), v.data(), dp.data(), B, S, H, d);
  k::AttnDV(p.data(), w.data(), dv.data(), B, S, H, d);
  k::SoftmaxRowsBwd(p.data(), dp.data(), ds.data(), B * H * S, S);
  k::AttnDQ(ds.data(), k2.data(), dq.data(), B, S, H, d);
  k::AttnDK(ds.data(), q.data(), dk.data(), B, S, H, d);

  auto loss = [&](const std::vector<float>& qq, const std::vector<float>& kk,
                  const std::vector<float>& vv) {
    std::vector<float> oo(n), pp(pn);
    k::AttnFwd(qq.data(), kk.data(), vv.data(), oo.data(), pp.data(), B, S, H,
               d);
    double l = 0.0;
    for (size_t i = 0; i < n; ++i) l += static_cast<double>(w[i]) * oo[i];
    return l;
  };
  const double eps = 1e-3;
  auto check = [&](const std::vector<float>& base, const std::vector<float>& g,
                   int which) {
    for (size_t i = 0; i < n; ++i) {
      std::vector<float> bp = base, bm = base;
      bp[i] += static_cast<float>(eps);
      bm[i] -= static_cast<float>(eps);
      const double lp = which == 0   ? loss(bp, k2, v)
                        : which == 1 ? loss(q, bp, v)
                                     : loss(q, k2, bp);
      const double lm = which == 0   ? loss(bm, k2, v)
                        : which == 1 ? loss(q, bm, v)
                                     : loss(q, k2, bm);
      EXPECT_NEAR(g[i], static_cast<float>((lp - lm) / (2 * eps)), 2e-3);
    }
  };
  check(q, dq, 0);
  check(k2, dk, 1);
  check(v, dv, 2);
}

TEST(Loss, SingleClassSoftmaxIsExactlyCertain) {
  // C = 1: every row's softmax is exactly 1 and the loss exactly 0 — the
  // denominator always contains the max element's exp(0).
  const size_t N = 3;
  const std::vector<float> logits = {2.0f, -5.0f, 100.0f};
  const std::vector<int32_t> labels = {0, 0, 0};
  float loss = -1.0f;
  std::vector<float> probs(N, 0.0f);
  k::SoftmaxXEntFwd(logits.data(), labels.data(), &loss, probs.data(), N, 1);
  EXPECT_EQ(loss, 0.0f);
  for (float p : probs) EXPECT_EQ(p, 1.0f);
}

TEST(Loss, NanLogitPropagatesToTheLossSlot) {
  // The engine's finite-loss guard depends on this: a NaN anywhere in the
  // logits must surface in the loss, not be scrubbed by the max reduction.
  const size_t N = 2, C = 3;
  std::vector<float> logits(N * C, 0.5f);
  logits[4] = std::numeric_limits<float>::quiet_NaN();
  const std::vector<int32_t> labels = {0, 2};
  float loss = 0.0f;
  std::vector<float> probs(N * C);
  k::SoftmaxXEntFwd(logits.data(), labels.data(), &loss, probs.data(), N, C);
  EXPECT_TRUE(std::isnan(loss));
}

TEST(ClipNorm, ZeroGradientAndZeroCapAreExact) {
  std::vector<float> g = {0.0f, 0.0f, 0.0f};
  k::ClipNorm(g.data(), g.size(), 1.0f);  // ||g|| = 0: no 0/0, unchanged
  for (float v : g) EXPECT_EQ(v, 0.0f);

  std::vector<float> h = {3.0f, -4.0f};
  k::ClipNorm(h.data(), h.size(), 0.0f);  // cap 0: scaled by exactly 0
  for (float v : h) EXPECT_EQ(v, 0.0f);
}

TEST(LayerNorm, ZeroVarianceRowNormalizesToBeta) {
  const size_t rows = 1, cols = 4;
  const std::vector<float> x(cols, 7.25f);  // constant row: variance 0
  const std::vector<float> gamma(cols, 2.0f);
  const std::vector<float> beta = {0.5f, -1.0f, 3.0f, 0.0f};
  std::vector<float> y(cols), mean(rows), rstd(rows);
  k::LayerNormFwd(x.data(), gamma.data(), beta.data(), y.data(), mean.data(),
                  rstd.data(), rows, cols);
  // (x - mu) is exactly 0 per element, so y == beta bit-for-bit.
  for (size_t c = 0; c < cols; ++c) EXPECT_EQ(y[c], beta[c]);
}

TEST(Activation, GeluAndSiluBackwardsMatchFiniteDifferences) {
  // The smooth activations' backward kernels were pinned only through the
  // compiled-plan system check; a direct central-difference unit test
  // covers the input space (both signs, near zero, the |x| ~ 1-5 shoulder)
  // without a compiler in the loop.
  const size_t n = 9;
  const std::vector<float> x = {-5.0f, -2.0f, -0.5f, -1e-3f, 0.0f,
                                1e-3f, 0.5f,  2.0f,  5.0f};
  const std::vector<float> dy(n, 1.0f);
  const double eps = 1e-3;
  std::vector<float> dx(n), fwd_p(n), fwd_m(n), xp(n), xm(n);

  k::GeluBwd(dy.data(), x.data(), dx.data(), n);
  for (size_t i = 0; i < n; ++i) {
    xp = x; xm = x;
    xp[i] += static_cast<float>(eps);
    xm[i] -= static_cast<float>(eps);
    k::GeluFwd(xp.data(), fwd_p.data(), n);
    k::GeluFwd(xm.data(), fwd_m.data(), n);
    EXPECT_NEAR(dx[i], static_cast<float>((fwd_p[i] - fwd_m[i]) / (2 * eps)),
                2e-3);
  }

  k::SiluBwd(dy.data(), x.data(), dx.data(), n);
  for (size_t i = 0; i < n; ++i) {
    xp = x; xm = x;
    xp[i] += static_cast<float>(eps);
    xm[i] -= static_cast<float>(eps);
    k::SiluFwd(xp.data(), fwd_p.data(), n);
    k::SiluFwd(xm.data(), fwd_m.data(), n);
    EXPECT_NEAR(dx[i], static_cast<float>((fwd_p[i] - fwd_m[i]) / (2 * eps)),
                2e-3);
  }
}

TEST(Gelu, BackwardStaysFiniteAtExtremeInputs) {
  // In the saturated-tanh regime the sech^2 factor is a true zero while u'
  // overflows; the dead term must be dropped, not evaluated as 0 * Inf.
  const std::vector<float> x = {1e30f, -1e30f};
  const std::vector<float> dy = {2.0f, 2.0f};
  std::vector<float> dx(2);
  k::GeluBwd(dy.data(), x.data(), dx.data(), 2);
  EXPECT_EQ(dx[0], 2.0f);  // derivative saturates to 1
  EXPECT_EQ(dx[1], 0.0f);  // derivative saturates to 0
}

TEST(Embed, GathersTableRowsByTokenId) {
  // table [4, 3]; tokens pick rows out of order and with a repeat.
  const std::vector<float> table = {0, 1, 2,  10, 11, 12,
                                    20, 21, 22, 30, 31, 32};
  const std::vector<int32_t> tokens = {2, 0, 3, 0};
  std::vector<float> out(tokens.size() * 3, -1.0f);
  k::EmbedFwd(tokens.data(), table.data(), out.data(), tokens.size(), 3);
  const std::vector<float> want = {20, 21, 22, 0, 1, 2,
                                   30, 31, 32, 0, 1, 2};
  for (size_t i = 0; i < want.size(); ++i) EXPECT_EQ(out[i], want[i]);
}

TEST(Attention, SingleTokenSequenceIsIdentityOverV) {
  // S = 1: the causal prefix is one element, P is exactly 1, and O copies V.
  const size_t B = 2, S = 1, H = 2, d = 3, n = B * S * H * d;
  const std::vector<float> q = RandnVector(n, 130);
  const std::vector<float> k2 = RandnVector(n, 131);
  const std::vector<float> v = RandnVector(n, 132);
  std::vector<float> o(n), p(B * H * S * S);
  k::AttnFwd(q.data(), k2.data(), v.data(), o.data(), p.data(), B, S, H, d);
  for (float pv : p) EXPECT_EQ(pv, 1.0f);
  for (size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], v[i]);
}

TEST(Rope, GoldenValuesPinTheConvention) {
  // The adjoint-identity and finite-difference tests are blind to a
  // convention bug implemented consistently in both kernels (flipped
  // rotation sign, off-by-one frequency, position offset). Pin the exact
  // definition — theta(s, pair p) = s * base^(-2p/d), rotation
  // [[cos,-sin],[sin,cos]] — with independently computed values.
  const size_t B = 1, S = 2, H = 1, d = 4;
  const std::vector<float> x = RandnVector(B * S * H * d, 72);
  std::vector<float> y(x.size());
  const float base = 10000.0f;
  k::RopeFwd(x.data(), y.data(), B, S, H, d, base);
  // Position s = 1, pair 0: theta = 1.
  const size_t r1 = H * d;  // row offset of position 1
  EXPECT_NEAR(y[r1 + 0],
              static_cast<float>(x[r1 + 0] * std::cos(1.0) -
                                 x[r1 + 1] * std::sin(1.0)),
              1e-5);
  EXPECT_NEAR(y[r1 + 1],
              static_cast<float>(x[r1 + 0] * std::sin(1.0) +
                                 x[r1 + 1] * std::cos(1.0)),
              1e-5);
  // Position s = 1, pair 1: theta = base^(-2/d) = 10000^(-1/2) = 0.01.
  const double t1 = 0.01;
  EXPECT_NEAR(y[r1 + 2],
              static_cast<float>(x[r1 + 2] * std::cos(t1) -
                                 x[r1 + 3] * std::sin(t1)),
              1e-5);
  EXPECT_NEAR(y[r1 + 3],
              static_cast<float>(x[r1 + 2] * std::sin(t1) +
                                 x[r1 + 3] * std::cos(t1)),
              1e-5);
}

TEST(Attention, TheTiledFamilyIsTheCachedFamilyBitForBit) {
  // E11 (#94): the tiled kernels keep a stats row instead of P and
  // recompute every probability, with the cached kernels' expressions in
  // the cached kernels' orders — so O, dQ, dK and dV are the cached
  // family's floats exactly, at any width, and no S x S buffer exists.
  struct Shape { size_t B, S, H, d; };
  for (const Shape sh : {Shape{1, 1, 1, 4}, Shape{2, 7, 3, 5},
                         Shape{2, 64, 2, 8}, Shape{1, 130, 4, 16}}) {
    const size_t n = sh.B * sh.S * sh.H * sh.d;
    const size_t pn = sh.B * sh.H * sh.S * sh.S;
    const size_t rows = sh.B * sh.H * sh.S;
    const auto q = RandnVector(n, 201 + sh.S);
    const auto k2 = RandnVector(n, 202 + sh.S);
    const auto v = RandnVector(n, 203 + sh.S);
    const auto w = RandnVector(n, 204 + sh.S);  // dL/dO
    std::vector<float> o(n), p(pn), dp(pn), ds(pn), dq(n), dk(n), dv(n);
    k::AttnFwd(q.data(), k2.data(), v.data(), o.data(), p.data(), sh.B, sh.S,
               sh.H, sh.d);
    k::AttnDP(w.data(), v.data(), dp.data(), sh.B, sh.S, sh.H, sh.d);
    k::AttnDV(p.data(), w.data(), dv.data(), sh.B, sh.S, sh.H, sh.d);
    k::SoftmaxRowsBwd(p.data(), dp.data(), ds.data(), rows, sh.S);
    k::AttnDQ(ds.data(), k2.data(), dq.data(), sh.B, sh.S, sh.H, sh.d);
    k::AttnDK(ds.data(), q.data(), dk.data(), sh.B, sh.S, sh.H, sh.d);
    for (const size_t threads : {size_t{1}, size_t{8}}) {
      ScopedThreads scoped(threads);
      std::vector<float> to(n, 9.0f), tq(n, 9.0f), tk(n, 9.0f), tv(n, 9.0f);
      std::vector<float> stats(rows * seeml::update::kAttnStatsWidth, 9.0f);
      k::AttnFwdTiled(q.data(), k2.data(), v.data(), to.data(), stats.data(),
                      sh.B, sh.S, sh.H, sh.d);
      k::AttnDQTiled(q.data(), k2.data(), v.data(), w.data(), stats.data(),
                     tq.data(), sh.B, sh.S, sh.H, sh.d);
      k::AttnDKTiled(q.data(), k2.data(), v.data(), w.data(), stats.data(),
                     tk.data(), sh.B, sh.S, sh.H, sh.d);
      k::AttnDVTiled(q.data(), k2.data(), w.data(), stats.data(), tv.data(),
                     sh.B, sh.S, sh.H, sh.d);
      EXPECT_BITWISE_EQ_F32(o, to);
      EXPECT_BITWISE_EQ_F32(dq, tq);
      EXPECT_BITWISE_EQ_F32(dk, tk);
      EXPECT_BITWISE_EQ_F32(dv, tv);
    }
  }
}

TEST(Attention, KernelsAreThreadCountInvariant) {
  // Shapes sized so every kernel decomposes into MANY chunks (units >>
  // grain) — with tiny shapes both widths run one chunk and the comparison
  // is vacuous. B*H*S = 256 units at grain ~5 gives ~52 chunks for the
  // attention family; the whole forward+backward chain and both RoPE
  // directions run under each width.
  const size_t B = 2, S = 64, H = 2, d = 8, n = B * S * H * d;
  const size_t pn = B * H * S * S;
  const std::vector<float> q = RandnVector(n, 95);
  const std::vector<float> k2 = RandnVector(n, 96);
  const std::vector<float> v = RandnVector(n, 97);
  const std::vector<float> w = RandnVector(n, 98);  // dL/dO
  auto attn_chain = [&](float* out) {
    // Layout: o[n], p[pn], dp[pn], ds[pn], dq[n], dk[n], dv[n].
    float* o = out;
    float* p = o + n;
    float* dp = p + pn;
    float* ds = dp + pn;
    float* dq = ds + pn;
    float* dk = dq + n;
    float* dv = dk + n;
    k::AttnFwd(q.data(), k2.data(), v.data(), o, p, B, S, H, d);
    k::AttnDP(w.data(), v.data(), dp, B, S, H, d);
    k::AttnDV(p, w.data(), dv, B, S, H, d);
    k::SoftmaxRowsBwd(p, dp, ds, B * H * S, S);
    k::AttnDQ(ds, k2.data(), dq, B, S, H, d);
    k::AttnDK(ds, q.data(), dk, B, S, H, d);
  };
  auto rope = [&](float* out) {
    k::RopeFwd(q.data(), out, B, S, H, d, 10000.0f);
    k::RopeBwd(w.data(), out + n, B, S, H, d, 10000.0f);
  };
  auto rms = [&](float* out) {
    std::vector<float> gamma(H * d, 1.0f);
    float* y = out;
    float* rstd = y + n;
    float* dx = rstd + B * S;
    k::RmsNormFwd(q.data(), gamma.data(), y, rstd, B * S, H * d);
    k::RmsNormBwd(w.data(), q.data(), gamma.data(), rstd, dx, B * S, H * d);
  };
  const size_t chain = 4 * n + 3 * pn;
  EXPECT_BITWISE_EQ_F32(RunAtWidth(1, chain, attn_chain),
                        RunAtWidth(8, chain, attn_chain));
  EXPECT_BITWISE_EQ_F32(RunAtWidth(1, 2 * n, rope), RunAtWidth(8, 2 * n, rope));
  EXPECT_BITWISE_EQ_F32(RunAtWidth(1, 2 * n + B * S, rms),
                        RunAtWidth(8, 2 * n + B * S, rms));
}

// --- The bitwise-safe kernel batch (E3, #82) ----------------------------------
// Each item must change memory traffic and nothing else: exact equality
// against the form it replaces, at one thread and at eight.

TEST(KernelBatch, RopeThroughTheTableIsBitIdenticalToRecomputing) {
  const size_t B = 3, S = 37, H = 5, d = 12, n = B * S * H * d;
  const auto x = RandnVector(n, 91);
  for (const float base : {10000.0f, 500000.0f}) {
    for (const size_t threads : {size_t{1}, size_t{8}}) {
      ScopedThreads scoped(threads);
      std::vector<float> table(S * d), plain(n), tabled(n);
      k::RopeTable(table.data(), S, d, base);
      k::RopeFwd(x.data(), plain.data(), B, S, H, d, base);
      k::RopeFwd(x.data(), tabled.data(), B, S, H, d, base, table.data());
      EXPECT_BITWISE_EQ_F32(plain, tabled);
      k::RopeBwd(x.data(), plain.data(), B, S, H, d, base);
      k::RopeBwd(x.data(), tabled.data(), B, S, H, d, base, table.data());
      EXPECT_BITWISE_EQ_F32(plain, tabled);
      // Position 0 is the identity rotation: cos 1, sin 0, for every pair.
      for (size_t c = 0; c < d / 2; ++c) {
        EXPECT_EQ(table[2 * c], 1.0f);
        EXPECT_EQ(table[2 * c + 1], 0.0f);
      }
    }
  }
}

TEST(KernelBatch, ReduceRowsTilesChangeNoBits) {
  // Columns straddle the 256-wide accumulation tile and the chunk grid;
  // the reference is the serial loop the kernel has always been equal to.
  for (const size_t cols : {size_t{1}, size_t{255}, size_t{256}, size_t{257},
                            size_t{1000}, size_t{4099}}) {
    const size_t rows = 129;
    const auto dy = RandnVector(rows * cols, 17 + cols);
    std::vector<float> want(cols, 0.0f);
    for (size_t r = 0; r < rows; ++r)
      for (size_t c = 0; c < cols; ++c) want[c] += dy[r * cols + c];
    for (const size_t threads : {size_t{1}, size_t{8}}) {
      ScopedThreads scoped(threads);
      std::vector<float> got(cols, -7.0f);  // must be overwritten, not added to
      k::ReduceRows(dy.data(), got.data(), rows, cols);
      EXPECT_BITWISE_EQ_F32(want, got);
    }
  }
}

TEST(KernelBatch, FusedClipStepsEqualClipThenStep) {
  const size_t n = 70001;  // several chunks of the norm reduction
  for (const float scale : {4.0f, 1e-3f}) {  // clipped, and inside the ball
    const auto g0 = RandnVector(n, 5, scale);
    const auto p0 = RandnVector(n, 6);
    for (const size_t threads : {size_t{1}, size_t{8}}) {
      ScopedThreads scoped(threads);
      const float clip = 1.0f;
      // AdamW, three consecutive steps so the moments carry.
      std::vector<float> pa = p0, ma(n, 0.0f), va(n, 0.0f);
      std::vector<float> pb = p0, mb(n, 0.0f), vb(n, 0.0f);
      for (uint64_t step = 1; step <= 3; ++step) {
        std::vector<float> g = g0;
        k::ClipNorm(g.data(), n, clip);
        EXPECT_TRUE(k::AdamWStep(pa.data(), g.data(), ma.data(), va.data(), n,
                                 1e-2f, 0.9f, 0.999f, 1e-8f, 0.01f, step));
        std::vector<float> raw = g0;
        EXPECT_TRUE(k::AdamWStep(pb.data(), raw.data(), mb.data(), vb.data(),
                                 n, 1e-2f, 0.9f, 0.999f, 1e-8f, 0.01f, step,
                                 clip));
        EXPECT_BITWISE_EQ_F32(raw, g0);  // the fused step never writes g
      }
      EXPECT_BITWISE_EQ_F32(pa, pb);
      EXPECT_BITWISE_EQ_F32(ma, mb);
      EXPECT_BITWISE_EQ_F32(va, vb);
      // SGD: g * s feeds an add directly — the contraction hazard.
      std::vector<float> sa = p0, sb = p0, g = g0;
      k::ClipNorm(g.data(), n, clip);
      EXPECT_TRUE(k::SgdStep(sa.data(), g.data(), n, 0.1f, 0.2f));
      EXPECT_TRUE(k::SgdStep(sb.data(), g0.data(), n, 0.1f, 0.2f, clip));
      EXPECT_BITWISE_EQ_F32(sa, sb);
    }
  }
}

TEST(KernelBatch, FusedClipRefusesANonFiniteNormAndTouchesNothing) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  // 3e38 squares to 9e76, finite in the f64 reduction: a huge gradient is
  // clipped, not refused.
  for (const float poison : {inf, -inf, nan, 3e38f}) {
    std::vector<float> g = {0.5f, poison, -0.25f};
    const std::vector<float> p0 = {1.0f, 2.0f, 3.0f};
    std::vector<float> p = p0, m(3, 0.125f), v(3, 0.5f);
    const bool finite = std::isfinite(poison);
    EXPECT_EQ(k::AdamWStep(p.data(), g.data(), m.data(), v.data(), 3, 1e-2f,
                           0.9f, 0.999f, 1e-8f, 0.0f, 1, /*clip_norm=*/1.0f),
              finite);
    if (!finite) {
      EXPECT_BITWISE_EQ_F32(p, p0);
      EXPECT_BITWISE_EQ_F32(m, std::vector<float>(3, 0.125f));
      EXPECT_BITWISE_EQ_F32(v, std::vector<float>(3, 0.5f));
    }
    std::vector<float> q = p0;
    EXPECT_EQ(k::SgdStep(q.data(), g.data(), 3, 0.1f, 0.0f, 1.0f), finite);
    if (!finite) EXPECT_BITWISE_EQ_F32(q, p0);
  }
}

// --- The GEMM redesign (E1, #80): traversal, never bits -----------------------
// The register-blocked, packed, 2D-partitioned cores must compute exactly
// what the kernels' reduction contract says, written here the naive way:
//   NN family   c += a0*b0 + a1*b1 + a2*b2 + a3*b3 over k-quads aligned to
//               k = 0, then one term at a time for the K % 4 tail;
//   NT family   eight lanes by k mod 8, combined in lane order, times alpha.
// Exact equality, on shapes chosen to cross every internal boundary: the
// four-row block (M % 4), the 64-row band, column bands (few rows, wide
// N), the N tile and the panel clamp (oversized tiles), the two-row dX
// pairing threshold (K = 64) and its odd row and leftover columns.

std::vector<float> ReferenceNN(const std::vector<float>& A,
                               const std::vector<float>& B, size_t M,
                               size_t N, size_t K, float alpha, bool a_t,
                               std::vector<float> C) {
  auto a_at = [&](size_t k, size_t m) {
    return a_t ? A[k * M + m] : A[m * K + k];
  };
  for (size_t m = 0; m < M; ++m)
    for (size_t n = 0; n < N; ++n) {
      float c = C[m * N + n];
      size_t k = 0;
      for (; k + 4 <= K; k += 4) {
        const float a0 = alpha * a_at(k, m), a1 = alpha * a_at(k + 1, m);
        const float a2 = alpha * a_at(k + 2, m), a3 = alpha * a_at(k + 3, m);
        c += a0 * B[k * N + n] + a1 * B[(k + 1) * N + n] +
             a2 * B[(k + 2) * N + n] + a3 * B[(k + 3) * N + n];
      }
      for (; k < K; ++k) {
        const float a = alpha * a_at(k, m);
        c += a * B[k * N + n];
      }
      C[m * N + n] = c;
    }
  return C;
}

std::vector<float> ReferenceNT(const std::vector<float>& A,
                               const std::vector<float>& B, size_t M,
                               size_t N, size_t K, float alpha) {
  std::vector<float> C(M * N);
  for (size_t m = 0; m < M; ++m)
    for (size_t n = 0; n < N; ++n) {
      float lane[8] = {};
      for (size_t k = 0; k < K; ++k) lane[k % 8] += A[m * K + k] * B[n * K + k];
      float sum = lane[0];
      for (size_t l = 1; l < 8; ++l) sum += lane[l];
      C[m * N + n] = alpha * sum;
    }
  return C;
}

struct GemmShape {
  size_t M, N, K;
};
constexpr GemmShape kGemmShapes[] = {
    {1, 1, 1},      {3, 5, 7},      {4, 4, 4},     {5, 9, 3},
    {67, 129, 66},  {130, 257, 63}, {131, 517, 203},  // ragged everywhere
    {8, 1100, 300},                                   // column bands
    {2, 700, 900},  {257, 33, 64},  {66, 131, 65},    // NT pairing edges
};

TEST(GemmRedesign, TheNNFamilyIsTheReferenceReductionExactly) {
  for (const GemmShape& sh : kGemmShapes) {
    const size_t M = sh.M, N = sh.N, K = sh.K;
    const auto A = RandnVector(M * K, 1000 + M);
    const auto B = RandnVector(K * N, 2000 + N);
    const auto C0 = RandnVector(M * N, 3000 + K);  // GemmAccNN's running C
    const std::vector<float> zeros(M * N, 0.0f);
    const auto want_nn = ReferenceNN(A, B, M, N, K, 1.0f, false, zeros);
    const auto want_tn = ReferenceNN(A, B, M, N, K, 1.0f, true, zeros);
    const auto want_acc = ReferenceNN(A, B, M, N, K, 0.375f, false, C0);
    // The runtime defaults, a tiling that forces the panel clamp, and the
    // smallest legal one.
    for (const k::GemmTiles tiles :
         {k::kDefaultGemmTiles, k::GemmTiles{4096, 100000}, k::GemmTiles{4, 1}}) {
      for (const size_t threads : {size_t{1}, size_t{8}}) {
        ScopedThreads scoped(threads);
        std::vector<float> got(M * N, 9.0f);  // must be overwritten
        k::GemmNN(A.data(), B.data(), got.data(), M, N, K, nullptr,
                  seeml::update::EpilogueAct::kNone, tiles);
        EXPECT_BITWISE_EQ_F32(want_nn, got);
        std::fill(got.begin(), got.end(), 9.0f);
        k::GemmTN(A.data(), B.data(), got.data(), M, N, K, tiles);
        EXPECT_BITWISE_EQ_F32(want_tn, got);
        got = C0;
        k::GemmAccNN(A.data(), B.data(), got.data(), M, N, K, 0.375f, tiles);
        EXPECT_BITWISE_EQ_F32(want_acc, got);
      }
    }
  }
}

TEST(GemmRedesign, TheAddendIsTheGemmThenAnAddExactly) {
  // kFlagGemmAddend (plan v14, E10): C = D + A@B must be the bits of the
  // two instructions it replaces — the GEMM into a transient, then an
  // elementwise add in either operand order — for all three f32 forms, at
  // any tiling and any width.
  for (const GemmShape& sh : kGemmShapes) {
    const size_t M = sh.M, N = sh.N, K = sh.K;
    const auto A = RandnVector(M * K, 7000 + M);
    const auto Bnn = RandnVector(K * N, 7100 + N);
    const auto D = RandnVector(M * N, 7200 + K);
    for (const k::GemmTiles tiles :
         {k::kDefaultGemmTiles, k::GemmTiles{4096, 100000}, k::GemmTiles{4, 1}}) {
      for (const size_t threads : {size_t{1}, size_t{8}}) {
        ScopedThreads scoped(threads);
        std::vector<float> plain(M * N), want(M * N), got(M * N, 9.0f);
        auto summed = [&] {
          for (size_t i = 0; i < M * N; ++i) want[i] = plain[i] + D[i];
        };
        k::GemmNN(A.data(), Bnn.data(), plain.data(), M, N, K, nullptr,
                  seeml::update::EpilogueAct::kNone, tiles);
        summed();
        k::GemmNN(A.data(), Bnn.data(), got.data(), M, N, K, nullptr,
                  seeml::update::EpilogueAct::kNone, tiles, D.data());
        EXPECT_BITWISE_EQ_F32(want, got);

        // NT reads B as [N, K]; TN reads A as [K, M]. Same element counts.
        k::GemmNT(A.data(), Bnn.data(), plain.data(), M, N, K, tiles);
        summed();
        std::fill(got.begin(), got.end(), 9.0f);
        k::GemmNT(A.data(), Bnn.data(), got.data(), M, N, K, tiles, D.data());
        EXPECT_BITWISE_EQ_F32(want, got);

        k::GemmTN(A.data(), Bnn.data(), plain.data(), M, N, K, tiles);
        summed();
        std::fill(got.begin(), got.end(), 9.0f);
        k::GemmTN(A.data(), Bnn.data(), got.data(), M, N, K, tiles, D.data());
        EXPECT_BITWISE_EQ_F32(want, got);
      }
    }
  }
}

TEST(GemmRedesign, TheNTFamilyIsTheReferenceReductionExactly) {
  for (const GemmShape& sh : kGemmShapes) {
    const size_t M = sh.M, N = sh.N, K = sh.K;
    const auto A = RandnVector(M * K, 4000 + M);
    const auto B = RandnVector(N * K, 5000 + N);
    const auto want = ReferenceNT(A, B, M, N, K, 1.0f);
    for (const k::GemmTiles tiles : {k::kDefaultGemmTiles, k::GemmTiles{8, 3}}) {
      for (const size_t threads : {size_t{1}, size_t{8}}) {
        ScopedThreads scoped(threads);
        std::vector<float> got(M * N, 9.0f);
        k::GemmNT(A.data(), B.data(), got.data(), M, N, K, tiles);
        EXPECT_BITWISE_EQ_F32(want, got);
      }
    }
  }
}

TEST(GemmRedesign, PackedInt8AndBf16WeightsWidenToTheSameBits) {
  // The panel widens a tile once instead of once per sweep; widening is
  // exact, so the q8 and bf16 GEMMs must equal the f32 GEMMs run over the
  // widened matrix (times the dequant scale, which rides alpha).
  for (const GemmShape& sh : {GemmShape{131, 517, 203}, GemmShape{6, 900, 70}}) {
    const size_t M = sh.M, N = sh.N, K = sh.K;
    const auto A = RandnVector(M * K, 77);
    std::vector<int8_t> q(K * N);
    std::vector<float> widened(K * N);
    for (size_t i = 0; i < q.size(); ++i) {
      q[i] = static_cast<int8_t>(static_cast<int>((i * 37) % 255) - 127);
      widened[i] = static_cast<float>(q[i]);
    }
    const float scale = 0.0131f;
    const std::vector<float> zeros(M * N, 0.0f);
    const auto want = ReferenceNN(A, widened, M, N, K, scale, false, zeros);
    for (const size_t threads : {size_t{1}, size_t{8}}) {
      ScopedThreads scoped(threads);
      std::vector<float> got(M * N, 9.0f);
      k::GemmNNQ8(A.data(), q.data(), got.data(), M, N, K, scale);
      EXPECT_BITWISE_EQ_F32(want, got);
    }
    // bf16: every value here is exactly representable, so widen(B) == B.
    std::vector<float> exact(K * N);
    std::vector<uint16_t> half(K * N);
    for (size_t i = 0; i < exact.size(); ++i) {
      exact[i] = static_cast<float>(static_cast<int>((i * 13) % 64) - 32) *
                 0.125f;
      half[i] = seeml::update::Float32ToBf16Bits(exact[i]);
    }
    const auto want_h = ReferenceNN(A, exact, M, N, K, 1.0f, false, zeros);
    // NT reads B as [N, K]: the same bytes, the other orientation.
    const auto A_nt = RandnVector(M * N, 78);  // A is [M, N-as-K] here
    const auto want_nt = ReferenceNT(A_nt, exact, M, K, N, 1.0f);
    for (const size_t threads : {size_t{1}, size_t{8}}) {
      ScopedThreads scoped(threads);
      std::vector<float> got(M * N, 9.0f);
      k::GemmNNBF16(A.data(), half.data(), got.data(), M, N, K);
      EXPECT_BITWISE_EQ_F32(want_h, got);
      std::vector<float> got_nt(M * K, 9.0f);
      k::GemmNTBF16(A_nt.data(), half.data(), got_nt.data(), M, K, N);
      EXPECT_BITWISE_EQ_F32(want_nt, got_nt);
    }
  }
}

// --- kFusedMap (E4, #83): a chain is the sequence it replaces, bit for bit ----

TEST(FusedMap, EveryStageMatchesItsStandaloneKernelInSequence) {
  namespace up = seeml::update;
  using up::FusedStage;
  using up::MakeFusedStage;
  // Sizes around the 1024-float block and the parallel grain; values that
  // exercise the activations' interesting ranges.
  for (const size_t n : {size_t{1}, size_t{1023}, size_t{1024}, size_t{1025},
                         size_t{70001}}) {
    const auto x = RandnVector(n, 11, 2.0f);
    const auto y = RandnVector(n, 12);
    const auto z = RandnVector(n, 13);
    const float a0 = 2.0f / 3.0f, a1 = -0.375f;
    for (const size_t threads : {size_t{1}, size_t{8}}) {
      ScopedThreads scoped(threads);
      // silu(x) -> + y -> * a0 -> z * (.)   [four stages, the run on the
      // right of the last product], against the four standalone kernels.
      std::vector<float> t1(n), t2(n), t3(n), want(n), got(n, 9.0f);
      k::SiluFwd(x.data(), t1.data(), n);
      k::AddEW(t1.data(), y.data(), t2.data(), n);
      k::Scale(t2.data(), t3.data(), a0, n);
      k::MulEW(z.data(), t3.data(), want.data(), n);
      const uint64_t prog =
          uint64_t{MakeFusedStage(FusedStage::kSilu)} |
          uint64_t{MakeFusedStage(FusedStage::kAdd, 1)} << 8 |
          uint64_t{MakeFusedStage(FusedStage::kScale, 0)} << 16 |
          uint64_t{MakeFusedStage(FusedStage::kMul, 2, true)} << 24;
      const float* others[3] = {nullptr, y.data(), z.data()};
      const float imm[2] = {a0, a1};
      k::FusedMap(x.data(), others, got.data(), n, prog, imm);
      EXPECT_BITWISE_EQ_F32(want, got);

      // The LoRA forward chain, scale -> add: a multiply feeding an add is
      // exactly where a contraction would change bits.
      k::Scale(x.data(), t1.data(), a1, n);
      k::AddEW(y.data(), t1.data(), want.data(), n);
      const uint64_t lora =
          uint64_t{MakeFusedStage(FusedStage::kScale, 1)} |
          uint64_t{MakeFusedStage(FusedStage::kAdd, 1, true)} << 8;
      std::fill(got.begin(), got.end(), 9.0f);
      k::FusedMap(x.data(), others, got.data(), n, lora, imm);
      EXPECT_BITWISE_EQ_F32(want, got);

      // gelu -> relu -> mul, two unary stages back to back on the block.
      k::GeluFwd(x.data(), t1.data(), n);
      k::ReluFwd(t1.data(), t2.data(), n);
      k::MulEW(t2.data(), z.data(), want.data(), n);
      const uint64_t acts =
          uint64_t{MakeFusedStage(FusedStage::kGelu)} |
          uint64_t{MakeFusedStage(FusedStage::kRelu)} << 8 |
          uint64_t{MakeFusedStage(FusedStage::kMul, 2)} << 16;
      std::fill(got.begin(), got.end(), 9.0f);
      k::FusedMap(x.data(), others, got.data(), n, acts, imm);
      EXPECT_BITWISE_EQ_F32(want, got);
    }
  }
}

}  // namespace
