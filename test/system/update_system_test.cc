// =============================================================================
// SeeAI Update Compiler — system verification suite.
//
//   1. Step-0 identity: B = 0 ⇒ the compiled update starts as the exact
//      source model (loss equals the ungrafted model's loss).
//   2. Gradient check: the AOT-compiled backward pass matches central finite
//      differences of the compiled forward pass (the mathematical oracle).
//   3. End-to-end update: train → loss decreases; merge → W' = W + (α/r)·A@B
//      to float precision; commit → the patched SMF matches, untouched
//      weights bit-identical, transactional output.
//   4. MSE regression path: dense-target training converges.
//   5. Distillation: a KL-loss plan against a teacher model compiles, runs,
//      and reduces the student/teacher divergence.
//   6. Composite loss: (1-w)·xent + w·kl trains against labels + teacher.
//   7. New operators: a gated GELU/SiLU/Mul/LayerNorm network gradient-checks
//      against finite differences of its own compiled forward pass.
//   8. Quantized base: int8-rodata plans shrink, gradient-check consistently,
//      train, and commit deltas onto the file's pristine f32 weights.
//   9. Poisoned corpus: non-finite losses abort the update. (The model must
//      route the poison through GELU/SiLU/LayerNorm — ReLU nets scrub
//      non-finites: relu(NaN) == 0.)
//
// Structural graft coverage lives in the unit suites (update_passes_test,
// update_compiler_test); this suite verifies the numerics end to end.
// =============================================================================

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "compiler/driver/update_compiler.h"
#include "runtime/feeder/dataset.h"
#include "runtime/engine/update_engine.h"
#include "compiler/frontend/ingressor/model_reader.h"
#include "compiler/frontend/egress/model_writer.h"
#include "source/parallel/parallel_for.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"
#include "test/support/scoped_temp_dir.h"

namespace {

using namespace seeml::update;
using seeml::update_rt::Dataset;
using seeml::update_rt::TrainOptions;
using seeml::update_rt::UpdateEngine;
using seeml::testing::BaseConfig;
using seeml::testing::FillSlots;
using seeml::testing::MakeClassificationData;
using seeml::testing::MakeMlp;
using seeml::testing::MakeRegressionData;
using seeml::testing::MakeUnlabeledData;
using seeml::testing::ReadArenaF32;
using seeml::testing::ScopedTempDir;
using seeml::testing::WriteArenaF32;

PlanHeader HeaderOf(const CompiledUpdate& compiled) {
  PlanHeader h;
  std::memcpy(&h, compiled.plan.data(), sizeof(h));
  return h;
}

TrainOptions Quiet() {
  TrainOptions options;
  options.log_every = 0;
  return options;
}

// =============================================================================
// 1. Step-0 identity: grafted model == source model (B = 0)
// =============================================================================

TEST(UpdateSystem, Step0Identity) {
  const int64_t in_dim = 6, hidden = 10, out_dim = 3, batch = 4;
  SmfModel model = MakeMlp(in_dim, hidden, out_dim, 2);

  UpdateConfig config = BaseConfig(batch);
  config.emit_optimizer = false;  // observe without mutating
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));

  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));

  std::mt19937_64 rng(99);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(batch * in_dim);
  for (auto& v : x) v = dist(rng);
  const std::vector<int32_t> labels = {0, 1, 2, 1};
  FillSlots(engine, x, labels);
  engine.ExecuteTrainOnce();
  const float grafted_loss = engine.LossValue();

  // Reference loss computed directly from the SMF weights (no LoRA).
  auto weights = [&](const char* name) {
    const SmfTensor* t = model.FindTensor(name);
    std::vector<float> v;
    if (t) {
      v.resize(t->data.size() / sizeof(float));
      std::memcpy(v.data(), t->data.data(), t->data.size());
    }
    return v;
  };
  const auto w1 = weights("w1"), b1 = weights("b1");
  const auto w2 = weights("w2"), b2 = weights("b2");
  ASSERT_FALSE(w1.empty());

  double ref_loss = 0.0;
  for (int64_t n = 0; n < batch; ++n) {
    std::vector<float> h(hidden, 0.0f), logits(out_dim, 0.0f);
    for (int64_t j = 0; j < hidden; ++j) {
      float acc = b1[j];
      for (int64_t i = 0; i < in_dim; ++i)
        acc += x[n * in_dim + i] * w1[i * hidden + j];
      h[j] = acc > 0 ? acc : 0;
    }
    for (int64_t c = 0; c < out_dim; ++c) {
      float acc = b2[c];
      for (int64_t j = 0; j < hidden; ++j) acc += h[j] * w2[j * out_dim + c];
      logits[c] = acc;
    }
    float mx = logits[0];
    for (float l : logits) mx = std::fmax(mx, l);
    double sum = 0.0;
    for (float l : logits) sum += std::exp(static_cast<double>(l - mx));
    ref_loss -= (logits[labels[n]] - mx) - std::log(sum);
  }
  ref_loss /= batch;

  EXPECT_NEAR(grafted_loss, ref_loss, 1e-4);
}

// =============================================================================
// 2. Finite-difference gradient check of the compiled backward pass
// =============================================================================

/// Central-difference check of every parameter's compiled gradient against
/// the compiled forward pass (the plan must be built emit_optimizer=false so
/// executions do not mutate the parameters). Nudges lora_B off zero first so
/// gradients w.r.t. A are non-degenerate.
///
/// A central difference only estimates the derivative where the loss is
/// smooth across the step; a ReLU pre-activation inside it (a kink) fails
/// the check with no bug present. Where the kinks fall depends on the data,
/// and std::normal_distribution is implemented differently by libstdc++ and
/// libc++ — so a check on normal_distribution data that passes on macOS can
/// land on a kink on Linux. `portable` nudges with raw mt19937_64 bits
/// (fully specified by the standard), for callers whose data is portable
/// too (PortableUniform), so both platforms check the same coordinates.
float PortableUniform(std::mt19937_64& rng) {  // [-1, 1), same everywhere
  return static_cast<float>(static_cast<double>(rng() >> 11) * 0x1p-52 - 1.0);
}

void GradientCheck(const CompiledUpdate& compiled, UpdateEngine& engine,
                   uint64_t nudge_seed, double tol = 2e-2,
                   bool portable = false) {
  std::mt19937_64 rng(nudge_seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (const auto& p : compiled.params)
    for (uint64_t i = 0; i < p.count; ++i)
      if (p.id.find(".lora_B") != std::string::npos)
        WriteArenaF32(engine, p.param_ref, i,
                      0.05f * (portable ? PortableUniform(rng) : dist(rng)));

  engine.ExecuteTrainOnce();
  const double eps = 2e-3;
  size_t checked = 0;

  for (const auto& p : compiled.params) {
    // Sample a few coordinates per parameter tensor.
    for (uint64_t i = 0; i < p.count; i += std::max<uint64_t>(1, p.count / 5)) {
      engine.ExecuteTrainOnce();  // refresh gradients for current params
      const float analytic = ReadArenaF32(engine, p.grad_ref, i);
      const float saved = ReadArenaF32(engine, p.param_ref, i);

      WriteArenaF32(engine, p.param_ref, i, saved + static_cast<float>(eps));
      engine.ExecuteTrainOnce();
      const double plus = engine.LossValue();
      WriteArenaF32(engine, p.param_ref, i, saved - static_cast<float>(eps));
      engine.ExecuteTrainOnce();
      const double minus = engine.LossValue();
      WriteArenaF32(engine, p.param_ref, i, saved);

      const double numeric = (plus - minus) / (2.0 * eps);
      const double denom =
          std::max(1e-4, std::fabs(numeric) + std::fabs(analytic));
      const double rel = std::fabs(numeric - analytic) / denom;
      // The loss is a single f32 (ulp ~1e-7 near 1.0), so the central
      // difference cannot resolve gradients below ~ulp/(2*eps) ≈ 5e-5.
      // Accept absolute disagreement inside that estimator noise floor;
      // demand relative agreement for everything the estimator can see.
      const double noise_floor = 1e-4;
      if (std::fabs(numeric - analytic) >= noise_floor && rel >= tol)
        ADD_FAILURE("param " + p.id + "[" + std::to_string(i) +
                    "]: analytic " + std::to_string(analytic) + " vs numeric " +
                    std::to_string(numeric));
      ++checked;
    }
  }
  EXPECT_GE(checked, 20u);
}

TEST(UpdateSystem, GradientsMatchFiniteDifferences) {
  const int64_t in_dim = 5, hidden = 7, out_dim = 3, batch = 2;
  SmfModel model = MakeMlp(in_dim, hidden, out_dim, 3);

  UpdateConfig config = BaseConfig(batch);
  config.lora.rank = 3;
  config.emit_optimizer = false;  // params stay fixed across executions
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));

  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));

  std::mt19937_64 rng(4242);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(batch * in_dim);
  for (auto& v : x) v = dist(rng);
  FillSlots(engine, x, {1, 2});

  GradientCheck(compiled, engine, 4242);
}

TEST(UpdateSystem, Bf16BaseGradientsMatchFiniteDifferences) {
  // The compiled backward through bf16-widened frozen weights is the exact
  // derivative of the compiled forward through the same widened weights —
  // the finite-difference suite at the decoder's loosened tolerance.
  const int64_t dim = 8, heads = 2, seq = 4, ffn = 10, vocab = 3;
  const int64_t batch = 8;
  SmfModel model = seeml::testing::MakeTinyDecoder(dim, heads, seq, ffn,
                                                   vocab, 24);
  UpdateConfig config = BaseConfig(batch);
  config.lora.rank = 2;
  config.emit_optimizer = false;
  config.bf16_base = true;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));
  size_t bf16_adapters = 0;
  for (const auto& ad : compiled.adapters) bf16_adapters += ad.bf16;
  EXPECT_EQ(bf16_adapters, compiled.adapters.size());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));
  std::mt19937_64 rng(2424);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(batch * dim);
  for (auto& v : x) v = dist(rng);
  FillSlots(engine, x, {1, 2, 0, 1, 2, 0, 1, 2});
  GradientCheck(compiled, engine, 2424, /*tol=*/3e-2);
}

TEST(UpdateSystem, Bf16BaseTrainsWithinADriftBudgetOfF32) {
  // Storage precision only: the same update on f32 and bf16 rodata must
  // track each other — the drift budget the roadmap asks for. bf16 rounds
  // every frozen weight to 8 mantissa bits (2^-9 relative), so the loss
  // trajectories agree to well under a percent on this MLP.
  const int64_t in_dim = 16, hidden = 32, out_dim = 4, batch = 16;
  SmfModel model = MakeMlp(in_dim, hidden, out_dim, 61);
  float curves[2][2];
  int i = 0;
  for (const bool bf16 : {false, true}) {
    UpdateConfig config = BaseConfig(batch);
    config.bf16_base = bf16;
    config.optimizer.lr = 5e-3f;
    ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                         UpdateCompiler(config).Compile(model));
    UpdateEngine engine;
    ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(),
                                    compiled.plan.size()));
    ASSERT_OK_AND_ASSIGN(Dataset data, MakeClassificationData(256, in_dim, 62));
    data.EnableShuffle(3);
    ASSERT_OK_AND_ASSIGN(auto report, engine.Train(data, 60, Quiet()));
    curves[i][0] = report.initial_avg_loss;
    curves[i][1] = report.final_avg_loss;
    ++i;
  }
  EXPECT_NEAR(curves[1][0], curves[0][0], 1e-2 * curves[0][0]);
  EXPECT_NEAR(curves[1][1], curves[0][1], 1e-2 * curves[0][1]);
  EXPECT_LT(curves[1][1], curves[1][0]);  // and it learns
}

TEST(UpdateSystem, AccumulatedGradientsMatchTheLargeBatchGradient) {
  // (b = 8, G = 4) folding four consecutive 8-row slices must equal the
  // (b = 32, G = 1) gradient over the same 32 rows. The 1/G seed keeps
  // every per-row adjoint bitwise-identical (division by a power of two);
  // only the row-sum association differs (four 8-row GEMMs folded vs one
  // 32-row GEMM), so the comparison is at f32 round-off, not bitwise.
  const int64_t in_dim = 5, hidden = 7, out_dim = 3;
  SmfModel model = MakeMlp(in_dim, hidden, out_dim, 71);
  UpdateConfig big = BaseConfig(32);
  big.lora.rank = 3;
  big.emit_optimizer = false;
  UpdateConfig micro = BaseConfig(8);
  micro.lora.rank = 3;
  micro.emit_optimizer = false;
  micro.grad_accum_steps = 4;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate cbig, UpdateCompiler(big).Compile(model));
  ASSERT_OK_AND_ASSIGN(CompiledUpdate cmicro,
                       UpdateCompiler(micro).Compile(model));
  ASSERT_EQ(HeaderOf(cmicro).step_instr_count, 0u);  // folds, no step

  std::mt19937_64 rng(7171);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(32 * in_dim);
  for (auto& v : x) v = dist(rng);
  std::vector<int32_t> labels(32);
  for (size_t i = 0; i < 32; ++i) labels[i] = static_cast<int32_t>(i % out_dim);

  // Identical adapter state on both engines: nudge lora_B off zero.
  auto nudge = [&](const CompiledUpdate& c, UpdateEngine& e) {
    std::mt19937_64 r(99);
    std::normal_distribution<float> d(0.0f, 1.0f);
    for (const auto& p : c.params)
      if (p.id.find(".lora_B") != std::string::npos)
        for (uint64_t i = 0; i < p.count; ++i)
          WriteArenaF32(e, p.param_ref, i, 0.05f * d(r));
  };
  UpdateEngine ebig, emicro;
  ASSERT_OK(ebig.LoadFromMemory(cbig.plan.data(), cbig.plan.size()));
  ASSERT_OK(emicro.LoadFromMemory(cmicro.plan.data(), cmicro.plan.size()));
  nudge(cbig, ebig);
  nudge(cmicro, emicro);
  // No optimizer, so the plan records no accumulation to the runtime: the
  // folds are driven by hand below, one micro-batch per execution.
  EXPECT_EQ(emicro.grad_accum_steps(), 1u);

  FillSlots(ebig, x, labels);
  ebig.ExecuteTrainOnce();
  for (int g = 0; g < 4; ++g) {
    std::vector<float> xs(x.begin() + g * 8 * in_dim,
                          x.begin() + (g + 1) * 8 * in_dim);
    std::vector<int32_t> ls(labels.begin() + g * 8, labels.begin() + (g + 1) * 8);
    FillSlots(emicro, xs, ls);
    emicro.ExecuteTrainOnce();
  }
  size_t checked = 0;
  for (const auto& pb : cbig.params) {
    const auto pm = std::find_if(cmicro.params.begin(), cmicro.params.end(),
                                 [&](const auto& q) { return q.id == pb.id; });
    ASSERT_TRUE(pm != cmicro.params.end());
    ASSERT_NE(pm->acc_ref, kNullRef);
    for (uint64_t i = 0; i < pb.count; ++i) {
      const float want = ReadArenaF32(ebig, pb.grad_ref, i);
      const float got = ReadArenaF32(emicro, pm->acc_ref, i);
      EXPECT_NEAR(got, want, 1e-6 + 1e-5 * std::fabs(want));
      ++checked;
    }
  }
  EXPECT_GT(checked, 20u);
}

TEST(UpdateSystem, MseGradientsMatchFiniteDifferences) {
  // G13 (#97): the MSE loss through the compiled backward against the
  // compiled forward — every other program-level FD check used xent or
  // the KL family.
  const int64_t in_dim = 5, hidden = 7, out_dim = 3, batch = 2;
  SmfModel model = MakeMlp(in_dim, hidden, out_dim, 63);
  UpdateConfig config = BaseConfig(batch);
  config.loss = LossKind::kMse;
  config.lora.rank = 3;
  config.emit_optimizer = false;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));
  std::mt19937_64 rng(6363);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(batch * in_dim);
  for (auto& v : x) v = dist(rng);
  FillSlots(engine, x, {});
  ASSERT_EQ(engine.header().label_bytes, batch * out_dim * sizeof(float));
  for (int64_t i = 0; i < batch * out_dim; ++i)
    WriteArenaF32(engine, engine.header().label_ref, i, dist(rng));
  GradientCheck(compiled, engine, 6363);
}

TEST(UpdateSystem, DistillationGradientsMatchFiniteDifferences) {
  // The T^2-scaled KL loss (#13) and the composite (1-w)·xent + w·kl, each
  // checked through the compiled backward against the compiled forward —
  // the program-level FD coverage these two losses never had.
  const int64_t in_dim = 5, hidden = 7, out_dim = 3, batch = 2;
  SmfModel student = MakeMlp(in_dim, hidden, out_dim, 61);
  SmfModel teacher = MakeMlp(in_dim, 9, out_dim, 62);

  for (const LossKind loss : {LossKind::kKLDistill, LossKind::kXEntPlusKL}) {
    UpdateConfig config = BaseConfig(batch);
    config.loss = loss;
    config.temperature = 2.0f;
    config.distill_weight = 0.4f;
    config.lora.rank = 3;
    config.emit_optimizer = false;  // params stay fixed across executions
    ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                         UpdateCompiler(config).Compile(student, &teacher));

    UpdateEngine engine;
    ASSERT_OK(
        engine.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));

    std::mt19937_64 rng(6161);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x(batch * in_dim);
    for (auto& v : x) v = dist(rng);
    std::vector<int32_t> labels;
    if (loss == LossKind::kXEntPlusKL) labels = {2, 0};
    FillSlots(engine, x, labels);

    GradientCheck(compiled, engine, 6161);
  }
}

// =============================================================================
// 3. End-to-end: train -> merge -> commit
// =============================================================================

TEST(UpdateSystem, TrainMergeCommit) {
  const int64_t in_dim = 8, hidden = 16, out_dim = 2, batch = 8;
  SmfModel model = MakeMlp(in_dim, hidden, out_dim, 5);

  UpdateConfig config = BaseConfig(batch);
  config.optimizer.lr = 5e-3f;
  config.optimizer.kind = OptimizerKind::kAdamW;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));

  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));

  ASSERT_OK_AND_ASSIGN(Dataset data, MakeClassificationData(512, in_dim, 11));
  ASSERT_OK_AND_ASSIGN(auto report, engine.Train(data, 400, Quiet()));
  EXPECT_LT(report.final_avg_loss, report.initial_avg_loss * 0.8f);

  // --- Merge: the delta must equal (α/r)·A@B to float precision. ------------
  ASSERT_OK(engine.RunMerge());
  for (const auto& a : compiled.adapters) {
    const auto* A =
        reinterpret_cast<const float*>(engine.arena() + RefOffset(a.a_ref));
    const auto* B =
        reinterpret_cast<const float*>(engine.arena() + RefOffset(a.b_ref));
    const auto* delta = reinterpret_cast<const float*>(
        engine.arena() + RefOffset(a.delta_ref));

    double max_err = 0.0, max_delta = 0.0;
    for (int64_t i = 0; i < a.k; ++i)
      for (int64_t j = 0; j < a.m; ++j) {
        double expected = 0.0;
        for (int64_t t = 0; t < a.r; ++t)
          expected += static_cast<double>(A[i * a.r + t]) * B[t * a.m + j];
        expected *= a.scale;
        max_err =
            std::max(max_err, std::fabs(expected - delta[i * a.m + j]));
        max_delta = std::max(max_delta, std::fabs(expected));
      }
    EXPECT_LT(max_err, 1e-4);
    // Training actually moved the weights (B left zero => no update at all).
    EXPECT_GT(max_delta, 1e-4);
  }

  // --- Commit: patched SMF, untouched tensors bit-identical. ----------------
  ScopedTempDir dir;
  const std::string src_path = dir.File("source_model.smf");
  const std::string out_path = dir.File("updated_model.smf");
  ASSERT_OK(SaveSmf(src_path, model));

  // Saving may relocate data offsets: recompile against the saved artifact so
  // the emit table matches the file the runtime patches (the real workflow).
  ASSERT_OK_AND_ASSIGN(SmfModel saved, LoadSmf(src_path));
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled2,
                       UpdateCompiler(config).Compile(saved));
  UpdateEngine engine2;
  ASSERT_OK(
      engine2.LoadFromMemory(compiled2.plan.data(), compiled2.plan.size()));
  ASSERT_OK_AND_ASSIGN(auto report2, engine2.Train(data, 400, Quiet()));
  EXPECT_TRUE(report2.improved());
  ASSERT_OK(engine2.RunMerge());
  ASSERT_OK(engine2.CommitToModel(src_path, out_path));

  ASSERT_OK_AND_ASSIGN(SmfModel updated, LoadSmf(out_path));
  for (const auto& a : compiled2.adapters) {
    const SmfTensor* before = saved.FindTensor(a.weight_name);
    const SmfTensor* t = updated.FindTensor(a.weight_name);
    ASSERT_NE(before, nullptr);
    ASSERT_NE(t, nullptr);
    // Commit adds the delta onto the file's pristine weights: W' = W + Δ.
    const auto* orig = reinterpret_cast<const float*>(before->data.data());
    const auto* patched = reinterpret_cast<const float*>(t->data.data());
    const auto* delta = reinterpret_cast<const float*>(
        engine2.arena() + RefOffset(a.delta_ref));
    bool all_equal = true;
    for (int64_t i = 0; i < a.k * a.m; ++i)
      if (patched[i] != orig[i] + delta[i]) all_equal = false;
    EXPECT_TRUE(all_equal);
  }
  // Bias tensors were not adapted: must be bit-identical to the source.
  for (const char* name : {"b1", "b2"}) {
    const SmfTensor* before = saved.FindTensor(name);
    const SmfTensor* after = updated.FindTensor(name);
    ASSERT_NE(before, nullptr);
    ASSERT_NE(after, nullptr);
    EXPECT_TRUE(before->data == after->data);
  }
}

// =============================================================================
// 4. MSE regression path: dense-target training converges
// =============================================================================

TEST(UpdateSystem, MseRegressionConverges) {
  const int64_t in_dim = 6, hidden = 12, out_dim = 2, batch = 8;
  SmfModel model = MakeMlp(in_dim, hidden, out_dim, 17);

  UpdateConfig config = BaseConfig(batch);
  config.loss = LossKind::kMse;
  config.optimizer.lr = 5e-3f;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));

  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));

  ASSERT_OK_AND_ASSIGN(Dataset data,
                       MakeRegressionData(256, in_dim, out_dim, 18));
  ASSERT_OK_AND_ASSIGN(auto report, engine.Train(data, 300, Quiet()));
  EXPECT_LT(report.final_avg_loss, report.initial_avg_loss * 0.9f);
}

// =============================================================================
// 5. Distillation from an open-weights teacher (no labels)
// =============================================================================

TEST(UpdateSystem, DistillationFromTeacherConverges) {
  const int64_t in_dim = 8, hidden = 12, out_dim = 4, batch = 8;
  SmfModel student = MakeMlp(in_dim, hidden, out_dim, 21);
  SmfModel teacher = MakeMlp(in_dim, 20, out_dim, 22);  // different capacity

  UpdateConfig config = BaseConfig(batch);
  config.loss = LossKind::kKLDistill;
  config.temperature = 2.0f;
  config.optimizer.lr = 5e-3f;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(student, &teacher));
  // Teacher weights ride along frozen; only the student's two MatMuls adapt.
  EXPECT_EQ(compiled.adapters.size(), 2u);

  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));

  // Unlabeled corpus: the teacher provides the training signal in-graph.
  ASSERT_OK_AND_ASSIGN(Dataset data, MakeUnlabeledData(256, in_dim, 31));
  ASSERT_OK_AND_ASSIGN(auto report, engine.Train(data, 300, Quiet()));
  EXPECT_LT(report.final_avg_loss, report.initial_avg_loss * 0.9f);
}

TEST(UpdateSystem, TeacherFusionChangesNoLossBits) {
  // The roadmap's fusion contract: a fused and an unfused compilation of
  // the same distillation update must produce bit-identical training — the
  // epilogue evaluates exactly the floats the standalone instruction
  // sequence would, just without the arena round-trips. Exact float
  // equality on every step's loss and on the eval loss, not tolerances.
  const int64_t in_dim = 8, hidden = 12, out_dim = 4, batch = 8;
  SmfModel student = MakeMlp(in_dim, hidden, out_dim, 41);
  SmfModel teacher = MakeMlp(in_dim, 20, out_dim, 42);

  UpdateConfig config = BaseConfig(batch);
  config.loss = LossKind::kKLDistill;
  config.optimizer.lr = 5e-3f;
  UpdateConfig unfused_config = config;
  unfused_config.fuse_epilogues = false;

  ASSERT_OK_AND_ASSIGN(CompiledUpdate fused,
                       UpdateCompiler(config).Compile(student, &teacher));
  ASSERT_OK_AND_ASSIGN(
      CompiledUpdate unfused,
      UpdateCompiler(unfused_config).Compile(student, &teacher));
  ASSERT_LT(HeaderOf(fused).train_instr_count,
            HeaderOf(unfused).train_instr_count);

  float losses[2][2];        // [engine][pre/post]
  std::vector<float> curves[2];
  const CompiledUpdate* plans[2] = {&fused, &unfused};
  for (int i = 0; i < 2; ++i) {
    UpdateEngine engine;
    ASSERT_OK(engine.LoadFromMemory(plans[i]->plan.data(),
                                    plans[i]->plan.size()));
    ASSERT_OK_AND_ASSIGN(Dataset data, MakeUnlabeledData(128, in_dim, 51));
    data.EnableShuffle(3);
    ASSERT_OK_AND_ASSIGN(float pre, engine.Evaluate(data));
    TrainOptions options = Quiet();
    options.record_loss_curve = true;
    ASSERT_OK_AND_ASSIGN(auto report, engine.Train(data, 40, options));
    ASSERT_OK_AND_ASSIGN(float post, engine.Evaluate(data));
    losses[i][0] = pre;
    losses[i][1] = post;
    curves[i] = report.loss_curve;
  }

  EXPECT_EQ(losses[0][0], losses[1][0]);
  EXPECT_EQ(losses[0][1], losses[1][1]);
  ASSERT_EQ(curves[0].size(), curves[1].size());
  for (size_t s = 0; s < curves[0].size(); ++s)
    EXPECT_EQ(curves[0][s], curves[1][s]);
}

// =============================================================================
// 6. Composite loss: (1-w)·xent + w·kl against labels + teacher
// =============================================================================

TEST(UpdateSystem, CompositeLossConverges) {
  const int64_t in_dim = 8, hidden = 12, out_dim = 2, batch = 8;
  SmfModel student = MakeMlp(in_dim, hidden, out_dim, 23);
  SmfModel teacher = MakeMlp(in_dim, 16, out_dim, 24);

  UpdateConfig config = BaseConfig(batch);
  config.loss = LossKind::kXEntPlusKL;
  config.distill_weight = 0.5f;
  config.temperature = 2.0f;
  config.optimizer.lr = 5e-3f;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(student, &teacher));

  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));

  ASSERT_OK_AND_ASSIGN(Dataset data, MakeClassificationData(256, in_dim, 25));
  ASSERT_OK_AND_ASSIGN(auto report, engine.Train(data, 300, Quiet()));
  EXPECT_TRUE(report.improved());
}

// =============================================================================
// 7. New operators: GELU / SiLU / Mul / LayerNorm gradient check
// =============================================================================

TEST(UpdateSystem, NewOperatorGradientsMatchFiniteDifferences) {
  const int64_t in_dim = 5, hidden = 8, out_dim = 3, batch = 2;
  SmfModel model = seeml::testing::MakeGatedNet(in_dim, hidden, out_dim, 17);

  UpdateConfig config = BaseConfig(batch);
  config.lora.rank = 3;
  config.emit_optimizer = false;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));
  EXPECT_EQ(compiled.adapters.size(), 3u);  // w1, w2, w3 all adapted

  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));

  std::mt19937_64 rng(4242);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(batch * in_dim);
  for (auto& v : x) v = dist(rng);
  // LayerNorm's per-row normalization amplifies central-difference
  // truncation noise on small gradients: allow 3% instead of 2%.
  FillSlots(engine, x, {1, 2});

  GradientCheck(compiled, engine, 4242, /*tol=*/3e-2);
}

// =============================================================================
// 7b. Transformer operators (plan v6): decoder gradient check + training
// =============================================================================

TEST(UpdateSystem, TransformerGradientsMatchFiniteDifferences) {
  // One pre-norm decoder block (RmsNorm/Rope/causal Attention/SwiGLU): the
  // finite-difference check exercises the whole v6 stack — parser, VJP
  // rules, lowering, validator, and the attention kernels — through the
  // compiled plan.
  const int64_t dim = 8, heads = 2, seq = 4, ffn = 10, vocab = 3;
  const int64_t batch = 8;  // 2 sequences of 4 positions
  SmfModel model = seeml::testing::MakeTinyDecoder(dim, heads, seq, ffn,
                                                   vocab, 23);

  UpdateConfig config = BaseConfig(batch);
  config.lora.rank = 2;
  config.emit_optimizer = false;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));
  EXPECT_EQ(compiled.adapters.size(), 8u);  // wq wk wv wo wg wu wd wh

  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));

  std::mt19937_64 rng(2323);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(batch * dim);
  for (auto& v : x) v = dist(rng);
  FillSlots(engine, x, {1, 2, 0, 1, 2, 0, 1, 2});
  // Normalization layers and the softmax inside attention amplify
  // central-difference truncation noise, as with LayerNorm above.
  GradientCheck(compiled, engine, 2323, /*tol=*/3e-2);
}

TEST(UpdateSystem, TransformerTrainsMergesAndCommits) {
  const int64_t dim = 8, heads = 2, seq = 4, ffn = 16, vocab = 2;
  const int64_t batch = 16;  // 4 sequences of 4 positions
  SmfModel model = seeml::testing::MakeTinyDecoder(dim, heads, seq, ffn,
                                                   vocab, 29);

  UpdateConfig config = BaseConfig(batch);
  config.optimizer.lr = 5e-3f;
  config.optimizer.kind = OptimizerKind::kAdamW;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));

  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));

  ASSERT_OK_AND_ASSIGN(Dataset data, MakeClassificationData(512, dim, 31));
  ASSERT_OK_AND_ASSIGN(auto report, engine.Train(data, 300, Quiet()));
  EXPECT_LT(report.final_avg_loss, report.initial_avg_loss * 0.9f);

  ASSERT_OK(engine.RunMerge());
  ScopedTempDir dir;
  const std::string src = dir.File("decoder.smf");
  ASSERT_OK(SaveSmf(src, model));
  // Rebind the commit to the saved file's identity, as a fleet plan would
  // be: recompile against the on-disk model.
  ASSERT_OK_AND_ASSIGN(SmfModel disk_model, LoadSmf(src));
  ASSERT_OK_AND_ASSIGN(CompiledUpdate bound,
                       UpdateCompiler(config).Compile(disk_model));
  UpdateEngine bound_engine;
  ASSERT_OK(
      bound_engine.LoadFromMemory(bound.plan.data(), bound.plan.size()));
  ASSERT_OK_AND_ASSIGN(auto bound_report,
                       bound_engine.Train(data, 200, Quiet()));
  EXPECT_LT(bound_report.final_avg_loss, bound_report.initial_avg_loss);
  ASSERT_OK(bound_engine.RunMerge());
  const std::string out = dir.File("decoder_updated.smf");
  ASSERT_OK(bound_engine.CommitToModel(src, out));
  // The committed file still loads and differs from the original weights.
  ASSERT_OK_AND_ASSIGN(SmfModel committed, LoadSmf(out));
  EXPECT_NE(committed.content_hash, disk_model.content_hash);
}

// =============================================================================
// 8. Quantized base weights (int8 rodata)
// =============================================================================

TEST(UpdateSystem, TheGateScoresTheModelThatShips) {
  // E12 (#95). Under --quantize-base / --bf16-base the adapter trains
  // against the narrow weights, but the commit patches the f32 source file:
  // the function that ships is W_f32 + delta. A plan compiled from a file
  // scores exactly that — its eval program reads the f32 weights from the
  // bound model — so the gated validation loss IS the committed model's
  // (up to the rounding of W + delta in the file), and the step-0 score is
  // the source model's, not the quantized source's. The in-plan proxy an
  // in-memory compile keeps is further from the truth: that gap is what
  // the gate used to certify.
  const int64_t in_dim = 8, hidden = 16, out_dim = 3, batch = 8;
  for (const bool bf16 : {false, true}) {
    SmfModel model = MakeMlp(in_dim, hidden, out_dim, 71);
    ScopedTempDir dir;
    const std::string src = dir.File("source.smf");
    ASSERT_OK(SaveSmf(src, model));
    ASSERT_OK_AND_ASSIGN(SmfModel saved, LoadSmf(src));
    UpdateConfig config = BaseConfig(batch);
    config.optimizer.lr = 5e-3f;
    config.quantize_base = !bf16;
    config.bf16_base = bf16;
    ASSERT_OK_AND_ASSIGN(CompiledUpdate shipped,
                         UpdateCompiler(config).Compile(saved));
    // SaveSmf stamps offsets and the hash onto `model`: the proxy compiles
    // a fresh, never-saved copy of the same network.
    SmfModel in_memory = MakeMlp(in_dim, hidden, out_dim, 71);
    ASSERT_OK_AND_ASSIGN(CompiledUpdate proxy,
                         UpdateCompiler(config).Compile(in_memory));
    EXPECT_TRUE(shipped.scores_shipped);
    EXPECT_FALSE(proxy.scores_shipped);

    ASSERT_OK_AND_ASSIGN(Dataset train, MakeClassificationData(96, in_dim, 72));
    ASSERT_OK_AND_ASSIGN(Dataset val, MakeClassificationData(40, in_dim, 73));
    TrainOptions options = Quiet();
    options.validation = &val;

    UpdateEngine engine;
    ASSERT_OK(engine.LoadFromMemory(shipped.plan.data(), shipped.plan.size()));
    EXPECT_TRUE(engine.scores_shipped());
    // Unbound, the plan refuses to score anything.
    EXPECT_ERROR_CONTAINS(engine.Evaluate(val), "BindSourceModel");
    ASSERT_OK(engine.BindSourceModel(src));
    ASSERT_OK_AND_ASSIGN(auto report, engine.Train(train, 60, options));
    ASSERT_OK(engine.RunMerge());
    const std::string out = dir.File("updated.smf");
    ASSERT_OK(engine.CommitToModel(src, out));

    UpdateEngine twin;  // the same training, scored by the in-plan proxy
    ASSERT_OK(twin.LoadFromMemory(proxy.plan.data(), proxy.plan.size()));
    ASSERT_OK_AND_ASSIGN(Dataset train2,
                         MakeClassificationData(96, in_dim, 72));
    ASSERT_OK_AND_ASSIGN(auto proxied, twin.Train(train2, 60, options));

    // The truth: an f32 plan compiled from each file, evaluated as is.
    auto f32_loss = [&](const std::string& path) -> float {
      auto m = LoadSmf(path);
      if (!m) return -1.0f;
      UpdateConfig f32 = BaseConfig(batch);
      auto c = UpdateCompiler(f32).Compile(*m);
      if (!c) return -1.0f;
      UpdateEngine e;
      if (!e.LoadFromMemory(c->plan.data(), c->plan.size())) return -1.0f;
      if (!e.BindSourceModel(path)) return -1.0f;
      auto l = e.Evaluate(val);
      return l ? *l : -1.0f;
    };
    const float source_loss = f32_loss(src);
    const float committed_loss = f32_loss(out);
    ASSERT_GT(source_loss, 0.0f);
    ASSERT_GT(committed_loss, 0.0f);
    // Step 0: B = 0, so the shipped eval is the source model exactly.
    EXPECT_EQ(report.val_initial_loss, source_loss);
    // The end: the gated score is the committed model's.
    EXPECT_NEAR(report.val_final_loss, committed_loss,
                2e-5 * (1.0 + committed_loss));
    // The proxy scored something else — at step 0 (the quantized source)
    // and at the end — and the shipped score is the closer.
    EXPECT_NE(proxied.val_initial_loss, source_loss);
    EXPECT_LT(std::fabs(report.val_final_loss - committed_loss),
              std::fabs(proxied.val_final_loss - committed_loss));
    // Training itself is unchanged: both ran against the narrow weights.
    EXPECT_EQ(report.final_avg_loss, proxied.final_avg_loss);
  }
}

TEST(UpdateSystem, QuantizedBaseTrainsAndCommitsWithoutBakingError) {
  const int64_t in_dim = 8, hidden = 16, out_dim = 2, batch = 8;
  SmfModel model = MakeMlp(in_dim, hidden, out_dim, 5);

  UpdateConfig config = BaseConfig(batch);
  config.optimizer.lr = 5e-3f;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate f32_plan,
                       UpdateCompiler(config).Compile(model));

  config.quantize_base = true;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));

  // MatMul weights quantize (biases stay f32): rodata shrinks.
  EXPECT_LT(compiled.rodata_size, f32_plan.rodata_size);
  for (const auto& a : compiled.adapters) EXPECT_GT(a.quant_scale, 0.0f);

  // The quantized network is the function being trained: its compiled
  // backward must still match finite differences of its compiled forward.
  // Weights, inputs and nudges are portable bits: this network has 128
  // ReLU pre-activations per batch, and on normal_distribution data a
  // finite-difference step lands on one of them for some platforms and not
  // others (see GradientCheck).
  {
    std::mt19937_64 rng(7);
    SmfModel gmodel = MakeMlp(in_dim, hidden, out_dim, 5);
    for (auto& t : gmodel.tensors) {
      if (!t.is_const) continue;
      std::vector<float> w(t.data.size() / sizeof(float));
      for (auto& v : w) v = 0.5f * PortableUniform(rng);
      std::memcpy(t.data.data(), w.data(), t.data.size());
    }
    UpdateConfig gc = config;
    gc.emit_optimizer = false;
    ASSERT_OK_AND_ASSIGN(CompiledUpdate gplan,
                         UpdateCompiler(gc).Compile(gmodel));
    UpdateEngine ge;
    ASSERT_OK(ge.LoadFromMemory(gplan.plan.data(), gplan.plan.size()));
    std::vector<float> x(batch * in_dim);
    for (auto& v : x) v = PortableUniform(rng);
    std::vector<int32_t> labels(batch);
    for (auto& l : labels) l = static_cast<int32_t>(rng() % out_dim);
    FillSlots(ge, x, labels);
    GradientCheck(gplan, ge, 7, /*tol=*/2e-2, /*portable=*/true);
  }

  // End-to-end on the saved artifact: train, merge, commit. The committed
  // weights must be the file's pristine f32 weights plus the delta — the
  // quantization error must NOT be baked in.
  ScopedTempDir dir;
  const std::string src_path = dir.File("quant_source.smf");
  ASSERT_OK(SaveSmf(src_path, model));
  ASSERT_OK_AND_ASSIGN(SmfModel saved, LoadSmf(src_path));
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled2,
                       UpdateCompiler(config).Compile(saved));

  UpdateEngine engine;
  ASSERT_OK(
      engine.LoadFromMemory(compiled2.plan.data(), compiled2.plan.size()));
  ASSERT_OK_AND_ASSIGN(Dataset data, MakeClassificationData(512, in_dim, 11));
  ASSERT_OK_AND_ASSIGN(auto report, engine.Train(data, 400, Quiet()));
  EXPECT_LT(report.final_avg_loss, report.initial_avg_loss * 0.8f);
  ASSERT_OK(engine.RunMerge());
  const std::string out_path = dir.File("quant_updated.smf");
  ASSERT_OK(engine.CommitToModel(src_path, out_path));

  ASSERT_OK_AND_ASSIGN(SmfModel updated, LoadSmf(out_path));
  for (const auto& a : compiled2.adapters) {
    const SmfTensor* before = saved.FindTensor(a.weight_name);
    const SmfTensor* after = updated.FindTensor(a.weight_name);
    ASSERT_NE(before, nullptr);
    ASSERT_NE(after, nullptr);
    const auto* orig = reinterpret_cast<const float*>(before->data.data());
    const auto* patched = reinterpret_cast<const float*>(after->data.data());
    const auto* delta = reinterpret_cast<const float*>(
        engine.arena() + RefOffset(a.delta_ref));
    bool exact = true;
    for (int64_t i = 0; i < a.k * a.m; ++i)
      if (patched[i] != orig[i] + delta[i]) exact = false;
    EXPECT_TRUE(exact);
  }
}

// =============================================================================
// 9. Poisoned corpus aborts the update
// =============================================================================

// =============================================================================
// 7c. Token-native input (plan v7): embedding gather end to end
// =============================================================================

TEST(UpdateSystem, TokenDecoderGradientsMatchFiniteDifferences) {
  const int64_t vocab = 11, dim = 8, heads = 2, seq = 4, ffn = 10;
  const int64_t batch = 8;  // 2 sequences of 4 token rows
  SmfModel model =
      seeml::testing::MakeTinyTokenDecoder(vocab, dim, heads, seq, ffn, 37);
  UpdateConfig config = BaseConfig(batch);
  config.lora.rank = 2;
  config.emit_optimizer = false;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));

  // Tokens follow the corpus's successor rule so labels are the true next
  // ids — the exact staging a token Dataset would produce.
  std::vector<int32_t> tokens(batch), labels(batch);
  int32_t t = 3;
  for (int64_t i = 0; i < batch; ++i) {
    tokens[i] = t;
    t = static_cast<int32_t>((3 * t + 1) % vocab);
    labels[i] = t;
  }
  seeml::testing::FillTokenSlots(engine, tokens, labels);
  GradientCheck(compiled, engine, 3737, /*tol=*/3e-2);
}

TEST(UpdateSystem, TokenDecoderResumeMatchesUninterruptedRun) {
  const int64_t vocab = 16, dim = 8, heads = 2, seq = 4, ffn = 16;
  const int64_t batch = 16;  // 4 records per step: on resume the feeder
                             // must advance in records, not rows
  SmfModel model =
      seeml::testing::MakeTinyTokenDecoder(vocab, dim, heads, seq, ffn, 41);
  UpdateConfig config = BaseConfig(batch);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));

  // The uninterrupted twin: 6 steps over a shuffled 10-record corpus, so
  // step 3 crosses an epoch boundary and the replay covers a reshuffle.
  UpdateEngine straight;
  ASSERT_OK(
      straight.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));
  ASSERT_OK_AND_ASSIGN(Dataset d1,
                       seeml::testing::MakeTokenCorpus(10, seq, vocab, 43));
  d1.EnableShuffle(7);
  ASSERT_OK(straight.Train(d1, 6, Quiet()));

  // The interrupted twin: 3 steps, checkpoint, process death, cold resume.
  ScopedTempDir dir;
  const std::string ck = dir.File("ck.bin");
  {
    UpdateEngine first;
    ASSERT_OK(
        first.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));
    ASSERT_OK_AND_ASSIGN(Dataset d2,
                         seeml::testing::MakeTokenCorpus(10, seq, vocab, 43));
    d2.EnableShuffle(7);
    ASSERT_OK(first.Train(d2, 3, Quiet()));
    ASSERT_OK(first.SaveCheckpoint(ck));
  }
  UpdateEngine resumed;
  ASSERT_OK(
      resumed.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));
  ASSERT_OK_AND_ASSIGN(Dataset d3,
                       seeml::testing::MakeTokenCorpus(10, seq, vocab, 43));
  d3.EnableShuffle(7);
  TrainOptions opt = Quiet();
  opt.checkpoint_path = ck;
  opt.resume = true;
  ASSERT_OK(resumed.Train(d3, 3, opt));

  // Guard against a vacuous pass: both engines must actually have trained
  // to step 6 before the byte comparison means anything.
  EXPECT_EQ(straight.step(), 6u);
  EXPECT_EQ(resumed.step(), 6u);
  EXPECT_EQ(std::memcmp(straight.arena(), resumed.arena(),
                        straight.header().persistent_size),
            0);
}

TEST(UpdateSystem, TokenDecoderTrainsMergesAndCommits) {
  const int64_t vocab = 16, dim = 8, heads = 2, seq = 4, ffn = 16;
  const int64_t batch = 16;  // 4 sequences per step
  SmfModel model =
      seeml::testing::MakeTinyTokenDecoder(vocab, dim, heads, seq, ffn, 41);
  UpdateConfig config = BaseConfig(batch);
  config.optimizer.lr = 5e-3f;
  config.optimizer.kind = OptimizerKind::kAdamW;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));

  ASSERT_OK_AND_ASSIGN(Dataset data,
                       seeml::testing::MakeTokenCorpus(256, 4, vocab, 43));
  ASSERT_OK_AND_ASSIGN(auto report, engine.Train(data, 300, Quiet()));
  // The successor rule is deterministic, so next-token loss is learnable.
  EXPECT_LT(report.final_avg_loss, report.initial_avg_loss * 0.7f);

  ASSERT_OK(engine.RunMerge());
  ScopedTempDir dir;
  const std::string src = dir.File("token_decoder.smf");
  ASSERT_OK(SaveSmf(src, model));
  ASSERT_OK_AND_ASSIGN(SmfModel disk_model, LoadSmf(src));
  ASSERT_OK_AND_ASSIGN(CompiledUpdate bound,
                       UpdateCompiler(config).Compile(disk_model));
  UpdateEngine bound_engine;
  ASSERT_OK(bound_engine.LoadFromMemory(bound.plan.data(), bound.plan.size()));
  ASSERT_OK_AND_ASSIGN(auto bound_report,
                       bound_engine.Train(data, 200, Quiet()));
  EXPECT_LT(bound_report.final_avg_loss, bound_report.initial_avg_loss);
  ASSERT_OK(bound_engine.RunMerge());
  const std::string out = dir.File("token_decoder_updated.smf");
  ASSERT_OK(bound_engine.CommitToModel(src, out));
  ASSERT_OK_AND_ASSIGN(SmfModel committed, LoadSmf(out));
  EXPECT_NE(committed.content_hash, disk_model.content_hash);
}

TEST(UpdateSystem, AdapterInitializationIsWellDistributed) {
  // Bounds on the counter-based randn's output space: correct scale (std
  // 1/sqrt(K)), near-zero mean, and no serial correlation between adjacent
  // elements — the regression surface of the overlapping-counter defect,
  // which coupled every sample's magnitude to its neighbor's angle while
  // keeping the marginals exact. Deterministic (seeded), so exact-bound
  // assertions cannot flake.
  const int64_t in_dim = 64, hidden = 96, out_dim = 4, batch = 4;
  SmfModel model = MakeMlp(in_dim, hidden, out_dim, 21);
  UpdateConfig config = BaseConfig(batch);
  config.lora.rank = 16;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));

  // w1's adapter A is [in_dim, rank] randn at std 1/sqrt(in_dim).
  const ParamDebugInfo* a_param = nullptr;
  for (const auto& p : compiled.params)
    if (p.id == "w1.lora_A") a_param = &p;
  ASSERT_NE(a_param, nullptr);
  const size_t n = a_param->count;
  ASSERT_EQ(n, static_cast<size_t>(in_dim * 16));

  std::vector<double> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = ReadArenaF32(engine, a_param->param_ref, i);
  double sum = 0.0, sq = 0.0, lag = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sum += v[i];
    sq += v[i] * v[i];
  }
  const double mean = sum / static_cast<double>(n);
  const double var = sq / static_cast<double>(n) - mean * mean;
  const double expect_std = 1.0 / std::sqrt(static_cast<double>(in_dim));
  for (size_t i = 0; i + 1 < n; ++i)
    lag += (v[i] - mean) * (v[i + 1] - mean);
  const double lag1 = lag / (static_cast<double>(n) * var);

  // 5-sigma-style bounds at n = 1024: mean within 5*std/sqrt(n), std
  // within 15%, |lag-1 autocorrelation| under 5/sqrt(n).
  EXPECT_TRUE(std::fabs(mean) < 5.0 * expect_std / std::sqrt(1.0 * n));
  EXPECT_TRUE(std::fabs(std::sqrt(var) / expect_std - 1.0) < 0.15);
  EXPECT_TRUE(std::fabs(lag1) < 5.0 / std::sqrt(1.0 * n));
}

TEST(UpdateSystem, ConcurrentEnginesShareThePoolDeterministically) {
  // Bounded nondeterminism at the process level: two engines training in
  // parallel threads share the global worker pool and each runs its own
  // feeder thread, yet every engine's result must be bitwise identical to
  // its serial run — chunk geometry depends on problem shape, never on
  // who else is submitting.
  const int64_t batch = 8;
  auto run = [&](uint64_t model_seed, uint64_t data_seed,
                 std::vector<uint8_t>* persist_out) {
    SmfModel model = MakeMlp(6, 12, 2, model_seed);
    UpdateConfig config = BaseConfig(batch);
    config.optimizer.lr = 5e-3f;
    auto compiled = UpdateCompiler(config).Compile(model);
    if (!compiled) return false;
    UpdateEngine engine;
    if (!engine.LoadFromMemory(compiled->plan.data(), compiled->plan.size()))
      return false;
    auto data = MakeClassificationData(128, 6, data_seed);
    if (!data) return false;
    if (!engine.Train(*data, 30, Quiet())) return false;
    persist_out->assign(engine.arena(),
                        engine.arena() + engine.header().persistent_size);
    return true;
  };

  std::vector<uint8_t> serial_a, serial_b, threaded_a, threaded_b;
  ASSERT_TRUE(run(31, 41, &serial_a));
  ASSERT_TRUE(run(32, 42, &serial_b));

  bool ok_a = false, ok_b = false;
  std::thread ta([&] { ok_a = run(31, 41, &threaded_a); });
  std::thread tb([&] { ok_b = run(32, 42, &threaded_b); });
  ta.join();
  tb.join();
  ASSERT_TRUE(ok_a);
  ASSERT_TRUE(ok_b);
  EXPECT_TRUE(threaded_a == serial_a);
  EXPECT_TRUE(threaded_b == serial_b);
}

TEST(UpdateSystem, NonFiniteLossAbortsTheUpdate) {
  const int64_t in_dim = 5, hidden = 8, out_dim = 3, batch = 8;
  SmfModel gated = seeml::testing::MakeGatedNet(in_dim, hidden, out_dim, 17);
  UpdateConfig config = BaseConfig(batch);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(gated));
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));

  // An infinite feature blows the logits up to inf/NaN through the gated
  // path (GELU/SiLU/LayerNorm propagate non-finites; ReLU would scrub them).
  std::vector<float> inputs(64 * in_dim, 1.0f);
  inputs[3] = std::numeric_limits<float>::infinity();
  std::vector<uint8_t> labels(64 * sizeof(int32_t), 0);
  ASSERT_OK_AND_ASSIGN(
      Dataset data, Dataset::FromMemory(std::move(inputs), std::move(labels),
                                        64, in_dim, 1, 0));
  EXPECT_ERROR_CONTAINS(engine.Train(data, 50, Quiet()), "non-finite");
}

// =============================================================================
// The bitwise-safe kernel batch (E3, #82): RoPE angle table, fused clip
// =============================================================================

TEST(UpdateSystem, KernelBatchChangesNoTrainingBits) {
  // Plan v12's two compiler-visible items, each switchable so this can be
  // said exactly: with or without the hoisted RoPE table, with or without
  // the clip folded into the optimizer step, at one thread or eight, under
  // accumulation or not — the same losses, the same validation loss and
  // the same persistent segment (adapters, AdamW moments, accumulators),
  // byte for byte.
  const int64_t vocab = 16, dim = 8, heads = 2, seq = 4, ffn = 16, batch = 16;
  SmfModel model =
      seeml::testing::MakeTinyTokenDecoder(vocab, dim, heads, seq, ffn, 41);
  for (const uint32_t accum : {1u, 2u}) {
    std::vector<float> want_curve;
    std::vector<uint8_t> want_state;
    float want_val = 0.0f;
    for (int variant = 0; variant < 8; ++variant) {
      UpdateConfig config = BaseConfig(batch);
      config.optimizer.lr = 5e-3f;
      config.optimizer.kind = OptimizerKind::kAdamW;
      config.optimizer.clip_norm = 0.05f;  // small enough to clip every step
      config.grad_accum_steps = accum;
      config.rope_table = (variant & 1) != 0;
      config.fuse_clip = (variant & 2) != 0;
      seeml::update::SetParallelThreadCount((variant & 4) ? 8 : 1);
      ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                           UpdateCompiler(config).Compile(model));
      const PlanHeader h = HeaderOf(compiled);
      UpdateEngine engine;
      ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(),
                                      compiled.plan.size()));
      ASSERT_OK_AND_ASSIGN(Dataset data,
                           seeml::testing::MakeTokenCorpus(128, 4, vocab, 43));
      data.EnableShuffle(9);
      TrainOptions options = Quiet();
      options.record_loss_curve = true;
      ASSERT_OK_AND_ASSIGN(auto report, engine.Train(data, 25, options));
      ASSERT_OK_AND_ASSIGN(float val, engine.Evaluate(data));
      std::vector<uint8_t> state(engine.arena(),
                                 engine.arena() + h.persistent_size);
      if (variant == 0) {
        want_curve = report.loss_curve;
        want_state = state;
        want_val = val;
        continue;
      }
      EXPECT_EQ(val, want_val);
      ASSERT_EQ(report.loss_curve.size(), want_curve.size());
      for (size_t i = 0; i < want_curve.size(); ++i)
        EXPECT_EQ(report.loss_curve[i], want_curve[i]);
      EXPECT_TRUE(state == want_state);
    }
  }
  seeml::update::SetParallelThreadCount(0);
}

TEST(UpdateSystem, RelaxedArithmeticChangesNoCpuBits) {
  // Plan v18: --precision certified-bf16 marks the frozen-weight GEMMs
  // kFlagRelaxed — a permission the CPU backend never takes. So at every
  // storage precision (f32, int8, bf16 base) the relaxed plan trains the
  // same losses, the same validation loss and the same persistent segment
  // as the exact plan, byte for byte: the format change is inert on the
  // reference backend, and what a certified kernel may change is bounded
  // elsewhere (tool/certify_numerics.py).
  const int64_t vocab = 16, dim = 8, heads = 2, seq = 4, ffn = 16, batch = 16;
  SmfModel model =
      seeml::testing::MakeTinyTokenDecoder(vocab, dim, heads, seq, ffn, 41);
  for (int storage = 0; storage < 3; ++storage) {
    std::vector<float> want_curve;
    std::vector<uint8_t> want_state;
    float want_val = 0.0f;
    uint64_t want_relaxed = 0;
    for (int relaxed = 0; relaxed < 2; ++relaxed) {
      UpdateConfig config = BaseConfig(batch);
      config.optimizer.lr = 5e-3f;
      config.optimizer.kind = OptimizerKind::kAdamW;
      config.quantize_base = storage == 1;
      config.bf16_base = storage == 2;
      config.precision = relaxed ? Precision::kCertifiedBf16
                                 : Precision::kF32;
      ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                           UpdateCompiler(config).Compile(model));
      const PlanHeader h = HeaderOf(compiled);
      UpdateEngine engine;
      ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(),
                                      compiled.plan.size()));
      ASSERT_OK_AND_ASSIGN(Dataset data,
                           seeml::testing::MakeTokenCorpus(128, 4, vocab, 43));
      data.EnableShuffle(9);
      TrainOptions options = Quiet();
      options.record_loss_curve = true;
      ASSERT_OK_AND_ASSIGN(auto report, engine.Train(data, 25, options));
      ASSERT_OK_AND_ASSIGN(float val, engine.Evaluate(data));
      std::vector<uint8_t> state(engine.arena(),
                                 engine.arena() + h.persistent_size);
      if (!relaxed) {
        EXPECT_EQ(compiled.relaxed_gemms, 0u);
        want_curve = report.loss_curve;
        want_state = state;
        want_val = val;
        continue;
      }
      EXPECT_GT(compiled.relaxed_gemms, 0u);
      ASSERT_EQ(report.loss_curve.size(), want_curve.size());
      if (seeml::update_rt::CpuBackendRelaxedIsExact() || storage != 0) {
        // The reference build, and the int8 / bf16 bases everywhere: the
        // bit is inert.
        EXPECT_EQ(val, want_val);
        for (size_t i = 0; i < want_curve.size(); ++i)
          EXPECT_EQ(report.loss_curve[i], want_curve[i]);
        EXPECT_TRUE(state == want_state);
      } else {
        // SEEML_ACCELERATE: the f32-weight GEMMs ran on SGEMM — close to
        // the exact run, and reproducible against itself.
        EXPECT_NEAR(val, want_val, 1e-3f * std::fabs(want_val) + 1e-4f);
        for (size_t i = 0; i < want_curve.size(); ++i)
          EXPECT_NEAR(report.loss_curve[i], want_curve[i],
                      1e-2f * std::fabs(want_curve[i]) + 1e-4f);
        UpdateEngine again;
        ASSERT_OK(again.LoadFromMemory(compiled.plan.data(),
                                       compiled.plan.size()));
        ASSERT_OK_AND_ASSIGN(Dataset data2, seeml::testing::MakeTokenCorpus(
                                                128, 4, vocab, 43));
        data2.EnableShuffle(9);
        ASSERT_OK_AND_ASSIGN(auto report2, again.Train(data2, 25, options));
        std::vector<uint8_t> state2(again.arena(),
                                    again.arena() + h.persistent_size);
        EXPECT_TRUE(state == state2);
        for (size_t i = 0; i < report.loss_curve.size(); ++i)
          EXPECT_EQ(report.loss_curve[i], report2.loss_curve[i]);
      }
      want_relaxed = compiled.relaxed_gemms;
    }
    EXPECT_GT(want_relaxed, 0u);
  }
}

TEST(UpdateSystem, ANonFiniteGradientNormAbortsTheStep) {
  // The fused clip is the one kernel that can refuse (E3): an overflowed
  // gradient used to be scaled by 0 * inf = NaN into the adapters and the
  // AdamW moments, surfacing only as the NEXT step's non-finite loss —
  // after a checkpoint could have persisted the poison.
  const int64_t in_dim = 4, batch = 8;
  SmfModel model = MakeMlp(in_dim, 8, 3, 61);
  UpdateConfig config = BaseConfig(batch);
  config.optimizer.clip_norm = 1.0f;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));
  const PlanHeader h = HeaderOf(compiled);
  // Inputs near FLT_MAX: finite logits are impossible, so the backward
  // pass hands the step a gradient whose norm is not finite.
  std::vector<float> inputs(64 * in_dim, 3e38f);
  std::vector<uint8_t> labels(64 * sizeof(int32_t), 0);
  ASSERT_OK_AND_ASSIGN(
      Dataset data, Dataset::FromMemory(std::move(inputs), std::move(labels),
                                        64, in_dim, 1, 0));
  auto trained = engine.Train(data, 5, Quiet());
  ASSERT_FALSE(trained.has_value());
  const bool named = trained.error().find("not finite") != std::string::npos ||
                     trained.error().find("non-finite") != std::string::npos;
  EXPECT_TRUE(named);
  // Whichever guard fired first, no tensor stepped on the poison: tensors
  // whose own norm was finite may have stepped before the refusal (a
  // failed Train ends the engine's usable state, and checkpoints are only
  // written after a whole step), but nothing persistent is non-finite.
  const auto* state = reinterpret_cast<const float*>(engine.arena());
  for (uint64_t i = 0; i < h.persistent_size / sizeof(float); ++i)
    ASSERT_TRUE(std::isfinite(state[i]));
}

// =============================================================================
// Tiled attention (E11, #94)
// =============================================================================

TEST(UpdateSystem, TiledAttentionTrainsTheSameBitsWithoutTheCache) {
  // The cached and the tiled family compile the same update to plans that
  // train to the same losses, validation loss and persistent segment, at
  // one thread and at eight; the tiled plan holds no S x S matrix, and the
  // FD check passes on it as on the cached one.
  const int64_t vocab = 16, dim = 8, heads = 2, seq = 8, ffn = 16;
  const int64_t batch = 16;
  SmfModel model =
      seeml::testing::MakeDecoderStack(dim, heads, seq, ffn, vocab, 2, 61);
  SmfModel tokens =
      seeml::testing::MakeTinyTokenDecoder(vocab, dim, heads, seq, ffn, 62);
  for (const SmfModel* m : {&tokens}) {
    std::vector<float> want_curve;
    std::vector<uint8_t> want_state;
    float want_val = 0.0f;
    uint64_t cached_arena = 0;
    for (int variant = 0; variant < 4; ++variant) {
      UpdateConfig config = BaseConfig(batch);
      config.optimizer.lr = 5e-3f;
      config.attention = (variant & 1) ? AttentionKind::kTiled
                                       : AttentionKind::kCached;
      seeml::update::SetParallelThreadCount((variant & 2) ? 8 : 1);
      ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                           UpdateCompiler(config).Compile(*m));
      EXPECT_EQ(compiled.attention_tiled, (variant & 1) != 0);
      EXPECT_GT(compiled.probs_cache_bytes, 0u);
      const PlanHeader h = HeaderOf(compiled);
      if (variant == 0) cached_arena = h.arena_size;
      if (variant == 1) EXPECT_LT(h.arena_size, cached_arena);
      UpdateEngine engine;
      ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(),
                                      compiled.plan.size()));
      ASSERT_OK_AND_ASSIGN(Dataset data,
                           seeml::testing::MakeTokenCorpus(128, seq, vocab, 47));
      data.EnableShuffle(5);
      TrainOptions options = Quiet();
      options.record_loss_curve = true;
      ASSERT_OK_AND_ASSIGN(auto report, engine.Train(data, 20, options));
      ASSERT_OK_AND_ASSIGN(float val, engine.Evaluate(data));
      std::vector<uint8_t> state(engine.arena(),
                                 engine.arena() + h.persistent_size);
      if (variant == 0) {
        want_curve = report.loss_curve;
        want_state = state;
        want_val = val;
        continue;
      }
      EXPECT_EQ(val, want_val);
      ASSERT_EQ(report.loss_curve.size(), want_curve.size());
      for (size_t i = 0; i < want_curve.size(); ++i)
        EXPECT_EQ(report.loss_curve[i], want_curve[i]);
      EXPECT_TRUE(state == want_state);
    }
  }
  seeml::update::SetParallelThreadCount(0);

  // The pre-embedded decoder stack: tiled by the auto rule once its caches
  // exceed a (deliberately tiny) budget, cached under a roomy one.
  UpdateConfig roomy = BaseConfig(batch);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate kept, UpdateCompiler(roomy).Compile(model));
  EXPECT_FALSE(kept.attention_tiled);
  UpdateConfig tight = roomy;
  tight.attention_cache_budget_bytes = kept.probs_cache_bytes - 1;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate tiled, UpdateCompiler(tight).Compile(model));
  EXPECT_TRUE(tiled.attention_tiled);
  EXPECT_LT(HeaderOf(tiled).arena_size, HeaderOf(kept).arena_size);
}

TEST(UpdateSystem, TiledAttentionGradientsMatchFiniteDifferences) {
  const int64_t vocab = 11, dim = 8, heads = 2, seq = 4, ffn = 10;
  const int64_t batch = 8;
  SmfModel model =
      seeml::testing::MakeTinyTokenDecoder(vocab, dim, heads, seq, ffn, 37);
  UpdateConfig config = BaseConfig(batch);
  config.lora.rank = 2;
  config.emit_optimizer = false;
  config.attention = AttentionKind::kTiled;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));
  EXPECT_TRUE(compiled.attention_tiled);
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(), compiled.plan.size()));
  std::vector<int32_t> tokens(batch), labels(batch);
  int32_t t = 3;
  for (int64_t i = 0; i < batch; ++i) {
    tokens[i] = t;
    t = static_cast<int32_t>((3 * t + 1) % vocab);
    labels[i] = t;
  }
  seeml::testing::FillTokenSlots(engine, tokens, labels);
  GradientCheck(compiled, engine, 3737, /*tol=*/3e-2);
}

// =============================================================================
// The GEMM addend (E10, #93): kFlagGemmAddend
// =============================================================================

TEST(UpdateSystem, TheGemmAddendChangesNoTrainingBits) {
  // Folded and unfolded compilations of the same update train to the same
  // losses, validation loss and persistent segment, at one thread and at
  // eight — an MLP, a token decoder, the decoder over int8 weights, and a
  // tied-weight net (fan-out sums on parameter gradients). alpha/r = 3/4
  // is deliberately not a power of two: the fold must be neutral for any
  // scale, not only where scaling happens to be exact.
  struct Case {
    SmfModel model;
    bool tokens;
    bool quantize;
    int64_t in_dim;
  };
  const int64_t vocab = 16, seq = 4, batch = 16;
  std::vector<Case> cases;
  cases.push_back({MakeMlp(8, 12, 4, 91), false, false, 8});
  cases.push_back({seeml::testing::MakeTinyTokenDecoder(vocab, 8, 2, seq, 16, 92),
                   true, false, 0});
  cases.push_back({seeml::testing::MakeTinyTokenDecoder(vocab, 8, 2, seq, 16, 93),
                   true, true, 0});
  cases.push_back({seeml::testing::MakeTiedMlp(8, 94), false, false, 8});
  for (const Case& c : cases) {
    std::vector<float> want_curve;
    std::vector<uint8_t> want_state;
    float want_val = 0.0f;
    uint64_t unfolded_instrs = 0;
    for (int variant = 0; variant < 4; ++variant) {
      UpdateConfig config = BaseConfig(batch);
      config.optimizer.lr = 5e-3f;
      config.lora.alpha = 3.0f;
      config.quantize_base = c.quantize;
      config.fuse_gemm_addend = (variant & 1) != 0;
      seeml::update::SetParallelThreadCount((variant & 2) ? 8 : 1);
      ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                           UpdateCompiler(config).Compile(c.model));
      const PlanHeader h = HeaderOf(compiled);
      if (variant == 0) unfolded_instrs = h.train_instr_count;
      if (variant == 1) EXPECT_LT(h.train_instr_count, unfolded_instrs);
      UpdateEngine engine;
      ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(),
                                      compiled.plan.size()));
      ASSERT_OK_AND_ASSIGN(
          Dataset data,
          c.tokens ? seeml::testing::MakeTokenCorpus(128, seq, vocab, 45)
                   : MakeClassificationData(128, c.in_dim, 46));
      data.EnableShuffle(5);
      TrainOptions options = Quiet();
      options.record_loss_curve = true;
      ASSERT_OK_AND_ASSIGN(auto report, engine.Train(data, 25, options));
      ASSERT_OK_AND_ASSIGN(float val, engine.Evaluate(data));
      std::vector<uint8_t> state(engine.arena(),
                                 engine.arena() + h.persistent_size);
      if (variant == 0) {
        want_curve = report.loss_curve;
        want_state = state;
        want_val = val;
        continue;
      }
      EXPECT_EQ(val, want_val);
      ASSERT_EQ(report.loss_curve.size(), want_curve.size());
      for (size_t i = 0; i < want_curve.size(); ++i)
        EXPECT_EQ(report.loss_curve[i], want_curve[i]);
      EXPECT_TRUE(state == want_state);
    }
  }
  seeml::update::SetParallelThreadCount(0);
}

// =============================================================================
// Elementwise chain fusion (E4, #83): kFusedMap
// =============================================================================

TEST(UpdateSystem, ElementwiseChainFusionChangesNoTrainingBits) {
  // A fused and an unfused compilation of the same update — an MLP, a
  // token decoder, and an MLP distilled under the composite loss (its
  // scale -> add is the chain) — train to the same losses, the same
  // validation loss and the same persistent segment, at one thread and at
  // eight; and where a chain exists the fused plan really is shorter. The
  // chain E4 was built on, scale -> add at every LoRA site, is gone since
  // E10: the scale moved to the rank-r activation and the add into the
  // GEMM, so on a plain LoRA update there is nothing left to fuse.
  struct Case {
    SmfModel model;
    bool tokens;
    bool distill = false;
  };
  const int64_t vocab = 16, seq = 4, batch = 16;
  std::vector<Case> cases;
  cases.push_back({MakeMlp(8, 12, 4, 81), false});
  cases.push_back(
      {seeml::testing::MakeTinyTokenDecoder(vocab, 8, 2, seq, 16, 82), true});
  cases.push_back({MakeMlp(8, 12, 4, 83), false, /*distill=*/true});
  SmfModel teacher = MakeMlp(8, 16, 4, 84);
  for (const Case& c : cases) {
    std::vector<float> want_curve;
    std::vector<uint8_t> want_state;
    float want_val = 0.0f;
    uint64_t unfused_instrs = 0;
    for (int variant = 0; variant < 4; ++variant) {
      UpdateConfig config = BaseConfig(batch);
      config.optimizer.lr = 5e-3f;
      config.fuse_elementwise = (variant & 1) != 0;
      if (c.distill) {
        config.loss = LossKind::kXEntPlusKL;
        config.distill_weight = 0.5f;
      }
      seeml::update::SetParallelThreadCount((variant & 2) ? 8 : 1);
      ASSERT_OK_AND_ASSIGN(
          CompiledUpdate compiled,
          UpdateCompiler(config).Compile(c.model,
                                         c.distill ? &teacher : nullptr));
      const PlanHeader h = HeaderOf(compiled);
      if (variant == 0) unfused_instrs = h.train_instr_count;
      if (variant == 1) {
        if (c.distill)
          EXPECT_LT(h.train_instr_count, unfused_instrs);
        else
          EXPECT_LE(h.train_instr_count, unfused_instrs);
      }
      UpdateEngine engine;
      ASSERT_OK(engine.LoadFromMemory(compiled.plan.data(),
                                      compiled.plan.size()));
      ASSERT_OK_AND_ASSIGN(
          Dataset data,
          c.tokens ? seeml::testing::MakeTokenCorpus(128, seq, vocab, 43)
                   : MakeClassificationData(128, 8, 44));
      data.EnableShuffle(5);
      TrainOptions options = Quiet();
      options.record_loss_curve = true;
      ASSERT_OK_AND_ASSIGN(auto report, engine.Train(data, 25, options));
      ASSERT_OK_AND_ASSIGN(float val, engine.Evaluate(data));
      std::vector<uint8_t> state(engine.arena(),
                                 engine.arena() + h.persistent_size);
      if (variant == 0) {
        want_curve = report.loss_curve;
        want_state = state;
        want_val = val;
        continue;
      }
      EXPECT_EQ(val, want_val);
      ASSERT_EQ(report.loss_curve.size(), want_curve.size());
      for (size_t i = 0; i < want_curve.size(); ++i)
        EXPECT_EQ(report.loss_curve[i], want_curve[i]);
      EXPECT_TRUE(state == want_state);
    }
  }
  seeml::update::SetParallelThreadCount(0);
}

}  // namespace
