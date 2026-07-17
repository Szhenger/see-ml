// =============================================================================
// SeeML Update Compiler — system verification suite.
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
//
// Structural graft coverage lives in the unit suites (update_passes_test,
// update_compiler_test); this suite verifies the numerics end to end.
// =============================================================================

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "compiler/backend/update_compiler.h"
#include "runtime/dataset.h"
#include "runtime/update_engine.h"
#include "source/smf.h"
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

  // Nudge B off its zero init so gradients w.r.t. A are non-degenerate.
  for (const auto& p : compiled.params)
    for (uint64_t i = 0; i < p.count; ++i)
      if (p.id.find(".lora_B") != std::string::npos)
        WriteArenaF32(engine, p.param_ref, i, 0.05f * dist(rng));

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
      if (rel >= 2e-2)
        ADD_FAILURE("param " + p.id + "[" + std::to_string(i) +
                    "]: analytic " + std::to_string(analytic) + " vs numeric " +
                    std::to_string(numeric));
      ++checked;
    }
  }
  EXPECT_GE(checked, 20u);
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

  // --- Merge: W' must equal W + (α/r)·A@B to float precision. ---------------
  ASSERT_OK(engine.RunMerge());
  for (const auto& a : compiled.adapters) {
    const auto* A =
        reinterpret_cast<const float*>(engine.arena() + RefOffset(a.a_ref));
    const auto* B =
        reinterpret_cast<const float*>(engine.arena() + RefOffset(a.b_ref));
    const auto* merged = reinterpret_cast<const float*>(
        engine.arena() + RefOffset(a.merged_ref));
    const SmfTensor* w_src = model.FindTensor(a.weight_name);
    ASSERT_NE(w_src, nullptr);
    const auto* W = reinterpret_cast<const float*>(w_src->data.data());

    double max_err = 0.0, max_delta = 0.0;
    for (int64_t i = 0; i < a.k; ++i)
      for (int64_t j = 0; j < a.m; ++j) {
        double delta = 0.0;
        for (int64_t t = 0; t < a.r; ++t)
          delta += static_cast<double>(A[i * a.r + t]) * B[t * a.m + j];
        const double expected = W[i * a.m + j] + a.scale * delta;
        max_err =
            std::max(max_err, std::fabs(expected - merged[i * a.m + j]));
        max_delta = std::max(max_delta, std::fabs(a.scale * delta));
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
    const SmfTensor* t = updated.FindTensor(a.weight_name);
    ASSERT_NE(t, nullptr);
    const auto* patched = reinterpret_cast<const float*>(t->data.data());
    const auto* merged = reinterpret_cast<const float*>(
        engine2.arena() + RefOffset(a.merged_ref));
    bool all_equal = true;
    for (int64_t i = 0; i < a.k * a.m; ++i)
      if (patched[i] != merged[i]) all_equal = false;
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

}  // namespace
