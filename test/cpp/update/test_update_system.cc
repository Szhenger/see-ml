// =============================================================================
// SeeML Update Compiler — system verification suite.
//
//   1. Grafting structure: adapters injected, consumers rewired, SSA valid.
//   2. Identity at step 0: B = 0 ⇒ the compiled update starts as the exact
//      source model (loss equals the ungrafted model's loss).
//   3. Gradient check: the AOT-compiled backward pass matches central finite
//      differences of the compiled forward pass (the mathematical oracle).
//   4. End-to-end update: train → loss decreases; merge → W' = W + (α/r)·A@B
//      to float precision; commit → the patched SMF matches, untouched
//      weights bit-identical, transactional output.
//   5. Distillation: a KL-loss plan against a teacher model compiles, runs,
//      and reduces the student/teacher divergence.
//
// Plain-assert harness: no external test framework required.
// =============================================================================

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "source/update/smf.h"
#include "source/update/update_compiler.h"
#include "source/update_runtime/dataset.h"
#include "source/update_runtime/update_engine.h"

namespace {

using namespace seeml::update;
using seeml::update_rt::Dataset;
using seeml::update_rt::UpdateEngine;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__,        \
                   __LINE__, #cond);                                       \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

#define CHECK_MSG(cond, ...)                                               \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n  ", __FILE__,      \
                   __LINE__, #cond);                                       \
      std::fprintf(stderr, __VA_ARGS__);                                   \
      std::fprintf(stderr, "\n");                                          \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

std::string TempDir() {
  auto dir = std::filesystem::temp_directory_path() / "seeml_update_test";
  std::filesystem::create_directories(dir);
  return dir.string();
}

// --- Model builders -----------------------------------------------------------

std::vector<uint8_t> AsBytes(const std::vector<float>& v) {
  std::vector<uint8_t> b(v.size() * sizeof(float));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

/// A deterministic 2-layer MLP:  in -> hidden (relu) -> out.
SmfModel MakeMlp(int64_t in_dim, int64_t hidden, int64_t out_dim,
                 uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.5f);
  auto randv = [&](size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
  };

  SmfModel m;
  m.input_name = "x";
  m.output_name = "logits";
  m.tensors.push_back({.name = "x", .dims = {-1, in_dim}, .is_const = false});
  m.tensors.push_back({.name = "w1",
                       .dims = {in_dim, hidden},
                       .is_const = true,
                       .data = AsBytes(randv(in_dim * hidden))});
  m.tensors.push_back({.name = "b1",
                       .dims = {hidden},
                       .is_const = true,
                       .data = AsBytes(randv(hidden))});
  m.tensors.push_back({.name = "w2",
                       .dims = {hidden, out_dim},
                       .is_const = true,
                       .data = AsBytes(randv(hidden * out_dim))});
  m.tensors.push_back({.name = "b2",
                       .dims = {out_dim},
                       .is_const = true,
                       .data = AsBytes(randv(out_dim))});
  for (auto& t : m.tensors)
    if (t.is_const) t.byte_size = t.data.size();

  m.ops.push_back({SmfOpKind::kMatMul, "mm1", {"x", "w1"}, "z1"});
  m.ops.push_back({SmfOpKind::kAddBias, "ab1", {"z1", "b1"}, "z1b"});
  m.ops.push_back({SmfOpKind::kRelu, "relu1", {"z1b"}, "h1"});
  m.ops.push_back({SmfOpKind::kMatMul, "mm2", {"h1", "w2"}, "z2"});
  m.ops.push_back({SmfOpKind::kAddBias, "ab2", {"z2", "b2"}, "logits"});
  return m;
}

/// A linearly-separable 2-class problem: label = 1[x·w_true > 0].
Dataset MakeClassificationData(uint64_t n, int64_t in_dim, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> w_true(in_dim);
  for (auto& w : w_true) w = dist(rng);

  std::vector<float> inputs(n * in_dim);
  std::vector<uint8_t> labels(n * sizeof(int32_t));
  auto* lab = reinterpret_cast<int32_t*>(labels.data());
  for (uint64_t i = 0; i < n; ++i) {
    float dot = 0.0f;
    for (int64_t d = 0; d < in_dim; ++d) {
      inputs[i * in_dim + d] = dist(rng);
      dot += inputs[i * in_dim + d] * w_true[d];
    }
    lab[i] = dot > 0.0f ? 1 : 0;
  }
  auto ds = Dataset::FromMemory(std::move(inputs), std::move(labels), n,
                                in_dim, /*label_kind=*/1, /*label_dim=*/0);
  CHECK(ds.has_value());
  return std::move(*ds);
}

float ReadArenaF32(UpdateEngine& e, uint64_t ref, uint64_t index = 0) {
  return reinterpret_cast<const float*>(e.arena() + RefOffset(ref))[index];
}

void WriteArenaF32(UpdateEngine& e, uint64_t ref, uint64_t index, float v) {
  reinterpret_cast<float*>(e.arena() + RefOffset(ref))[index] = v;
}

void FillSlots(UpdateEngine& e, const std::vector<float>& x,
               const std::vector<int32_t>& labels) {
  CHECK(x.size() == e.header().input_floats);
  std::memcpy(e.arena() + RefOffset(e.header().input_ref), x.data(),
              x.size() * sizeof(float));
  if (!labels.empty())
    std::memcpy(e.arena() + RefOffset(e.header().label_ref), labels.data(),
                labels.size() * sizeof(int32_t));
}

UpdateConfig BaseConfig(int64_t batch) {
  UpdateConfig config;
  config.batch = batch;
  config.loss = LossKind::kSoftmaxXEnt;
  config.lora.rank = 4;
  config.lora.alpha = 8.0f;
  config.lora.seed = 7;
  return config;
}

// =============================================================================
// 1. Grafting structure
// =============================================================================

void TestGraftStructure() {
  SmfModel model = MakeMlp(6, 10, 3, 1);
  UpdateConfig config = BaseConfig(4);
  auto compiled = UpdateCompiler(config).Compile(model);
  CHECK_MSG(compiled.has_value(), "%s", compiled.error().c_str());

  // Both student MatMuls adapted; A/B pairs tracked with gradients.
  CHECK(compiled->adapters.size() == 2);
  CHECK(compiled->params.size() == 4);  // {A,B} x 2 layers
  for (const auto& a : compiled->adapters) {
    CHECK(a.r == 4);
    CHECK(IsRodataRef(a.weight_rodata_ref));
    CHECK(!IsRodataRef(a.a_ref) && !IsRodataRef(a.b_ref));
    // Persistent segment holds the adapters.
    CHECK(RefOffset(a.a_ref) < compiled->persistent_size);
    CHECK(RefOffset(a.b_ref) < compiled->persistent_size);
  }
  // The training program contains forward + backward + optimizer code.
  CHECK(compiled->train_instruction_count > 10);
  // Merge program: copy + gemm_acc per adapter.
  CHECK(compiled->merge_instruction_count == 4);

  // Target filtering restricts grafting.
  config.lora.target_filters = {"w2"};
  auto filtered = UpdateCompiler(config).Compile(model);
  CHECK_MSG(filtered.has_value(), "%s", filtered.error().c_str());
  CHECK(filtered->adapters.size() == 1);
  CHECK(filtered->adapters[0].weight_name == "w2");

  std::printf("ok  TestGraftStructure\n");
}

// =============================================================================
// 2. Step-0 identity: grafted model == source model (B = 0)
// =============================================================================

void TestStep0Identity() {
  const int64_t in_dim = 6, hidden = 10, out_dim = 3, batch = 4;
  SmfModel model = MakeMlp(in_dim, hidden, out_dim, 2);

  UpdateConfig config = BaseConfig(batch);
  config.emit_optimizer = false;  // observe without mutating
  auto compiled = UpdateCompiler(config).Compile(model);
  CHECK_MSG(compiled.has_value(), "%s", compiled.error().c_str());

  UpdateEngine engine;
  CHECK(engine.LoadFromMemory(compiled->plan.data(), compiled->plan.size())
            .has_value());

  std::mt19937_64 rng(99);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(batch * in_dim);
  for (auto& v : x) v = dist(rng);
  std::vector<int32_t> labels = {0, 1, 2, 1};
  FillSlots(engine, x, labels);
  engine.ExecuteTrainOnce();
  const float grafted_loss = engine.LossValue();

  // Reference loss computed directly from the SMF weights (no LoRA).
  auto weights = [&](const char* name) {
    const SmfTensor* t = model.FindTensor(name);
    CHECK(t != nullptr);
    std::vector<float> v(t->data.size() / sizeof(float));
    std::memcpy(v.data(), t->data.data(), t->data.size());
    return v;
  };
  auto w1 = weights("w1"), b1 = weights("b1");
  auto w2 = weights("w2"), b2 = weights("b2");

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

  CHECK_MSG(std::fabs(grafted_loss - ref_loss) < 1e-4,
            "grafted %.6f vs reference %.6f", grafted_loss, ref_loss);
  std::printf("ok  TestStep0Identity (loss %.6f)\n", grafted_loss);
}

// =============================================================================
// 3. Finite-difference gradient check of the compiled backward pass
// =============================================================================

void TestGradientsAgainstFiniteDifferences() {
  const int64_t in_dim = 5, hidden = 7, out_dim = 3, batch = 2;
  SmfModel model = MakeMlp(in_dim, hidden, out_dim, 3);

  UpdateConfig config = BaseConfig(batch);
  config.lora.rank = 3;
  config.emit_optimizer = false;  // params stay fixed across executions
  auto compiled = UpdateCompiler(config).Compile(model);
  CHECK_MSG(compiled.has_value(), "%s", compiled.error().c_str());

  UpdateEngine engine;
  CHECK(engine.LoadFromMemory(compiled->plan.data(), compiled->plan.size())
            .has_value());

  std::mt19937_64 rng(4242);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(batch * in_dim);
  for (auto& v : x) v = dist(rng);
  std::vector<int32_t> labels = {1, 2};
  FillSlots(engine, x, labels);

  // Nudge B off its zero init so gradients w.r.t. A are non-degenerate.
  for (const auto& p : compiled->params)
    for (uint64_t i = 0; i < p.count; ++i)
      if (p.id.find(".lora_B") != std::string::npos)
        WriteArenaF32(engine, p.param_ref, i, 0.05f * dist(rng));

  engine.ExecuteTrainOnce();
  const double eps = 2e-3;
  size_t checked = 0;

  for (const auto& p : compiled->params) {
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
      CHECK_MSG(rel < 2e-2,
                "param %s[%llu]: analytic %.6g vs numeric %.6g (rel %.3g)",
                p.id.c_str(), (unsigned long long)i, analytic, numeric, rel);
      ++checked;
    }
  }
  CHECK(checked >= 20);
  std::printf("ok  TestGradientsAgainstFiniteDifferences (%zu coords)\n",
              checked);
}

// =============================================================================
// 4. End-to-end: train -> merge -> commit
// =============================================================================

void TestEndToEndUpdate() {
  const int64_t in_dim = 8, hidden = 16, out_dim = 2, batch = 8;
  SmfModel model = MakeMlp(in_dim, hidden, out_dim, 5);

  UpdateConfig config = BaseConfig(batch);
  config.optimizer.lr = 5e-3f;
  config.optimizer.kind = OptimizerKind::kAdamW;
  auto compiled = UpdateCompiler(config).Compile(model);
  CHECK_MSG(compiled.has_value(), "%s", compiled.error().c_str());

  UpdateEngine engine;
  CHECK(engine.LoadFromMemory(compiled->plan.data(), compiled->plan.size())
            .has_value());

  Dataset data = MakeClassificationData(512, in_dim, 11);
  seeml::update_rt::TrainOptions options;
  options.log_every = 0;
  auto report = engine.Train(data, 400, options);
  CHECK_MSG(report.has_value(), "%s", report.error().c_str());
  CHECK_MSG(report->final_avg_loss < report->initial_avg_loss * 0.8f,
            "loss %.4f -> %.4f: update did not learn", report->initial_avg_loss,
            report->final_avg_loss);

  // --- Merge: W' must equal W + (α/r)·A@B to float precision. ---------------
  CHECK(engine.RunMerge().has_value());
  for (const auto& a : compiled->adapters) {
    const auto* A =
        reinterpret_cast<const float*>(engine.arena() + RefOffset(a.a_ref));
    const auto* B =
        reinterpret_cast<const float*>(engine.arena() + RefOffset(a.b_ref));
    const auto* merged = reinterpret_cast<const float*>(engine.arena() +
                                                        RefOffset(a.merged_ref));
    const SmfTensor* w_src = model.FindTensor(a.weight_name);
    CHECK(w_src != nullptr);
    const auto* W = reinterpret_cast<const float*>(w_src->data.data());

    double max_err = 0.0, max_delta = 0.0;
    for (int64_t i = 0; i < a.k; ++i)
      for (int64_t j = 0; j < a.m; ++j) {
        double delta = 0.0;
        for (int64_t t = 0; t < a.r; ++t)
          delta += static_cast<double>(A[i * a.r + t]) * B[t * a.m + j];
        const double expected = W[i * a.m + j] + a.scale * delta;
        max_err = std::max(
            max_err, std::fabs(expected - merged[i * a.m + j]));
        max_delta = std::max(max_delta, std::fabs(a.scale * delta));
      }
    CHECK_MSG(max_err < 1e-4, "merge mismatch %.3g for %s", max_err,
              a.weight_name.c_str());
    // Training actually moved the weights (B left zero => no update at all).
    CHECK_MSG(max_delta > 1e-4, "adapter for %s never learned",
              a.weight_name.c_str());
  }

  // --- Commit: patched SMF, untouched tensors bit-identical. ----------------
  const std::string dir = TempDir();
  const std::string src_path = dir + "/source_model.smf";
  const std::string out_path = dir + "/updated_model.smf";
  CHECK(SaveSmf(src_path, model).has_value());

  // Saving may relocate data offsets: recompile against the saved artifact so
  // the emit table matches the file the runtime patches (the real workflow).
  auto saved = LoadSmf(src_path);
  CHECK(saved.has_value());
  auto compiled2 = UpdateCompiler(config).Compile(*saved);
  CHECK_MSG(compiled2.has_value(), "%s", compiled2.error().c_str());
  UpdateEngine engine2;
  CHECK(engine2.LoadFromMemory(compiled2->plan.data(), compiled2->plan.size())
            .has_value());
  auto report2 = engine2.Train(data, 400, options);
  CHECK(report2.has_value() && report2->improved());
  CHECK(engine2.RunMerge().has_value());
  CHECK(engine2.CommitToModel(src_path, out_path).has_value());

  auto updated = LoadSmf(out_path);
  CHECK_MSG(updated.has_value(), "%s", updated.error().c_str());
  for (const auto& a : compiled2->adapters) {
    const SmfTensor* t = updated->FindTensor(a.weight_name);
    CHECK(t != nullptr);
    const auto* patched = reinterpret_cast<const float*>(t->data.data());
    const auto* merged = reinterpret_cast<const float*>(
        engine2.arena() + RefOffset(a.merged_ref));
    for (int64_t i = 0; i < a.k * a.m; ++i)
      CHECK_MSG(patched[i] == merged[i], "commit mismatch in %s",
                a.weight_name.c_str());
  }
  // Bias tensors were not adapted: must be bit-identical to the source.
  for (const char* name : {"b1", "b2"}) {
    const SmfTensor* before = saved->FindTensor(name);
    const SmfTensor* after = updated->FindTensor(name);
    CHECK(before && after && before->data == after->data);
  }

  std::printf("ok  TestEndToEndUpdate (loss %.4f -> %.4f)\n",
              report->initial_avg_loss, report->final_avg_loss);
}

// =============================================================================
// 5. Distillation from an open-weights teacher (no labels)
// =============================================================================

void TestDistillationFromTeacher() {
  const int64_t in_dim = 8, hidden = 12, out_dim = 4, batch = 8;
  SmfModel student = MakeMlp(in_dim, hidden, out_dim, 21);
  SmfModel teacher = MakeMlp(in_dim, 20, out_dim, 22);  // different capacity

  UpdateConfig config = BaseConfig(batch);
  config.loss = LossKind::kKLDistill;
  config.temperature = 2.0f;
  config.optimizer.lr = 5e-3f;
  auto compiled = UpdateCompiler(config).Compile(student, &teacher);
  CHECK_MSG(compiled.has_value(), "%s", compiled.error().c_str());
  // Teacher weights ride along frozen; only the student's two MatMuls adapt.
  CHECK(compiled->adapters.size() == 2);

  UpdateEngine engine;
  CHECK(engine.LoadFromMemory(compiled->plan.data(), compiled->plan.size())
            .has_value());

  // Unlabeled corpus: the teacher provides the training signal in-graph.
  std::mt19937_64 rng(31);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  const uint64_t n = 256;
  std::vector<float> inputs(n * in_dim);
  for (auto& v : inputs) v = dist(rng);
  auto data = Dataset::FromMemory(std::move(inputs), {}, n, in_dim,
                                  /*label_kind=*/0, /*label_dim=*/0);
  CHECK(data.has_value());

  seeml::update_rt::TrainOptions options;
  options.log_every = 0;
  auto report = engine.Train(*data, 300, options);
  CHECK_MSG(report.has_value(), "%s", report.error().c_str());
  CHECK_MSG(report->final_avg_loss < report->initial_avg_loss * 0.9f,
            "KL %.4f -> %.4f: distillation did not converge",
            report->initial_avg_loss, report->final_avg_loss);

  std::printf("ok  TestDistillationFromTeacher (KL %.4f -> %.4f)\n",
              report->initial_avg_loss, report->final_avg_loss);
}

}  // namespace

int main() {
  TestGraftStructure();
  TestStep0Identity();
  TestGradientsAgainstFiniteDifferences();
  TestEndToEndUpdate();
  TestDistillationFromTeacher();
  std::printf("all update-system tests passed\n");
  return 0;
}
