// =============================================================================
// Metal backend tests (G1b): every program family the compiler emits,
// trained on the CPU reference and on the Metal backend from identical
// plans, data and seeds, agrees at tolerance — losses, eval program, and
// the persistent segment (adapters + AdamW moments) — while the Metal
// backend agrees with ITSELF bitwise run-to-run (the per-backend
// determinism doctrine, docs/roadmap.md Project 5). Plus the sync
// contract: CPU-resident instructions interleaved with pending GPU work
// see a coherent arena. Hardware-gated: on a Metal-less host the suite
// passes vacuously with a note.
// =============================================================================

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "test/framework/seetest.h"
#include "test/support/builders.h"

#if defined(__APPLE__)

#include "compiler/driver/update_compiler.h"
#include "runtime/engine/update_engine.h"
#include "runtime/executor/backend.h"
#include "runtime/executor/metal_backend.h"
#include "source/plan/update_types.h"

namespace {

using namespace seeml::update;
using namespace seeml::update_rt;
using namespace seeml::testing;

bool Skip() {
  if (MetalBackendAvailable()) return false;
  std::puts("  [ note ] no Metal device; the Metal backend is untested here");
  return true;
}

struct Run {
  std::vector<float> curve;
  std::vector<float> persistent;
  float eval_loss = 0.0f;
  std::string device;
};

using DataMaker = std::expected<Dataset, std::string> (*)(uint64_t seed);

std::expected<Run, std::string> TrainOn(BackendKind kind,
                                        const std::vector<uint8_t>& plan,
                                        DataMaker make, uint64_t steps) {
  UpdateEngine engine;
  if (auto r = engine.SelectBackend(kind); !r) return std::unexpected(r.error());
  if (auto r = engine.LoadFromMemory(plan.data(), plan.size()); !r)
    return std::unexpected(r.error());
  auto data = make(17);
  if (!data) return std::unexpected(data.error());
  data->EnableShuffle(3);
  TrainOptions options;
  options.log_every = 0;
  options.record_loss_curve = true;
  auto report = engine.Train(*data, steps, options);
  if (!report) return std::unexpected(report.error());
  Run run;
  run.curve = report->loss_curve;
  const size_t floats = engine.header().persistent_size / sizeof(float);
  run.persistent.resize(floats);
  std::memcpy(run.persistent.data(), engine.arena(), floats * sizeof(float));
  auto eval = engine.Evaluate(*data);
  if (!eval) return std::unexpected(eval.error());
  run.eval_loss = *eval;
  run.device = engine.backend_device();
  return run;
}

void ExpectClose(const char* what, const std::vector<float>& got,
                 const std::vector<float>& want, double rel, double abs_floor) {
  ASSERT_EQ(got.size(), want.size());
  size_t worst = 0;
  double worst_err = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double err = std::fabs(static_cast<double>(got[i]) - want[i]);
    const double tol = abs_floor + rel * std::fabs(static_cast<double>(want[i]));
    if (err > tol && err > worst_err) {
      worst_err = err;
      worst = i;
    }
  }
  if (worst_err > 0.0)
    ADD_FAILURE(std::string(what) + "[" + std::to_string(worst) + "]: metal " +
                std::to_string(got[worst]) + " vs cpu " +
                std::to_string(want[worst]));
}

// --- Fixture families --------------------------------------------------------

std::expected<Dataset, std::string> ClassData(uint64_t seed) {
  return MakeClassificationData(64, 16, seed);
}
std::expected<Dataset, std::string> DecoderData(uint64_t seed) {
  return MakeClassificationData(64, 8, seed);
}
std::expected<Dataset, std::string> TokenData(uint64_t seed) {
  return MakeTokenCorpus(32, 4, 16, seed);
}
std::expected<Dataset, std::string> RegressionData(uint64_t seed) {
  return MakeRegressionData(64, 16, 3, seed);
}
std::expected<Dataset, std::string> UnlabeledData(uint64_t seed) {
  return MakeUnlabeledData(64, 8, seed);
}

struct Family {
  const char* name;
  std::vector<uint8_t> plan;
  DataMaker data;
};

std::vector<Family> Families() {
  std::vector<Family> out;
  auto add = [&](const char* name, std::expected<CompiledUpdate, std::string> c,
                 DataMaker data) {
    if (!c) {
      ADD_FAILURE(std::string(name) + ": " + c.error());
      return;
    }
    out.push_back({name, c->plan, data});
  };
  {  // MLP, cross-entropy, AdamW, clip + cosine schedule: the classic path
    SmfModel m = MakeMlp(16, 32, 4, 31);
    UpdateConfig c = BaseConfig(16);
    c.optimizer.clip_norm = 0.5f;
    c.optimizer.lr_schedule = LrSchedule::kCosineWithWarmup;
    c.optimizer.warmup_steps = 2;
    c.default_steps = 12;
    add("mlp_xent_adamw_clip_cosine", UpdateCompiler(c).Compile(m), ClassData);
  }
  {  // gradient accumulation: the grad + step programs, kAccumulate on GPU
    SmfModel m = MakeMlp(16, 32, 4, 39);
    UpdateConfig c = BaseConfig(8);
    c.grad_accum_steps = 2;
    c.optimizer.clip_norm = 0.5f;
    add("mlp_xent_grad_accum_2", UpdateCompiler(c).Compile(m), ClassData);
  }
  {  // same model, SGD, quantized base: the q8 GEMM pair
    SmfModel m = MakeMlp(16, 32, 4, 32);
    UpdateConfig c = BaseConfig(16);
    c.optimizer.kind = OptimizerKind::kSgd;
    c.quantize_base = true;
    add("mlp_xent_sgd_q8", UpdateCompiler(c).Compile(m), ClassData);
  }
  {  // gated net: gelu / silu / mul / layer_norm + fused epilogues
    SmfModel m = MakeGatedNet(16, 24, 4, 33);
    UpdateConfig c = BaseConfig(16);
    add("gated_gelu_silu_layernorm", UpdateCompiler(c).Compile(m), ClassData);
  }
  {  // tiny decoder: rms_norm / rope / attention / add / swiglu
    SmfModel m = MakeTinyDecoder(8, 2, 4, 16, 2, 34);
    UpdateConfig c = BaseConfig(16);
    add("decoder_rmsnorm_rope_attention", UpdateCompiler(c).Compile(m),
        DecoderData);
  }
  {  // token-native decoder: the embedding gather stays on the CPU
    SmfModel m = MakeTinyTokenDecoder(16, 8, 2, 4, 16, 35);
    UpdateConfig c = BaseConfig(16);
    add("token_decoder_embed", UpdateCompiler(c).Compile(m), TokenData);
  }
  {  // MSE regression
    SmfModel m = MakeMlp(16, 32, 3, 36);
    UpdateConfig c = BaseConfig(16);
    c.loss = LossKind::kMse;
    add("mlp_mse", UpdateCompiler(c).Compile(m), RegressionData);
  }
  {  // KL distillation against a teacher (composite of GPU + CPU loss)
    SmfModel s = MakeMlp(8, 12, 4, 37), t = MakeMlp(8, 20, 4, 38);
    UpdateConfig c = BaseConfig(16);
    c.loss = LossKind::kKLDistill;
    c.temperature = 2.0f;
    add("mlp_kl_distill", UpdateCompiler(c).Compile(s, &t), UnlabeledData);
  }
  return out;
}

TEST(MetalBackend, EveryProgramFamilyMatchesCpuAtTolerance) {
  if (Skip()) return;
  for (const Family& f : Families()) {
    auto cpu = TrainOn(BackendKind::kCpu, f.plan, f.data, 8);
    auto gpu = TrainOn(BackendKind::kMetal, f.plan, f.data, 8);
    ASSERT_OK(cpu);
    ASSERT_OK(gpu);
    std::printf("  [ %-32s ] metal: %s\n", f.name, gpu->device.c_str());
    // Losses: GPU f32 reductions vs the CPU's double partials, compounded
    // over 8 steps of AdamW — the per-backend doctrine prices this at
    // tolerance, and the release gate (#65) at 1% after 300 steps.
    ExpectClose((std::string(f.name) + " loss curve").c_str(), gpu->curve,
                cpu->curve, 2e-3, 1e-4);
    EXPECT_NEAR(gpu->eval_loss, cpu->eval_loss,
                1e-4 + 2e-3 * std::fabs(cpu->eval_loss));
    // The persistent segment: adapters and optimizer moments. Second
    // moments are squares of tiny gradients, hence the absolute floor.
    ExpectClose((std::string(f.name) + " persistent").c_str(), gpu->persistent,
                cpu->persistent, 5e-3, 1e-6);
  }
}

TEST(MetalBackend, IsBitwiseReproducibleRunToRun) {
  if (Skip()) return;
  SmfModel m = MakeTinyDecoder(8, 2, 4, 16, 2, 41);
  UpdateConfig c = BaseConfig(16);
  c.optimizer.clip_norm = 1.0f;
  auto compiled = UpdateCompiler(c).Compile(m);
  ASSERT_OK(compiled);
  auto first = TrainOn(BackendKind::kMetal, compiled->plan, DecoderData, 20);
  auto again = TrainOn(BackendKind::kMetal, compiled->plan, DecoderData, 20);
  ASSERT_OK(first);
  ASSERT_OK(again);
  ASSERT_EQ(first->curve.size(), 20u);
  EXPECT_EQ(std::memcmp(first->curve.data(), again->curve.data(),
                        first->curve.size() * sizeof(float)),
            0);
  EXPECT_EQ(first->eval_loss, again->eval_loss);
  EXPECT_TRUE(first->persistent == again->persistent);
}

TEST(MetalBackend, ZeroCopyResidencyWhenThePlanIsPageAligned) {
  if (Skip()) return;
  SmfModel m = MakeMlp(16, 32, 4, 42);
  auto compiled = UpdateCompiler(BaseConfig(16)).Compile(m);
  ASSERT_OK(compiled);
  // A page-aligned copy of the blob (what the .incbin stub produces) lets
  // the backend wrap rodata zero-copy; the heap vector forces the copy.
  ASSERT_EQ(compiled->plan.size() % kSeeuRodataAlignment, 0u);
  void* aligned = std::aligned_alloc(kSeeuRodataAlignment,
                                     compiled->plan.size());
  ASSERT_TRUE(aligned != nullptr);
  std::memcpy(aligned, compiled->plan.data(), compiled->plan.size());
  {
    UpdateEngine engine;
    ASSERT_OK(engine.SelectBackend(BackendKind::kMetal));
    ASSERT_OK(engine.LoadFromMemory(static_cast<const uint8_t*>(aligned),
                                    compiled->plan.size()));
    EXPECT_STR_CONTAINS(engine.backend_device(), "zero-copy");
  }
  {
    std::vector<uint8_t> shifted(compiled->plan.size() + 64);
    std::memcpy(shifted.data() + 64, compiled->plan.data(),
                compiled->plan.size());
    UpdateEngine engine;
    ASSERT_OK(engine.SelectBackend(BackendKind::kMetal));
    ASSERT_OK(engine.LoadFromMemory(shifted.data() + 64,
                                    compiled->plan.size()));
    EXPECT_STR_CONTAINS(engine.backend_device(), "copied");
  }
  std::free(aligned);
}

TEST(MetalBackend, CpuInstructionsWaitForThePendingGpuWorkTheyTouch) {
  if (Skip()) return;
  auto metal = CreateMetalBackend();
  ASSERT_OK(metal);
  // A 16 KiB arena; slots (floats): a @0, c @512, tokens @1024 (i32),
  // b @2048 (zeros), loss @3072, d @4096. Rodata: a [4][8] table.
  constexpr size_t kArenaBytes = 16384;
  uint8_t* arena = static_cast<uint8_t*>(std::aligned_alloc(16384, kArenaBytes));
  ASSERT_TRUE(arena != nullptr);
  std::memset(arena, 0, kArenaBytes);
  std::vector<float> table(32);
  for (size_t i = 0; i < 32; ++i) table[i] = 0.25f * static_cast<float>(i);
  int32_t tokens[4] = {3, 0, 2, 1};
  std::memcpy(arena + 1024, tokens, sizeof(tokens));
  ASSERT_OK((*metal)->Bind(arena, kArenaBytes,
                           reinterpret_cast<const uint8_t*>(table.data()),
                           table.size() * sizeof(float),
                           table.size() * sizeof(float)));
  auto f32 = [](float v) { return uint64_t{std::bit_cast<uint32_t>(v)}; };
  UpdateInstruction fill;  // GPU: a = 2.0
  fill.opcode = static_cast<uint16_t>(OpCode::kFill);
  fill.in[0] = MakeArenaRef(0); fill.in[1] = f32(2.0f); fill.out[0] = 32;
  UpdateInstruction scale;  // GPU: c = 3 * a  (reads a)
  scale.opcode = static_cast<uint16_t>(OpCode::kScale);
  scale.in[0] = MakeArenaRef(0); scale.in[1] = MakeArenaRef(512);
  scale.in[2] = f32(3.0f); scale.out[0] = 32;
  UpdateInstruction embed;  // CPU: a = table[tokens] (WRITES a: must wait)
  embed.opcode = static_cast<uint16_t>(OpCode::kEmbedFwd);
  embed.in[0] = MakeArenaRef(1024); embed.in[1] = MakeRodataRef(0);
  embed.in[2] = MakeArenaRef(0); embed.out[0] = 4;
  embed.out[1] = (uint64_t{4} << 32) | 8;
  UpdateInstruction mse;  // CPU: loss = mean((a - b)^2) (reads a)
  mse.opcode = static_cast<uint16_t>(OpCode::kMseFwd);
  mse.in[0] = MakeArenaRef(0); mse.in[1] = MakeArenaRef(2048);
  mse.in[2] = MakeArenaRef(3072); mse.out[0] = 32;
  UpdateInstruction half;  // GPU: d = 0.5 * c
  half.opcode = static_cast<uint16_t>(OpCode::kScale);
  half.in[0] = MakeArenaRef(512); half.in[1] = MakeArenaRef(4096);
  half.in[2] = f32(0.5f); half.out[0] = 32;
  StepParams params;
  for (const UpdateInstruction* ins : {&fill, &scale, &embed, &mse, &half})
    ASSERT_OK((*metal)->Execute(*ins, params));
  ASSERT_OK((*metal)->Flush());
  const float* a = reinterpret_cast<const float*>(arena);
  const float* c = reinterpret_cast<const float*>(arena + 512);
  const float* d = reinterpret_cast<const float*>(arena + 4096);
  double want_loss = 0.0;
  for (size_t r = 0; r < 4; ++r)
    for (size_t j = 0; j < 8; ++j) {
      const float t = table[static_cast<size_t>(tokens[r]) * 8 + j];
      EXPECT_EQ(a[r * 8 + j], t);  // the gather landed after the scale read
      want_loss += static_cast<double>(t) * t;
    }
  for (size_t i = 0; i < 32; ++i) {
    EXPECT_EQ(c[i], 6.0f);  // scaled the 2.0 fill, not the gathered rows
    EXPECT_EQ(d[i], 3.0f);
  }
  EXPECT_NEAR(*reinterpret_cast<const float*>(arena + 3072), want_loss / 32.0,
              1e-5);
  std::free(arena);
}

TEST(MetalBackend, ActivationsStayFiniteAtLargeMagnitude) {
  // The GPU's *fast* tanh returns NaN above ~44, which through the GELU
  // means NaN for any pre-activation past ~10.25; the library must be
  // compiled with the precise transcendental functions. Sweep every
  // activation forward and backward over [-40, 40] on both backends.
  if (Skip()) return;
  auto metal = CreateMetalBackend();
  ASSERT_OK(metal);
  constexpr size_t kArenaBytes = 16384, kN = 801;
  uint8_t* arena = static_cast<uint8_t*>(std::aligned_alloc(16384, kArenaBytes));
  ASSERT_TRUE(arena != nullptr);
  std::memset(arena, 0, kArenaBytes);
  float* x = reinterpret_cast<float*>(arena);          // [0, 3204)
  float* dy = reinterpret_cast<float*>(arena + 3264);  // up to 6468
  float* out = reinterpret_cast<float*>(arena + 6528); // up to 9732
  for (size_t i = 0; i < kN; ++i) {
    x[i] = -40.0f + 0.1f * static_cast<float>(i);
    dy[i] = 0.5f + 0.001f * static_cast<float>(i);
  }
  ASSERT_OK((*metal)->Bind(arena, kArenaBytes, nullptr, 0, 0));
  auto cpu = CreateCpuBackend();
  ASSERT_OK(cpu->Bind(arena, kArenaBytes, nullptr, 0, 0));
  const OpCode fwd[] = {OpCode::kReluFwd, OpCode::kGeluFwd, OpCode::kSiluFwd};
  const OpCode bwd[] = {OpCode::kReluBwd, OpCode::kGeluBwd, OpCode::kSiluBwd};
  StepParams params;
  for (int which = 0; which < 3; ++which) {
    for (int pass = 0; pass < 2; ++pass) {
      UpdateInstruction ins;
      ins.opcode = static_cast<uint16_t>(pass == 0 ? fwd[which] : bwd[which]);
      if (pass == 0) {
        ins.in[0] = MakeArenaRef(0);
        ins.in[1] = MakeArenaRef(6528);
      } else {
        ins.in[0] = MakeArenaRef(3264);
        ins.in[1] = MakeArenaRef(0);
        ins.in[2] = MakeArenaRef(6528);
      }
      ins.out[0] = kN;
      std::vector<float> want(kN), got(kN);
      ASSERT_OK(cpu->Execute(ins, params));
      std::memcpy(want.data(), out, kN * sizeof(float));
      std::memset(out, 0x7f, kN * sizeof(float));  // NaN-ish stale bytes
      ASSERT_OK((*metal)->Execute(ins, params));
      ASSERT_OK((*metal)->Flush());
      std::memcpy(got.data(), out, kN * sizeof(float));
      for (size_t i = 0; i < kN; ++i) {
        EXPECT_TRUE(std::isfinite(got[i]));
        EXPECT_NEAR(got[i], want[i], 1e-4 + 2e-4 * std::fabs(want[i]));
      }
    }
  }
  std::free(arena);
}

TEST(MetalBackend, MergeDeltasMatchCpuAtTolerance) {
  if (Skip()) return;
  SmfModel m = MakeMlp(16, 32, 4, 43);
  auto compiled = UpdateCompiler(BaseConfig(16)).Compile(m);
  ASSERT_OK(compiled);
  std::vector<float> deltas[2];
  int i = 0;
  for (BackendKind kind : {BackendKind::kCpu, BackendKind::kMetal}) {
    UpdateEngine engine;
    ASSERT_OK(engine.SelectBackend(kind));
    ASSERT_OK(engine.LoadFromMemory(compiled->plan.data(),
                                    compiled->plan.size()));
    auto data = ClassData(17);
    ASSERT_OK(data);
    TrainOptions options;
    options.log_every = 0;
    ASSERT_OK(engine.Train(*data, 6, options));
    ASSERT_OK(engine.RunMerge());
    for (const auto& ad : compiled->adapters)
      for (int64_t e = 0; e < ad.k * ad.m; ++e)
        deltas[i].push_back(
            ReadArenaF32(engine, ad.delta_ref, static_cast<uint64_t>(e)));
    ++i;
  }
  ExpectClose("merge deltas", deltas[1], deltas[0], 5e-3, 1e-6);
}

}  // namespace

#else  // !__APPLE__

namespace {
TEST(MetalBackend, UnsupportedPlatformIsVacuouslyGreen) { EXPECT_TRUE(true); }
}  // namespace

#endif
