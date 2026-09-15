// =============================================================================
// seeml-bench — the benchmark harness of docs/benchmarks.md. Usage:
//
//   seeml-bench --out bench.json [--threads 1,8] [--steps-lo 20]
//               [--steps-hi 80] [--repeats 3] [--fixtures name,name,...]
//               [--peak-gflops F] [--backend cpu|metal|auto]
//               [--gemm-tiles K,N | --kernel-policy table.json
//                [--target-host KEY]] [--version]
//
// The kernel policy (the CPU GEMM tiles the compiler writes into every
// plan, v11) is resolved once, exactly as seeml-update-compile resolves
// it, and applied to every fixture: the harness measures the geometry a
// package would ship with. --gemm-tiles is the offline tuner's arm switch
// (tool/autotune.py sweeps it); the report records the resolved policy,
// the host key the table is keyed on, the host description, and the
// analytic tiling (SuggestGemmTiling) so the tuner can measure it as an
// arm.
//
// Compiles the standard fixture set in-process (the seeded builders the
// test suites share), trains each plan at every requested thread width, and
// emits one JSON object per run — Tier A throughput (rows/s, per-step
// latency by steps-regression: run two step counts, slope = per-step cost,
// intercept = fixed lifecycle cost), the fwd/bwd/optimizer split from the
// engine's SEEML_STEP_TIMING instrumentation (zeros unless the runtime was
// built with it), per-step GEMM FLOPs read out of the compiled plan's own
// instruction stream, and the Tier D lifecycle latencies (compile, load,
// merge, checkpoint save).
//
// Alongside the internal numbers, every cell carries the fields the external
// fine-tuning harnesses print, under their definitions, so a SeeML run can
// be set beside an MLX-LM `lora` log or a llama-bench table without unit
// translation:
//   tokens_per_s / samples_per_s  MLX-LM "Tokens/sec": target rows per wall
//                                 second (every row of a SeeML batch is a
//                                 loss target, so this equals rows_per_s);
//   it_per_s                      MLX-LM "It/sec" (optimizer steps per s);
//   step_ms_min / step_ms_max     llama-bench's "± spread" over the repeats;
//   train_loss_first / _last      MLX-LM "Train loss" (windowed means);
//   trained_tokens                MLX-LM "Trained Tokens" over the sweep;
//   peak_rss_bytes                MLX-LM "Peak mem" (OS-observed, process-
//                                 cumulative, so it rises monotonically
//                                 across fixtures);
//   mfu                           PaLM/Megatron model-FLOPs utilization —
//                                 achieved GEMM FLOP/s over the host peak
//                                 given by --peak-gflops (omitted otherwise:
//                                 a peak the harness guessed is not a
//                                 standard).
//
// Everything is pinned: fixture seeds, dataset seeds, thread widths (via
// SetParallelThreadCount, overriding SEEML_THREADS). Medians of --repeats
// runs. Argument parsing is strict, as in seeml-update-compile: an unknown
// flag, a missing value, or trailing garbage is a hard error (exit 2).
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <sys/utsname.h>

#include "compiler/backend/architecture/host_arch.h"
#include "compiler/backend/tuner/kernel_policy_table.h"
#include "compiler/diagnostics/logger.h"
#include "compiler/driver/update_compiler.h"
#include "runtime/engine/update_engine.h"
#include "source/identity/version.h"
#include "source/parallel/parallel_for.h"
#include "source/plan/update_types.h"
#include "test/support/builders.h"

namespace {

namespace up = seeml::update;
namespace rt = seeml::update_rt;
namespace tf = seeml::testing;
using Clock = std::chrono::steady_clock;

double MsSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// OS-observed peak resident set of this process, in bytes. Linux reports
/// ru_maxrss in kilobytes, macOS in bytes.
uint64_t PeakRssBytes() {
  rusage ru{};
  if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#if defined(__APPLE__)
  return static_cast<uint64_t>(ru.ru_maxrss);
#else
  return static_cast<uint64_t>(ru.ru_maxrss) * 1024u;
#endif
}

/// "sysname machine" — a number without its host string is not a benchmark.
std::string HostString() {
  utsname u{};
  if (uname(&u) != 0) return "unknown";
  return std::string(u.sysname) + " " + u.machine;
}

/// Minimal JSON string escaping for the host strings the report carries.
std::string JsonEscape(const std::string& s) {
  std::string out;
  for (const char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
      out += buf;
    } else {
      out += c;
    }
  }
  return out;
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  // Even counts take the mean of the two middle samples — the textbook
  // median, rather than a biased upper-middle. Odd counts (the default
  // repeats=3, which CI uses) are unaffected. A baseline recorded with an
  // EVEN --repeats by an older binary is not comparable to this one —
  // re-seed it (nightly: workflow_dispatch with reseed_baseline=true).
  return n % 2 == 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// --- The standard fixture set -----------------------------------------------
// Shapes follow the issue-#56 baseline table; seeds are arbitrary but fixed
// forever — changing one invalidates every stored baseline.

struct Fixture {
  const char* name;
  const char* kind;  // "feature" | "decoder" | "token"
  int64_t batch_rows;
  int64_t seq;  // rows per sequence (1 for feature plans)
  up::SmfModel (*model)();
  std::expected<rt::Dataset, std::string> (*data)();
  bool quantize_base = false;  // int8 frozen weights (the field-run config)
  // Frontier-shaped fixtures run only when named by --fixtures: at model
  // scale a CPU sweep takes the better part of an hour, so the default set
  // (and the nightly gate's keys) stay the small shapes.
  bool frontier = false;
};

const Fixture kFixtures[] = {
    {"mlp_64x512x3", "feature", 32, 1,
     [] { return tf::MakeMlpStack(64, 512, 3, 4, 101); },
     [] { return tf::MakeClassificationData(4096, 64, 102); }},
    {"mlp_128x1024x2", "feature", 32, 1,
     [] { return tf::MakeMlpStack(128, 1024, 2, 4, 103); },
     [] { return tf::MakeClassificationData(4096, 128, 104); }},
    {"dec_v256_d128_s32", "decoder", 128, 32,
     [] { return tf::MakeDecoderStack(128, 8, 32, 512, 256, 4, 105); },
     [] { return tf::MakeClassificationData(8192, 128, 106); }},
    {"dec_v256_d128_s128", "decoder", 128, 128,
     [] { return tf::MakeDecoderStack(128, 8, 128, 512, 256, 4, 107); },
     [] { return tf::MakeClassificationData(8192, 128, 108); }},
    {"dec_v512_d192_s32", "decoder", 128, 32,
     [] { return tf::MakeDecoderStack(192, 8, 32, 768, 512, 4, 109); },
     [] { return tf::MakeClassificationData(8192, 192, 110); }},
    {"tok_v64_d64_s16", "token", 128, 16,
     [] { return tf::MakeTinyTokenDecoder(64, 64, 4, 16, 256, 111); },
     [] { return tf::MakeTokenCorpus(1024, 16, 64, 112); }},
    // SmolLM-135M's geometry (vocab 49152, D=576, 9 heads, ffn 1536, 30
    // blocks), q8 base, 512 tokens per step: the frontier row of
    // docs/benchmarks.md. Seeded synthetic weights and corpus, so the
    // per-step cost is the model's; the loss is not a quality number.
    {"tok_smollm135m_q8", "token", 512, 128,
     [] { return tf::MakeTokenDecoderStack(49152, 576, 9, 128, 1536, 30, 113); },
     [] { return tf::MakeTokenCorpus(64, 128, 49152, 114); },
     /*quantize_base=*/true, /*frontier=*/true},
};

/// Sums 2·M·N·K over every GEMM-family instruction in the plan's training
/// stream — the plan itself is the ground truth for per-step matmul work.
uint64_t GemmFlopsPerStep(const std::vector<uint8_t>& plan);

/// Micro-batches per optimizer step recorded in the plan (1 = none).
uint64_t GradAccumOf(const std::vector<uint8_t>& plan) {
  up::PlanHeader h{};
  if (plan.size() < sizeof(h)) return 1;
  std::memcpy(&h, plan.data(), sizeof(h));
  return h.grad_accum_steps > 1 ? h.grad_accum_steps : 1;
}

uint64_t GemmFlopsPerStep(const std::vector<uint8_t>& plan) {
  up::PlanHeader h;
  std::memcpy(&h, plan.data(), sizeof h);
  const auto* ins = reinterpret_cast<const up::UpdateInstruction*>(
      plan.data() + h.train_instr_offset);
  uint64_t flops = 0;
  for (uint64_t i = 0; i < h.train_instr_count; ++i) {
    switch (static_cast<up::OpCode>(ins[i].opcode)) {
      case up::OpCode::kGemmNN:
      case up::OpCode::kGemmNT:
      case up::OpCode::kGemmTN:
      case up::OpCode::kGemmAccNN:
      case up::OpCode::kGemmNNQ8:
      case up::OpCode::kGemmNTQ8:
      case up::OpCode::kGemmNNBF16:
      case up::OpCode::kGemmNTBF16:
        flops += 2 * ins[i].out[0] * ins[i].out[1] * ins[i].out[2];
        break;
      default:
        break;
    }
  }
  return flops;
}

// --- Strict argument cursor (the seeml-update-compile discipline) -----------

struct Args {
  std::vector<std::string> rest;
  explicit Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) rest.emplace_back(argv[i]);
  }
  bool Take(const std::string& flag) {
    auto it = std::find(rest.begin(), rest.end(), flag);
    if (it == rest.end()) return false;
    rest.erase(it);
    return true;
  }
  std::expected<std::string, std::string> TakeValue(const std::string& flag,
                                                    std::string def) {
    auto it = std::find(rest.begin(), rest.end(), flag);
    if (it == rest.end()) return def;
    if (it + 1 == rest.end())
      return std::unexpected(flag + " is missing its value");
    std::string v = *(it + 1);
    rest.erase(it, it + 2);
    return v;
  }
};

std::expected<uint64_t, std::string> ParseU64(const std::string& flag,
                                              const std::string& v) {
  if (v.empty()) return std::unexpected(flag + " must be a positive integer");
  uint64_t out = 0;
  for (char c : v) {
    if (c < '0' || c > '9')
      return std::unexpected(flag + ": '" + v + "' is not a whole number");
    const auto digit = static_cast<uint64_t>(c - '0');
    if (out > (UINT64_MAX - digit) / 10)
      return std::unexpected(flag + ": '" + v + "' does not fit 64 bits");
    out = out * 10 + digit;
  }
  if (out == 0) return std::unexpected(flag + " must be >= 1");
  return out;
}

std::expected<std::vector<std::string>, std::string> SplitCsv(
    const std::string& flag, const std::string& v) {
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= v.size()) {
    size_t comma = v.find(',', start);
    if (comma == std::string::npos) comma = v.size();
    if (comma == start)
      return std::unexpected(flag + ": empty element in '" + v + "'");
    out.push_back(v.substr(start, comma - start));
    start = comma + 1;
  }
  return out;
}

/// Strict positive decimal (for --peak-gflops): the whole string must parse.
std::expected<double, std::string> ParsePositiveF64(const std::string& flag,
                                                    const std::string& v) {
  if (v.empty()) return std::unexpected(flag + " must be a positive number");
  char* end = nullptr;
  const double out = std::strtod(v.c_str(), &end);
  if (end == v.c_str() || *end != '\0' || !(out > 0.0) || out > 1e300)
    return std::unexpected(flag + ": '" + v + "' is not a positive number");
  return out;
}

int Fail(const std::string& msg) {
  std::fprintf(stderr, "seeml-bench: %s\n", msg.c_str());
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  // Six in-process compiles would otherwise narrate every pass at INFO;
  // the harness's own stderr lines are the progress report.
  seeml::diag::Logger::SetLevel(seeml::diag::LogLevel::kWarn);
  Args args(argc, argv);
  if (args.Take("--version")) {
    std::printf("seeml-bench %s\n", seeml::update::kSeemlVersion);
    return 0;
  }

  auto out_path = args.TakeValue("--out", "");
  auto threads_csv = args.TakeValue("--threads", "1,8");
  auto lo_s = args.TakeValue("--steps-lo", "20");
  auto hi_s = args.TakeValue("--steps-hi", "80");
  auto repeats_s = args.TakeValue("--repeats", "3");
  auto fixtures_csv = args.TakeValue("--fixtures", "");
  auto peak_s = args.TakeValue("--peak-gflops", "");
  auto backend_s = args.TakeValue("--backend", "cpu");
  auto tiles_s = args.TakeValue("--gemm-tiles", "");
  auto policy_s = args.TakeValue("--kernel-policy", "");
  auto target_host_s = args.TakeValue("--target-host", "");
  for (auto* v : {&out_path, &threads_csv, &lo_s, &hi_s, &repeats_s,
                  &fixtures_csv, &peak_s, &tiles_s, &policy_s,
                  &target_host_s})
    if (!*v) return Fail(v->error());
  if (!args.rest.empty()) return Fail("unknown argument '" + args.rest[0] +
                                      "'");
  if (out_path->empty())
    return Fail("--out is required\nusage: seeml-bench --out bench.json "
                "[--threads 1,8] [--steps-lo 20] [--steps-hi 80] "
                "[--backend cpu|metal|auto] "
                "[--repeats 3] [--fixtures name,...] [--peak-gflops F] "
                "[--gemm-tiles K,N | --kernel-policy table.json "
                "[--target-host KEY]] [--version]");

  auto lo = ParseU64("--steps-lo", *lo_s);
  auto hi = ParseU64("--steps-hi", *hi_s);
  auto repeats = ParseU64("--repeats", *repeats_s);
  for (auto* v : {&lo, &hi, &repeats})
    if (!*v) return Fail(v->error());
  if (*hi <= *lo) return Fail("--steps-hi must exceed --steps-lo");
  // 0 = not given: mfu is then omitted rather than computed against a guess.
  double peak_gflops = 0.0;
  if (!peak_s->empty()) {
    auto p = ParsePositiveF64("--peak-gflops", *peak_s);
    if (!p) return Fail(p.error());
    peak_gflops = *p;
  }

  if (!backend_s) return Fail(backend_s.error());
  const auto backend_kind = rt::ParseBackendKind(*backend_s);
  if (!backend_kind)
    return Fail("--backend must be cpu, metal or auto, got '" + *backend_s +
                "'");
  // The report records the backend that actually ran (the per-backend
  // gate keys rows by it): resolve `auto` once, the way every fixture's
  // engine will, so the field never says "auto".
  std::string backend_resolved;
  {
    rt::UpdateEngine probe;
    if (auto ok = probe.SelectBackend(*backend_kind); !ok)
      return Fail(ok.error());
    backend_resolved = probe.backend_name();
    if (!probe.backend_note().empty())
      std::fprintf(stderr, "seeml-bench: %s\n", probe.backend_note().c_str());
  }
  // The kernel policy, resolved the way the compiler resolves it, so the
  // harness measures what a package ships; every fixture compiles with it.
  up::KernelPolicyRequest policy_request;
  if (!tiles_s->empty()) {
    auto tiles = up::ParseGemmTilesFlag(*tiles_s);
    if (!tiles) return Fail(tiles.error());
    policy_request.explicit_tiles = *tiles;
  }
  if (!policy_s->empty()) policy_request.table_path = *policy_s;
  if (!target_host_s->empty()) policy_request.target_host = *target_host_s;
  auto policy = up::ResolveKernelPolicy(policy_request);
  if (!policy) return Fail(policy.error());
  if (!policy->note.empty())
    std::fprintf(stderr, "seeml-bench: %s\n", policy->note.c_str());
  // The tiles that actually run: a zero header field is the runtime's
  // compiled-in default, which this binary knows.
  rt::kernels::GemmTiles effective_tiles;
  if (policy->tiles.gemm_tile_k) effective_tiles.k = policy->tiles.gemm_tile_k;
  if (policy->tiles.gemm_tile_n) effective_tiles.n = policy->tiles.gemm_tile_n;
  const up::HostArchInfo host_arch = up::DetectHostArch();
  const up::GemmTiling analytic = up::SuggestGemmTiling(host_arch);

  auto thread_names = SplitCsv("--threads", *threads_csv);
  if (!thread_names) return Fail(thread_names.error());
  std::vector<uint64_t> threads;
  for (const auto& t : *thread_names) {
    auto n = ParseU64("--threads", t);
    if (!n) return Fail(n.error());
    threads.push_back(*n);
  }

  std::vector<const Fixture*> selected;
  if (fixtures_csv->empty()) {
    for (const Fixture& f : kFixtures)
      if (!f.frontier) selected.push_back(&f);
  } else {
    auto names = SplitCsv("--fixtures", *fixtures_csv);
    if (!names) return Fail(names.error());
    for (const auto& n : *names) {
      const Fixture* found = nullptr;
      for (const Fixture& f : kFixtures)
        if (n == f.name) found = &f;
      if (!found) return Fail("--fixtures: unknown fixture '" + n + "'");
      selected.push_back(found);
    }
  }

  // Stage the report beside its destination and rename it into place only
  // when every fixture completed: a failed run must not leave a truncated
  // bench.json that a later gate could mistake for a report (nor a stray
  // checkpoint temp), so both are cleaned up on any early return.
  const std::string tmp_path = *out_path + ".tmp";
  struct Staging {
    std::string tmp, ckpt;
    FILE* file = nullptr;
    bool committed = false;
    ~Staging() {
      if (committed) return;
      if (file) std::fclose(file);
      std::remove(tmp.c_str());
      if (!ckpt.empty()) std::remove(ckpt.c_str());
    }
  } staging{tmp_path, "", nullptr, false};
  FILE* out = std::fopen(tmp_path.c_str(), "w");
  if (!out) return Fail("cannot open '" + tmp_path + "' for writing");
  staging.file = out;
  // Schema 2 adds the external-standard fields (see the header comment);
  // every schema-1 key is unchanged, so stored baselines remain comparable.
  // Schema 3 adds the kernel policy and the host identity it is keyed on;
  // every schema-2 key is unchanged, so stored baselines stay comparable
  // (bench_compare.py refuses to compare runs whose policies differ).
  std::fprintf(out,
               "{\n  \"seeml_version\": \"%s\",\n  \"schema\": 3,\n"
               "  \"host\": \"%s\",\n  \"backend\": \"%s\",\n"
               "  \"host_key\": \"%s\",\n"
               "  \"host_arch\": {\"isa\": \"%s\", \"cpu_model\": \"%s\", "
               "\"physical_cores\": %zu, \"l1d_bytes\": %" PRIu64
               ", \"l2_bytes\": %" PRIu64 ", \"simd_width_f32\": %zu},\n"
               "  \"kernel_policy\": {\"source\": \"%s\", \"gemm_tile_k\": %zu, "
               "\"gemm_tile_n\": %zu},\n"
               "  \"analytic_gemm_tiles\": {\"k\": %zu, \"n\": %zu},\n"
               "  \"config\": {\"steps_lo\": %" PRIu64 ", \"steps_hi\": %"
               PRIu64 ", \"repeats\": %" PRIu64 ", \"threads\": \"%s\", "
               "\"peak_gflops\": %.1f},\n"
               "  \"fixtures\": {",
               seeml::update::kSeemlVersion, HostString().c_str(),
               backend_resolved.c_str(), JsonEscape(policy->host_key).c_str(),
               std::string(host_arch.isa).c_str(),
               JsonEscape(host_arch.cpu_model).c_str(),
               host_arch.physical_cores, host_arch.l1d_bytes,
               host_arch.l2_bytes, host_arch.simd_width_f32,
               policy->source.c_str(), effective_tiles.k, effective_tiles.n,
               analytic.kc, analytic.nc, *lo, *hi,
               *repeats, threads_csv->c_str(), peak_gflops);

  bool first_fixture = true;
  for (const Fixture* f : selected) {
    std::fprintf(stderr, "seeml-bench: %s\n", f->name);
    const up::SmfModel model = f->model();
    up::UpdateConfig config = tf::BaseConfig(f->batch_rows);
    config.lora.rank = 8;
    config.lora.alpha = 16.0f;
    config.quantize_base = f->quantize_base;
    config.gemm_tile_k = policy->tiles.gemm_tile_k;
    config.gemm_tile_n = policy->tiles.gemm_tile_n;

    const auto t_compile = Clock::now();
    auto compiled = up::UpdateCompiler(config).Compile(model);
    if (!compiled) return Fail(f->name + (": " + compiled.error()));
    const double compile_ms = MsSince(t_compile);

    rt::UpdateEngine engine;
    if (auto ok = engine.SelectBackend(*backend_kind); !ok)
      return Fail(f->name + (": " + ok.error()));
    const auto t_load = Clock::now();
    if (auto ok = engine.LoadFromMemory(compiled->plan.data(),
                                        compiled->plan.size());
        !ok)
      return Fail(f->name + (": " + ok.error()));
    const double load_ms = MsSince(t_load);

    auto data = f->data();
    if (!data) return Fail(f->name + (": " + data.error()));

    rt::TrainOptions quiet;
    quiet.log_every = 0;

    uint64_t trainable_params = 0;
    for (const auto& p : compiled->params) trainable_params += p.count;

    std::fprintf(out,
                 "%s\n    \"%s\": {\n"
                 "      \"kind\": \"%s\", \"batch_rows\": %" PRId64
                 ", \"seq\": %" PRId64 ",\n"
                 "      \"plan_bytes\": %zu, \"arena_bytes\": %" PRIu64
                 ", \"persistent_bytes\": %" PRIu64 ", \"rodata_bytes\": %"
                 PRIu64 ",\n"
                 "      \"train_instructions\": %" PRIu64
                 ", \"gemm_flops_per_step\": %" PRIu64
                 ", \"trainable_params\": %" PRIu64 ",\n"
                 "      \"compile_ms\": %.2f, \"load_ms\": %.3f,\n"
                 "      \"threads\": {",
                 first_fixture ? "" : ",", f->name, f->kind, f->batch_rows,
                 f->seq, compiled->plan.size(), compiled->arena_size,
                 compiled->persistent_size, compiled->rodata_size,
                 compiled->train_instruction_count,
                 GemmFlopsPerStep(compiled->plan) * GradAccumOf(compiled->plan), trainable_params,
                 compile_ms, load_ms);
    first_fixture = false;

    // MLX-LM's "Train loss" / "Trained Tokens" over the whole sweep: the
    // engine trains continuously across thread widths and repeats, so the
    // first window is the fixture's starting loss and the last its ending
    // loss — a sanity anchor that the timed work was real training.
    bool have_first_loss = false;
    float train_loss_first = 0.0f, train_loss_last = 0.0f;
    uint64_t trained_steps = 0;
    auto note = [&](const rt::TrainReport& r) {
      if (!have_first_loss) {
        train_loss_first = r.initial_avg_loss;
        have_first_loss = true;
      }
      train_loss_last = r.final_avg_loss;
      trained_steps += r.steps;
    };

    bool first_thread = true;
    for (uint64_t t : threads) {
      up::SetParallelThreadCount(t);
      // Warm the pool, the caches, and the feeder before timing.
      if (auto r = engine.Train(*data, 4, quiet); !r)
        return Fail(f->name + (": " + r.error()));
      else
        note(*r);

      std::vector<double> per_step_ms, lifecycle_ms;
      engine.ResetStepTimings();
      for (uint64_t rep = 0; rep < *repeats; ++rep) {
        const auto t_lo = Clock::now();
        auto r_lo = engine.Train(*data, *lo, quiet);
        const double wall_lo = MsSince(t_lo);
        if (!r_lo) return Fail(f->name + (": " + r_lo.error()));
        note(*r_lo);
        const auto t_hi = Clock::now();
        auto r_hi = engine.Train(*data, *hi, quiet);
        const double wall_hi = MsSince(t_hi);
        if (!r_hi) return Fail(f->name + (": " + r_hi.error()));
        note(*r_hi);
        const double slope = (wall_hi - wall_lo) /
                             static_cast<double>(*hi - *lo);
        per_step_ms.push_back(slope);
        lifecycle_ms.push_back(wall_lo - slope * static_cast<double>(*lo));
      }
      const double step_ms = Median(per_step_ms);
      const double rows_per_s =
          step_ms > 0.0 ? 1000.0 * static_cast<double>(f->batch_rows) /
                              step_ms
                        : 0.0;
      const double gflops =
          step_ms > 0.0
              ? static_cast<double>(GemmFlopsPerStep(compiled->plan) * GradAccumOf(compiled->plan)) /
                    (step_ms * 1e6)
              : 0.0;
      const rt::StepTimings st = engine.step_timings();
      const double denom = st.steps ? static_cast<double>(st.steps) : 1.0;
      // External-standard fields. Every row of a SeeML batch is a loss
      // target (token plans: next-token label per position; feature plans:
      // one label per row), so MLX-LM's masked "Tokens/sec" is exactly
      // rows_per_s here; the key name says which unit the fixture trains in.
      const char* unit_key = std::strcmp(f->kind, "feature") == 0
                                 ? "samples_per_s"
                                 : "tokens_per_s";
      const double it_per_s = step_ms > 0.0 ? 1000.0 / step_ms : 0.0;
      const auto [lo_it, hi_it] =
          std::minmax_element(per_step_ms.begin(), per_step_ms.end());
      std::fprintf(out,
                   "%s\n        \"%" PRIu64 "\": {\"step_ms\": %.3f, "
                   "\"lifecycle_ms\": %.2f, \"rows_per_s\": %.0f, "
                   "\"gemm_gflops\": %.1f, \"fwd_ms\": %.3f, "
                   "\"bwd_ms\": %.3f, \"opt_ms\": %.3f,\n"
                   "              \"%s\": %.0f, \"it_per_s\": %.3f, "
                   "\"step_ms_min\": %.3f, \"step_ms_max\": %.3f",
                   first_thread ? "" : ",", t, step_ms,
                   Median(lifecycle_ms), rows_per_s, gflops,
                   1000.0 * st.fwd_seconds / denom,
                   1000.0 * st.bwd_seconds / denom,
                   1000.0 * st.opt_seconds / denom, unit_key, rows_per_s,
                   it_per_s, *lo_it, *hi_it);
      if (peak_gflops > 0.0)
        std::fprintf(out, ", \"mfu\": %.4f", gflops / peak_gflops);
      std::fprintf(out, "}");
      first_thread = false;
    }

    const auto t_merge = Clock::now();
    if (auto ok = engine.RunMerge(); !ok)
      return Fail(f->name + (": " + ok.error()));
    const double merge_ms = MsSince(t_merge);
    const std::string ckpt = *out_path + "." + f->name + ".ckpt.tmp";
    staging.ckpt = ckpt;
    const auto t_ckpt = Clock::now();
    if (auto ok = engine.SaveCheckpoint(ckpt); !ok)
      return Fail(f->name + (": " + ok.error()));
    const double ckpt_ms = MsSince(t_ckpt);
    std::remove(ckpt.c_str());
    staging.ckpt.clear();
    // Tier C's "peak RSS vs arena" and MLX-LM's "Peak mem", measured rather
    // than asserted: the OS-observed peak over (arena + plan).
    const uint64_t peak_rss = PeakRssBytes();
    const double planned = static_cast<double>(compiled->arena_size) +
                           static_cast<double>(compiled->plan.size());
    std::fprintf(out,
                 "\n      },\n      \"merge_ms\": %.3f, "
                 "\"checkpoint_save_ms\": %.3f,\n"
                 "      \"train_loss_first\": %.6f, "
                 "\"train_loss_last\": %.6f, \"trained_tokens\": %" PRIu64
                 ",\n      \"peak_rss_bytes\": %" PRIu64
                 ", \"rss_over_planned\": %.3f\n    }",
                 merge_ms, ckpt_ms, static_cast<double>(train_loss_first),
                 static_cast<double>(train_loss_last),
                 trained_steps * static_cast<uint64_t>(f->batch_rows),
                 peak_rss,
                 planned > 0.0 ? static_cast<double>(peak_rss) / planned
                               : 0.0);
  }

  std::fprintf(out, "\n  }\n}\n");
  // A truncated report must not exit 0 — same discipline as --report in
  // seeml-update-compile.
  const bool write_failed = std::ferror(out) != 0;
  const bool close_failed = std::fclose(out) != 0;
  staging.file = nullptr;
  if (write_failed || close_failed)
    return Fail("failed writing '" + tmp_path + "'");
  if (std::rename(tmp_path.c_str(), out_path->c_str()) != 0)
    return Fail("cannot move '" + tmp_path + "' to '" + *out_path + "'");
  staging.committed = true;
  std::fprintf(stderr, "seeml-bench: wrote %s\n", out_path->c_str());
  return 0;
}
