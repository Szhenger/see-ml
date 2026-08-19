// =============================================================================
// seeml-bench — the benchmark harness of docs/benchmarks.md. Usage:
//
//   seeml-bench --out bench.json [--threads 1,8] [--steps-lo 20]
//               [--steps-hi 80] [--repeats 3] [--fixtures name,name,...]
//               [--version]
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
// Everything is pinned: fixture seeds, dataset seeds, thread widths (via
// SetParallelThreadCount, overriding SEEML_THREADS). Medians of --repeats
// runs. Argument parsing is strict, as in seeml-update-compile: an unknown
// flag, a missing value, or trailing garbage is a hard error (exit 2).
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <expected>
#include <string>
#include <vector>

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

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
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
};

/// Sums 2·M·N·K over every GEMM-family instruction in the plan's training
/// stream — the plan itself is the ground truth for per-step matmul work.
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
    out = out * 10 + static_cast<uint64_t>(c - '0');
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
  for (auto* v : {&out_path, &threads_csv, &lo_s, &hi_s, &repeats_s,
                  &fixtures_csv})
    if (!*v) return Fail(v->error());
  if (!args.rest.empty()) return Fail("unknown argument '" + args.rest[0] +
                                      "'");
  if (out_path->empty())
    return Fail("--out is required\nusage: seeml-bench --out bench.json "
                "[--threads 1,8] [--steps-lo 20] [--steps-hi 80] "
                "[--repeats 3] [--fixtures name,...] [--version]");

  auto lo = ParseU64("--steps-lo", *lo_s);
  auto hi = ParseU64("--steps-hi", *hi_s);
  auto repeats = ParseU64("--repeats", *repeats_s);
  for (auto* v : {&lo, &hi, &repeats})
    if (!*v) return Fail(v->error());
  if (*hi <= *lo) return Fail("--steps-hi must exceed --steps-lo");

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
    for (const Fixture& f : kFixtures) selected.push_back(&f);
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

  FILE* out = std::fopen(out_path->c_str(), "w");
  if (!out) return Fail("cannot open '" + *out_path + "' for writing");
  std::fprintf(out,
               "{\n  \"seeml_version\": \"%s\",\n  \"schema\": 1,\n"
               "  \"config\": {\"steps_lo\": %" PRIu64 ", \"steps_hi\": %"
               PRIu64 ", \"repeats\": %" PRIu64 ", \"threads\": \"%s\"},\n"
               "  \"fixtures\": {",
               seeml::update::kSeemlVersion, *lo, *hi, *repeats,
               threads_csv->c_str());

  bool first_fixture = true;
  for (const Fixture* f : selected) {
    std::fprintf(stderr, "seeml-bench: %s\n", f->name);
    const up::SmfModel model = f->model();
    up::UpdateConfig config = tf::BaseConfig(f->batch_rows);
    config.lora.rank = 8;
    config.lora.alpha = 16.0f;

    const auto t_compile = Clock::now();
    auto compiled = up::UpdateCompiler(config).Compile(model);
    if (!compiled) return Fail(f->name + (": " + compiled.error()));
    const double compile_ms = MsSince(t_compile);

    rt::UpdateEngine engine;
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

    std::fprintf(out,
                 "%s\n    \"%s\": {\n"
                 "      \"kind\": \"%s\", \"batch_rows\": %" PRId64
                 ", \"seq\": %" PRId64 ",\n"
                 "      \"plan_bytes\": %zu, \"arena_bytes\": %" PRIu64
                 ", \"persistent_bytes\": %" PRIu64 ", \"rodata_bytes\": %"
                 PRIu64 ",\n"
                 "      \"train_instructions\": %" PRIu64
                 ", \"gemm_flops_per_step\": %" PRIu64 ",\n"
                 "      \"compile_ms\": %.2f, \"load_ms\": %.3f,\n"
                 "      \"threads\": {",
                 first_fixture ? "" : ",", f->name, f->kind, f->batch_rows,
                 f->seq, compiled->plan.size(), compiled->arena_size,
                 compiled->persistent_size, compiled->rodata_size,
                 compiled->train_instruction_count,
                 GemmFlopsPerStep(compiled->plan), compile_ms, load_ms);
    first_fixture = false;

    bool first_thread = true;
    for (uint64_t t : threads) {
      up::SetParallelThreadCount(t);
      // Warm the pool, the caches, and the feeder before timing.
      if (auto r = engine.Train(*data, 4, quiet); !r)
        return Fail(f->name + (": " + r.error()));

      std::vector<double> per_step_ms, lifecycle_ms;
      engine.ResetStepTimings();
      for (uint64_t rep = 0; rep < *repeats; ++rep) {
        const auto t_lo = Clock::now();
        if (auto r = engine.Train(*data, *lo, quiet); !r)
          return Fail(f->name + (": " + r.error()));
        const double wall_lo = MsSince(t_lo);
        const auto t_hi = Clock::now();
        if (auto r = engine.Train(*data, *hi, quiet); !r)
          return Fail(f->name + (": " + r.error()));
        const double wall_hi = MsSince(t_hi);
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
              ? static_cast<double>(GemmFlopsPerStep(compiled->plan)) /
                    (step_ms * 1e6)
              : 0.0;
      const rt::StepTimings st = engine.step_timings();
      const double denom = st.steps ? static_cast<double>(st.steps) : 1.0;
      std::fprintf(out,
                   "%s\n        \"%" PRIu64 "\": {\"step_ms\": %.3f, "
                   "\"lifecycle_ms\": %.2f, \"rows_per_s\": %.0f, "
                   "\"gemm_gflops\": %.1f, \"fwd_ms\": %.3f, "
                   "\"bwd_ms\": %.3f, \"opt_ms\": %.3f}",
                   first_thread ? "" : ",", t, step_ms,
                   Median(lifecycle_ms), rows_per_s, gflops,
                   1000.0 * st.fwd_seconds / denom,
                   1000.0 * st.bwd_seconds / denom,
                   1000.0 * st.opt_seconds / denom);
      first_thread = false;
    }

    const auto t_merge = Clock::now();
    if (auto ok = engine.RunMerge(); !ok)
      return Fail(f->name + (": " + ok.error()));
    const double merge_ms = MsSince(t_merge);
    const std::string ckpt = *out_path + "." + f->name + ".ckpt.tmp";
    const auto t_ckpt = Clock::now();
    if (auto ok = engine.SaveCheckpoint(ckpt); !ok)
      return Fail(f->name + (": " + ok.error()));
    const double ckpt_ms = MsSince(t_ckpt);
    std::remove(ckpt.c_str());
    std::fprintf(out,
                 "\n      },\n      \"merge_ms\": %.3f, "
                 "\"checkpoint_save_ms\": %.3f\n    }",
                 merge_ms, ckpt_ms);
  }

  std::fprintf(out, "\n  }\n}\n");
  // A truncated report must not exit 0 — same discipline as --report in
  // seeml-update-compile.
  if (std::ferror(out) || std::fclose(out) != 0)
    return Fail("failed writing '" + *out_path + "'");
  std::fprintf(stderr, "seeml-bench: wrote %s\n", out_path->c_str());
  return 0;
}
