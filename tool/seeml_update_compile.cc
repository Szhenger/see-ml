// =============================================================================
// seeml-update-compile — the SeeML Model Update Compiler CLI.
//
//   seeml-update-compile
//       --source  model.smf            the on-device model to update
//       --out     out_dir/             emission directory
//       [--data-batch 32]              compiled batch size
//       [--loss xent|mse|kl|xent+kl]   training objective (default xent)
//       [--teacher open_model.smf]     open-weights teacher (kl / xent+kl)
//       [--distill-weight 0.5]         KL weight for the composite loss
//       [--temperature 2.0]            distillation temperature
//       [--lora-rank 8] [--lora-alpha 16] [--lora-seed 42]
//       [--targets substr,substr]      restrict adapters to matching weights
//       [--optimizer adamw|sgd] [--lr 1e-3] [--weight-decay 0.01]
//       [--clip-norm 0]                per-tensor L2 gradient clip (0 = off)
//       [--lr-schedule const|cosine]   runtime LR schedule
//       [--warmup 0]                   warmup steps (cosine schedule)
//       [--min-lr-factor 0]            cosine floor as a fraction of --lr
//       [--quantize-base]              int8-quantize frozen weights in rodata
//       [--steps 1000]                 default step count baked into the plan
//       [--report report.json]         machine-readable compile report
//       [--build]                      run build.sh after emission
//
// Every numeric flag is parsed strictly: trailing garbage, overflow, or an
// unknown flag is a hard error, never a silent default.
//
// Output: update_plan.seeu + generated TUs + vendored runtime sources +
// build.sh; with --build, the linked self-contained `model_update` binary.
// =============================================================================

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "compiler/backend/native_emitter.h"
#include "compiler/backend/update_compiler.h"
#include "compiler/frontend/ingressor/model_reader.h"

namespace {

using namespace seeml::update;

int Fail(const std::string& msg) {
  std::fprintf(stderr, "seeml-update-compile: %s\n", msg.c_str());
  return 1;
}

void PrintUsage() {
  std::fprintf(stderr,
               "usage: seeml-update-compile --source model.smf --out dir/\n"
               "  [--data-batch N] [--loss xent|mse|kl|xent+kl]\n"
               "  [--teacher t.smf] [--distill-weight W] [--temperature T]\n"
               "  [--lora-rank R] [--lora-alpha A] [--lora-seed S]\n"
               "  [--targets substr,...] [--optimizer adamw|sgd] [--lr LR]\n"
               "  [--weight-decay WD] [--clip-norm C]\n"
               "  [--lr-schedule const|cosine] [--warmup N]\n"
               "  [--min-lr-factor F] [--quantize-base] [--steps N]\n"
               "  [--report out.json] [--build]\n");
}

/// Strict argument cursor: every flag must be known, every value must parse
/// completely. This is the difference between `--lr 1e-3` and `--lr abc`
/// silently training at lr = 0.
class Args {
 public:
  Args(int argc, char** argv) : argc_(argc), argv_(argv) {}

  bool Take(const char* flag) {
    for (int i = 1; i < argc_; ++i)
      if (!taken_[i] && std::strcmp(argv_[i], flag) == 0) {
        taken_[i] = true;
        return true;
      }
    return false;
  }

  std::optional<std::string> TakeValue(const char* flag) {
    for (int i = 1; i + 1 < argc_; ++i)
      if (!taken_[i] && std::strcmp(argv_[i], flag) == 0) {
        taken_[i] = taken_[i + 1] = true;
        return std::string(argv_[i + 1]);
      }
    return std::nullopt;
  }

  /// Any argv slot not consumed by a Take* call is an error.
  std::optional<std::string> FirstUnknown() const {
    for (int i = 1; i < argc_; ++i)
      if (!taken_[i]) return std::string(argv_[i]);
    return std::nullopt;
  }

 private:
  int argc_;
  char** argv_;
  bool taken_[256] = {};
};

bool ParseI64(const std::string& s, int64_t* out) {
  errno = 0;
  char* end = nullptr;
  const long long v = std::strtoll(s.c_str(), &end, 10);
  if (errno != 0 || end == s.c_str() || *end != '\0') return false;
  *out = v;
  return true;
}

bool ParseU64(const std::string& s, uint64_t* out) {
  errno = 0;
  char* end = nullptr;
  if (!s.empty() && s[0] == '-') return false;
  const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
  if (errno != 0 || end == s.c_str() || *end != '\0') return false;
  *out = v;
  return true;
}

bool ParseF32(const std::string& s, float* out) {
  errno = 0;
  char* end = nullptr;
  const float v = std::strtof(s.c_str(), &end);
  if (errno != 0 || end == s.c_str() || *end != '\0') return false;
  *out = v;
  return true;
}

std::string JsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);  // most names escape nothing
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 255) return Fail("too many arguments");
  Args args(argc, argv);

  if (args.Take("--help") || args.Take("-h")) {
    PrintUsage();
    return 0;
  }

  const auto source_path = args.TakeValue("--source");
  const auto out_dir = args.TakeValue("--out");
  if (!source_path || !out_dir) {
    PrintUsage();
    return Fail("--source and --out are required");
  }

  UpdateConfig config;

  // --- Numeric / enum flags, all strictly parsed. -----------------------------
  if (auto v = args.TakeValue("--data-batch")) {
    if (!ParseI64(*v, &config.batch) || config.batch <= 0)
      return Fail("--data-batch must be a positive integer, got '" + *v + "'");
  }
  if (auto v = args.TakeValue("--steps")) {
    if (!ParseU64(*v, &config.default_steps))
      return Fail("--steps must be a non-negative integer, got '" + *v + "'");
  }
  if (auto v = args.TakeValue("--loss")) {
    if (*v == "xent") config.loss = LossKind::kSoftmaxXEnt;
    else if (*v == "mse") config.loss = LossKind::kMse;
    else if (*v == "kl") config.loss = LossKind::kKLDistill;
    else if (*v == "xent+kl") config.loss = LossKind::kXEntPlusKL;
    else return Fail("unknown --loss '" + *v + "'");
  }
  if (auto v = args.TakeValue("--distill-weight")) {
    if (!ParseF32(*v, &config.distill_weight) || config.distill_weight < 0.0f ||
        config.distill_weight > 1.0f)
      return Fail("--distill-weight must be in [0, 1], got '" + *v + "'");
  }
  if (auto v = args.TakeValue("--temperature")) {
    if (!ParseF32(*v, &config.temperature) || config.temperature <= 0.0f)
      return Fail("--temperature must be positive, got '" + *v + "'");
  }
  if (auto v = args.TakeValue("--lora-rank")) {
    if (!ParseI64(*v, &config.lora.rank) || config.lora.rank <= 0)
      return Fail("--lora-rank must be a positive integer, got '" + *v + "'");
  }
  if (auto v = args.TakeValue("--lora-alpha")) {
    if (!ParseF32(*v, &config.lora.alpha) || config.lora.alpha <= 0.0f)
      return Fail("--lora-alpha must be positive, got '" + *v + "'");
  }
  if (auto v = args.TakeValue("--lora-seed")) {
    if (!ParseU64(*v, &config.lora.seed))
      return Fail("--lora-seed must be a non-negative integer, got '" + *v +
                  "'");
  }
  if (auto v = args.TakeValue("--targets")) {
    std::stringstream ss(*v);
    std::string item;
    while (std::getline(ss, item, ','))
      if (!item.empty()) config.lora.target_filters.push_back(item);
  }
  if (auto v = args.TakeValue("--optimizer")) {
    if (*v == "adamw") config.optimizer.kind = OptimizerKind::kAdamW;
    else if (*v == "sgd") config.optimizer.kind = OptimizerKind::kSgd;
    else return Fail("unknown --optimizer '" + *v + "'");
  }
  if (auto v = args.TakeValue("--lr")) {
    if (!ParseF32(*v, &config.optimizer.lr) || config.optimizer.lr <= 0.0f)
      return Fail("--lr must be positive, got '" + *v + "'");
  }
  if (auto v = args.TakeValue("--weight-decay")) {
    if (!ParseF32(*v, &config.optimizer.weight_decay) ||
        config.optimizer.weight_decay < 0.0f)
      return Fail("--weight-decay must be non-negative, got '" + *v + "'");
  }
  if (auto v = args.TakeValue("--clip-norm")) {
    if (!ParseF32(*v, &config.optimizer.clip_norm) ||
        config.optimizer.clip_norm < 0.0f)
      return Fail("--clip-norm must be non-negative, got '" + *v + "'");
  }
  if (auto v = args.TakeValue("--lr-schedule")) {
    if (*v == "const") config.optimizer.lr_schedule = LrSchedule::kConstant;
    else if (*v == "cosine")
      config.optimizer.lr_schedule = LrSchedule::kCosineWithWarmup;
    else return Fail("unknown --lr-schedule '" + *v + "'");
  }
  if (auto v = args.TakeValue("--warmup")) {
    if (!ParseU64(*v, &config.optimizer.warmup_steps))
      return Fail("--warmup must be a non-negative integer, got '" + *v + "'");
  }
  if (auto v = args.TakeValue("--min-lr-factor")) {
    if (!ParseF32(*v, &config.optimizer.min_lr_factor) ||
        config.optimizer.min_lr_factor < 0.0f ||
        config.optimizer.min_lr_factor > 1.0f)
      return Fail("--min-lr-factor must be in [0, 1], got '" + *v + "'");
  }
  config.quantize_base = args.Take("--quantize-base");

  const auto teacher_path = args.TakeValue("--teacher");
  const auto report_path = args.TakeValue("--report");
  const bool want_build = args.Take("--build");

  if (auto unknown = args.FirstUnknown())
    return Fail("unknown argument '" + *unknown + "' (see --help)");

  // --- Ingest ---------------------------------------------------------------
  // Student and teacher load concurrently: one file's read overlaps the
  // other's hashing and payload copies.
  std::vector<std::string> model_paths{*source_path};
  if (teacher_path) model_paths.push_back(*teacher_path);
  auto models = LoadSmfMany(model_paths);
  if (!models) return Fail(models.error());

  SmfModel source_model = std::move((*models)[0]);
  SmfModel teacher_model;
  const SmfModel* teacher = nullptr;
  if (teacher_path) {
    teacher_model = std::move((*models)[1]);
    teacher = &teacher_model;
  }

  // --- Compile ----------------------------------------------------------------
  UpdateCompiler compiler(config);
  auto compiled = compiler.Compile(source_model, teacher);
  if (!compiled) return Fail(compiled.error());

  std::fprintf(stderr,
               "seeml-update-compile: %zu adapter(s) | %" PRIu64 " train + %"
               PRIu64 " eval + %" PRIu64 " merge instrs | arena %" PRIu64
               " B (%" PRIu64 " B persistent) | rodata %" PRIu64 " B\n",
               compiled->adapters.size(), compiled->train_instruction_count,
               compiled->eval_instruction_count,
               compiled->merge_instruction_count, compiled->arena_size,
               compiled->persistent_size, compiled->rodata_size);

  // --- Emit ------------------------------------------------------------------
  const std::string repo_root =
      std::filesystem::path(argv[0]).parent_path().parent_path().string();
  auto paths = EmitNativePackage(
      compiled->plan, *out_dir,
      repo_root.empty() || !std::filesystem::exists(repo_root + "/source")
          ? "."
          : repo_root);
  if (!paths) return Fail(paths.error());

  std::fprintf(stderr, "seeml-update-compile: emitted %s\n",
               paths->plan_file.c_str());

  // --- Machine-readable report -------------------------------------------------
  if (report_path) {
    std::FILE* f = std::fopen(report_path->c_str(), "w");
    if (!f) return Fail("cannot write report '" + *report_path + "'");
    std::fprintf(f,
                 "{\n"
                 "  \"plan_file\": \"%s\",\n"
                 "  \"arena_bytes\": %" PRIu64 ",\n"
                 "  \"persistent_bytes\": %" PRIu64 ",\n"
                 "  \"rodata_bytes\": %" PRIu64 ",\n"
                 "  \"train_instructions\": %" PRIu64 ",\n"
                 "  \"eval_instructions\": %" PRIu64 ",\n"
                 "  \"merge_instructions\": %" PRIu64 ",\n"
                 "  \"quantized_base\": %s,\n"
                 "  \"adapters\": [",
                 JsonEscape(paths->plan_file).c_str(), compiled->arena_size,
                 compiled->persistent_size, compiled->rodata_size,
                 compiled->train_instruction_count,
                 compiled->eval_instruction_count,
                 compiled->merge_instruction_count,
                 config.quantize_base ? "true" : "false");
    for (size_t i = 0; i < compiled->adapters.size(); ++i) {
      const auto& a = compiled->adapters[i];
      std::fprintf(f,
                   "%s\n    {\"weight\": \"%s\", \"k\": %" PRId64
                   ", \"m\": %" PRId64 ", \"rank\": %" PRId64
                   ", \"scale\": %g, \"quant_scale\": %g}",
                   i ? "," : "", JsonEscape(a.weight_name).c_str(), a.k, a.m,
                   a.r, a.scale, a.quant_scale);
    }
    std::fprintf(f, "\n  ]\n}\n");
    std::fclose(f);
  }

  if (want_build) {
    const std::string cmd = "sh '" + paths->build_script + "'";
    if (std::system(cmd.c_str()) != 0) return Fail("build.sh failed");
  }
  return 0;
}
