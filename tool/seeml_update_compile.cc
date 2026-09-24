// =============================================================================
// seeml-update-compile — the SeeAI Model Update Compiler CLI.
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
//       [--min-lr-factor 0.1]          cosine floor as a fraction of --lr
//       [--allow-zero-lr]              permit --min-lr-factor 0 (the last
//                                      step then trains at LR 0)
//       [--quantize-base]              int8-quantize frozen weights in rodata
//       [--bf16-base]                  store frozen weights as bfloat16 rodata
//       [--precision f32|certified-bf16] relaxed frozen-weight GEMMs (v18):
//                                      the package then needs a numerics
//                                      certificate (tool/certify_numerics.py)
//                                      (2x smaller, f32 compute; exclusive with
//                                      --quantize-base)
//       [--steps 1000]                 default optimizer-step count baked into the plan
//       [--grad-accum 1]               micro-batches accumulated per optimizer step
//                                      (activations scale with --data-batch, the
//                                      effective batch is data-batch x grad-accum)
//       [--kernel-policy table.json]   the host-keyed kernel-policy table
//                                      tool/autotune.py measured; the entry
//                                      for this host (or --target-host KEY)
//                                      sets the plan's CPU GEMM tiles
//       [--target-host KEY]            look the table up by this host key
//                                      instead of the compile host's
//       [--gemm-tiles K,N]             set the CPU GEMM tiles explicitly
//                                      (throughput only, never bits)
//       [--report report.json]         machine-readable compile report
//       [--no-embed]                   skip the decimal byte-array TU; embed
//                                      the plan with tool/pack_update.py
//       [--build]                      run build.sh after emission
//       [--version]                    print the release version
//
// Every numeric flag is parsed strictly: trailing garbage, overflow, NaN/Inf,
// or an unknown flag is a hard error, never a silent default.
//
// Output: update_plan.seeu + generated TUs + vendored runtime sources +
// build.sh; with --build, the linked self-contained `model_update` binary.
// --no-embed leaves the plan on disk once (the .seeu) and no embedded TU:
// `python3 tool/pack_update.py <out> --build` then embeds it as an .incbin
// assembly stub, which is the route that scales past 100M parameters.
// =============================================================================

#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "compiler/backend/architecture/host_arch.h"
#include "compiler/backend/packaging/native_emitter.h"
#include "compiler/backend/architecture/kernel_policy_table.h"
#include "compiler/driver/update_compiler.h"
#include "compiler/frontend/ingressor/model_reader.h"
#include "source/identity/version.h"

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
               "  [--lr-schedule const|cosine] [--warmup N] [--allow-zero-lr]\n"
               "  [--min-lr-factor F] [--quantize-base | --bf16-base]\n"
               "  [--precision f32|certified-bf16]\n"
               "  [--attention auto|cached|tiled] [--attention-cache-budget-mib N]\n"
               "  [--steps N]\n"
               "  [--grad-accum G]\n"
               "  [--no-fuse-epilogue] [--no-fuse-elementwise] [--no-fuse-addend]\n"
               "  [--no-rope-table]\n"
               "  [--no-fuse-clip]\n"
               "  [--report out.json] [--dump-sir out.txt]\n"
               "  [--kernel-policy table.json] [--target-host KEY]\n"
               "  [--gemm-tiles K,N]\n"
               "  [--no-embed] [--build] [--version]\n");
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
        // A following flag is a missing value, not a value: consuming it
        // ("--targets --quantize-base") would silently disable one option
        // and misconfigure the other — the exact silent default the banner
        // forbids. Remember the culprit so the error names the real
        // mistake ("requires a value"), not "unknown argument".
        if (std::strncmp(argv_[i + 1], "--", 2) == 0) {
          if (!missing_value_) missing_value_ = flag;
          return std::nullopt;
        }
        taken_[i] = taken_[i + 1] = true;
        return std::string(argv_[i + 1]);
      }
    // The flag as the very last argument has no value slot at all.
    if (argc_ > 1 && !taken_[argc_ - 1] &&
        std::strcmp(argv_[argc_ - 1], flag) == 0 && !missing_value_)
      missing_value_ = flag;
    return std::nullopt;
  }

  /// First flag seen with no usable value; set lazily by TakeValue.
  const std::optional<std::string>& MissingValue() const {
    return missing_value_;
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
  std::optional<std::string> missing_value_;
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

// Finite only: strtof accepts "nan", "inf" and "infinity", and every range
// check below is a comparison a NaN silently passes, so a `--lr nan` would
// otherwise bake NaN into the plan header and poison the first step.
bool ParseF32(const std::string& s, float* out) {
  errno = 0;
  char* end = nullptr;
  const float v = std::strtof(s.c_str(), &end);
  if (errno != 0 || end == s.c_str() || *end != '\0' || !std::isfinite(v))
    return false;
  *out = v;
  return true;
}

std::string JsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);  // most names escape nothing
  for (char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      // Control characters (a newline in a path, say) are invalid raw JSON.
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x",
                    static_cast<unsigned>(static_cast<unsigned char>(c)));
      out += buf;
    } else {
      out += c;
    }
  }
  return out;
}

/// Single-quotes `s` for POSIX sh, escaping embedded quotes ('\''): the
/// path comes from user-supplied --out, and a bare quote in it would
/// terminate the quoting and execute the remainder as shell code.
std::string ShellQuote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'')
      out += "'\\''";
    else
      out += c;
  }
  out += '\'';
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

  if (args.Take("--version")) {
    std::printf("seeml-update-compile %s\n", kSeemlVersion);
    return 0;
  }

  const auto source_path = args.TakeValue("--source");
  const auto out_dir = args.TakeValue("--out");
  if (!source_path || !out_dir) {
    if (const auto& m = args.MissingValue())
      return Fail(*m + " requires a value");
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
    if (!ParseU64(*v, &config.default_steps) || config.default_steps == 0)
      return Fail("--steps must be a positive integer, got '" + *v + "'");
  }
  if (auto v = args.TakeValue("--grad-accum")) {
    uint64_t g = 0;
    if (!ParseU64(*v, &g) || g < 1 || g > 0xFFFFFFFFull)
      return Fail("--grad-accum must be a positive integer, got '" + *v + "'");
    config.grad_accum_steps = static_cast<uint32_t>(g);
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
  // The cosine schedule's two knobs are errors under `const` (E8, #91): a
  // flag that cannot apply is never silently ignored — the header's rule.
  const bool cosine =
      config.optimizer.lr_schedule == LrSchedule::kCosineWithWarmup;
  if (auto v = args.TakeValue("--warmup")) {
    if (!cosine)
      return Fail("--warmup applies to --lr-schedule cosine only");
    if (!ParseU64(*v, &config.optimizer.warmup_steps))
      return Fail("--warmup must be a non-negative integer, got '" + *v + "'");
  }
  if (auto v = args.TakeValue("--min-lr-factor")) {
    if (!cosine)
      return Fail("--min-lr-factor applies to --lr-schedule cosine only");
    if (!ParseF32(*v, &config.optimizer.min_lr_factor) ||
        config.optimizer.min_lr_factor < 0.0f ||
        config.optimizer.min_lr_factor > 1.0f)
      return Fail("--min-lr-factor must be in [0, 1], got '" + *v + "'");
  }
  config.optimizer.allow_zero_lr = args.Take("--allow-zero-lr");
  if (config.optimizer.allow_zero_lr && !cosine)
    return Fail("--allow-zero-lr applies to --lr-schedule cosine only");
  config.quantize_base = args.Take("--quantize-base");
  config.bf16_base = args.Take("--bf16-base");
  // Arithmetic (v18, F2 / F4): the relaxed GEMM family is a permission the
  // compiler grants per instruction; the package then needs a certificate.
  if (auto v = args.TakeValue("--precision")) {
    if (*v == "f32") config.precision = Precision::kF32;
    else if (*v == "certified-bf16")
      config.precision = Precision::kCertifiedBf16;
    else
      return Fail("unknown --precision '" + *v +
                  "' (f32 or certified-bf16)");
  }
  // Attention memory (E11, plan v15): auto tiles when the probability
  // caches of all layers would exceed the budget; the two families compute
  // the same bits, so this is a memory decision alone.
  if (auto v = args.TakeValue("--attention")) {
    if (*v == "auto") config.attention = AttentionKind::kAuto;
    else if (*v == "cached") config.attention = AttentionKind::kCached;
    else if (*v == "tiled") config.attention = AttentionKind::kTiled;
    else return Fail("unknown --attention '" + *v + "'");
  }
  if (auto v = args.TakeValue("--attention-cache-budget-mib")) {
    uint64_t mib = 0;
    if (!ParseU64(*v, &mib) || mib > (1ull << 40))
      return Fail("--attention-cache-budget-mib must be a whole number of "
                  "MiB, got '" + *v + "'");
    if (config.attention != AttentionKind::kAuto)
      return Fail("--attention-cache-budget-mib applies to --attention auto only");
    config.attention_cache_budget_bytes = mib << 20;
  }
  if (config.quantize_base && config.bf16_base)
    return Fail("--quantize-base and --bf16-base are mutually exclusive "
                "(one storage precision per weight)");
  config.fuse_epilogues = !args.Take("--no-fuse-epilogue");
  // Debug: the final SIR, rendered only when a destination is named.
  const auto dump_sir_path = args.TakeValue("--dump-sir");
  config.dump_sir = dump_sir_path.has_value();
  config.fuse_elementwise = !args.Take("--no-fuse-elementwise");
  config.fuse_gemm_addend = !args.Take("--no-fuse-addend");
  config.rope_table = !args.Take("--no-rope-table");
  config.fuse_clip = !args.Take("--no-fuse-clip");

  // --- Kernel policy: explicit tiles, the tuner's table, or the defaults.
  KernelPolicyRequest policy_request;
  if (auto v = args.TakeValue("--gemm-tiles")) {
    auto tiles = ParseGemmTilesFlag(*v);
    if (!tiles) return Fail(tiles.error());
    policy_request.explicit_tiles = *tiles;
  }
  policy_request.table_path = args.TakeValue("--kernel-policy");
  policy_request.target_host = args.TakeValue("--target-host");

  const auto teacher_path = args.TakeValue("--teacher");
  const auto report_path = args.TakeValue("--report");
  const bool no_embed = args.Take("--no-embed");
  const bool want_build = args.Take("--build");

  if (const auto& m = args.MissingValue())
    return Fail(*m + " requires a value");
  if (auto unknown = args.FirstUnknown())
    return Fail("unknown argument '" + *unknown + "' (see --help)");
  // Without an embedded TU the generated build.sh has nothing to link the
  // plan from; the packer is the step that supplies it, and it drives the
  // build itself. Refuse here rather than let build.sh fail after emission.
  if (no_embed && want_build)
    return Fail("--build needs the embedded plan TU that --no-embed skips; "
                "run `python3 tool/pack_update.py <out> --build` instead");

  // Resolved after the argument check so a malformed table is reported
  // against a well-formed command line. A table without this host is a
  // note (the defaults apply); a table that does not parse is an error.
  auto policy = ResolveKernelPolicy(policy_request);
  if (!policy) return Fail(policy.error());
  config.gemm_tile_k = policy->tiles.gemm_tile_k;
  config.gemm_tile_n = policy->tiles.gemm_tile_n;
  if (policy->source == "default")
    std::fprintf(stderr,
                 "seeml-update-compile: kernel policy default (the runtime's "
                 "GEMM tiles; host \"%s\")\n",
                 policy->host_key.c_str());
  else
    std::fprintf(stderr,
                 "seeml-update-compile: kernel policy from %s: GEMM tiles K %u "
                 "N %u (host \"%s\")\n",
                 policy->source.c_str(), policy->tiles.gemm_tile_k,
                 policy->tiles.gemm_tile_n, policy->host_key.c_str());

  // --- Ingest ---------------------------------------------------------------
  // Student and teacher load concurrently: one file's read overlaps the
  // other's hashing and payload copies.
  std::vector<std::string> model_paths{*source_path};
  if (teacher_path) model_paths.push_back(*teacher_path);
  auto models = LoadSmfMany(model_paths);
  if (!models) return Fail(models.error());

  SmfModel source_model = std::move((*models)[0]);
  SmfModel teacher_model;
  SmfModel* teacher = nullptr;
  if (teacher_path) {
    teacher_model = std::move((*models)[1]);
    teacher = &teacher_model;
  }

  // --- Compile ----------------------------------------------------------------
  UpdateCompiler compiler(config);
  // The consuming compile: nothing below reads a weight again, so each
  // payload is released as plan assembly packs it (one resident copy of
  // the weights while the blob is built, not two).
  auto compiled = compiler.Compile(std::move(source_model), teacher);
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
  EmitOptions emit_options;
  emit_options.relaxed_plan = compiled->relaxed_gemms > 0;
  emit_options.embed_plan_tu = !no_embed;
  auto paths = EmitNativePackage(
      compiled->plan, *out_dir,
      repo_root.empty() || !std::filesystem::exists(repo_root + "/source")
          ? "."
          : repo_root,
      emit_options);
  if (!paths) return Fail(paths.error());

  std::fprintf(stderr, "seeml-update-compile: emitted %s\n",
               paths->plan_file.c_str());
  if (no_embed)
    std::fprintf(stderr,
                 "seeml-update-compile: no embedded TU (--no-embed); next:"
                 " python3 tool/pack_update.py %s --build\n",
                 out_dir->c_str());

  if (dump_sir_path) {
    std::ofstream f(*dump_sir_path, std::ios::binary | std::ios::trunc);
    f << compiled->sir_dump;
    f.close();
    if (f.fail()) return Fail("short write to '" + *dump_sir_path + "'");
  }

  // --- Machine-readable report -------------------------------------------------
  if (report_path) {
    std::FILE* f = std::fopen(report_path->c_str(), "w");
    if (!f) return Fail("cannot write report '" + *report_path + "'");
    PlanHeader plan_header;
    std::memcpy(&plan_header, compiled->plan.data(), sizeof(plan_header));
    const HostArchInfo host_arch = DetectHostArch();
    // null when no embedded TU was written (--no-embed): the packer's stub
    // is the package's TU then, and the emitter has no path to report.
    const std::string embedded_tu_json =
        paths->embedded_tu.empty()
            ? "null"
            : "\"" + JsonEscape(paths->embedded_tu) + "\"";
    std::fprintf(f,
                 "{\n"
                 "  \"schema\": 1,\n"
                 "  \"seeml_version\": \"%s\",\n"
                 "  \"plan_file\": \"%s\",\n"
                 "  \"plan_bytes\": %zu,\n"
                 "  \"plan_version\": %u,\n"
                 "  \"plan_hash\": \"%016" PRIx64 "\",\n"
                 "  \"source_model_hash\": \"%016" PRIx64 "\",\n"
                 "  \"host_arch\": {\"isa\": \"%s\", \"cpu_model\": \"%s\", "
                 "\"physical_cores\": %zu, \"l1d_bytes\": %" PRIu64
                 ", \"l2_bytes\": %" PRIu64 ", \"simd_width_f32\": %zu},\n"
                 "  \"arena_bytes\": %" PRIu64 ",\n"
                 "  \"persistent_bytes\": %" PRIu64 ",\n"
                 "  \"rodata_bytes\": %" PRIu64 ",\n"
                 "  \"train_instructions\": %" PRIu64 ",\n"
                 "  \"eval_instructions\": %" PRIu64 ",\n"
                 "  \"merge_instructions\": %" PRIu64 ",\n"
                 "  \"step_instructions\": %" PRIu64 ",\n"
                 "  \"grad_accum_steps\": %u,\n"
                 "  \"effective_batch\": %" PRIu64 ",\n"
                 "  \"quantized_base\": %s,\n"
                 "  \"bf16_base\": %s,\n"
                 "  \"precision\": \"%s\",\n"
                 "  \"relaxed_gemms\": %" PRIu64 ",\n"
                 "  \"validation_scores\": \"%s\",\n"
                 "  \"attention\": {\"family\": \"%s\", "
                 "\"cached_probs_bytes\": %" PRIu64 "},\n"
                 "  \"embedded_tu\": %s,\n"
                 "  \"kernel_policy\": {\"source\": \"%s\", \"host_key\": "
                 "\"%s\", \"gemm_tile_k\": %u, \"gemm_tile_n\": %u},\n"
                 "  \"adapters\": [",
                 kSeemlVersion, JsonEscape(paths->plan_file).c_str(),
                 compiled->plan.size(), plan_header.version,
                 plan_header.plan_hash, plan_header.source_model_hash,
                 std::string(host_arch.isa).c_str(),
                 JsonEscape(host_arch.cpu_model).c_str(),
                 host_arch.physical_cores, host_arch.l1d_bytes,
                 host_arch.l2_bytes, host_arch.simd_width_f32,
                 compiled->arena_size,
                 compiled->persistent_size, compiled->rodata_size,
                 compiled->train_instruction_count,
                 compiled->eval_instruction_count,
                 compiled->merge_instruction_count,
                 compiled->step_instruction_count, compiled->grad_accum_steps,
                 static_cast<uint64_t>(config.batch) * compiled->grad_accum_steps,
                 config.quantize_base ? "true" : "false",
                 config.bf16_base ? "true" : "false",
                 config.precision == Precision::kCertifiedBf16
                     ? "certified-bf16" : "f32",
                 compiled->relaxed_gemms,
                 compiled->scores_shipped ? "shipped" : "plan",
                 compiled->attention_tiled ? "tiled" : "cached",
                 compiled->probs_cache_bytes,
                 embedded_tu_json.c_str(), policy->source.c_str(),
                 JsonEscape(policy->host_key).c_str(),
                 compiled->gemm_tile_k, compiled->gemm_tile_n);
    for (size_t i = 0; i < compiled->adapters.size(); ++i) {
      const auto& a = compiled->adapters[i];
      std::fprintf(f,
                   "%s\n    {\"weight\": \"%s\", \"k\": %" PRId64
                   ", \"m\": %" PRId64 ", \"rank\": %" PRId64
                   ", \"scale\": %g, \"quant_scale\": %g, \"bf16\": %s}",
                   i ? "," : "", JsonEscape(a.weight_name).c_str(), a.k, a.m,
                   a.r, a.scale, a.quant_scale, a.bf16 ? "true" : "false");
    }
    // The package's exact contents (P6): what the packer vendors against.
    std::fprintf(f, "\n  ],\n  \"vendored_sources\": [");
    for (size_t i = 0; i < paths->vendored_sources.size(); ++i)
      std::fprintf(f, "%s\"%s\"", i ? ", " : "",
                   JsonEscape(paths->vendored_sources[i]).c_str());
    std::fprintf(f, "]");
    // Every pass and driver phase, in order, with its wall time: the
    // measurement behind any compile-side performance claim (E5, #84).
    std::fprintf(f, ",\n  \"passes\": [");
    for (size_t i = 0; i < compiled->pass_timings.size(); ++i) {
      const auto& t = compiled->pass_timings[i];
      std::fprintf(f, "%s\n    {\"name\": \"%s\", \"ops\": %zu, \"ms\": %.3f}",
                   i ? "," : "", JsonEscape(t.name).c_str(), t.ops_after,
                   t.ms);
    }
    std::fprintf(f, "\n  ]\n}\n");
    // ferror catches any failed fprintf above; fclose catches the final
    // flush. A truncated report that exits 0 hands downstream automation
    // invalid JSON with a success status — the silent default this tool's
    // header promises never to have.
    const bool write_failed = std::ferror(f) != 0;
    if (std::fclose(f) != 0 || write_failed)
      return Fail("short write to report '" + *report_path + "'");
  }

  if (want_build) {
    const std::string cmd = "sh " + ShellQuote(paths->build_script);
    if (std::system(cmd.c_str()) != 0) return Fail("build.sh failed");
  }
  return 0;
}
