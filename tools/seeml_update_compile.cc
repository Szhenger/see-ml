// =============================================================================
// seeml-update-compile — the SeeML Model Update Compiler CLI.
//
//   seeml-update-compile
//       --source  model.smf            the on-device model to update
//       --data-batch 32                compiled batch size
//       --loss    xent|mse|kl|xent+kl  training objective
//       [--teacher open_model.smf]     open-weights teacher (kl / xent+kl)
//       [--distill-weight 0.5]         KL weight for the composite loss
//       [--temperature 2.0]            distillation temperature
//       [--lora-rank 8] [--lora-alpha 16] [--lora-seed 42]
//       [--targets substr,substr]      restrict adapters to matching weights
//       [--optimizer adamw|sgd] [--lr 1e-3] [--weight-decay 0.01]
//       [--steps 1000]                 default step count baked into the plan
//       --out     out_dir/             emission directory
//       [--build]                      run build.sh after emission
//
// Output: update_plan.seeu + generated TUs + build.sh; with --build, a set of
// .o files and the linked `model_update` executable.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>

#include "source/update/native_emitter.h"
#include "source/update/smf.h"
#include "source/update/update_compiler.h"

namespace {

const char* Arg(int argc, char** argv, const char* flag,
                const char* dflt = nullptr) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
  return dflt;
}

bool Has(int argc, char** argv, const char* flag) {
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0) return true;
  return false;
}

int Fail(const std::string& msg) {
  std::fprintf(stderr, "seeml-update-compile: %s\n", msg.c_str());
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace seeml::update;

  const char* source_path = Arg(argc, argv, "--source");
  const char* out_dir = Arg(argc, argv, "--out");
  if (!source_path || !out_dir)
    return Fail("--source and --out are required (see header for usage)");

  UpdateConfig config;
  config.batch = std::strtoll(Arg(argc, argv, "--data-batch", "32"), nullptr, 10);
  config.default_steps =
      std::strtoull(Arg(argc, argv, "--steps", "1000"), nullptr, 10);

  const std::string loss = Arg(argc, argv, "--loss", "xent");
  if (loss == "xent") config.loss = LossKind::kSoftmaxXEnt;
  else if (loss == "mse") config.loss = LossKind::kMse;
  else if (loss == "kl") config.loss = LossKind::kKLDistill;
  else if (loss == "xent+kl") config.loss = LossKind::kXEntPlusKL;
  else return Fail("unknown --loss '" + loss + "'");

  config.distill_weight =
      std::strtof(Arg(argc, argv, "--distill-weight", "0.5"), nullptr);
  config.temperature =
      std::strtof(Arg(argc, argv, "--temperature", "2.0"), nullptr);

  config.lora.rank = std::strtoll(Arg(argc, argv, "--lora-rank", "8"), nullptr, 10);
  config.lora.alpha = std::strtof(Arg(argc, argv, "--lora-alpha", "16"), nullptr);
  config.lora.seed =
      std::strtoull(Arg(argc, argv, "--lora-seed", "42"), nullptr, 10);
  if (const char* targets = Arg(argc, argv, "--targets")) {
    std::stringstream ss(targets);
    std::string item;
    while (std::getline(ss, item, ','))
      if (!item.empty()) config.lora.target_filters.push_back(item);
  }

  const std::string opt = Arg(argc, argv, "--optimizer", "adamw");
  if (opt == "adamw") config.optimizer.kind = OptimizerKind::kAdamW;
  else if (opt == "sgd") config.optimizer.kind = OptimizerKind::kSgd;
  else return Fail("unknown --optimizer '" + opt + "'");
  config.optimizer.lr = std::strtof(Arg(argc, argv, "--lr", "1e-3"), nullptr);
  config.optimizer.weight_decay =
      std::strtof(Arg(argc, argv, "--weight-decay", "0.01"), nullptr);

  // --- Ingest ---------------------------------------------------------------
  auto source = LoadSmf(source_path);
  if (!source) return Fail(source.error());

  SmfModel teacher_model;
  const SmfModel* teacher = nullptr;
  if (const char* teacher_path = Arg(argc, argv, "--teacher")) {
    auto t = LoadSmf(teacher_path);
    if (!t) return Fail(t.error());
    teacher_model = std::move(*t);
    teacher = &teacher_model;
  }

  // --- Compile ----------------------------------------------------------------
  UpdateCompiler compiler(config);
  auto compiled = compiler.Compile(*source, teacher);
  if (!compiled) return Fail(compiled.error());

  std::fprintf(stderr,
               "seeml-update-compile: %zu adapter(s) | %llu train + %llu merge "
               "instrs | arena %llu B (%llu B persistent)\n",
               compiled->adapters.size(),
               (unsigned long long)compiled->train_instruction_count,
               (unsigned long long)compiled->merge_instruction_count,
               (unsigned long long)compiled->arena_size,
               (unsigned long long)compiled->persistent_size);

  // --- Emit ------------------------------------------------------------------
  const std::string repo_root =
      std::filesystem::path(argv[0]).parent_path().parent_path().string();
  auto paths = EmitNativePackage(
      compiled->plan, out_dir,
      repo_root.empty() || !std::filesystem::exists(repo_root + "/source")
          ? "."
          : repo_root);
  if (!paths) return Fail(paths.error());

  std::fprintf(stderr, "seeml-update-compile: emitted %s\n",
               paths->plan_file.c_str());

  if (Has(argc, argv, "--build")) {
    const std::string cmd = "sh '" + paths->build_script + "'";
    if (std::system(cmd.c_str()) != 0) return Fail("build.sh failed");
  }
  return 0;
}
