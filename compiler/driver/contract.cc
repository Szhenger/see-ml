#include "compiler/driver/contract.h"

#include "compiler/diagnostics/architecting/error.h"
#include "compiler/diagnostics/generating/error.h"
#include "compiler/diagnostics/parsing/error.h"
#include "compiler/diagnostics/passing/error.h"
#include "compiler/diagnostics/tokenizing/error.h"
#include "compiler/diagnostics/updating/error.h"

namespace seeml::update {

namespace {

namespace diag = seeml::diag;
namespace generating = seeml::diag::generating;

/// Every unit registered across the six diagnostics process modules. A
/// message that cannot be attributed to one of these escaped the partition.
constexpr std::string_view kRegisteredUnits[] = {
    diag::tokenizing::kContainer,      diag::tokenizing::kIngressor,
    diag::parsing::kUnit,              diag::passing::kPassManager,
    diag::passing::kConvLowering,      diag::updating::kAutodiff,
    diag::updating::kLoraGrafter,      diag::updating::kMergeBuilder,
    diag::updating::kOptimizer,        diag::updating::kEpilogueFuser,
    diag::architecting::kHostArch,
    diag::architecting::kAutotuner,    diag::generating::kDriver,
    diag::generating::kArenaBinder,    diag::generating::kInstructionLowering,
    diag::generating::kNativeEmitter,
};

std::unexpected<std::string> Broken(std::string_view boundary,
                                    std::string_view what) {
  std::string m;
  m.reserve(boundary.size() + what.size() + 12);
  m.append(boundary).append(" contract: ").append(what);
  return generating::Error(generating::kDriver, m);
}

}  // namespace

bool WellFormedDiagnostic(std::string_view message) {
  for (std::string_view unit : kRegisteredUnits) {
    if (message.size() > unit.size() + 2 && message.starts_with(unit) &&
        message.substr(unit.size(), 2) == ": ")
      return true;
  }
  return false;
}

std::expected<void, std::string> VerifyFrontendContract(
    const seeml::sir::Block& block, const GraphBuild& build) {
  if (!build.input || !build.output)
    return Broken("frontend", "graph build lacks an input or output value");
  if (auto v = block.verify(); !v)
    return Broken("frontend", "forward SIR is corrupt: " + v.error());
  for (const auto& [value, tensor] : build.weight_sources) {
    if (!value || !tensor)
      return Broken("frontend", "null weight-source entry");
    const seeml::sir::Operation* def = value->definingOp();
    if (!def || def->mnemonic() != "sc_mem.weight")
      return Broken("frontend", "weight source '" + std::string(value->id()) +
                                    "' is not an sc_mem.weight declaration");
    if (!tensor->is_const || tensor->dims.empty())
      return Broken("frontend", "weight source '" + std::string(value->id()) +
                                    "' maps to a non-constant SMF tensor");
  }
  return {};
}

std::expected<void, std::string> VerifyAnalysisContract(
    const seeml::sir::Block& block, std::span<const GraftedAdapter> adapters,
    const std::unordered_map<seeml::sir::Value*, seeml::sir::Value*>&
        param_grads,
    const MergeProgram& merge) {
  if (adapters.empty())
    return Broken("analysis", "no adapters were grafted");
  if (auto v = block.verify(); !v)
    return Broken("analysis", "training SIR is corrupt: " + v.error());

  for (const GraftedAdapter& a : adapters) {
    if (!a.frozen_weight || !a.A || !a.B)
      return Broken("analysis",
                    "adapter '" + a.id_base + "' has null values");
    for (seeml::sir::Value* p : {a.A, a.B}) {
      const seeml::sir::Operation* def = p->definingOp();
      if (!def || def->mnemonic() != "sc_mem.param")
        return Broken("analysis", "trainable '" + std::string(p->id()) +
                                      "' is not an sc_mem.param declaration");
      auto g = param_grads.find(p);
      if (g == param_grads.end() || !g->second)
        return Broken("analysis", "no gradient reached trainable '" +
                                      std::string(p->id()) + "'");
    }
  }
  if (param_grads.size() != 2 * adapters.size())
    return Broken("analysis",
                  "gradient count does not match the trainable set (" +
                      std::to_string(param_grads.size()) + " gradients for " +
                      std::to_string(adapters.size()) + " adapter(s))");

  if (!merge.block)
    return Broken("analysis", "merge program was never built");
  if (auto v = merge.block->verify(); !v)
    return Broken("analysis", "merge program is corrupt: " + v.error());
  if (merge.outputs.size() != adapters.size())
    return Broken("analysis", "merge program covers " +
                                  std::to_string(merge.outputs.size()) +
                                  " of " + std::to_string(adapters.size()) +
                                  " adapter(s)");
  for (const auto& [mirror, original] : merge.aliases)
    if (!mirror || !original)
      return Broken("analysis", "merge program has a null alias");
  return {};
}

std::expected<void, std::string> VerifyGeneratedPlan(
    const CompiledUpdate& compiled, size_t num_adapters) {
  if (compiled.plan.empty())
    return Broken("backend", "assembled plan is empty");
  if (compiled.train_instruction_count == 0 ||
      compiled.eval_instruction_count == 0 ||
      compiled.merge_instruction_count == 0)
    return Broken("backend", "a lowered program is empty (train " +
                                 std::to_string(
                                     compiled.train_instruction_count) +
                                 ", eval " +
                                 std::to_string(
                                     compiled.eval_instruction_count) +
                                 ", merge " +
                                 std::to_string(
                                     compiled.merge_instruction_count) + ")");
  if (compiled.eval_instruction_count > compiled.train_instruction_count)
    return Broken("backend",
                  "eval program is larger than the training program that "
                  "contains it");
  if (compiled.merge_instruction_count < num_adapters)
    return Broken("backend", "fewer merge instructions than adapters");
  if (compiled.arena_size == 0 ||
      compiled.arena_size < compiled.persistent_size)
    return Broken("backend", "arena cannot contain its persistent segment");
  if (compiled.rodata_size == 0)
    return Broken("backend", "no frozen weight was packed to rodata");
  if (compiled.adapters.size() != num_adapters)
    return Broken("backend", "adapter debug hooks cover " +
                                 std::to_string(compiled.adapters.size()) +
                                 " of " + std::to_string(num_adapters) +
                                 " adapter(s)");
  if (compiled.params.size() != 2 * num_adapters)
    return Broken("backend", "parameter debug hooks do not cover the "
                             "trainable set");
  return {};
}

}  // namespace seeml::update
