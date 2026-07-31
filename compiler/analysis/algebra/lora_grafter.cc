#include "compiler/analysis/algebra/lora_grafter.h"

#include <cmath>
#include <string_view>
#include <unordered_map>

#include "compiler/diagnostics/updating/error.h"

namespace seeml::update {

namespace sir = seeml::sir;
namespace updating = seeml::diag::updating;

namespace {

bool IsTeacherValue(const sir::Value* v) {
  return v->id().starts_with("t::");
}

int64_t Dim(const sir::Value* v, size_t i) { return v->shape().dims.at(i); }

}  // namespace

std::expected<std::vector<GraftedAdapter>, std::string> LoraGrafter::Run(
    sir::Block& block) {
  if (spec_.rank <= 0)
    return updating::Error(updating::kLoraGrafter, "rank must be positive");

  // Snapshot the target ops first: grafting mutates the op list.
  std::vector<sir::Operation*> targets;
  block.walk([&](sir::Operation* op) {
    if (op->mnemonic() != "sc_high.matmul" || op->numOperands() != 2) return;
    sir::Value* w = op->operand(1);
    const sir::Operation* def = w->definingOp();
    // Eligible: multiplies a *frozen* base weight (not an adapter, not an
    // activation), and is part of the student graph.
    if (!def || def->mnemonic() != "sc_mem.weight") return;
    if (IsTeacherValue(w) || IsTeacherValue(op->result(0))) return;
    if (!spec_.target_filters.empty()) {
      bool matched = false;
      for (const auto& f : spec_.target_filters)
        if (w->id().find(f) != std::string_view::npos) matched = true;
      if (!matched) return;
    }
    targets.push_back(op);
  });

  if (targets.empty())
    return updating::Error(updating::kLoraGrafter,
                           "no eligible MatMul targets found");

  std::vector<GraftedAdapter> adapters;
  adapters.reserve(targets.size());
  std::unordered_map<std::string, size_t> graft_sites;
  // One adapter pair per *unique* frozen weight. A tied weight consumed by
  // several MatMuls must share its A/B: per-site pairs would train fine but
  // commit W + ΣΔ_i to the single file weight, polluting every site with
  // every other site's delta. With a shared pair, each site computes
  // x_i @ (W + Δ) during training and the committed weight is exactly that.
  // Autodiff sums the pair's gradients across sites (Accumulate on fan-out).
  std::unordered_map<sir::Value*, size_t> adapter_for_weight;
  const float scale = spec_.alpha / static_cast<float>(spec_.rank);

  size_t adapter_index = 0;
  for (sir::Operation* target : targets) {
    sir::Value* x = target->operand(0);  // [N, K]
    sir::Value* w = target->operand(1);  // [K, M]
    sir::Value* c = target->result(0);   // [N, M]

    const int64_t n = Dim(x, 0);
    const int64_t k = Dim(w, 0);
    const int64_t m = Dim(w, 1);
    const int64_t r = spec_.rank;

    // Record C's consumers *before* creating the add op, so the rewire step
    // below naturally excludes the new LoRA subgraph itself.
    std::vector<sir::Operation*> consumers(c->users().begin(),
                                           c->users().end());

    // Later graft sites of a tied weight get suffixed ids for their compute
    // subgraph — Block::verify enforces id uniqueness, and ambiguous ids
    // would make ParamDebugInfo and the SIR dump lie about which site is
    // which.
    std::string base = std::string(w->id());
    if (const size_t site = graft_sites[base]++; site > 0)
      base += "@" + std::to_string(site);

    std::vector<std::unique_ptr<sir::Operation>> new_ops;
    sir::Value* a = nullptr;
    sir::Value* b = nullptr;
    const auto existing = adapter_for_weight.find(w);
    const bool is_new_adapter = existing == adapter_for_weight.end();
    if (is_new_adapter) {
      // A ~ N(0, 1/sqrt(K)), B = 0  =>  delta starts at zero: the grafted
      // model is bit-identical to the source model until training moves B.
      auto a_op = std::make_unique<sir::Operation>("sc_mem.param");
      a_op->setAttribute("trainable", int64_t{1});
      a_op->setAttribute("init", std::string("randn"));
      a_op->setAttribute("std", 1.0f / std::sqrt(static_cast<float>(k)));
      a_op->setAttribute("seed",
                         static_cast<int64_t>(spec_.seed + adapter_index));
      a = a_op->addResult(base + ".lora_A", sir::DataType::F32,
                          sir::Shape{k, r});

      auto b_op = std::make_unique<sir::Operation>("sc_mem.param");
      b_op->setAttribute("trainable", int64_t{1});
      b_op->setAttribute("init", std::string("zeros"));
      b = b_op->addResult(base + ".lora_B", sir::DataType::F32,
                          sir::Shape{r, m});
      new_ops.push_back(std::move(a_op));
      new_ops.push_back(std::move(b_op));
    } else {
      a = adapters[existing->second].A;
      b = adapters[existing->second].B;
    }

    auto t_op = std::make_unique<sir::Operation>("sc_high.matmul");
    t_op->addOperand(x);
    t_op->addOperand(a);
    sir::Value* t = t_op->addResult(base + ".lora_t", sir::DataType::F32,
                                    sir::Shape{n, r});

    auto u_op = std::make_unique<sir::Operation>("sc_high.matmul");
    u_op->addOperand(t);
    u_op->addOperand(b);
    sir::Value* u = u_op->addResult(base + ".lora_u", sir::DataType::F32,
                                    sir::Shape{n, m});

    auto s_op = std::make_unique<sir::Operation>("sc_high.scale");
    s_op->setAttribute("alpha", scale);
    s_op->addOperand(u);
    sir::Value* s = s_op->addResult(base + ".lora_s", sir::DataType::F32,
                                    sir::Shape{n, m});

    auto add_op = std::make_unique<sir::Operation>("sc_high.add");
    add_op->addOperand(c);
    add_op->addOperand(s);
    sir::Value* c_prime = add_op->addResult(base + ".lora_out",
                                            sir::DataType::F32,
                                            sir::Shape{n, m});

    new_ops.push_back(std::move(t_op));
    new_ops.push_back(std::move(u_op));
    new_ops.push_back(std::move(s_op));
    new_ops.push_back(std::move(add_op));
    block.insertOpsAfter(target, std::move(new_ops));

    // Rewire the original consumers of C onto C'.
    for (sir::Operation* consumer : consumers)
      for (size_t i = 0; i < consumer->numOperands(); ++i)
        if (consumer->operand(i) == c) consumer->setOperand(i, c_prime);

    if (is_new_adapter) {
      adapter_for_weight[w] = adapters.size();
      adapters.push_back({.frozen_weight = w, .A = a, .B = b, .scale = scale,
                          .id_base = base});
      ++adapter_index;
    }
  }

  seeml::diag::Note(updating::kLoraGrafter,
                    "grafted " + std::to_string(adapters.size()) +
                        " adapter(s), rank=" + std::to_string(spec_.rank));
  return adapters;
}

}  // namespace seeml::update
