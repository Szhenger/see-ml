#include "compiler/backend/update_compiler.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "compiler/backend/arena_binder.h"
#include "compiler/backend/instruction_lowering.h"
#include "compiler/diagnostics/logger.h"
#include "compiler/frontend/forward_builder.h"
#include "compiler/frontend/sir.h"
#include "compiler/analysis/update_passes.h"
#include "source/hash.h"
#include "source/parallel_for.h"

namespace seeml::update {

namespace sir = seecpp::sir;
using seecpp::utility::Logger;

// =============================================================================
// UpdateCompiler::Compile
// =============================================================================

std::expected<CompiledUpdate, std::string> UpdateCompiler::Compile(
    const SmfModel& source, const SmfModel* teacher) {
  const bool wants_teacher = config_.loss == LossKind::kKLDistill ||
                             config_.loss == LossKind::kXEntPlusKL;
  if (wants_teacher && !teacher)
    return std::unexpected(
        "UpdateCompiler: the selected loss requires a teacher model");

  const SmfTensor* in_tensor = source.FindTensor(source.input_name);
  if (!in_tensor || in_tensor->dims.empty())
    return std::unexpected("UpdateCompiler: source model lacks input metadata");
  const int64_t in_dim = in_tensor->dims.back();
  const int64_t batch = config_.batch;

  // --- 1. Forward SIR: student (+ frozen teacher sharing the input). -------
  sir::Block block;
  GraphBuild build;
  build.input =
      block.addArgument(sir::DataType::F32, sir::Shape{batch, in_dim});

  auto student_out = BuildForward(block, source, "", build.input, batch, build);
  if (!student_out) return std::unexpected(student_out.error());
  build.output = *student_out;
  const int64_t out_dim = build.output->shape().dims.at(1);

  sir::Value* teacher_out = nullptr;
  if (wants_teacher) {
    const SmfTensor* t_in = teacher->FindTensor(teacher->input_name);
    if (!t_in || t_in->dims.empty() || t_in->dims.back() != in_dim)
      return std::unexpected(
          "UpdateCompiler: teacher input dimensionality mismatch");
    auto t_out = BuildForward(block, *teacher, "t::", build.input, batch, build);
    if (!t_out) return std::unexpected(t_out.error());
    teacher_out = *t_out;
    if (teacher_out->shape().dims.at(1) != out_dim)
      return std::unexpected(
          "UpdateCompiler: teacher output dimensionality mismatch");
  }

  // --- 2. Loss grafting. ----------------------------------------------------
  sir::Value* labels = nullptr;
  uint32_t label_kind = 0;
  uint64_t label_bytes = 0;
  sir::Value* loss = nullptr;

  auto make_xent = [&]() {
    labels = block.addArgument(sir::DataType::I32, sir::Shape{batch});
    label_kind = 1;
    label_bytes = static_cast<uint64_t>(batch) * sizeof(int32_t);
    sir::Operation* op = block.appendOp("sc_high.softmax_xent");
    op->addOperand(build.output);
    op->addOperand(labels);
    sir::Value* l = op->addResult("loss.xent", sir::DataType::F32, sir::Shape{});
    op->addResult("loss.xent_probs", sir::DataType::F32,
                  sir::Shape{batch, out_dim});
    return l;
  };
  auto make_kl = [&]() {
    sir::Operation* op = block.appendOp("sc_high.kl_distill");
    op->setAttribute("temperature", config_.temperature);
    op->addOperand(build.output);
    op->addOperand(teacher_out);
    sir::Value* l = op->addResult("loss.kl", sir::DataType::F32, sir::Shape{});
    op->addResult("loss.kl_ps", sir::DataType::F32, sir::Shape{batch, out_dim});
    op->addResult("loss.kl_pt", sir::DataType::F32, sir::Shape{batch, out_dim});
    return l;
  };

  switch (config_.loss) {
    case LossKind::kSoftmaxXEnt:
      loss = make_xent();
      break;
    case LossKind::kMse: {
      labels =
          block.addArgument(sir::DataType::F32, sir::Shape{batch, out_dim});
      label_kind = 2;
      label_bytes =
          static_cast<uint64_t>(batch) * static_cast<uint64_t>(out_dim) *
          sizeof(float);
      sir::Operation* op = block.appendOp("sc_high.mse");
      op->addOperand(build.output);
      op->addOperand(labels);
      loss = op->addResult("loss.mse", sir::DataType::F32, sir::Shape{});
      break;
    }
    case LossKind::kKLDistill:
      loss = make_kl();
      break;
    case LossKind::kXEntPlusKL: {
      sir::Value* xent = make_xent();
      sir::Value* kl = make_kl();
      sir::Operation* sx = block.appendOp("sc_high.scale");
      sx->setAttribute("alpha", 1.0f - config_.distill_weight);
      sx->addOperand(xent);
      sir::Value* wx =
          sx->addResult("loss.xent_w", sir::DataType::F32, sir::Shape{});
      sir::Operation* sk = block.appendOp("sc_high.scale");
      sk->setAttribute("alpha", config_.distill_weight);
      sk->addOperand(kl);
      sir::Value* wk =
          sk->addResult("loss.kl_w", sir::DataType::F32, sir::Shape{});
      sir::Operation* total = block.appendOp("sc_high.add");
      total->addOperand(wx);
      total->addOperand(wk);
      loss = total->addResult("loss.total", sir::DataType::F32, sir::Shape{});
      break;
    }
  }

  // --- 3. LoRA grafting. ----------------------------------------------------
  LoraGrafter grafter(config_.lora);
  auto adapters = grafter.Run(block);
  if (!adapters) return std::unexpected(adapters.error());

  std::vector<sir::Value*> trainables;
  for (const GraftedAdapter& a : *adapters) {
    trainables.push_back(a.A);
    trainables.push_back(a.B);
  }

  // Snapshot the primal program (adapted forward + loss, nothing else): it
  // becomes the evaluation program, lowered against the same arena binding.
  // Running it computes the current validation loss without touching any
  // parameter — the basis of the runtime's regression gate.
  std::vector<sir::Operation*> primal_ops;
  primal_ops.reserve(block.numOps());
  block.walk([&](sir::Operation* op) { primal_ops.push_back(op); });

  // --- 4. Reverse-mode autodiff pruned to the adapters. ---------------------
  TrainableAutodiff autodiff;
  auto param_grads = autodiff.Run(block, loss, trainables);
  if (!param_grads) return std::unexpected(param_grads.error());

  // --- 5. Optimizer synthesis (one program execution == one full step). -----
  if (config_.emit_optimizer) {
    OptimizerSynthesizer opt(config_.optimizer);
    if (auto r = opt.Run(block, *param_grads); !r)
      return std::unexpected(r.error());
  }

  if (!block.validate())
    return std::unexpected("UpdateCompiler: training program failed SSA "
                           "dominance validation");

  // --- 6. Merge program: Δ = (α/r)·A@B (commit adds Δ to the model file). ---
  MergeBuilder merger;
  auto merge = merger.Run(*adapters);
  if (!merge) return std::unexpected(merge.error());
  if (!merge->block->validate())
    return std::unexpected("UpdateCompiler: merge program failed validation");

  // --- 7. Frozen-weight quantization selection. ------------------------------
  std::unordered_map<const sir::Value*, float> quant_scales;
  if (config_.quantize_base) {
    quant_scales = SelectQuantizedWeights(block, build);
    Logger::Info("UpdateCompiler: quantized " +
                 std::to_string(quant_scales.size()) +
                 " frozen weight(s) to int8 rodata");
  }

  // --- 8. Segmented arena binding. -------------------------------------------
  // Pin the loss slot and every parameter gradient: they are read outside the
  // program (engine / optimizer-less debug builds) and must survive the step.
  std::unordered_set<const sir::Value*> pinned;
  pinned.insert(loss);
  for (const auto& [p, g] : *param_grads) pinned.insert(g);

  auto binding = BindArena(block, build, pinned, quant_scales);
  if (!binding) return std::unexpected(binding.error());

  // Bind the merge program into the same arena: A/B mirrors resolve through
  // the alias map; delta outputs are pinned fresh transients in the workspace
  // region (training has finished when the merge runs, so reuse is safe).
  std::unordered_set<const sir::Value*> merge_pinned;
  for (const auto& [delta, adapter] : merge->outputs) merge_pinned.insert(delta);

  std::unordered_map<const sir::Value*, uint64_t> merge_bound;
  for (const auto& [mirror, original] : merge->aliases) {
    auto it = binding->refs.find(original);
    if (it == binding->refs.end())
      return std::unexpected("UpdateCompiler: merge alias to unbound value");
    merge_bound[mirror] = it->second;
  }
  const uint64_t merge_high = LinearScanTransients(
      *merge->block, AlignUp(binding->io_end), merge_bound, merge_pinned,
      merge_bound);
  binding->arena_size = std::max(binding->arena_size, merge_high);

  // --- 9. Lowering: train, eval (primal prefix), and merge programs. ---------
  auto resolve_train =
      [&](const sir::Value* v) -> std::expected<uint64_t, std::string> {
    if (auto it = binding->refs.find(v); it != binding->refs.end())
      return it->second;
    return std::unexpected("UpdateCompiler: unbound value '" +
                           std::string(v->id()) + "'");
  };
  std::vector<sir::Operation*> train_ops;
  train_ops.reserve(block.numOps());
  block.walk([&](sir::Operation* op) { train_ops.push_back(op); });
  auto train_instrs = LowerOps(train_ops, resolve_train, quant_scales);
  if (!train_instrs) return std::unexpected(train_instrs.error());

  auto eval_instrs = LowerOps(primal_ops, resolve_train, quant_scales);
  if (!eval_instrs) return std::unexpected(eval_instrs.error());

  auto resolve_merge =
      [&](const sir::Value* v) -> std::expected<uint64_t, std::string> {
    if (auto it = merge_bound.find(v); it != merge_bound.end())
      return it->second;
    return std::unexpected("UpdateCompiler: unbound merge value '" +
                           std::string(v->id()) + "'");
  };
  std::vector<sir::Operation*> merge_ops;
  merge_ops.reserve(merge->block->numOps());
  merge->block->walk([&](sir::Operation* op) { merge_ops.push_back(op); });
  auto merge_instrs = LowerOps(merge_ops, resolve_merge, quant_scales);
  if (!merge_instrs) return std::unexpected(merge_instrs.error());

  // --- 10. Persistent-segment initial image (deterministic, seeded). --------
  // Every randn parameter owns an independent per-tensor RNG stream and a
  // disjoint byte range of the image, so generation parallelizes per tensor
  // with output identical to the serial loop.
  std::vector<uint8_t> persist_init(binding->persistent_size, 0);
  std::vector<const ParamInit*> randn_params;
  for (const ParamInit& p : binding->params)
    if (p.init == "randn") randn_params.push_back(&p);  // zeros already
  ParallelFor(randn_params.size(), 1, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) {
      const ParamInit& p = *randn_params[i];
      std::mt19937_64 rng(p.seed);
      std::normal_distribution<float> dist(0.0f, p.std);
      auto* dst = reinterpret_cast<float*>(persist_init.data() + p.offset);
      const uint64_t count = static_cast<uint64_t>(p.value->shape().volume());
      for (uint64_t j = 0; j < count; ++j) dst[j] = dist(rng);
    }
  });

  // --- 11. Emit table: delta arena ranges -> source-file byte ranges. -------
  std::vector<EmitEntry> emit_table;
  for (const auto& [delta, adapter] : merge->outputs) {
    auto src = build.weight_sources.find(adapter->frozen_weight);
    if (src == build.weight_sources.end())
      return std::unexpected("UpdateCompiler: adapter weight lacks SMF source");
    emit_table.push_back({.smf_data_offset = src->second->data_offset,
                          .byte_size = src->second->byte_size,
                          .arena_offset = RefOffset(merge_bound.at(delta))});
  }

  // --- 12. Plan assembly. -----------------------------------------------------
  PlanHeader header;
  header.arena_size = AlignUp(binding->arena_size);
  header.persistent_size = binding->persistent_size;
  header.input_ref = binding->refs.at(build.input);
  header.input_floats = static_cast<uint64_t>(batch) *
                        static_cast<uint64_t>(in_dim);
  header.label_ref = labels ? binding->refs.at(labels) : kNullRef;
  header.label_bytes = label_bytes;
  header.label_kind = label_kind;
  header.optimizer_kind = static_cast<uint32_t>(config_.optimizer.kind);
  header.loss_ref = binding->refs.at(loss);
  header.lr = config_.optimizer.lr;
  header.beta1 = config_.optimizer.beta1;
  header.beta2 = config_.optimizer.beta2;
  header.eps = config_.optimizer.eps;
  header.weight_decay = config_.optimizer.weight_decay;
  header.batch = static_cast<uint64_t>(batch);
  header.default_steps = config_.default_steps;
  header.lr_schedule = static_cast<uint32_t>(config_.optimizer.lr_schedule);
  header.warmup_steps = config_.optimizer.warmup_steps;
  header.min_lr_factor = config_.optimizer.min_lr_factor;
  header.clip_norm = config_.optimizer.clip_norm;
  header.source_model_hash = source.content_hash;

  uint64_t off = AlignUp(sizeof(PlanHeader));
  header.train_instr_offset = off;
  header.train_instr_count = train_instrs->size();
  off += train_instrs->size() * sizeof(UpdateInstruction);
  header.merge_instr_offset = off;
  header.merge_instr_count = merge_instrs->size();
  off += merge_instrs->size() * sizeof(UpdateInstruction);
  header.eval_instr_offset = off;
  header.eval_instr_count = eval_instrs->size();
  off += eval_instrs->size() * sizeof(UpdateInstruction);
  off = AlignUp(off);
  header.rodata_offset = off;
  header.rodata_size = binding->rodata.size();
  off = AlignUp(off + binding->rodata.size());
  header.persist_init_offset = off;
  header.persist_init_size = persist_init.size();
  off = AlignUp(off + persist_init.size());
  header.emit_table_offset = off;
  header.emit_count = emit_table.size();
  off += emit_table.size() * sizeof(EmitEntry);

  CompiledUpdate result;
  result.plan.resize(off, 0);
  auto put = [&](uint64_t at, const void* src, size_t n) {
    std::memcpy(result.plan.data() + at, src, n);
  };
  put(0, &header, sizeof(header));
  put(header.train_instr_offset, train_instrs->data(),
      train_instrs->size() * sizeof(UpdateInstruction));
  put(header.merge_instr_offset, merge_instrs->data(),
      merge_instrs->size() * sizeof(UpdateInstruction));
  put(header.eval_instr_offset, eval_instrs->data(),
      eval_instrs->size() * sizeof(UpdateInstruction));
  if (!binding->rodata.empty())
    put(header.rodata_offset, binding->rodata.data(), binding->rodata.size());
  if (!persist_init.empty())
    put(header.persist_init_offset, persist_init.data(), persist_init.size());
  if (!emit_table.empty())
    put(header.emit_table_offset, emit_table.data(),
        emit_table.size() * sizeof(EmitEntry));

  // Integrity seal: hash the whole blob with the hash field zeroed (it is,
  // so far), then patch the result in. Initialize() re-derives and compares.
  const uint64_t plan_hash = Fnv1a64(result.plan.data(), result.plan.size());
  std::memcpy(result.plan.data() + offsetof(PlanHeader, plan_hash), &plan_hash,
              sizeof(plan_hash));

  // --- 13. Debug hooks + report. ----------------------------------------------
  std::ostringstream dump;
  block.print(dump);
  dump << "  --- merge program ---\n";
  merge->block->print(dump);
  result.sir_dump = dump.str();

  for (const auto& [p, g] : *param_grads)
    result.params.push_back(
        {.id = std::string(p->id()),
         .param_ref = binding->refs.at(p),
         .grad_ref = binding->refs.at(g),
         .count = static_cast<uint64_t>(p->shape().volume())});
  std::sort(result.params.begin(), result.params.end(),
            [](const ParamDebugInfo& a, const ParamDebugInfo& b) {
              return a.id < b.id;
            });

  for (size_t i = 0; i < adapters->size(); ++i) {
    const GraftedAdapter& a = (*adapters)[i];
    const auto q = quant_scales.find(a.frozen_weight);
    result.adapters.push_back(
        {.weight_name = std::string(a.frozen_weight->id()),
         .weight_rodata_ref = binding->refs.at(a.frozen_weight),
         .a_ref = binding->refs.at(a.A),
         .b_ref = binding->refs.at(a.B),
         .delta_ref = merge_bound.at(merge->outputs[i].first),
         .k = a.frozen_weight->shape().dims.at(0),
         .m = a.frozen_weight->shape().dims.at(1),
         .r = a.A->shape().dims.at(1),
         .scale = a.scale,
         .quant_scale = q != quant_scales.end() ? q->second : 0.0f});
  }

  result.arena_size = header.arena_size;
  result.persistent_size = header.persistent_size;
  result.train_instruction_count = header.train_instr_count;
  result.merge_instruction_count = header.merge_instr_count;
  result.eval_instruction_count = header.eval_instr_count;
  result.rodata_size = header.rodata_size;

  Logger::Info("UpdateCompiler: plan compiled — " +
               std::to_string(header.train_instr_count) + " train + " +
               std::to_string(header.eval_instr_count) + " eval + " +
               std::to_string(header.merge_instr_count) + " merge instrs, " +
               "arena " + std::to_string(header.arena_size) + " B (" +
               std::to_string(header.persistent_size) + " B persistent), " +
               "rodata " + std::to_string(header.rodata_size) + " B");
  return result;
}

}  // namespace seeml::update
