#include "source/update/update_compiler.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "source/sir/sir.h"
#include "source/update/update_passes.h"
#include "source/utility/logger/logger.h"

namespace seeml::update {

namespace sir = seecpp::sir;
using seecpp::utility::Logger;

namespace {

constexpr uint64_t kAlign = 64;
uint64_t AlignUp(uint64_t v) { return (v + kAlign - 1) & ~(kAlign - 1); }

uint64_t F32Bits(float f) { return std::bit_cast<uint32_t>(f); }

// -----------------------------------------------------------------------------
// Forward graph construction (SMF -> SIR)
// -----------------------------------------------------------------------------

struct GraphBuild {
  sir::Value* input = nullptr;    // shared batch input block argument
  sir::Value* output = nullptr;   // network output (logits)
  // Frozen weight values -> their SMF tensor (for rodata packing + commit).
  std::unordered_map<const sir::Value*, const SmfTensor*> weight_sources;
};

std::expected<sir::Value*, std::string> BuildForward(
    sir::Block& block, const SmfModel& model, const std::string& prefix,
    sir::Value* input, int64_t batch, GraphBuild& build) {
  std::unordered_map<std::string, sir::Value*> values;
  values[model.input_name] = input;

  auto materialize_weight =
      [&](const std::string& name) -> std::expected<sir::Value*, std::string> {
    const SmfTensor* t = model.FindTensor(name);
    if (!t || !t->is_const)
      return std::unexpected("UpdateCompiler: '" + name +
                             "' is not a constant tensor in the model");
    sir::Operation* op = block.appendOp("sc_mem.weight");
    op->setAttribute("smf_offset", static_cast<int64_t>(t->data_offset));
    sir::Value* v = op->addResult(prefix + name, sir::DataType::F32,
                                  sir::Shape(t->dims));
    build.weight_sources[v] = t;
    return v;
  };

  auto resolve =
      [&](const std::string& name) -> std::expected<sir::Value*, std::string> {
    if (auto it = values.find(name); it != values.end()) return it->second;
    return materialize_weight(name);
  };

  for (const SmfOp& op : model.ops) {
    switch (op.kind) {
      case SmfOpKind::kMatMul: {
        if (op.inputs.size() != 2)
          return std::unexpected("UpdateCompiler: MatMul '" + op.name +
                                 "' needs 2 inputs");
        auto x = resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        auto w = resolve(op.inputs[1]);
        if (!w) return std::unexpected(w.error());
        const int64_t k_x = (*x)->shape().dims.at(1);
        const int64_t k_w = (*w)->shape().dims.at(0);
        if (k_x != k_w)
          return std::unexpected("UpdateCompiler: MatMul '" + op.name +
                                 "' inner dimensions disagree (" +
                                 std::to_string(k_x) + " vs " +
                                 std::to_string(k_w) + ")");
        sir::Operation* mm = block.appendOp("sc_high.matmul");
        mm->addOperand(*x);
        mm->addOperand(*w);
        values[op.output] = mm->addResult(
            prefix + op.output, sir::DataType::F32,
            sir::Shape{batch, (*w)->shape().dims.at(1)});
        break;
      }
      case SmfOpKind::kAddBias: {
        if (op.inputs.size() != 2)
          return std::unexpected("UpdateCompiler: AddBias '" + op.name +
                                 "' needs 2 inputs");
        auto x = resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        auto b = resolve(op.inputs[1]);
        if (!b) return std::unexpected(b.error());
        sir::Operation* ab = block.appendOp("sc_high.add_bias");
        ab->addOperand(*x);
        ab->addOperand(*b);
        values[op.output] = ab->addResult(prefix + op.output,
                                          sir::DataType::F32, (*x)->shape());
        break;
      }
      case SmfOpKind::kRelu: {
        auto x = resolve(op.inputs.at(0));
        if (!x) return std::unexpected(x.error());
        sir::Operation* r = block.appendOp("sc_high.relu");
        r->addOperand(*x);
        values[op.output] = r->addResult(prefix + op.output,
                                         sir::DataType::F32, (*x)->shape());
        break;
      }
    }
  }

  auto out = values.find(model.output_name);
  if (out == values.end())
    return std::unexpected("UpdateCompiler: model output '" +
                           model.output_name + "' was never produced");
  return out->second;
}

// -----------------------------------------------------------------------------
// Segmented arena binding
// -----------------------------------------------------------------------------

struct ParamInit {
  const sir::Value* value;
  uint64_t offset;  // arena offset
  uint64_t bytes;
  std::string init;  // "randn" | "zeros"
  float std = 0.0f;
  uint64_t seed = 0;
};

struct ArenaBinding {
  std::unordered_map<const sir::Value*, uint64_t> refs;  // value -> ref word
  uint64_t persistent_size = 0;
  uint64_t io_end = 0;         // end of [persistent | io] prefix
  uint64_t arena_size = 0;     // total (after transient scan, both programs)
  std::vector<ParamInit> params;             // in allocation order
  std::vector<uint8_t> rodata;               // packed frozen weights
};

uint64_t ValueBytes(const sir::Value* v) {
  return AlignUp(v->shape().byteSize(v->dtype()));
}

/// Liveness-driven linear-scan allocation for transient values, mirroring the
/// middle-end ArenaMapper but segment-aware: values in `pinned` are never
/// reclaimed (loss slot, parameter gradients, merged weights).
uint64_t LinearScanTransients(
    sir::Block& block, uint64_t base,
    const std::unordered_map<const sir::Value*, uint64_t>& already_bound,
    const std::unordered_set<const sir::Value*>& pinned,
    std::unordered_map<const sir::Value*, uint64_t>& refs_out) {
  struct Interval {
    const sir::Value* value;
    size_t start, end;
    uint64_t bytes;
  };

  std::unordered_map<const sir::Value*, size_t> birth, death;
  std::vector<const sir::Value*> order;
  size_t tick = 0;
  block.walk([&](sir::Operation* op) {
    if (!op->mnemonic().starts_with("sc_mem.")) {
      for (const auto& res : op->results()) {
        if (already_bound.contains(res.get())) continue;
        birth[res.get()] = tick;
        death[res.get()] = tick;
        order.push_back(res.get());
      }
    }
    for (sir::Value* operand : op->operands())
      if (auto it = death.find(operand); it != death.end()) it->second = tick;
    ++tick;
  });

  std::vector<Interval> intervals;
  intervals.reserve(order.size());
  for (const sir::Value* v : order)
    intervals.push_back({v, birth[v],
                         pinned.contains(v) ? SIZE_MAX : death[v],
                         ValueBytes(v)});
  std::sort(intervals.begin(), intervals.end(),
            [](const Interval& a, const Interval& b) {
              return a.start < b.start;
            });

  struct ActiveBlock {
    uint64_t start, end;
    size_t free_after;
  };
  std::vector<ActiveBlock> active;
  uint64_t high_water = base;

  for (const Interval& iv : intervals) {
    std::erase_if(active, [&](const ActiveBlock& ab) {
      return ab.free_after != SIZE_MAX && ab.free_after < iv.start;
    });
    std::sort(active.begin(), active.end(),
              [](const ActiveBlock& a, const ActiveBlock& b) {
                return a.start < b.start;
              });
    uint64_t offset = base;
    for (const ActiveBlock& ab : active) {
      if (offset + iv.bytes <= ab.start) break;  // first fit
      offset = std::max(offset, ab.end);
    }
    refs_out[iv.value] = MakeArenaRef(offset);
    active.push_back({offset, offset + iv.bytes, iv.end});
    high_water = std::max(high_water, offset + iv.bytes);
  }
  return high_water;
}

std::expected<ArenaBinding, std::string> BindArena(
    sir::Block& train_block, const GraphBuild& build,
    const std::unordered_set<const sir::Value*>& pinned) {
  ArenaBinding binding;

  // --- PERSISTENT segment: trainable adapters + optimizer state at offset 0.
  uint64_t cursor = 0;
  train_block.walk([&](sir::Operation* op) {
    if (op->mnemonic() != "sc_mem.param") return;
    const sir::Value* v = op->result(0);
    const uint64_t bytes = ValueBytes(v);
    binding.refs[v] = MakeArenaRef(cursor);
    binding.params.push_back(
        {.value = v,
         .offset = cursor,
         .bytes = bytes,
         .init = op->getAttrAs<std::string>("init").value_or("zeros"),
         .std = op->getAttrAs<float>("std").value_or(0.02f),
         .seed = static_cast<uint64_t>(
             op->getAttrAs<int64_t>("seed").value_or(0))});
    cursor += bytes;
  });
  binding.persistent_size = cursor;

  // --- IO segment: batch input + label slots, right after PERSISTENT.
  for (const auto& arg : train_block.arguments()) {
    binding.refs[arg.get()] = MakeArenaRef(cursor);
    cursor += ValueBytes(arg.get());
  }
  binding.io_end = cursor;

  // --- RODATA: pack every frozen weight, dedup-free sequential layout.
  train_block.walk([&](sir::Operation* op) {
    if (op->mnemonic() != "sc_mem.weight") return;
    const sir::Value* v = op->result(0);
    auto src = build.weight_sources.find(v);
    if (src == build.weight_sources.end()) return;  // caught below
    const uint64_t offset = AlignUp(binding.rodata.size());
    binding.rodata.resize(offset + src->second->byte_size, 0);
    std::memcpy(binding.rodata.data() + offset, src->second->data.data(),
                src->second->byte_size);
    binding.refs[v] = MakeRodataRef(offset);
  });

  bool missing_source = false;
  train_block.walk([&](sir::Operation* op) {
    if (op->mnemonic() == "sc_mem.weight" &&
        !binding.refs.contains(op->result(0)))
      missing_source = true;
  });
  if (missing_source)
    return std::unexpected(
        "UpdateCompiler: frozen weight without SMF backing data");

  // --- TRANSIENT segment: liveness-scanned workspace after the IO prefix.
  binding.arena_size =
      LinearScanTransients(train_block, AlignUp(binding.io_end), binding.refs,
                           pinned, binding.refs);
  return binding;
}

// -----------------------------------------------------------------------------
// Instruction lowering (SIR -> UpdateInstruction stream)
// -----------------------------------------------------------------------------

using ResolveFn = std::function<std::expected<uint64_t, std::string>(
    const sir::Value*)>;

std::expected<std::vector<UpdateInstruction>, std::string> LowerBlock(
    sir::Block& block, const ResolveFn& resolve) {
  std::vector<UpdateInstruction> instrs;
  std::string error;

  auto ref = [&](const sir::Value* v) -> uint64_t {
    auto r = resolve(v);
    if (!r) {
      if (error.empty()) error = r.error();
      return kNullRef;
    }
    return r.value();
  };
  auto vol = [](const sir::Value* v) {
    return static_cast<uint64_t>(v->shape().volume());
  };

  block.walk([&](sir::Operation* op) {
    if (!error.empty()) return;
    const std::string_view m = op->mnemonic();
    if (m.starts_with("sc_mem.")) return;  // storage declaration, no code

    UpdateInstruction ins;
    auto set = [&](OpCode oc) { ins.opcode = static_cast<uint16_t>(oc); };

    if (m == "sc_high.matmul" || m == "sc_low.matmul_nt" ||
        m == "sc_low.matmul_tn") {
      const sir::Value* a = op->operand(0);
      const sir::Value* c = op->result(0);
      set(m == "sc_high.matmul"  ? OpCode::kGemmNN
          : m == "sc_low.matmul_nt" ? OpCode::kGemmNT
                                    : OpCode::kGemmTN);
      ins.in[0] = ref(a);
      ins.in[1] = ref(op->operand(1));
      ins.in[2] = ref(c);
      ins.out[0] = static_cast<uint64_t>(c->shape().dims.at(0));  // M
      ins.out[1] = static_cast<uint64_t>(c->shape().dims.at(1));  // N
      ins.out[2] = static_cast<uint64_t>(                          // K
          m == "sc_low.matmul_tn" ? a->shape().dims.at(0)
                                  : a->shape().dims.at(1));
    } else if (m == "sc_low.gemm_acc") {
      const sir::Value* a = op->operand(0);
      const sir::Value* b = op->operand(1);
      set(OpCode::kGemmAccNN);
      ins.in[0] = ref(a);
      ins.in[1] = ref(b);
      ins.in[2] = ref(op->operand(2));
      ins.in[3] = F32Bits(op->getAttrAs<float>("alpha").value_or(1.0f));
      ins.out[0] = static_cast<uint64_t>(a->shape().dims.at(0));
      ins.out[1] = static_cast<uint64_t>(b->shape().dims.at(1));
      ins.out[2] = static_cast<uint64_t>(a->shape().dims.at(1));
    } else if (m == "sc_high.add") {
      set(OpCode::kAddEW);
      ins.in[0] = ref(op->operand(0));
      ins.in[1] = ref(op->operand(1));
      ins.in[2] = ref(op->result(0));
      ins.out[0] = vol(op->result(0));
    } else if (m == "sc_high.add_bias") {
      set(OpCode::kAddBias);
      ins.in[0] = ref(op->operand(0));
      ins.in[1] = ref(op->operand(1));
      ins.in[2] = ref(op->result(0));
      ins.out[0] = static_cast<uint64_t>(op->result(0)->shape().dims.at(0));
      ins.out[1] = static_cast<uint64_t>(op->result(0)->shape().dims.at(1));
    } else if (m == "sc_high.relu") {
      set(OpCode::kReluFwd);
      ins.in[0] = ref(op->operand(0));
      ins.in[1] = ref(op->result(0));
      ins.out[0] = vol(op->result(0));
    } else if (m == "sc_low.relu_grad") {
      set(OpCode::kReluBwd);
      ins.in[0] = ref(op->operand(0));
      ins.in[1] = ref(op->operand(1));
      ins.in[2] = ref(op->result(0));
      ins.out[0] = vol(op->result(0));
    } else if (m == "sc_high.scale" || m == "sc_low.scale") {
      set(OpCode::kScale);
      ins.in[0] = ref(op->operand(0));
      ins.in[1] = ref(op->result(0));
      ins.in[2] = F32Bits(op->getAttrAs<float>("alpha").value_or(1.0f));
      ins.out[0] = vol(op->result(0));
    } else if (m == "sc_low.reduce_rows") {
      set(OpCode::kReduceRows);
      ins.in[0] = ref(op->operand(0));
      ins.in[1] = ref(op->result(0));
      ins.out[0] = static_cast<uint64_t>(op->operand(0)->shape().dims.at(0));
      ins.out[1] = static_cast<uint64_t>(op->operand(0)->shape().dims.at(1));
    } else if (m == "sc_high.softmax_xent") {
      set(OpCode::kSoftmaxXEntFwd);
      ins.in[0] = ref(op->operand(0));   // logits
      ins.in[1] = ref(op->operand(1));   // labels (i32)
      ins.in[2] = ref(op->result(0));    // loss
      ins.in[3] = ref(op->result(1));    // probs cache
      ins.out[0] = static_cast<uint64_t>(op->operand(0)->shape().dims.at(0));
      ins.out[1] = static_cast<uint64_t>(op->operand(0)->shape().dims.at(1));
    } else if (m == "sc_low.softmax_xent_grad") {
      set(OpCode::kSoftmaxXEntBwd);
      ins.in[0] = ref(op->operand(0));   // probs
      ins.in[1] = ref(op->operand(1));   // labels
      ins.in[2] = ref(op->operand(2));   // seed
      ins.in[3] = ref(op->result(0));    // dlogits
      ins.out[0] = static_cast<uint64_t>(op->result(0)->shape().dims.at(0));
      ins.out[1] = static_cast<uint64_t>(op->result(0)->shape().dims.at(1));
    } else if (m == "sc_high.mse") {
      set(OpCode::kMseFwd);
      ins.in[0] = ref(op->operand(0));
      ins.in[1] = ref(op->operand(1));
      ins.in[2] = ref(op->result(0));
      ins.out[0] = vol(op->operand(0));
    } else if (m == "sc_low.mse_grad") {
      set(OpCode::kMseBwd);
      ins.in[0] = ref(op->operand(0));
      ins.in[1] = ref(op->operand(1));
      ins.in[2] = ref(op->operand(2));
      ins.in[3] = ref(op->result(0));
      ins.out[0] = vol(op->result(0));
    } else if (m == "sc_high.kl_distill") {
      const sir::Value* s = op->operand(0);
      set(OpCode::kKLDistillFwd);
      ins.in[0] = ref(s);
      ins.in[1] = ref(op->operand(1));   // teacher logits
      ins.in[2] = ref(op->result(0));    // loss
      ins.in[3] = ref(op->result(1));    // p_s
      ins.out[0] = ref(op->result(2));   // p_t
      ins.out[1] = (static_cast<uint64_t>(s->shape().dims.at(0)) << 32) |
                   static_cast<uint64_t>(s->shape().dims.at(1));
      ins.out[2] = F32Bits(op->getAttrAs<float>("temperature").value_or(1.0f));
    } else if (m == "sc_low.kl_grad") {
      const sir::Value* d = op->result(0);
      set(OpCode::kKLDistillBwd);
      ins.in[0] = ref(op->operand(0));   // p_s
      ins.in[1] = ref(op->operand(1));   // p_t
      ins.in[2] = ref(op->operand(2));   // seed
      ins.in[3] = ref(d);                // dlogits
      ins.out[0] = (static_cast<uint64_t>(d->shape().dims.at(0)) << 32) |
                   static_cast<uint64_t>(d->shape().dims.at(1));
      ins.out[1] = F32Bits(op->getAttrAs<float>("temperature").value_or(1.0f));
    } else if (m == "sc_low.sgd_step") {
      set(OpCode::kSgdStep);
      ins.in[0] = ref(op->operand(0));
      ins.in[1] = ref(op->operand(1));
      ins.out[0] = vol(op->operand(0));
    } else if (m == "sc_low.adamw_step") {
      set(OpCode::kAdamWStep);
      ins.in[0] = ref(op->operand(0));
      ins.in[1] = ref(op->operand(1));
      ins.in[2] = ref(op->operand(2));
      ins.in[3] = ref(op->operand(3));
      ins.out[0] = vol(op->operand(0));
    } else if (m == "sc_low.fill") {
      set(OpCode::kFill);
      ins.in[0] = ref(op->result(0));
      ins.in[1] = F32Bits(op->getAttrAs<float>("value").value_or(0.0f));
      ins.out[0] = vol(op->result(0));
    } else if (m == "sc_low.copy") {
      set(OpCode::kCopy);
      ins.in[0] = ref(op->operand(0));
      ins.in[1] = ref(op->result(0));
      ins.out[0] = vol(op->result(0));
    } else {
      error = "UpdateCompiler: cannot lower '" + std::string(m) + "'";
      return;
    }
    instrs.push_back(ins);
  });

  if (!error.empty()) return std::unexpected(error);
  return instrs;
}

}  // namespace

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

  // --- 6. Merge program: W' = W + (α/r)·A@B. --------------------------------
  MergeBuilder merger;
  auto merge = merger.Run(*adapters);
  if (!merge) return std::unexpected(merge.error());
  if (!merge->block->validate())
    return std::unexpected("UpdateCompiler: merge program failed validation");

  // --- 7. Segmented arena binding. -------------------------------------------
  // Pin the loss slot and every parameter gradient: they are read outside the
  // program (engine / optimizer-less debug builds) and must survive the step.
  std::unordered_set<const sir::Value*> pinned;
  pinned.insert(loss);
  for (const auto& [p, g] : *param_grads) pinned.insert(g);

  auto binding = BindArena(block, build, pinned);
  if (!binding) return std::unexpected(binding.error());

  // Bind the merge program into the same arena: mirrors resolve through the
  // alias map; merged outputs are pinned fresh transients in the workspace
  // region (training has finished when the merge runs, so reuse is safe).
  std::unordered_set<const sir::Value*> merge_pinned;
  for (const auto& [merged, adapter] : merge->outputs) merge_pinned.insert(merged);

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

  // --- 8. Lowering. ----------------------------------------------------------
  auto resolve_train =
      [&](const sir::Value* v) -> std::expected<uint64_t, std::string> {
    if (auto it = binding->refs.find(v); it != binding->refs.end())
      return it->second;
    return std::unexpected("UpdateCompiler: unbound value '" +
                           std::string(v->id()) + "'");
  };
  auto train_instrs = LowerBlock(block, resolve_train);
  if (!train_instrs) return std::unexpected(train_instrs.error());

  auto resolve_merge =
      [&](const sir::Value* v) -> std::expected<uint64_t, std::string> {
    if (auto it = merge_bound.find(v); it != merge_bound.end())
      return it->second;
    return std::unexpected("UpdateCompiler: unbound merge value '" +
                           std::string(v->id()) + "'");
  };
  auto merge_instrs = LowerBlock(*merge->block, resolve_merge);
  if (!merge_instrs) return std::unexpected(merge_instrs.error());

  // --- 9. Persistent-segment initial image (deterministic, seeded). ---------
  std::vector<uint8_t> persist_init(binding->persistent_size, 0);
  for (const ParamInit& p : binding->params) {
    if (p.init != "randn") continue;  // zeros already
    std::mt19937_64 rng(p.seed);
    std::normal_distribution<float> dist(0.0f, p.std);
    auto* dst = reinterpret_cast<float*>(persist_init.data() + p.offset);
    const uint64_t count =
        static_cast<uint64_t>(p.value->shape().volume());
    for (uint64_t i = 0; i < count; ++i) dst[i] = dist(rng);
  }

  // --- 10. Emit table: merged arena ranges -> source-file byte ranges. ------
  std::vector<EmitEntry> emit_table;
  for (const auto& [merged, adapter] : merge->outputs) {
    auto src = build.weight_sources.find(adapter->frozen_weight);
    if (src == build.weight_sources.end())
      return std::unexpected("UpdateCompiler: adapter weight lacks SMF source");
    emit_table.push_back({.smf_data_offset = src->second->data_offset,
                          .byte_size = src->second->byte_size,
                          .arena_offset = RefOffset(merge_bound.at(merged))});
  }

  // --- 11. Plan assembly. -----------------------------------------------------
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

  uint64_t off = AlignUp(sizeof(PlanHeader));
  header.train_instr_offset = off;
  header.train_instr_count = train_instrs->size();
  off += train_instrs->size() * sizeof(UpdateInstruction);
  header.merge_instr_offset = off;
  header.merge_instr_count = merge_instrs->size();
  off += merge_instrs->size() * sizeof(UpdateInstruction);
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
  if (!binding->rodata.empty())
    put(header.rodata_offset, binding->rodata.data(), binding->rodata.size());
  if (!persist_init.empty())
    put(header.persist_init_offset, persist_init.data(), persist_init.size());
  if (!emit_table.empty())
    put(header.emit_table_offset, emit_table.data(),
        emit_table.size() * sizeof(EmitEntry));

  // --- 12. Debug hooks + report. ----------------------------------------------
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
    result.adapters.push_back(
        {.weight_name = std::string(a.frozen_weight->id()),
         .weight_rodata_ref = binding->refs.at(a.frozen_weight),
         .a_ref = binding->refs.at(a.A),
         .b_ref = binding->refs.at(a.B),
         .merged_ref = merge_bound.at(merge->outputs[i].first),
         .k = a.frozen_weight->shape().dims.at(0),
         .m = a.frozen_weight->shape().dims.at(1),
         .r = a.A->shape().dims.at(1),
         .scale = a.scale});
  }

  result.arena_size = header.arena_size;
  result.persistent_size = header.persistent_size;
  result.train_instruction_count = header.train_instr_count;
  result.merge_instruction_count = header.merge_instr_count;

  Logger::Info("UpdateCompiler: plan compiled — " +
               std::to_string(header.train_instr_count) + " train instrs, " +
               std::to_string(header.merge_instr_count) + " merge instrs, " +
               "arena " + std::to_string(header.arena_size) + " B (" +
               std::to_string(header.persistent_size) + " B persistent)");
  return result;
}

}  // namespace seeml::update
