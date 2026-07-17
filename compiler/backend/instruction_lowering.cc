#include "compiler/backend/instruction_lowering.h"

#include <bit>

namespace seeml::update {

namespace sir = seecpp::sir;

namespace {

uint64_t F32Bits(float f) { return std::bit_cast<uint32_t>(f); }

}  // namespace

std::expected<std::vector<UpdateInstruction>, std::string> LowerOps(
    const std::vector<sir::Operation*>& ops, const ResolveFn& resolve,
    const std::unordered_map<const sir::Value*, float>& quant_scales) {
  std::vector<UpdateInstruction> instrs;
  instrs.reserve(ops.size());  // ~1 instruction per non-storage op
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

  for (sir::Operation* op : ops) {
    if (!error.empty()) break;
    const std::string_view m = op->mnemonic();
    if (m.starts_with("sc_mem.")) continue;  // storage declaration, no code

    UpdateInstruction ins;
    auto set = [&](OpCode oc) { ins.opcode = static_cast<uint16_t>(oc); };

    if (m == "sc_high.matmul" || m == "sc_low.matmul_nt" ||
        m == "sc_low.matmul_tn") {
      const sir::Value* a = op->operand(0);
      const sir::Value* c = op->result(0);
      // GEMMs whose B operand is a quantized frozen weight take the q8
      // opcode and carry the dequant scale in in[3].
      const auto q = m != "sc_low.matmul_tn"
                         ? quant_scales.find(op->operand(1))
                         : quant_scales.end();
      const bool q8 = q != quant_scales.end();
      set(m == "sc_high.matmul"     ? (q8 ? OpCode::kGemmNNQ8 : OpCode::kGemmNN)
          : m == "sc_low.matmul_nt" ? (q8 ? OpCode::kGemmNTQ8 : OpCode::kGemmNT)
                                    : OpCode::kGemmTN);
      ins.in[0] = ref(a);
      ins.in[1] = ref(op->operand(1));
      ins.in[2] = ref(c);
      if (q8) ins.in[3] = F32Bits(q->second);
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
    } else if (m == "sc_high.relu" || m == "sc_high.gelu" ||
               m == "sc_high.silu") {
      set(m == "sc_high.relu"   ? OpCode::kReluFwd
          : m == "sc_high.gelu" ? OpCode::kGeluFwd
                                : OpCode::kSiluFwd);
      ins.in[0] = ref(op->operand(0));
      ins.in[1] = ref(op->result(0));
      ins.out[0] = vol(op->result(0));
    } else if (m == "sc_low.relu_grad" || m == "sc_low.gelu_grad" ||
               m == "sc_low.silu_grad") {
      set(m == "sc_low.relu_grad"   ? OpCode::kReluBwd
          : m == "sc_low.gelu_grad" ? OpCode::kGeluBwd
                                    : OpCode::kSiluBwd);
      ins.in[0] = ref(op->operand(0));
      ins.in[1] = ref(op->operand(1));
      ins.in[2] = ref(op->result(0));
      ins.out[0] = vol(op->result(0));
    } else if (m == "sc_high.mul" || m == "sc_low.mul") {
      set(OpCode::kMulEW);
      ins.in[0] = ref(op->operand(0));
      ins.in[1] = ref(op->operand(1));
      ins.in[2] = ref(op->result(0));
      ins.out[0] = vol(op->result(0));
    } else if (m == "sc_high.layer_norm") {
      const sir::Value* y = op->result(0);
      set(OpCode::kLayerNormFwd);
      ins.in[0] = ref(op->operand(0));   // x
      ins.in[1] = ref(op->operand(1));   // gamma
      ins.in[2] = ref(op->operand(2));   // beta
      ins.in[3] = ref(y);
      ins.out[0] = (static_cast<uint64_t>(y->shape().dims.at(0)) << 32) |
                   static_cast<uint64_t>(y->shape().dims.at(1));
      ins.out[1] = ref(op->result(1));   // mean cache
      ins.out[2] = ref(op->result(2));   // rstd cache
    } else if (m == "sc_low.layer_norm_grad") {
      const sir::Value* dx = op->result(0);
      set(OpCode::kLayerNormBwd);
      ins.in[0] = ref(op->operand(0));   // dy
      ins.in[1] = ref(op->operand(1));   // x
      ins.in[2] = ref(op->operand(2));   // gamma
      ins.in[3] = ref(dx);
      ins.out[0] = ref(op->operand(3));  // mean cache
      ins.out[1] = ref(op->operand(4));  // rstd cache
      ins.out[2] = (static_cast<uint64_t>(dx->shape().dims.at(0)) << 32) |
                   static_cast<uint64_t>(dx->shape().dims.at(1));
    } else if (m == "sc_low.clip_norm") {
      set(OpCode::kClipNorm);
      ins.in[0] = ref(op->operand(0));
      ins.in[1] = F32Bits(op->getAttrAs<float>("max_norm").value_or(0.0f));
      ins.out[0] = vol(op->operand(0));
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
      break;
    }
    instrs.push_back(ins);
  }

  if (!error.empty()) return std::unexpected(error);
  return instrs;
}

}  // namespace seeml::update
