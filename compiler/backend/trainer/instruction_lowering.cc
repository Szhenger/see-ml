#include "compiler/backend/trainer/instruction_lowering.h"
#include "source/language/model_format.h"

#include <bit>

#include "compiler/diagnostics/generating/error.h"

namespace seeml::update {

namespace sir = seeml::sir;
namespace generating = seeml::diag::generating;

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
  // Packed dim words carry two dims in 32-bit halves. Shapes are int64:
  // a half that does not fit must be a refusal, never a silent truncation —
  // the validator cannot tell a truncated word from an honest small one.
  auto pack32 = [&](int64_t hi, int64_t lo) -> uint64_t {
    if (hi < 0 || lo < 0 || hi > 0xFFFFFFFFll || lo > 0xFFFFFFFFll) {
      if (error.empty())
        error = "dimension pair (" + std::to_string(hi) + ", " +
                std::to_string(lo) + ") exceeds the 32-bit ISA dim fields";
      return 0;
    }
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
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
      // Fused epilogue (plan v5): a third operand is the fused bias, whose
      // ref rides the free in[3]; "epilogue_act" selects the activation.
      // Both are produced only by GemmEpilogueFuser, and only on
      // sc_high.matmul — the fuser also guarantees a bias never lands on a
      // q8 GEMM (in[3] is its scale), so a clash here is a compiler bug.
      const bool fused_bias =
          m == "sc_high.matmul" && op->numOperands() == 3;
      if (fused_bias && q8) {
        error = "fused bias on a quantized GEMM ('" +
                std::string(c->id()) + "')";
        break;
      }
      if (fused_bias) ins.in[3] = ref(op->operand(2));
      EpilogueAct act = EpilogueAct::kNone;
      if (auto act_name = op->getAttrAs<std::string>("epilogue_act")) {
        act = *act_name == "relu"   ? EpilogueAct::kRelu
              : *act_name == "gelu" ? EpilogueAct::kGelu
              : *act_name == "silu" ? EpilogueAct::kSilu
                                    : EpilogueAct::kNone;
        if (act == EpilogueAct::kNone) {
          error = "unknown epilogue activation '" + *act_name + "'";
          break;
        }
      }
      ins.flags = MakeEpilogueFlags(fused_bias, act);
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
      ins.out[0] = pack32(y->shape().dims.at(0), y->shape().dims.at(1));
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
      ins.out[2] = pack32(dx->shape().dims.at(0), dx->shape().dims.at(1));
    } else if (m == "sc_high.embedding") {
      const sir::Value* out = op->result(0);
      const sir::Value* table = op->operand(1);
      set(OpCode::kEmbedFwd);
      ins.in[0] = ref(op->operand(0));   // i32 tokens
      ins.in[1] = ref(table);            // rodata [V, D]
      ins.in[2] = ref(out);
      ins.out[0] = static_cast<uint64_t>(out->shape().dims.at(0));
      ins.out[1] =
          pack32(table->shape().dims.at(0), table->shape().dims.at(1));
    } else if (m == "sc_high.rms_norm") {
      const sir::Value* y = op->result(0);
      set(OpCode::kRmsNormFwd);
      ins.in[0] = ref(op->operand(0));   // x
      ins.in[1] = ref(op->operand(1));   // gamma
      ins.in[2] = ref(y);
      ins.in[3] = ref(op->result(1));    // rstd cache
      ins.out[0] = pack32(y->shape().dims.at(0), y->shape().dims.at(1));
    } else if (m == "sc_low.rms_norm_grad") {
      const sir::Value* dx = op->result(0);
      set(OpCode::kRmsNormBwd);
      ins.in[0] = ref(op->operand(0));   // dy
      ins.in[1] = ref(op->operand(1));   // x
      ins.in[2] = ref(op->operand(2));   // gamma
      ins.in[3] = ref(dx);
      ins.out[0] = ref(op->operand(3));  // rstd cache
      ins.out[1] = pack32(dx->shape().dims.at(0), dx->shape().dims.at(1));
    } else if (m == "sc_high.rope" || m == "sc_low.rope_grad" ||
               m == "sc_high.attention" || m == "sc_low.attn_dp" ||
               m == "sc_low.attn_dv" || m == "sc_low.attn_dq" ||
               m == "sc_low.attn_dk") {
      // Shared sequence geometry, packed as B<<32|S and H<<32|d. Derived
      // from a designated [T, H*d] activation of the op plus the heads/seq
      // attributes the frontend validated (seq | T, heads | D).
      const sir::Value* act = m == "sc_high.attention" ? op->operand(0)
                              : m == "sc_low.attn_dp"  ? op->operand(0)
                                                       : op->result(0);
      const int64_t heads = op->getAttrAs<int64_t>("heads").value_or(0);
      const int64_t seq = op->getAttrAs<int64_t>("seq").value_or(0);
      const int64_t rows = act->shape().dims.at(0);
      const int64_t width = act->shape().dims.at(1);
      if (heads <= 0 || seq <= 0 || rows % seq != 0 || width % heads != 0) {
        error = "malformed sequence geometry on '" + std::string(m) + "'";
        break;
      }
      const uint64_t bs = pack32(rows / seq, seq);
      const uint64_t hd = pack32(heads, width / heads);
      if (m == "sc_high.rope" || m == "sc_low.rope_grad") {
        set(m == "sc_high.rope" ? OpCode::kRopeFwd : OpCode::kRopeBwd);
        ins.in[0] = ref(op->operand(0));
        ins.in[1] = ref(op->result(0));
        ins.out[0] = bs;
        ins.out[1] = hd;
        // The parser always sets "base" (SMF v5 attr1, or the format
        // default); the fallback only covers SIR built by hand in tests.
        ins.out[2] = F32Bits(
            op->getAttrAs<float>("base").value_or(kSmfDefaultRopeBase));
      } else if (m == "sc_high.attention") {
        set(OpCode::kAttnFwd);
        ins.in[0] = ref(op->operand(0));   // q
        ins.in[1] = ref(op->operand(1));   // k
        ins.in[2] = ref(op->operand(2));   // v
        ins.in[3] = ref(op->result(0));    // o
        ins.out[0] = ref(op->result(1));   // probs cache
        ins.out[1] = bs;
        ins.out[2] = hd;
      } else {
        set(m == "sc_low.attn_dp"   ? OpCode::kAttnDP
            : m == "sc_low.attn_dv" ? OpCode::kAttnDV
            : m == "sc_low.attn_dq" ? OpCode::kAttnDQ
                                    : OpCode::kAttnDK);
        ins.in[0] = ref(op->operand(0));
        ins.in[1] = ref(op->operand(1));
        ins.in[2] = ref(op->result(0));
        ins.out[0] = bs;
        ins.out[1] = hd;
      }
    } else if (m == "sc_low.softmax_rows_grad") {
      const sir::Value* ds = op->result(0);
      set(OpCode::kSoftmaxRowsBwd);
      ins.in[0] = ref(op->operand(0));   // probs
      ins.in[1] = ref(op->operand(1));   // dp
      ins.in[2] = ref(ds);
      ins.out[0] = pack32(ds->shape().dims.at(0), ds->shape().dims.at(1));
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
      ins.out[1] = pack32(s->shape().dims.at(0), s->shape().dims.at(1));
      ins.out[2] = F32Bits(op->getAttrAs<float>("temperature").value_or(1.0f));
    } else if (m == "sc_low.kl_grad") {
      const sir::Value* d = op->result(0);
      set(OpCode::kKLDistillBwd);
      ins.in[0] = ref(op->operand(0));   // p_s
      ins.in[1] = ref(op->operand(1));   // p_t
      ins.in[2] = ref(op->operand(2));   // seed
      ins.in[3] = ref(d);                // dlogits
      ins.out[0] = pack32(d->shape().dims.at(0), d->shape().dims.at(1));
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
      error = "cannot lower '" + std::string(m) + "'";
      break;
    }
    instrs.push_back(ins);
  }

  if (!error.empty())
    return generating::Error(generating::kInstructionLowering, error);
  return instrs;
}

}  // namespace seeml::update
