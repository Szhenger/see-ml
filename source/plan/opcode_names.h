#ifndef SEEML_SOURCE_PLAN_OPCODE_NAMES_H_
#define SEEML_SOURCE_PLAN_OPCODE_NAMES_H_

#include <cstdint>

#include "source/plan/instruction.h"

// =============================================================================
// The ONE table of opcode names (P6, #86): the disassembler prints them, the
// ABI manifest (tool/seeml_abi.cc -> tool/seeml/abi.json) publishes them, and
// the Python plane's interpreter and certifier are tested against that
// manifest — so "softmax_xent.fwd" is spelled once. The switch has no
// default on purpose: adding an OpCode without naming it here is a compile
// error (-Wswitch), which is what keeps the manifest complete.
// Build-host tools only; the runtime never needs a name.
// =============================================================================

namespace seeml::update {

/// The highest opcode value; the manifest walks 0..kOpCodeMax.
inline constexpr uint16_t kOpCodeMax = static_cast<uint16_t>(OpCode::kAttnDVTiled);

/// The opcode's name, or nullptr for a value that is not an OpCode.
inline const char* OpCodeName(uint16_t opcode) {
  if (opcode > kOpCodeMax) return nullptr;
  switch (static_cast<OpCode>(opcode)) {
    case OpCode::kNop:
      return "nop";
    case OpCode::kGemmNN:
      return "gemm.nn";
    case OpCode::kGemmNT:
      return "gemm.nt";
    case OpCode::kGemmTN:
      return "gemm.tn";
    case OpCode::kGemmAccNN:
      return "gemm.acc_nn";
    case OpCode::kAddEW:
      return "add.ew";
    case OpCode::kAddBias:
      return "add.bias";
    case OpCode::kReluFwd:
      return "relu.fwd";
    case OpCode::kReluBwd:
      return "relu.bwd";
    case OpCode::kScale:
      return "scale";
    case OpCode::kReduceRows:
      return "reduce.rows";
    case OpCode::kSoftmaxXEntFwd:
      return "softmax_xent.fwd";
    case OpCode::kSoftmaxXEntBwd:
      return "softmax_xent.bwd";
    case OpCode::kMseFwd:
      return "mse.fwd";
    case OpCode::kMseBwd:
      return "mse.bwd";
    case OpCode::kKLDistillFwd:
      return "kl_distill.fwd";
    case OpCode::kKLDistillBwd:
      return "kl_distill.bwd";
    case OpCode::kSgdStep:
      return "sgd.step";
    case OpCode::kAdamWStep:
      return "adamw.step";
    case OpCode::kFill:
      return "fill";
    case OpCode::kCopy:
      return "copy";
    case OpCode::kMulEW:
      return "mul.ew";
    case OpCode::kGeluFwd:
      return "gelu.fwd";
    case OpCode::kGeluBwd:
      return "gelu.bwd";
    case OpCode::kSiluFwd:
      return "silu.fwd";
    case OpCode::kSiluBwd:
      return "silu.bwd";
    case OpCode::kLayerNormFwd:
      return "layer_norm.fwd";
    case OpCode::kLayerNormBwd:
      return "layer_norm.bwd";
    case OpCode::kClipNorm:
      return "clip.norm";
    case OpCode::kGemmNNQ8:
      return "gemm.nn.q8";
    case OpCode::kGemmNTQ8:
      return "gemm.nt.q8";
    case OpCode::kRmsNormFwd:
      return "rms_norm.fwd";
    case OpCode::kRmsNormBwd:
      return "rms_norm.bwd";
    case OpCode::kRopeFwd:
      return "rope.fwd";
    case OpCode::kRopeBwd:
      return "rope.bwd";
    case OpCode::kAttnFwd:
      return "attn.fwd";
    case OpCode::kAttnDP:
      return "attn.dp";
    case OpCode::kAttnDV:
      return "attn.dv";
    case OpCode::kSoftmaxRowsBwd:
      return "softmax_rows.bwd";
    case OpCode::kAttnDQ:
      return "attn.dq";
    case OpCode::kAttnDK:
      return "attn.dk";
    case OpCode::kEmbedFwd:
      return "embed.fwd";
    case OpCode::kAccumulate:
      return "accumulate";
    case OpCode::kGemmNNBF16:
      return "gemm.nn.bf16";
    case OpCode::kGemmNTBF16:
      return "gemm.nt.bf16";
    case OpCode::kRopeTable:
      return "rope.table";
    case OpCode::kAttnFwdTiled: return "attn.fwd.tiled";
    case OpCode::kAttnDQTiled: return "attn.dq.tiled";
    case OpCode::kAttnDKTiled: return "attn.dk.tiled";
    case OpCode::kAttnDVTiled: return "attn.dv.tiled";
    case OpCode::kFusedMap:
      return "fused.map";
  }
  return nullptr;
}

}  // namespace seeml::update

#endif  // SEEML_SOURCE_PLAN_OPCODE_NAMES_H_
