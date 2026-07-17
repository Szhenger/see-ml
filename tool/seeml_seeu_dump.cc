// =============================================================================
// seeml-seeu-dump — .seeu Update Plan disassembler.
//
//   seeml-seeu-dump plan.seeu [--instrs]
//
// Prints the plan header (memory contract, I/O slots, hyperparameters,
// integrity hashes, section table) and, with --instrs, disassembles the
// train / eval / merge instruction streams. This is the field-debugging
// tool: it depends only on update_types.h + hash.h so it builds anywhere.
// =============================================================================

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "compiler/backend/update_types.h"
#include "source/hash.h"

namespace {

using namespace seeml::update;

const char* OpName(uint16_t opcode) {
  switch (static_cast<OpCode>(opcode)) {
    case OpCode::kNop:            return "nop";
    case OpCode::kGemmNN:         return "gemm.nn";
    case OpCode::kGemmNT:         return "gemm.nt";
    case OpCode::kGemmTN:         return "gemm.tn";
    case OpCode::kGemmAccNN:      return "gemm.acc_nn";
    case OpCode::kAddEW:          return "add.ew";
    case OpCode::kAddBias:        return "add.bias";
    case OpCode::kReluFwd:        return "relu.fwd";
    case OpCode::kReluBwd:        return "relu.bwd";
    case OpCode::kScale:          return "scale";
    case OpCode::kReduceRows:     return "reduce.rows";
    case OpCode::kSoftmaxXEntFwd: return "softmax_xent.fwd";
    case OpCode::kSoftmaxXEntBwd: return "softmax_xent.bwd";
    case OpCode::kMseFwd:         return "mse.fwd";
    case OpCode::kMseBwd:         return "mse.bwd";
    case OpCode::kKLDistillFwd:   return "kl_distill.fwd";
    case OpCode::kKLDistillBwd:   return "kl_distill.bwd";
    case OpCode::kSgdStep:        return "sgd.step";
    case OpCode::kAdamWStep:      return "adamw.step";
    case OpCode::kFill:           return "fill";
    case OpCode::kCopy:           return "copy";
    case OpCode::kMulEW:          return "mul.ew";
    case OpCode::kGeluFwd:        return "gelu.fwd";
    case OpCode::kGeluBwd:        return "gelu.bwd";
    case OpCode::kSiluFwd:        return "silu.fwd";
    case OpCode::kSiluBwd:        return "silu.bwd";
    case OpCode::kLayerNormFwd:   return "layer_norm.fwd";
    case OpCode::kLayerNormBwd:   return "layer_norm.bwd";
    case OpCode::kClipNorm:       return "clip.norm";
    case OpCode::kGemmNNQ8:       return "gemm.nn.q8";
    case OpCode::kGemmNTQ8:       return "gemm.nt.q8";
  }
  return "<unknown>";
}

void PrintRef(uint64_t ref) {
  if (ref == kNullRef) {
    std::printf("  <null>          ");
    return;
  }
  std::printf("  %s+0x%08" PRIx64, IsRodataRef(ref) ? "ro" : "ar",
              RefOffset(ref));
}

void Disassemble(const char* title, const UpdateInstruction* instrs,
                 uint64_t count) {
  std::printf("\n%s (%" PRIu64 " instructions)\n", title, count);
  for (uint64_t i = 0; i < count; ++i) {
    const UpdateInstruction& ins = instrs[i];
    std::printf("  %4" PRIu64 "  %-18s", i, OpName(ins.opcode));
    for (uint64_t in : ins.in)
      if (in != kNullRef) PrintRef(in);
    std::printf("   dims/aux: %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
                ins.out[0], ins.out[1], ins.out[2]);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: seeml-seeu-dump plan.seeu [--instrs]\n");
    return 2;
  }
  bool want_instrs = false;
  for (int i = 2; i < argc; ++i)
    if (std::strcmp(argv[i], "--instrs") == 0) want_instrs = true;

  std::ifstream f(argv[1], std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "seeml-seeu-dump: cannot open '%s'\n", argv[1]);
    return 1;
  }
  std::vector<uint8_t> plan((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
  if (plan.size() < sizeof(PlanHeader)) {
    std::fprintf(stderr, "seeml-seeu-dump: file smaller than a plan header\n");
    return 1;
  }
  PlanHeader h;
  std::memcpy(&h, plan.data(), sizeof(h));
  if (h.magic != kSeeuMagic) {
    std::fprintf(stderr, "seeml-seeu-dump: bad magic\n");
    return 1;
  }

  // Verify the integrity seal the same way the runtime does.
  uint64_t state = kFnvOffsetBasis;
  constexpr size_t kHashAt = offsetof(PlanHeader, plan_hash);
  constexpr uint8_t kZero[sizeof(uint64_t)] = {};
  state = Fnv1a64(plan.data(), kHashAt, state);
  state = Fnv1a64(kZero, sizeof(kZero), state);
  state = Fnv1a64(plan.data() + kHashAt + sizeof(uint64_t),
                  plan.size() - kHashAt - sizeof(uint64_t), state);

  std::printf("seeu plan: %s\n", argv[1]);
  std::printf("  version            %u\n", h.version);
  std::printf("  plan_hash          %016" PRIx64 "  (%s)\n", h.plan_hash,
              state == h.plan_hash ? "verified" : "MISMATCH — corrupt");
  std::printf("  source_model_hash  %016" PRIx64 "%s\n", h.source_model_hash,
              h.source_model_hash ? "" : "  (unbound)");
  std::printf("  arena              %" PRIu64 " B (%" PRIu64 " B persistent)\n",
              h.arena_size, h.persistent_size);
  std::printf("  rodata             %" PRIu64 " B\n", h.rodata_size);
  std::printf("  batch              %" PRIu64 "\n", h.batch);
  std::printf("  input slot         ar+0x%08" PRIx64 "  %" PRIu64 " floats\n",
              RefOffset(h.input_ref), h.input_floats);
  if (h.label_kind)
    std::printf("  label slot         ar+0x%08" PRIx64 "  %" PRIu64
                " B/batch (kind %u)\n",
                RefOffset(h.label_ref), h.label_bytes, h.label_kind);
  std::printf("  loss slot          ar+0x%08" PRIx64 "\n",
              RefOffset(h.loss_ref));
  std::printf("  optimizer          %s  lr %g  wd %g  clip %g\n",
              h.optimizer_kind == 1 ? "adamw" : "sgd", h.lr, h.weight_decay,
              h.clip_norm);
  std::printf("  lr schedule        %s  warmup %" PRIu64 "  min_factor %g\n",
              h.lr_schedule == 1 ? "cosine+warmup" : "constant",
              h.warmup_steps, h.min_lr_factor);
  std::printf("  default steps      %" PRIu64 "\n", h.default_steps);
  std::printf("  programs           train %" PRIu64 " | eval %" PRIu64
              " | merge %" PRIu64 " instrs\n",
              h.train_instr_count, h.eval_instr_count, h.merge_instr_count);
  std::printf("  emit table         %" PRIu64 " entr%s\n", h.emit_count,
              h.emit_count == 1 ? "y" : "ies");

  if (h.emit_table_offset + h.emit_count * sizeof(EmitEntry) <= plan.size()) {
    for (uint64_t i = 0; i < h.emit_count; ++i) {
      EmitEntry e;
      std::memcpy(&e, plan.data() + h.emit_table_offset + i * sizeof(e),
                  sizeof(e));
      std::printf("    [%2" PRIu64 "] smf+0x%08" PRIx64 "  %8" PRIu64
                  " B  <- delta ar+0x%08" PRIx64 "\n",
                  i, e.smf_data_offset, e.byte_size, e.arena_offset);
    }
  }

  if (want_instrs) {
    auto stream = [&](uint64_t off) {
      return reinterpret_cast<const UpdateInstruction*>(plan.data() + off);
    };
    if (h.train_instr_offset + h.train_instr_count * 64 <= plan.size())
      Disassemble("train", stream(h.train_instr_offset), h.train_instr_count);
    if (h.eval_instr_offset + h.eval_instr_count * 64 <= plan.size())
      Disassemble("eval", stream(h.eval_instr_offset), h.eval_instr_count);
    if (h.merge_instr_offset + h.merge_instr_count * 64 <= plan.size())
      Disassemble("merge", stream(h.merge_instr_offset), h.merge_instr_count);
  }
  return 0;
}
