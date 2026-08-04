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

#include "source/plan/update_types.h"
#include "source/identity/hash.h"

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

// Overflow-checked section bound over fully untrusted header fields: the
// unchecked form `offset + count * elem <= size` wraps modulo 2^64 for a
// crafted count and "passes", walking the disassembler off the buffer —
// in the one tool whose job is inspecting corrupt plans.
bool SectionOk(uint64_t offset, uint64_t count, uint64_t elem, size_t size) {
  if (elem != 0 && count > UINT64_MAX / elem) return false;
  const uint64_t bytes = count * elem;
  return offset <= size && bytes <= size - offset;
}

void PrintRef(uint64_t ref) {
  if (ref == kNullRef) {
    std::printf("  <null>          ");
    return;
  }
  std::printf("  %s+0x%08" PRIx64, IsRodataRef(ref) ? "ro" : "ar",
              RefOffset(ref));
}

float ImmBitsToF32(uint64_t bits) {
  float f = 0.0f;
  const uint32_t u = static_cast<uint32_t>(bits);
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

/// The in[] slot carrying f32 immediate bits rather than a tensor ref, per
/// the ISA in source/plan/instruction.h; -1 when every slot is a ref.
/// Decoding immediates as refs printed "ar+0x3f800000" for alpha = 1.0 — an
/// apparently valid ~1 GB arena reference a field debugger would chase.
int ImmInSlot(uint16_t opcode) {
  switch (static_cast<OpCode>(opcode)) {
    case OpCode::kScale:     return 2;
    case OpCode::kFill:      return 1;
    case OpCode::kClipNorm:  return 1;
    case OpCode::kGemmAccNN: return 3;
    case OpCode::kGemmNNQ8:  return 3;
    case OpCode::kGemmNTQ8:  return 3;
    default:                 return -1;
  }
}

void Disassemble(const char* title, const UpdateInstruction* instrs,
                 uint64_t count) {
  std::printf("\n%s (%" PRIu64 " instructions)\n", title, count);
  for (uint64_t i = 0; i < count; ++i) {
    const UpdateInstruction& ins = instrs[i];
    std::printf("  %4" PRIu64 "  %-18s", i, OpName(ins.opcode));
    const int imm = ImmInSlot(ins.opcode);
    for (int s = 0; s < 4; ++s) {
      if (s == imm)
        std::printf("  imm(%-11g)", ImmBitsToF32(ins.in[s]));
      else if (ins.in[s] != kNullRef)
        PrintRef(ins.in[s]);
    }
    // out[] words that hold tensor refs are printed as refs; everything
    // else stays raw dims/aux.
    switch (static_cast<OpCode>(ins.opcode)) {
      case OpCode::kLayerNormFwd:
        std::printf("   stats:");
        PrintRef(ins.out[1]);
        PrintRef(ins.out[2]);
        std::printf("   rows/cols: %" PRIu64 " %" PRIu64 "\n",
                    ins.out[0] >> 32, ins.out[0] & 0xFFFFFFFFu);
        break;
      case OpCode::kLayerNormBwd:
        std::printf("   stats:");
        PrintRef(ins.out[0]);
        PrintRef(ins.out[1]);
        std::printf("   rows/cols: %" PRIu64 " %" PRIu64 "\n",
                    ins.out[2] >> 32, ins.out[2] & 0xFFFFFFFFu);
        break;
      case OpCode::kKLDistillFwd:
        std::printf("   p_t:");
        PrintRef(ins.out[0]);
        std::printf("   n/c: %" PRIu64 " %" PRIu64 "  T %g\n",
                    ins.out[1] >> 32, ins.out[1] & 0xFFFFFFFFu,
                    ImmBitsToF32(ins.out[2]));
        break;
      case OpCode::kKLDistillBwd:
        std::printf("   n/c: %" PRIu64 " %" PRIu64 "  T %g\n",
                    ins.out[0] >> 32, ins.out[0] & 0xFFFFFFFFu,
                    ImmBitsToF32(ins.out[1]));
        break;
      default:
        std::printf("   dims/aux: %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
                    ins.out[0], ins.out[1], ins.out[2]);
        break;
    }
  }
}

/// Overflow-checked section bounds. Header fields are untrusted — this tool
/// exists to debug corrupt plans and deliberately keeps going after a hash
/// MISMATCH — and offset + count * elem can wrap in uint64 exactly for the
/// inputs the check exists to reject (the runtime's contract.cc uses the
/// same MulOk/RangeOk discipline).
bool SectionInBounds(uint64_t offset, uint64_t count, uint64_t elem_bytes,
                     uint64_t plan_size) {
  if (elem_bytes != 0 && count > UINT64_MAX / elem_bytes) return false;
  const uint64_t bytes = count * elem_bytes;
  return offset <= plan_size && bytes <= plan_size - offset;
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
  // Sized single read (plans embed the frozen weights, so they can be MBs).
  f.seekg(0, std::ios::end);
  const std::streamoff end = f.tellg();
  if (end < 0) {
    std::fprintf(stderr, "seeml-seeu-dump: cannot stat '%s'\n", argv[1]);
    return 1;
  }
  f.seekg(0);
  std::vector<uint8_t> plan(static_cast<size_t>(end));
  if (!plan.empty() &&
      !f.read(reinterpret_cast<char*>(plan.data()),
              static_cast<std::streamsize>(plan.size()))) {
    std::fprintf(stderr, "seeml-seeu-dump: cannot read '%s'\n", argv[1]);
    return 1;
  }
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
  // The header layout is version-specific: interpreting an older plan
  // through the current struct prints authoritative-looking garbage (and
  // feeds garbage counts to the section walks below).
  if (h.version != kSeeuVersion) {
    std::fprintf(stderr,
                 "seeml-seeu-dump: plan version %u, but this tool "
                 "understands only version %u — refusing to interpret the "
                 "header\n",
                 h.version, kSeeuVersion);
    return 1;
  }

  // Verify the integrity seal the same way the runtime does.
  const uint64_t state = PlanSelfHash(plan.data(), plan.size(),
                                      offsetof(PlanHeader, plan_hash));

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

  if (SectionInBounds(h.emit_table_offset, h.emit_count, sizeof(EmitEntry),
                      plan.size())) {
    for (uint64_t i = 0; i < h.emit_count; ++i) {
      EmitEntry e;
      std::memcpy(&e, plan.data() + h.emit_table_offset + i * sizeof(e),
                  sizeof(e));
      std::printf("    [%2" PRIu64 "] smf+0x%08" PRIx64 "  %8" PRIu64
                  " B  <- delta ar+0x%08" PRIx64 "\n",
                  i, e.smf_data_offset, e.byte_size, e.arena_offset);
    }
  } else {
    std::fprintf(stderr,
                 "seeml-seeu-dump: emit table exceeds the file — skipped\n");
  }

  if (want_instrs) {
    auto stream = [&](uint64_t off) {
      return reinterpret_cast<const UpdateInstruction*>(plan.data() + off);
    };
    if (SectionInBounds(h.train_instr_offset, h.train_instr_count,
                        sizeof(UpdateInstruction), plan.size()))
      Disassemble("train", stream(h.train_instr_offset), h.train_instr_count);
    if (SectionInBounds(h.eval_instr_offset, h.eval_instr_count,
                        sizeof(UpdateInstruction), plan.size()))
      Disassemble("eval", stream(h.eval_instr_offset), h.eval_instr_count);
    if (SectionInBounds(h.merge_instr_offset, h.merge_instr_count,
                        sizeof(UpdateInstruction), plan.size()))
      Disassemble("merge", stream(h.merge_instr_offset), h.merge_instr_count);
  }
  return 0;
}
