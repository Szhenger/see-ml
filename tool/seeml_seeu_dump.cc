// =============================================================================
// seeml-seeu-dump — .seeu Update Plan disassembler.
//
//   seeml-seeu-dump plan.seeu [--instrs | --json] [--version]
//
// Prints the plan header (memory contract, I/O slots, hyperparameters,
// integrity hashes, section table) and, with --instrs, disassembles the
// train / eval / merge instruction streams. This is the field-debugging
// tool: it depends only on update_types.h + hash.h + version.h so it
// builds anywhere.
//
// --json prints the same decode as one JSON object — every header field by
// its struct name, the four instruction streams (opcode, its name, flags,
// in[4], out[3] as integers) and the emit table — and exits 1 on a plan
// whose seal does not verify. It is the C++ decoder's view of a plan in a
// form another program can check itself against (P6, #86): the Python
// plane's plan reader is tested equal to it on every fixture, so the
// most-validated format in the tree has one decoder and one checked mirror,
// not two independent ones.
// =============================================================================

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "source/plan/opcode_names.h"
#include "source/plan/update_types.h"
#include "source/identity/hash.h"
#include "source/identity/version.h"

namespace {

using namespace seeml::update;

const char* OpName(uint16_t opcode) {
  const char* name = OpCodeName(opcode);  // source/plan/opcode_names.h
  return name ? name : "<unknown>";
}

void PrintRef(uint64_t ref) {
  if (ref == kNullRef) {
    std::printf("  <null>          ");
    return;
  }
  std::printf("  %s+0x%08" PRIx64,
              IsRodataRef(ref)   ? "ro"
              : IsSourceRef(ref) ? "src"  // v17: the source model file
                                 : "ar",
              RefOffset(ref));
}

float ImmBitsToF32(uint64_t bits) {
  float f = 0.0f;
  const uint32_t u = static_cast<uint32_t>(bits);
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

// The KL temperature word's high half is the v8 loss scale; zero (pre-v8)
// reads as 1.0, mirroring the engine.
float KlScaleOf(uint64_t word) {
  const uint64_t hi = word >> 32;
  return hi == 0 ? 1.0f : ImmBitsToF32(hi);
}

/// The in[] slot carrying f32 immediate bits rather than a tensor ref, per
/// the ISA in source/plan/instruction.h; -1 when every slot is a ref.
/// Decoding immediates as refs printed "ar+0x3f800000" for alpha = 1.0 — an
/// apparently valid ~1 GB arena reference a field debugger would chase.
int ImmInSlot(uint16_t opcode, uint16_t flags = 0) {
  // v17: a q8 GEMM with per-column scales carries a ref in in[3].
  if (flags & kFlagQ8ColScale) return -1;
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

/// Compact decode of the flag word, composed from its parts: the v5
/// epilogue ("bias", "relu", "bias+gelu", ...), the v14 GEMM "addend" and
/// the v17 per-column int8 scales ("cols"). Returned by value, so any
/// number of calls may share one expression.
std::string EpilogueName(uint16_t flags) {
  static const char* const kActs[] = {"", "relu", "gelu", "silu"};
  std::string name;
  auto add = [&name](const char* part) {
    if (!name.empty()) name += "+";
    name += part;
  };
  if (flags & kFlagEpilogueBias) add("bias");
  if (const uint16_t act = (flags & kFlagEpilogueActMask) >>
                           kFlagEpilogueActShift)
    add(kActs[act]);
  if (flags & kFlagGemmAddend) add("addend");
  if (flags & kFlagQ8ColScale) add("cols");
  if (flags & static_cast<uint16_t>(~kKnownFlagsMask)) add("unknown-flags");
  return name;
}

bool SectionInBounds(uint64_t off, uint64_t count, uint64_t elem,
                     uint64_t size);

uint32_t F32BitsOf(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

/// The whole plan as one JSON object (see the banner). u64 words print as
/// integers: refs and packed dim words are exact, and kNullRef is 2^64 - 1.
int DumpJson(const std::vector<uint8_t>& plan, const PlanHeader& h,
             bool sealed) {
  if (!sealed) {
    std::fprintf(stderr, "seeml-seeu-dump: plan_hash mismatch — corrupt\n");
    return 1;
  }
  std::printf("{\n  \"header\": {");
#define U(f) std::printf("%s\"" #f "\": %" PRIu64, first ? "" : ", ", \
                        static_cast<uint64_t>(h.f)), first = false
#define F(f) std::printf(", \"" #f "_bits\": %u", \
                        static_cast<unsigned>(F32BitsOf(h.f)))
  bool first = true;
  U(magic); U(version); U(arena_size); U(persistent_size); U(input_ref);
  U(input_floats); U(label_ref); U(label_bytes); U(label_kind);
  U(optimizer_kind); U(loss_ref); U(train_instr_offset); U(train_instr_count);
  U(merge_instr_offset); U(merge_instr_count); U(rodata_offset);
  U(rodata_size); U(persist_init_offset); U(persist_init_size);
  U(emit_table_offset); U(emit_count); U(gemm_tile_k); U(batch);
  U(default_steps); U(eval_instr_offset); U(eval_instr_count);
  U(source_model_hash); U(plan_hash); U(lr_schedule); U(gemm_tile_n);
  U(warmup_steps); U(input_kind); U(grad_accum_steps); U(seq_len);
  U(step_instr_offset); U(step_instr_count);
  // Floats travel as their f32 bit patterns: exact, and free of any
  // decimal round trip a reader would have to reproduce.
  F(lr); F(beta1); F(beta2); F(eps); F(weight_decay); F(min_lr_factor);
  F(clip_norm);
#undef U
#undef F
  std::printf("},\n  \"sections\": {");
  const struct {
    const char* name;
    uint64_t off, count;
  } sections[] = {{"train", h.train_instr_offset, h.train_instr_count},
                  {"step", h.step_instr_offset, h.step_instr_count},
                  {"eval", h.eval_instr_offset, h.eval_instr_count},
                  {"merge", h.merge_instr_offset, h.merge_instr_count}};
  for (size_t s = 0; s < 4; ++s) {
    if (!SectionInBounds(sections[s].off, sections[s].count,
                         sizeof(UpdateInstruction), plan.size())) {
      std::fprintf(stderr, "seeml-seeu-dump: %s section exceeds the file\n",
                   sections[s].name);
      return 1;
    }
    std::printf("%s\n    \"%s\": [", s ? "," : "", sections[s].name);
    for (uint64_t i = 0; i < sections[s].count; ++i) {
      UpdateInstruction ins;  // copied out: the offset may be unaligned
      std::memcpy(&ins, plan.data() + sections[s].off + i * sizeof(ins),
                  sizeof(ins));
      std::printf("%s\n      {\"opcode\": %u, \"name\": \"%s\", \"flags\": %u, "
                  "\"imm\": %u, "
                  "\"in\": [%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64
                  "], \"out\": [%" PRIu64 ", %" PRIu64 ", %" PRIu64 "]}",
                  i ? "," : "", ins.opcode, OpName(ins.opcode), ins.flags,
                  ins.imm,
                  ins.in[0], ins.in[1], ins.in[2], ins.in[3], ins.out[0],
                  ins.out[1], ins.out[2]);
    }
    std::printf("%s]", sections[s].count ? "\n    " : "");
  }
  std::printf("\n  },\n  \"emit\": [");
  if (!SectionInBounds(h.emit_table_offset, h.emit_count, sizeof(EmitEntry),
                       plan.size())) {
    std::fprintf(stderr, "seeml-seeu-dump: emit table exceeds the file\n");
    return 1;
  }
  for (uint64_t i = 0; i < h.emit_count; ++i) {
    EmitEntry e;
    std::memcpy(&e, plan.data() + h.emit_table_offset + i * sizeof(e),
                sizeof(e));
    std::printf("%s\n    {\"smf_data_offset\": %" PRIu64 ", \"byte_size\": %"
                PRIu64 ", \"arena_offset\": %" PRIu64 "}",
                i ? "," : "", e.smf_data_offset, e.byte_size, e.arena_offset);
  }
  std::printf("%s]\n}\n", h.emit_count ? "\n  " : "");
  return std::ferror(stdout) ? 1 : 0;
}

void Disassemble(const char* title, const UpdateInstruction* instrs,
                 uint64_t count) {
  std::printf("\n%s (%" PRIu64 " instructions)\n", title, count);
  for (uint64_t i = 0; i < count; ++i) {
    const UpdateInstruction& ins = instrs[i];
    std::printf("  %4" PRIu64 "  %-18s", i, OpName(ins.opcode));
    if (ins.flags)
      std::printf(" epi(%s)", EpilogueName(ins.flags).c_str());
    const int imm = ImmInSlot(ins.opcode, ins.flags);
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
        std::printf("   rows/cols: %" PRIu64 " %" PRIu64 "  eps %g\n",
                    ins.out[0] >> 32, ins.out[0] & 0xFFFFFFFFu,
                    static_cast<double>(NormEpsOf(ins.imm)));
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
        std::printf("   n/c: %" PRIu64 " %" PRIu64 "  T %g  scale %g\n",
                    ins.out[1] >> 32, ins.out[1] & 0xFFFFFFFFu,
                    ImmBitsToF32(ins.out[2] & 0xFFFFFFFFu),
                    KlScaleOf(ins.out[2]));
        break;
      case OpCode::kKLDistillBwd:
        std::printf("   n/c: %" PRIu64 " %" PRIu64 "  T %g  scale %g\n",
                    ins.out[0] >> 32, ins.out[0] & 0xFFFFFFFFu,
                    ImmBitsToF32(ins.out[1] & 0xFFFFFFFFu),
                    KlScaleOf(ins.out[1]));
        break;
      case OpCode::kRmsNormFwd:
        std::printf("   rows/cols: %" PRIu64 " %" PRIu64 "  eps %g\n",
                    ins.out[0] >> 32, ins.out[0] & 0xFFFFFFFFu,
                    static_cast<double>(NormEpsOf(ins.imm)));
        break;
      case OpCode::kRmsNormBwd:
        std::printf("   rstd:");
        PrintRef(ins.out[0]);
        std::printf("   rows/cols: %" PRIu64 " %" PRIu64 "\n",
                    ins.out[1] >> 32, ins.out[1] & 0xFFFFFFFFu);
        break;
      case OpCode::kRopeFwd:
      case OpCode::kRopeBwd:
        std::printf("   B/S: %" PRIu64 " %" PRIu64 "  H/d: %" PRIu64
                    " %" PRIu64 "  base %g\n",
                    ins.out[0] >> 32, ins.out[0] & 0xFFFFFFFFu,
                    ins.out[1] >> 32, ins.out[1] & 0xFFFFFFFFu,
                    ImmBitsToF32(ins.out[2]));
        break;
      case OpCode::kAttnFwd:
        std::printf("   probs:");
        PrintRef(ins.out[0]);
        std::printf("   B/S: %" PRIu64 " %" PRIu64 "  H/d: %" PRIu64
                    " %" PRIu64 "\n",
                    ins.out[1] >> 32, ins.out[1] & 0xFFFFFFFFu,
                    ins.out[2] >> 32, ins.out[2] & 0xFFFFFFFFu);
        break;
      case OpCode::kAttnDP:
      case OpCode::kAttnDV:
      case OpCode::kAttnDQ:
      case OpCode::kAttnDK:
        std::printf("   B/S: %" PRIu64 " %" PRIu64 "  H/d: %" PRIu64
                    " %" PRIu64 "\n",
                    ins.out[0] >> 32, ins.out[0] & 0xFFFFFFFFu,
                    ins.out[1] >> 32, ins.out[1] & 0xFFFFFFFFu);
        break;
      case OpCode::kAttnFwdTiled:
        std::printf("   B/S: %" PRIu64 " %" PRIu64 "  H/d: %" PRIu64
                    " %" PRIu64 "  stats: out[0]\n",
                    ins.out[1] >> 32, ins.out[1] & 0xFFFFFFFFu,
                    ins.out[2] >> 32, ins.out[2] & 0xFFFFFFFFu);
        break;
      case OpCode::kAttnDQTiled:
      case OpCode::kAttnDKTiled:
      case OpCode::kAttnDVTiled: {
        const auto g = UnpackAttnGeometry(ins.out[2]);
        std::printf("   B/S: %" PRIu64 " %" PRIu64 "  H/d: %" PRIu64
                    " %" PRIu64 "  stats: out[0]  result: out[1]\n",
                    g.B, g.S, g.H, g.d);
        break;
      }
      case OpCode::kSoftmaxRowsBwd:
        std::printf("   rows/cols: %" PRIu64 " %" PRIu64 "\n",
                    ins.out[0] >> 32, ins.out[0] & 0xFFFFFFFFu);
        break;
      case OpCode::kEmbedFwd:
        std::printf("   rows: %" PRIu64 "  V/D: %" PRIu64 " %" PRIu64 "\n",
                    ins.out[0], ins.out[1] >> 32, ins.out[1] & 0xFFFFFFFFu);
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
  if (argc >= 2 && (std::strcmp(argv[1], "--help") == 0 ||
                    std::strcmp(argv[1], "-h") == 0)) {
    std::printf("usage: seeml-seeu-dump plan.seeu [--instrs | --json] [--version]\n");
    return 0;
  }
  if (argc >= 2 && std::strcmp(argv[1], "--version") == 0) {
    std::printf("seeml-seeu-dump %s\n", kSeemlVersion);
    return 0;
  }
  if (argc < 2) {
    std::fprintf(stderr, "usage: seeml-seeu-dump plan.seeu [--instrs | --json]\n");
    return 2;
  }
  bool want_instrs = false;
  bool want_json = false;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--instrs") == 0) {
      want_instrs = true;
    } else if (std::strcmp(argv[i], "--json") == 0) {
      want_json = true;
    } else {
      // A typo'd flag must not silently print header-only output with
      // exit 0 — the sibling compile CLI treats every unconsumed argv slot
      // as a hard error for the same reason.
      std::fprintf(stderr, "seeml-seeu-dump: unknown argument '%s'\n",
                   argv[i]);
      return 2;
    }
  }

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
  // Accept exactly the versions the runtime loads (schema.h): every plan in
  // the readable range shares this header layout and the PlanSelfHash seal,
  // and a field tool that refuses artifacts the fleet runs is useless for
  // debugging them. Below the floor the layout/seal differ — interpreting
  // such a plan through the current struct prints authoritative-looking
  // garbage (and feeds garbage counts to the section walks below).
  if (h.version < kSeeuOldestReadable || h.version > kSeeuVersion) {
    std::fprintf(stderr,
                 "seeml-seeu-dump: plan version %u, but this tool "
                 "understands versions %u..%u — refusing to interpret the "
                 "header\n",
                 h.version, kSeeuOldestReadable, kSeeuVersion);
    return 1;
  }

  // Verify the integrity seal the same way the runtime does.
  const uint64_t state = PlanSelfHash(plan.data(), plan.size(),
                                      offsetof(PlanHeader, plan_hash));
  const bool sealed = state == h.plan_hash;
  if (want_json) return DumpJson(plan, h, sealed);

  std::printf("seeu plan: %s\n", argv[1]);
  std::printf("  version            %u\n", h.version);
  std::printf("  plan_hash          %016" PRIx64 "  (%s)\n", h.plan_hash,
              sealed ? "verified" : "MISMATCH — corrupt");
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
  if (h.gemm_tile_k || h.gemm_tile_n)
    std::printf("  gemm tiles         K %u  N %u  (v11; 0 = runtime default)\n",
                h.gemm_tile_k, h.gemm_tile_n);
  if (h.grad_accum_steps > 1)
    std::printf("  grad accumulation  %u micro-batches per optimizer step "
                "(effective batch %" PRIu64 ")\n",
                h.grad_accum_steps, h.batch * h.grad_accum_steps);
  std::printf("  programs           train %" PRIu64 " | eval %" PRIu64
              " | merge %" PRIu64 " | step %" PRIu64 " instrs\n",
              h.train_instr_count, h.eval_instr_count, h.merge_instr_count,
              h.step_instr_count);
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
    // Copy each stream out before decoding: a corrupt header may place an
    // offset at any alignment, and a reinterpret_cast there is UB (and a
    // real SIGBUS on strict-alignment targets). The dumper's job is to
    // describe bad plans, not to crash on them.
    std::vector<UpdateInstruction> copy;
    auto stream = [&](uint64_t off, uint64_t count) {
      copy.resize(static_cast<size_t>(count));
      if (count)
        std::memcpy(copy.data(), plan.data() + off,
                    static_cast<size_t>(count) * sizeof(UpdateInstruction));
      return copy.data();
    };
    if (SectionInBounds(h.train_instr_offset, h.train_instr_count,
                        sizeof(UpdateInstruction), plan.size()))
      Disassemble("train", stream(h.train_instr_offset, h.train_instr_count),
                  h.train_instr_count);
    if (SectionInBounds(h.eval_instr_offset, h.eval_instr_count,
                        sizeof(UpdateInstruction), plan.size()))
      Disassemble("eval", stream(h.eval_instr_offset, h.eval_instr_count),
                  h.eval_instr_count);
    if (SectionInBounds(h.merge_instr_offset, h.merge_instr_count,
                        sizeof(UpdateInstruction), plan.size()))
      Disassemble("merge", stream(h.merge_instr_offset, h.merge_instr_count),
                  h.merge_instr_count);
    if (h.step_instr_count &&
        SectionInBounds(h.step_instr_offset, h.step_instr_count,
                        sizeof(UpdateInstruction), plan.size()))
      Disassemble("step", stream(h.step_instr_offset, h.step_instr_count),
                  h.step_instr_count);
  }
  // The dump above is still useful forensics for a corrupt plan, but a
  // script must not need to parse text to learn the seal failed.
  return sealed ? 0 : 1;
}
