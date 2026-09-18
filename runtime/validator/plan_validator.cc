#include "runtime/validator/plan_validator.h"

#include <bit>
#include <cmath>

#include "runtime/diagnostics/validating/error.h"

namespace seeml::update_rt {

namespace up = seeml::update;

namespace {

/// Proves a kFusedMap micro-program (plan v13): 1..4 known stages,
/// zero-terminated with nothing after the terminator; binary stages name
/// operand slot 1 or 2 (recorded in `uses_slot`), scale stages immediate 0
/// or 1, unary stages carry no argument; every immediate a scale names is
/// finite and every one none names is zero.
bool FusedProgramOk(uint64_t stages_word, uint64_t imm_word,
                    bool uses_slot[3]) {
  if ((stages_word >> 32) != 0) return false;
  constexpr uint8_t kKnownBits = static_cast<uint8_t>(
      up::kFusedStageKindMask |
      (up::kFusedStageArgMask << up::kFusedStageArgShift) |
      up::kFusedStageRunIsRight);
  bool uses_imm[2] = {false, false};
  size_t stages = 0;
  for (; stages < up::kFusedMapMaxStages; ++stages) {
    const auto stage = static_cast<uint8_t>(stages_word >> (8 * stages));
    const up::FusedStage kind = up::FusedStageKind(stage);
    if (kind == up::FusedStage::kEnd) {
      if (stage != 0) return false;
      break;
    }
    if ((stage & ~kKnownBits) != 0) return false;
    const uint8_t arg = up::FusedStageArg(stage);
    if (kind == up::FusedStage::kAdd || kind == up::FusedStage::kMul) {
      if (arg != 1 && arg != 2) return false;
      uses_slot[arg] = true;
    } else if (kind == up::FusedStage::kScale) {
      if (arg > 1 || (stage & up::kFusedStageRunIsRight)) return false;
      uses_imm[arg] = true;
    } else if (kind == up::FusedStage::kRelu ||
               kind == up::FusedStage::kGelu ||
               kind == up::FusedStage::kSilu) {
      if ((stage & ~up::kFusedStageKindMask) != 0) return false;
    } else {
      return false;
    }
  }
  if (stages == 0 || (stages_word >> (8 * stages)) != 0) return false;
  for (int i = 0; i < 2; ++i) {
    const uint32_t bits = static_cast<uint32_t>(imm_word >> (32 * i));
    if (uses_imm[i] ? !std::isfinite(std::bit_cast<float>(bits)) : bits != 0)
      return false;
  }
  return true;
}

std::expected<void, std::string> ValidateInstructionImpl(
    const up::UpdateInstruction& ins, uint64_t arena_size,
    uint64_t rodata_size, uint32_t plan_version, bool allow_source,
    InstructionExtents* extents) {
  // Flags discipline before any operand math. Pre-v5 plans predate the
  // flags vocabulary: a nonzero word there is corruption, not a feature.
  // From v5 on, every set bit must be a defined epilogue bit AND defined
  // for this opcode — Execute() applies flags blindly, so an unknown or
  // misplaced bit must die here, loudly, not skip silently.
  if (plan_version < up::kSeeuFlagsVersion) {
    if (ins.flags != 0)
      return diag::validating::Error(
          "instruction carries flags in a pre-v" +
          std::to_string(up::kSeeuFlagsVersion) + " plan (opcode " +
          std::to_string(ins.opcode) + ")");
  } else {
    const auto opcode = static_cast<up::OpCode>(ins.opcode);
    const bool addend_ok =
        plan_version >= up::kSeeuGemmAddendVersion &&
        (opcode == up::OpCode::kGemmNN || opcode == up::OpCode::kGemmNT ||
         opcode == up::OpCode::kGemmTN);
    const bool colscale_ok =
        plan_version >= up::kSeeuShippedEvalVersion &&
        (opcode == up::OpCode::kGemmNNQ8 || opcode == up::OpCode::kGemmNTQ8);
    const uint16_t allowed = static_cast<uint16_t>(
        ((opcode == up::OpCode::kGemmNN || opcode == up::OpCode::kGemmNNBF16)
             ? up::kEpilogueFlagsMask
         : opcode == up::OpCode::kGemmNNQ8 ? up::kFlagEpilogueActMask
                                           : uint16_t{0}) |
        (addend_ok ? up::kFlagGemmAddend : uint16_t{0}) |
        (colscale_ok ? up::kFlagQ8ColScale : uint16_t{0}));
    if (ins.flags & static_cast<uint16_t>(~allowed))
      return diag::validating::Error(
          "unknown or misplaced instruction flags " +
          std::to_string(ins.flags) + " (opcode " +
          std::to_string(ins.opcode) + ")");
    // One ref slot, one tenant: the addend and the epilogue both want in[3].
    if ((ins.flags & up::kFlagGemmAddend) &&
        (ins.flags & up::kEpilogueFlagsMask))
      return diag::validating::Error(
          "GEMM addend combined with a bias / activation epilogue (opcode " +
          std::to_string(ins.opcode) + ")");
  }
  // The imm word (v16): defined on the normalization forwards only, as a
  // finite positive epsilon's f32 bits (0 = 1e-5). Before v16 it was a pad
  // word every compiler wrote as zero; a nonzero value anywhere it is not
  // defined would be ignored by Execute(), so it dies here.
  if (ins.imm != 0) {
    const auto opcode = static_cast<up::OpCode>(ins.opcode);
    const bool norm_fwd = opcode == up::OpCode::kLayerNormFwd ||
                          opcode == up::OpCode::kRmsNormFwd;
    if (!norm_fwd || plan_version < up::kSeeuNormEpsVersion)
      return diag::validating::Error(
          "instruction carries an imm word its opcode does not define "
          "(opcode " + std::to_string(ins.opcode) + ", plan v" +
          std::to_string(plan_version) + ")");
    const float eps = std::bit_cast<float>(ins.imm);
    if (!std::isfinite(eps) || !(eps > 0.0f))
      return diag::validating::Error(
          "normalization epsilon must be a finite positive float");
  }
  // Every kernel is compiled with SEEML_RESTRICT pointers: a written range
  // overlapping any *other* operand of the same instruction is undefined
  // behavior, not a wrong answer. Bounds alone don't rule that out, so each
  // validated operand's byte range is recorded and proven disjoint from
  // every written range before the instruction is accepted. A single ref
  // that is read and written through one pointer (SGD's param, GemmAcc's C)
  // is one operand, not an alias.
  using OperandRange = OperandExtent;
  OperandRange* ranges = extents->ranges;
  size_t& num_ranges = extents->count;
  num_ranges = 0;

  // elem_bytes: f32/i32 operands are 4 bytes; quantized weights are 1.
  // A ref is admitted only if its extent is nonzero (every kernel
  // dereferences element 0 unconditionally, so a zero-extent operand is not
  // "harmlessly empty" — it is an unchecked pointer), aligned to its
  // element size (the kernels cast the raw offset to a typed pointer), and
  // in bounds for its address space.
  auto ref_ok_w = [&](uint64_t ref, uint64_t elems, bool write,
                      uint64_t elem_bytes) {
    if (ref == up::kNullRef) return false;
    if (write && up::IsRodataRef(ref)) return false;
    // A source ref (v17): read-only f32, in the eval program only.
    const bool source = up::IsSourceRef(ref);
    if (source && (write || !allow_source ||
                   plan_version < up::kSeeuShippedEvalVersion))
      return false;
    if (elems == 0) return false;
    uint64_t bytes = 0;
    if (!MulOk(elems, elem_bytes, &bytes)) return false;
    // Execute() reinterpret_casts the ref to its element type and
    // dereferences directly, so alignment is part of "safe to dispatch
    // blindly": a misaligned offset is UB, and a bus error on the
    // strict-alignment targets this runtime ships to.
    if (up::RefOffset(ref) % elem_bytes != 0) return false;
    const uint64_t space = source ? kSourceSpaceLimit
                           : up::IsRodataRef(ref) ? rodata_size
                                                  : arena_size;
    if (!RangeOk(up::RefOffset(ref), bytes, space)) return false;
    ranges[num_ranges++] = {up::RefOffset(ref), bytes, write,
                            up::IsRodataRef(ref), source};
    return true;
  };
  auto ref_ok = [&](uint64_t ref, uint64_t elems, bool write) {
    return ref_ok_w(ref, elems, write, sizeof(float));
  };
  auto fail = [&] {
    return diag::validating::Error("instruction operand out of bounds "
                           "(opcode " +
                           std::to_string(ins.opcode) + ")");
  };
  // Accept only if no written range overlaps another operand's range in the
  // same address space. Called in place of a bare success return by every
  // case that records operands.
  auto disjoint = [&]() -> std::expected<void, std::string> {
    for (size_t i = 0; i < num_ranges; ++i)
      for (size_t j = i + 1; j < num_ranges; ++j) {
        const OperandRange& a = ranges[i];
        const OperandRange& b = ranges[j];
        if (!(a.write || b.write) || a.rodata != b.rodata ||
            a.source != b.source)
          continue;
        if (a.bytes == 0 || b.bytes == 0) continue;
        if (a.off < b.off + b.bytes && b.off < a.off + a.bytes)
          return diag::validating::Error(
              "instruction operands alias a written range (opcode " +
              std::to_string(ins.opcode) + ")");
      }
    return {};
  };

  const uint64_t d0 = ins.out[0], d1 = ins.out[1], d2 = ins.out[2];
  // The KL temperature word's high half is the loss scale (v8). Pre-v8
  // plans never wrote it: a nonzero high word there is corruption, exactly
  // like flags before v5. From v8 on, zero means "absent" (scale 1.0) and
  // any written scale must be a finite positive float — Execute() multiplies
  // by it blindly, and a zero or NaN scale would train on nothing, silently.
  auto kl_scale_ok = [&](uint64_t word) -> std::expected<void, std::string> {
    const uint64_t hi = word >> 32;
    if (plan_version < up::kSeeuKlScaleVersion) {
      if (hi != 0)
        return diag::validating::Error(
            "kl_distill carries a loss scale in a pre-v" +
            std::to_string(up::kSeeuKlScaleVersion) + " plan");
      return {};
    }
    if (hi == 0) return {};
    const float scale = std::bit_cast<float>(static_cast<uint32_t>(hi));
    if (!std::isfinite(scale) || scale <= 0.0f)
      return diag::validating::Error(
          "kl_distill loss scale must be a finite positive float");
    return {};
  };
  uint64_t mk = 0, kn = 0, mn = 0, nc = 0;
  // Transformer opcodes exist only from v6: no earlier compiler emits them,
  // so their appearance in an older plan is corruption, not a feature.
  if (ins.opcode >= static_cast<uint16_t>(up::OpCode::kRmsNormFwd) &&
      ins.opcode <= static_cast<uint16_t>(up::OpCode::kAttnDK) &&
      plan_version < up::kSeeuTransformerVersion)
    return diag::validating::Error(
        "transformer opcode " + std::to_string(ins.opcode) + " in a pre-v" +
        std::to_string(up::kSeeuTransformerVersion) + " plan");
  if ((ins.opcode == static_cast<uint16_t>(up::OpCode::kGemmNNBF16) ||
       ins.opcode == static_cast<uint16_t>(up::OpCode::kGemmNTBF16)) &&
      plan_version < up::kSeeuBf16Version)
    return diag::validating::Error(
        "bf16 GEMM opcode in a pre-v" +
        std::to_string(up::kSeeuBf16Version) + " plan");
  if (ins.opcode >= static_cast<uint16_t>(up::OpCode::kAttnFwdTiled) &&
      ins.opcode <= static_cast<uint16_t>(up::OpCode::kAttnDVTiled) &&
      plan_version < up::kSeeuTiledAttentionVersion)
    return diag::validating::Error(
        "tiled attention opcode " + std::to_string(ins.opcode) +
        " in a pre-v" + std::to_string(up::kSeeuTiledAttentionVersion) +
        " plan");
  if (ins.opcode == static_cast<uint16_t>(up::OpCode::kAccumulate) &&
      plan_version < up::kSeeuGradAccumVersion)
    return diag::validating::Error(
        "accumulate opcode in a pre-v" +
        std::to_string(up::kSeeuGradAccumVersion) + " plan");
  if (ins.opcode == static_cast<uint16_t>(up::OpCode::kFusedMap) &&
      plan_version < up::kSeeuFusedMapVersion)
    return diag::validating::Error(
        "fused_map opcode in a pre-v" +
        std::to_string(up::kSeeuFusedMapVersion) + " plan");
  if (ins.opcode == static_cast<uint16_t>(up::OpCode::kRopeTable) &&
      plan_version < up::kSeeuKernelBatchVersion)
    return diag::validating::Error(
        "rope_table opcode in a pre-v" +
        std::to_string(up::kSeeuKernelBatchVersion) + " plan");
  // The fused clip threshold on the optimizer steps (v12, out[1]): absent
  // below v12 — a nonzero word there is corruption — and from v12 on either
  // zero or a finite positive f32, since Execute() clips by it blindly.
  auto clip_word_ok = [&](uint64_t word) -> std::expected<void, std::string> {
    if (word == 0) return {};
    if (plan_version < up::kSeeuKernelBatchVersion)
      return diag::validating::Error(
          "optimizer step carries a clip threshold in a pre-v" +
          std::to_string(up::kSeeuKernelBatchVersion) + " plan");
    const float clip = std::bit_cast<float>(static_cast<uint32_t>(word));
    if ((word >> 32) != 0 || !std::isfinite(clip) || clip <= 0.0f)
      return diag::validating::Error(
          "optimizer step clip threshold must be a finite positive float");
    return {};
  };
  if (ins.opcode == static_cast<uint16_t>(up::OpCode::kEmbedFwd) &&
      plan_version < up::kSeeuTokenVersion)
    return diag::validating::Error(
        "token opcode " + std::to_string(ins.opcode) + " in a pre-v" +
        std::to_string(up::kSeeuTokenVersion) + " plan");
  // Shared geometry for the transformer family: activations are
  // [B*S, H*d] f32, the probability matrix [B*H*S, S]. Derived with the
  // same overflow-safe chain the kernels' loop bounds imply.
  auto attn_geometry = [&](uint64_t bs_word, uint64_t hd_word, uint64_t* td,
                           uint64_t* pn) {
    const uint64_t B = bs_word >> 32, S = bs_word & 0xFFFFFFFFu;
    const uint64_t H = hd_word >> 32, d = hd_word & 0xFFFFFFFFu;
    uint64_t t = 0, dm = 0, bhs = 0;
    if (B == 0 || S == 0 || H == 0 || d == 0) return false;
    if (!MulOk(B, S, &t) || !MulOk(H, d, &dm) || !MulOk(t, dm, td)) return false;
    if (!MulOk(t, H, &bhs) || !MulOk(bhs, S, pn)) return false;
    return true;
  };
  switch (static_cast<up::OpCode>(ins.opcode)) {
    case up::OpCode::kNop:
      return disjoint();
    case up::OpCode::kGemmNN:
    case up::OpCode::kGemmNT:
    case up::OpCode::kGemmTN:
    case up::OpCode::kGemmAccNN:
    case up::OpCode::kGemmNNQ8:
    case up::OpCode::kGemmNTQ8:
    case up::OpCode::kGemmNNBF16:
    case up::OpCode::kGemmNTBF16: {
      // Every layout variant reads M*K (A) and K*N (B), writes M*N (C).
      const auto oc = static_cast<up::OpCode>(ins.opcode);
      const bool q8 =
          oc == up::OpCode::kGemmNNQ8 || oc == up::OpCode::kGemmNTQ8;
      const bool bf16 =
          oc == up::OpCode::kGemmNNBF16 || oc == up::OpCode::kGemmNTBF16;
      if (!MulOk(d0, d2, &mk) || !MulOk(d2, d1, &kn) || !MulOk(d0, d1, &mn))
        return fail();
      // Narrow-storage B must live in rodata: only the compiler's own
      // packing produces it — 1 byte per element for int8, 2 for bf16.
      if ((q8 || bf16) && !up::IsRodataRef(ins.in[1])) return fail();
      const uint64_t b_elem = q8 ? 1 : bf16 ? 2 : sizeof(float);
      if (!ref_ok(ins.in[0], mk, false) ||
          !ref_ok_w(ins.in[1], kn, false, b_elem) ||
          !ref_ok(ins.in[2], mn, true))
        return fail();
      // Fused-bias epilogue (flags proved valid for this opcode above):
      // in[3] is a read of N floats and joins the overlap discipline
      // against the written C range.
      if ((ins.flags & up::kFlagEpilogueBias) && !ref_ok(ins.in[3], d1, false))
        return fail();
      // Per-column int8 scales (v17): a rodata vector, one float per
      // output column of W — N in the forward (NN), the reduction extent K
      // in the dX GEMM (NT), where W is read as [N, K].
      if (ins.flags & up::kFlagQ8ColScale) {
        const bool nt = oc == up::OpCode::kGemmNTQ8;
        if (!up::IsRodataRef(ins.in[3]) ||
            !ref_ok(ins.in[3], nt ? d2 : d1, false))
          return fail();
      }
      // The addend (v14, f32 GEMMs only — proved above): in[3] is a read of
      // M*N floats, disjoint from the written C like every other operand.
      if ((ins.flags & up::kFlagGemmAddend) && !ref_ok(ins.in[3], mn, false))
        return fail();
      return disjoint();
    }
    case up::OpCode::kAddEW:
    case up::OpCode::kMulEW:
    case up::OpCode::kReluBwd:
    case up::OpCode::kGeluBwd:
    case up::OpCode::kSiluBwd:
      if (!ref_ok(ins.in[0], d0, false) || !ref_ok(ins.in[1], d0, false) ||
          !ref_ok(ins.in[2], d0, true))
        return fail();
      return disjoint();
    case up::OpCode::kAddBias:
      if (!MulOk(d0, d1, &mn)) return fail();
      if (!ref_ok(ins.in[0], mn, false) || !ref_ok(ins.in[1], d1, false) ||
          !ref_ok(ins.in[2], mn, true))
        return fail();
      return disjoint();
    case up::OpCode::kReluFwd:
    case up::OpCode::kGeluFwd:
    case up::OpCode::kSiluFwd:
    case up::OpCode::kScale:
    case up::OpCode::kCopy:
      if (!ref_ok(ins.in[0], d0, false) || !ref_ok(ins.in[1], d0, true))
        return fail();
      return disjoint();
    case up::OpCode::kLayerNormFwd: {
      const uint64_t rows = d0 >> 32, cols = d0 & 0xFFFFFFFFu;
      if (!MulOk(rows, cols, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], cols, false) ||
          !ref_ok(ins.in[2], cols, false) || !ref_ok(ins.in[3], nc, true) ||
          !ref_ok(ins.out[1], rows, true) || !ref_ok(ins.out[2], rows, true))
        return fail();
      return disjoint();
    }
    case up::OpCode::kLayerNormBwd: {
      const uint64_t rows = d2 >> 32, cols = d2 & 0xFFFFFFFFu;
      if (!MulOk(rows, cols, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], nc, false) ||
          !ref_ok(ins.in[2], cols, false) || !ref_ok(ins.in[3], nc, true) ||
          !ref_ok(ins.out[0], rows, false) || !ref_ok(ins.out[1], rows, false))
        return fail();
      return disjoint();
    }
    case up::OpCode::kClipNorm:
      if (!ref_ok(ins.in[0], d0, true)) return fail();
      return disjoint();
    case up::OpCode::kReduceRows:
      if (!MulOk(d0, d1, &mn)) return fail();
      if (!ref_ok(ins.in[0], mn, false) || !ref_ok(ins.in[1], d1, true))
        return fail();
      return disjoint();
    case up::OpCode::kSoftmaxXEntFwd:
      if (!MulOk(d0, d1, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], d0, false) ||
          !ref_ok(ins.in[2], 1, true) || !ref_ok(ins.in[3], nc, true))
        return fail();
      return disjoint();
    case up::OpCode::kSoftmaxXEntBwd:
      if (!MulOk(d0, d1, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], d0, false) ||
          !ref_ok(ins.in[2], 1, false) || !ref_ok(ins.in[3], nc, true))
        return fail();
      return disjoint();
    case up::OpCode::kMseFwd:
      if (!ref_ok(ins.in[0], d0, false) || !ref_ok(ins.in[1], d0, false) ||
          !ref_ok(ins.in[2], 1, true))
        return fail();
      return disjoint();
    case up::OpCode::kMseBwd:
      if (!ref_ok(ins.in[0], d0, false) || !ref_ok(ins.in[1], d0, false) ||
          !ref_ok(ins.in[2], 1, false) || !ref_ok(ins.in[3], d0, true))
        return fail();
      return disjoint();
    case up::OpCode::kKLDistillFwd:
      if (!MulOk(d1 >> 32, d1 & 0xFFFFFFFFu, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], nc, false) ||
          !ref_ok(ins.in[2], 1, true) || !ref_ok(ins.in[3], nc, true) ||
          !ref_ok(ins.out[0], nc, true))
        return fail();
      if (auto r = kl_scale_ok(d2); !r) return r;
      return disjoint();
    case up::OpCode::kKLDistillBwd:
      if (!MulOk(d0 >> 32, d0 & 0xFFFFFFFFu, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], nc, false) ||
          !ref_ok(ins.in[2], 1, false) || !ref_ok(ins.in[3], nc, true))
        return fail();
      if (auto r = kl_scale_ok(d1); !r) return r;
      return disjoint();
    case up::OpCode::kSgdStep:
      if (auto r = clip_word_ok(d1); !r) return r;
      if (!ref_ok(ins.in[0], d0, true) || !ref_ok(ins.in[1], d0, false))
        return fail();
      return disjoint();
    case up::OpCode::kAccumulate:  // dst (read+write, one operand), src
      if (!ref_ok(ins.in[0], d0, true) || !ref_ok(ins.in[1], d0, false))
        return fail();
      return disjoint();
    case up::OpCode::kAdamWStep:
      if (auto r = clip_word_ok(d1); !r) return r;
      if (!ref_ok(ins.in[0], d0, true) || !ref_ok(ins.in[1], d0, false) ||
          !ref_ok(ins.in[2], d0, true) || !ref_ok(ins.in[3], d0, true))
        return fail();
      return disjoint();
    case up::OpCode::kFill:
      if (!ref_ok(ins.in[0], d0, true)) return fail();
      return disjoint();
    case up::OpCode::kRmsNormFwd: {
      const uint64_t rows = d0 >> 32, cols = d0 & 0xFFFFFFFFu;
      if (!MulOk(rows, cols, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], cols, false) ||
          !ref_ok(ins.in[2], nc, true) || !ref_ok(ins.in[3], rows, true))
        return fail();
      return disjoint();
    }
    case up::OpCode::kRmsNormBwd: {
      const uint64_t rows = d1 >> 32, cols = d1 & 0xFFFFFFFFu;
      if (!MulOk(rows, cols, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], nc, false) ||
          !ref_ok(ins.in[2], cols, false) || !ref_ok(ins.in[3], nc, true) ||
          !ref_ok(ins.out[0], rows, false))
        return fail();
      return disjoint();
    }
    case up::OpCode::kRopeFwd:
    case up::OpCode::kRopeBwd: {
      uint64_t td = 0, pn = 0;
      if (!attn_geometry(d0, d1, &td, &pn)) return fail();
      // The rotation pairs (2c, 2c+1) require an even head width.
      if ((d1 & 0xFFFFFFFFu) % 2 != 0) return fail();
      // The rotary base rides out[2] as f32 bits (SMF v5 attr1, or the
      // lowering default). A non-finite or <= 1 base would make every
      // angle degenerate or NaN — reject it here, not as a NaN loss later.
      const float base =
          std::bit_cast<float>(static_cast<uint32_t>(ins.out[2]));
      if ((ins.out[2] >> 32) != 0 || !std::isfinite(base) || base <= 1.0f)
        return fail();
      if (!ref_ok(ins.in[0], td, false) || !ref_ok(ins.in[1], td, true))
        return fail();
      // The optional angle table (v12): [S, d/2, 2] = S*d floats, the
      // extent kRopeTable writes for this geometry. Below v12 the slot is
      // kNullRef by construction; anything else there is corruption.
      if (ins.in[2] != up::kNullRef) {
        if (plan_version < up::kSeeuKernelBatchVersion)
          return diag::validating::Error(
              "rope carries an angle table in a pre-v" +
              std::to_string(up::kSeeuKernelBatchVersion) + " plan");
        uint64_t table = 0;
        if (!MulOk(d0 & 0xFFFFFFFFu, d1 & 0xFFFFFFFFu, &table) ||
            !ref_ok(ins.in[2], table, false))
          return fail();
      }
      return disjoint();
    }
    case up::OpCode::kFusedMap: {
      // The micro-program is proven like everything else (FusedProgramOk);
      // then every operand slot a stage names must be present and count
      // floats long, and every slot none names absent.
      bool uses_slot[3] = {false, false, false};
      if (!FusedProgramOk(d1, d2, uses_slot)) return fail();
      if (!ref_ok(ins.in[0], d0, false)) return fail();
      for (int slot = 1; slot <= 2; ++slot) {
        if (uses_slot[slot] ? !ref_ok(ins.in[slot], d0, false)
                            : ins.in[slot] != up::kNullRef)
          return fail();
      }
      if (!ref_ok(ins.in[3], d0, true)) return fail();
      return disjoint();
    }
    case up::OpCode::kRopeTable: {
      // out[0] = S<<32|d, out[1] = base bits; the table is S*d floats.
      const uint64_t S = d0 >> 32, d = d0 & 0xFFFFFFFFu;
      const float base = std::bit_cast<float>(static_cast<uint32_t>(d1));
      uint64_t table = 0;
      if (S == 0 || d == 0 || d % 2 != 0 || (d1 >> 32) != 0 ||
          !std::isfinite(base) || base <= 1.0f || !MulOk(S, d, &table))
        return fail();
      if (!ref_ok(ins.in[0], table, true)) return fail();
      return disjoint();
    }
    case up::OpCode::kAttnFwd: {
      uint64_t td = 0, pn = 0;
      if (!attn_geometry(d1, d2, &td, &pn)) return fail();
      if (!ref_ok(ins.in[0], td, false) || !ref_ok(ins.in[1], td, false) ||
          !ref_ok(ins.in[2], td, false) || !ref_ok(ins.in[3], td, true) ||
          !ref_ok(ins.out[0], pn, true))
        return fail();
      return disjoint();
    }
    case up::OpCode::kAttnDP: {
      uint64_t td = 0, pn = 0;
      if (!attn_geometry(d0, d1, &td, &pn)) return fail();
      if (!ref_ok(ins.in[0], td, false) || !ref_ok(ins.in[1], td, false) ||
          !ref_ok(ins.in[2], pn, true))
        return fail();
      return disjoint();
    }
    case up::OpCode::kAttnDV: {
      uint64_t td = 0, pn = 0;
      if (!attn_geometry(d0, d1, &td, &pn)) return fail();
      if (!ref_ok(ins.in[0], pn, false) || !ref_ok(ins.in[1], td, false) ||
          !ref_ok(ins.in[2], td, true))
        return fail();
      return disjoint();
    }
    case up::OpCode::kSoftmaxRowsBwd: {
      const uint64_t rows = d0 >> 32, cols = d0 & 0xFFFFFFFFu;
      if (!MulOk(rows, cols, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], nc, false) ||
          !ref_ok(ins.in[2], nc, true))
        return fail();
      return disjoint();
    }
    case up::OpCode::kAttnDQ:
    case up::OpCode::kAttnDK: {
      uint64_t td = 0, pn = 0;
      if (!attn_geometry(d0, d1, &td, &pn)) return fail();
      if (!ref_ok(ins.in[0], pn, false) || !ref_ok(ins.in[1], td, false) ||
          !ref_ok(ins.in[2], td, true))
        return fail();
      return disjoint();
    }
    case up::OpCode::kAttnFwdTiled: {
      // q, k, v read and o written at [B*S, H*d]; stats written at
      // [B*H*S, kAttnStatsWidth] — the family's geometry words.
      uint64_t td = 0, pn = 0, st = 0;
      if (!attn_geometry(d1, d2, &td, &pn)) return fail();
      const uint64_t bhs = (d1 >> 32) * (d1 & 0xFFFFFFFFu) * (d2 >> 32);
      if (!MulOk(bhs, up::kAttnStatsWidth, &st)) return fail();
      if (!ref_ok(ins.in[0], td, false) || !ref_ok(ins.in[1], td, false) ||
          !ref_ok(ins.in[2], td, false) || !ref_ok(ins.in[3], td, true) ||
          !ref_ok(ins.out[0], st, true))
        return fail();
      return disjoint();
    }
    case up::OpCode::kAttnDQTiled:
    case up::OpCode::kAttnDKTiled:
    case up::OpCode::kAttnDVTiled: {
      // q, k, v, dO read; the result written; stats read — and, for dQ,
      // written too (the delta column), through the one pointer it is.
      // Geometry is the packed 16-bit word: every field must be nonzero
      // and the products must be sound.
      const auto g = up::UnpackAttnGeometry(d2);
      const uint64_t bs = (g.B << 32) | g.S, hd = (g.H << 32) | g.d;
      uint64_t td = 0, pn = 0, st = 0;
      if (!attn_geometry(bs, hd, &td, &pn)) return fail();
      if (!MulOk(g.B * g.S * g.H, up::kAttnStatsWidth, &st)) return fail();
      const bool dq = ins.opcode == static_cast<uint16_t>(up::OpCode::kAttnDQTiled);
      if (!ref_ok(ins.in[0], td, false) || !ref_ok(ins.in[1], td, false) ||
          !ref_ok(ins.in[2], td, false) || !ref_ok(ins.in[3], td, false) ||
          !ref_ok(ins.out[0], st, dq) || !ref_ok(ins.out[1], td, true))
        return fail();
      return disjoint();
    }
    case up::OpCode::kEmbedFwd: {
      // out[t, :] = table[tokens[t], :] with T = d0 rows and a [V, D]
      // table packed as d1 = V<<32|D. The table must be rodata: only the
      // compiler's own packing produces it, and the gather's row bound
      // (tokens[t] < V) is the FEEDER contract's runtime obligation — the
      // extents proven here are the buffers, not the indices.
      const uint64_t vocab = d1 >> 32, dim = d1 & 0xFFFFFFFFu;
      uint64_t vd = 0, td = 0;
      if (!MulOk(vocab, dim, &vd) || !MulOk(d0, dim, &td)) return fail();
      if (!up::IsRodataRef(ins.in[1])) return fail();
      if (!ref_ok(ins.in[0], d0, false) ||  // i32 tokens: 4-byte elements
          !ref_ok(ins.in[1], vd, false) || !ref_ok(ins.in[2], td, true))
        return fail();
      return disjoint();
    }
  }
  // Every known opcode returned through disjoint() above; anything else
  // must be a load error — Execute() would silently skip it.
  return diag::validating::Error("unknown opcode " +
                                 std::to_string(ins.opcode));
}

}  // namespace

std::expected<void, std::string> ValidateInstruction(
    const up::UpdateInstruction& ins, uint64_t arena_size,
    uint64_t rodata_size, uint32_t plan_version, bool allow_source) {
  InstructionExtents extents;
  return ValidateInstructionImpl(ins, arena_size, rodata_size, plan_version,
                                 allow_source, &extents);
}

std::expected<InstructionExtents, std::string> DescribeInstruction(
    const up::UpdateInstruction& ins, uint64_t arena_size,
    uint64_t rodata_size, uint32_t plan_version, bool allow_source) {
  InstructionExtents extents;
  if (auto r = ValidateInstructionImpl(ins, arena_size, rodata_size,
                                       plan_version, allow_source, &extents);
      !r)
    return std::unexpected(r.error());
  return extents;
}

}  // namespace seeml::update_rt
