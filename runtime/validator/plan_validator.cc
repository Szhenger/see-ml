#include "runtime/validator/plan_validator.h"

#include "runtime/diagnostics/validating/error.h"

namespace seeml::update_rt {

namespace up = seeml::update;

std::expected<void, std::string> ValidateInstruction(
    const up::UpdateInstruction& ins, uint64_t arena_size,
    uint64_t rodata_size) {
  // Every kernel is compiled with SEEML_RESTRICT pointers: a written range
  // overlapping any *other* operand of the same instruction is undefined
  // behavior, not a wrong answer. Bounds alone don't rule that out, so each
  // validated operand's byte range is recorded and proven disjoint from
  // every written range before the instruction is accepted. A single ref
  // that is read and written through one pointer (SGD's param, GemmAcc's C)
  // is one operand, not an alias.
  struct OperandRange {
    uint64_t off, bytes;
    bool write, rodata;
  };
  OperandRange ranges[8];
  size_t num_ranges = 0;

  // elem_bytes: f32/i32 operands are 4 bytes; quantized weights are 1.
  auto ref_ok_w = [&](uint64_t ref, uint64_t elems, bool write,
                      uint64_t elem_bytes) {
    if (ref == up::kNullRef) return false;
    if (write && up::IsRodataRef(ref)) return false;
    uint64_t bytes = 0;
    if (!MulOk(elems, elem_bytes, &bytes)) return false;
    // Execute() reinterpret_casts the ref to its element type and
    // dereferences directly, so alignment is part of "safe to dispatch
    // blindly": a misaligned offset is UB, and a bus error on the
    // strict-alignment targets this runtime ships to.
    if (up::RefOffset(ref) % elem_bytes != 0) return false;
    const uint64_t space = up::IsRodataRef(ref) ? rodata_size : arena_size;
    if (!RangeOk(up::RefOffset(ref), bytes, space)) return false;
    ranges[num_ranges++] = {up::RefOffset(ref), bytes, write,
                            up::IsRodataRef(ref)};
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
        if (!(a.write || b.write) || a.rodata != b.rodata) continue;
        if (a.bytes == 0 || b.bytes == 0) continue;
        if (a.off < b.off + b.bytes && b.off < a.off + a.bytes)
          return diag::validating::Error(
              "instruction operands alias a written range (opcode " +
              std::to_string(ins.opcode) + ")");
      }
    return {};
  };

  const uint64_t d0 = ins.out[0], d1 = ins.out[1], d2 = ins.out[2];
  uint64_t mk = 0, kn = 0, mn = 0, nc = 0;
  switch (static_cast<up::OpCode>(ins.opcode)) {
    case up::OpCode::kNop:
      return disjoint();
    case up::OpCode::kGemmNN:
    case up::OpCode::kGemmNT:
    case up::OpCode::kGemmTN:
    case up::OpCode::kGemmAccNN:
    case up::OpCode::kGemmNNQ8:
    case up::OpCode::kGemmNTQ8: {
      // Every layout variant reads M*K (A) and K*N (B), writes M*N (C).
      const bool q8 = static_cast<up::OpCode>(ins.opcode) ==
                          up::OpCode::kGemmNNQ8 ||
                      static_cast<up::OpCode>(ins.opcode) ==
                          up::OpCode::kGemmNTQ8;
      if (!MulOk(d0, d2, &mk) || !MulOk(d2, d1, &kn) || !MulOk(d0, d1, &mn))
        return fail();
      // Quantized B must live in rodata: only the compiler's own int8
      // packing produces it, and it is 1 byte per element.
      if (q8 && !up::IsRodataRef(ins.in[1])) return fail();
      if (!ref_ok(ins.in[0], mk, false) ||
          !ref_ok_w(ins.in[1], kn, false, q8 ? 1 : sizeof(float)) ||
          !ref_ok(ins.in[2], mn, true))
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
      return disjoint();
    case up::OpCode::kKLDistillBwd:
      if (!MulOk(d0 >> 32, d0 & 0xFFFFFFFFu, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], nc, false) ||
          !ref_ok(ins.in[2], 1, false) || !ref_ok(ins.in[3], nc, true))
        return fail();
      return disjoint();
    case up::OpCode::kSgdStep:
      if (!ref_ok(ins.in[0], d0, true) || !ref_ok(ins.in[1], d0, false))
        return fail();
      return disjoint();
    case up::OpCode::kAdamWStep:
      if (!ref_ok(ins.in[0], d0, true) || !ref_ok(ins.in[1], d0, false) ||
          !ref_ok(ins.in[2], d0, true) || !ref_ok(ins.in[3], d0, true))
        return fail();
      return disjoint();
    case up::OpCode::kFill:
      if (!ref_ok(ins.in[0], d0, true)) return fail();
      return disjoint();
  }
  return diag::validating::Error("unknown opcode " +
                         std::to_string(ins.opcode));
}

}  // namespace seeml::update_rt
