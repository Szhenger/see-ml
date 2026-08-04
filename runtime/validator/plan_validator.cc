#include "runtime/validator/plan_validator.h"

#include "runtime/diagnostics/validating/error.h"

namespace seeml::update_rt {

namespace up = seeml::update;

std::expected<void, std::string> ValidateInstruction(
    const up::UpdateInstruction& ins, uint64_t arena_size,
    uint64_t rodata_size) {
  // Byte ranges admitted so far, for the aliasing proof after the switch.
  // No opcode carries more than 6 validated operands (LayerNormFwd).
  struct Extent {
    uint64_t begin = 0, bytes = 0;
    bool write = false, rodata = false;
  };
  Extent extents[8];
  size_t n_extents = 0;

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
    if (elems == 0) return false;
    uint64_t bytes = 0;
    if (!MulOk(elems, elem_bytes, &bytes)) return false;
    const uint64_t offset = up::RefOffset(ref);
    if (offset % elem_bytes != 0) return false;
    const uint64_t space = up::IsRodataRef(ref) ? rodata_size : arena_size;
    if (!RangeOk(offset, bytes, space)) return false;
    extents[n_extents++] =
        Extent{offset, bytes, write, up::IsRodataRef(ref)};
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

  const uint64_t d0 = ins.out[0], d1 = ins.out[1], d2 = ins.out[2];
  uint64_t mk = 0, kn = 0, mn = 0, nc = 0;
  switch (static_cast<up::OpCode>(ins.opcode)) {
    case up::OpCode::kNop:
      break;
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
      break;
    }
    case up::OpCode::kAddEW:
    case up::OpCode::kMulEW:
    case up::OpCode::kReluBwd:
    case up::OpCode::kGeluBwd:
    case up::OpCode::kSiluBwd:
      if (!ref_ok(ins.in[0], d0, false) || !ref_ok(ins.in[1], d0, false) ||
          !ref_ok(ins.in[2], d0, true))
        return fail();
      break;
    case up::OpCode::kAddBias:
      if (!MulOk(d0, d1, &mn)) return fail();
      if (!ref_ok(ins.in[0], mn, false) || !ref_ok(ins.in[1], d1, false) ||
          !ref_ok(ins.in[2], mn, true))
        return fail();
      break;
    case up::OpCode::kReluFwd:
    case up::OpCode::kGeluFwd:
    case up::OpCode::kSiluFwd:
    case up::OpCode::kScale:
    case up::OpCode::kCopy:
      if (!ref_ok(ins.in[0], d0, false) || !ref_ok(ins.in[1], d0, true))
        return fail();
      break;
    case up::OpCode::kLayerNormFwd: {
      const uint64_t rows = d0 >> 32, cols = d0 & 0xFFFFFFFFu;
      if (!MulOk(rows, cols, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], cols, false) ||
          !ref_ok(ins.in[2], cols, false) || !ref_ok(ins.in[3], nc, true) ||
          !ref_ok(ins.out[1], rows, true) || !ref_ok(ins.out[2], rows, true))
        return fail();
      break;
    }
    case up::OpCode::kLayerNormBwd: {
      const uint64_t rows = d2 >> 32, cols = d2 & 0xFFFFFFFFu;
      if (!MulOk(rows, cols, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], nc, false) ||
          !ref_ok(ins.in[2], cols, false) || !ref_ok(ins.in[3], nc, true) ||
          !ref_ok(ins.out[0], rows, false) || !ref_ok(ins.out[1], rows, false))
        return fail();
      break;
    }
    case up::OpCode::kClipNorm:
      if (!ref_ok(ins.in[0], d0, true)) return fail();
      break;
    case up::OpCode::kReduceRows:
      if (!MulOk(d0, d1, &mn)) return fail();
      if (!ref_ok(ins.in[0], mn, false) || !ref_ok(ins.in[1], d1, true))
        return fail();
      break;
    case up::OpCode::kSoftmaxXEntFwd:
      if (!MulOk(d0, d1, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], d0, false) ||
          !ref_ok(ins.in[2], 1, true) || !ref_ok(ins.in[3], nc, true))
        return fail();
      break;
    case up::OpCode::kSoftmaxXEntBwd:
      if (!MulOk(d0, d1, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], d0, false) ||
          !ref_ok(ins.in[2], 1, false) || !ref_ok(ins.in[3], nc, true))
        return fail();
      break;
    case up::OpCode::kMseFwd:
      if (!ref_ok(ins.in[0], d0, false) || !ref_ok(ins.in[1], d0, false) ||
          !ref_ok(ins.in[2], 1, true))
        return fail();
      break;
    case up::OpCode::kMseBwd:
      if (!ref_ok(ins.in[0], d0, false) || !ref_ok(ins.in[1], d0, false) ||
          !ref_ok(ins.in[2], 1, false) || !ref_ok(ins.in[3], d0, true))
        return fail();
      break;
    case up::OpCode::kKLDistillFwd:
      if (!MulOk(d1 >> 32, d1 & 0xFFFFFFFFu, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], nc, false) ||
          !ref_ok(ins.in[2], 1, true) || !ref_ok(ins.in[3], nc, true) ||
          !ref_ok(ins.out[0], nc, true))
        return fail();
      break;
    case up::OpCode::kKLDistillBwd:
      if (!MulOk(d0 >> 32, d0 & 0xFFFFFFFFu, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], nc, false) ||
          !ref_ok(ins.in[2], 1, false) || !ref_ok(ins.in[3], nc, true))
        return fail();
      break;
    case up::OpCode::kSgdStep:
      if (!ref_ok(ins.in[0], d0, true) || !ref_ok(ins.in[1], d0, false))
        return fail();
      break;
    case up::OpCode::kAdamWStep:
      if (!ref_ok(ins.in[0], d0, true) || !ref_ok(ins.in[1], d0, false) ||
          !ref_ok(ins.in[2], d0, true) || !ref_ok(ins.in[3], d0, true))
        return fail();
      break;
    case up::OpCode::kFill:
      if (!ref_ok(ins.in[0], d0, true)) return fail();
      break;
    default:
      return diag::validating::Error("unknown opcode " +
                                     std::to_string(ins.opcode));
  }

  // Aliasing proof: the kernels are compiled with no-alias (restrict)
  // qualifiers on the promise the arena binder keeps for compiled plans; a
  // foreign plan must prove it here or blind dispatch is UB. Any operand
  // pair where at least one side is written must be disjoint (read-read
  // overlap is harmless — nothing is modified through either pointer).
  for (size_t i = 0; i < n_extents; ++i)
    for (size_t j = i + 1; j < n_extents; ++j) {
      const Extent& a = extents[i];
      const Extent& b = extents[j];
      if (!a.write && !b.write) continue;
      if (a.rodata != b.rodata) continue;
      if (a.begin < b.begin + b.bytes && b.begin < a.begin + a.bytes)
        return diag::validating::Error(
            "instruction operands alias (opcode " +
            std::to_string(ins.opcode) + ")");
    }
  return {};
}

}  // namespace seeml::update_rt
