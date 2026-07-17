#ifndef SEEML_RUNTIME_PLAN_VALIDATOR_H_
#define SEEML_RUNTIME_PLAN_VALIDATOR_H_

#include <cstdint>
#include <expected>
#include <string>

#include "source/update_types.h"

// =============================================================================
// Load-time validation of a .seeu plan's instruction streams.
//
// Every operand ref of every instruction is checked against the address
// space it targets (arena or rodata) before the first step runs — after
// this, the engine's Execute() dispatch can trust the programs blindly.
// The overflow-safe helpers are exposed for the engine's own section and
// I/O-slot checks, so plan bounds math is written exactly one way.
// =============================================================================

namespace seeml::update_rt {

/// Overflow-checked u64 multiply for file-supplied offsets and sizes.
inline bool MulOk(uint64_t a, uint64_t b, uint64_t* out) {
  if (b != 0 && a > UINT64_MAX / b) return false;
  *out = a * b;
  return true;
}

/// True iff [off, off + bytes) lies inside [0, size) without wrapping.
inline bool RangeOk(uint64_t off, uint64_t bytes, uint64_t size) {
  return off <= size && bytes <= size - off;
}

/// Checks that every ref operand of `ins` lies fully inside its address space
/// (arena or rodata), with byte extents derived from the instruction's dims
/// the same way the kernels derive their loop bounds, and that writes only
/// target the mutable arena. Rejects unknown opcodes, which Execute() would
/// silently skip.
[[nodiscard]] std::expected<void, std::string> ValidateInstruction(
    const seeml::update::UpdateInstruction& ins, uint64_t arena_size,
    uint64_t rodata_size);

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_PLAN_VALIDATOR_H_
