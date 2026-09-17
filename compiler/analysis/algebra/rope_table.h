#ifndef SEEML_COMPILER_ANALYSIS_ALGEBRA_ROPE_TABLE_H_
#define SEEML_COMPILER_ANALYSIS_ALGEBRA_ROPE_TABLE_H_

#include <cstddef>
#include <expected>
#include <string>

#include "compiler/frontend/representation/sir.h"

// =============================================================================
// RopeTableHoister — common-subexpression elimination for the one thing
// every RoPE op recomputes: the angles. cos/sin of angle(s, c) =
// s * base^(-2c/d) depend on the position and the pair alone, yet each
// rotation kernel derived them again for every (b, h) unit, of every
// rotation, forward and backward — ~99% of the family's transcendental
// work (Performance Audit finding #3; E3, #82).
//
// The pass declares one `sc_low.rope_table` per distinct (seq, head width,
// base) ahead of its first user and hands its result to every
// `sc_high.rope` as operand 1; autodiff forwards the same value to the
// `sc_low.rope_grad` it emits. The table is an ordinary transient: the
// arena binder keeps it live from the first forward rotation to the last
// backward one, so the step-0 memory gate prices it like any other value
// (S*d floats), and the eval program — the primal snapshot — carries the
// table op too.
//
// Runs on the primal block, before autodiff. Bitwise-neutral: the runtime
// builds the table with the recurrence the rotation kernels run, through
// the same libm calls, ON THE DEVICE — the compiler never evaluates a
// transcendental on the device's behalf.
// =============================================================================

namespace seeml::update {

class RopeTableHoister {
 public:
  /// Returns the number of tables declared (0 for a model with no RoPE).
  [[nodiscard]] std::expected<size_t, std::string> Run(
      seeml::sir::Block& block);
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_ALGEBRA_ROPE_TABLE_H_
