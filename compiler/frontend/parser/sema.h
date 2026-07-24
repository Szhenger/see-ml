#ifndef SEEML_COMPILER_FRONTEND_PARSER_SEMA_H_
#define SEEML_COMPILER_FRONTEND_PARSER_SEMA_H_

#include <expected>
#include <string>

#include "compiler/frontend/ingressor/model_format.h"
#include "compiler/frontend/sir.h"

// =============================================================================
// Sema — semantic analysis for the forward parser, at two levels:
//
//   graph  CheckGraph validates the SMF op list as a whole before any SIR is
//          built: every op output must introduce a fresh name (no duplicate
//          outputs, no shadowing of a tensor or the graph input), every
//          consumed name that an op produces must be produced by an EARLIER
//          op (the topological-order contract the single forward pass relies
//          on), and the declared model output must be produced by an op —
//          not be a frozen tensor or the input echoed back.
//   op     CheckMatMul / CheckAddBias / CheckMul / CheckLayerNorm validate
//          rank, inner-dimension, and affine-width agreement between one op
//          and its resolved SIR operands.
//
// Structural checks (operand counts, names that exist nowhere) stay in the
// parser and resolver.
// =============================================================================

namespace seeml::update::sema {

/// Whole-graph validation (see banner). Cheap — two O(tensors + ops) passes
/// over string_views into the model — and turns what would be late,
/// misleading parse failures into precise errors before any SIR exists.
[[nodiscard]] std::expected<void, std::string> CheckGraph(
    const SmfModel& model);

/// MatMul: both operands rank-2, inner dimensions agree.
[[nodiscard]] std::expected<void, std::string> CheckMatMul(
    const SmfOp& op, const seeml::sir::Value& x, const seeml::sir::Value& w);

/// AddBias: rank-1 bias whose width matches the input's last dimension.
[[nodiscard]] std::expected<void, std::string> CheckAddBias(
    const SmfOp& op, const seeml::sir::Value& x, const seeml::sir::Value& b);

/// Mul: elementwise — operand shapes must agree exactly.
[[nodiscard]] std::expected<void, std::string> CheckMul(
    const SmfOp& op, const seeml::sir::Value& x, const seeml::sir::Value& y);

/// LayerNorm: rank-2 input; rank-1 gamma/beta matching the last dimension.
[[nodiscard]] std::expected<void, std::string> CheckLayerNorm(
    const SmfOp& op, const seeml::sir::Value& x,
    const seeml::sir::Value& gamma, const seeml::sir::Value& beta);

}  // namespace seeml::update::sema

#endif  // SEEML_COMPILER_FRONTEND_PARSER_SEMA_H_
