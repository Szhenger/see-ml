#ifndef SEEML_COMPILER_BACKEND_INSTRUCTION_LOWERING_H_
#define SEEML_COMPILER_BACKEND_INSTRUCTION_LOWERING_H_

#include <cstdint>
#include <expected>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "compiler/frontend/sir.h"
#include "source/update_types.h"

// =============================================================================
// Instruction lowering — SIR ops to the .seeu UpdateInstruction stream.
//
// Purely mechanical: every sc_high/sc_low op maps to one opcode, with its
// operand refs resolved through the caller-supplied ResolveFn (the arena
// binding for the training/eval programs, the alias-aware map for the merge
// program). Storage declarations (sc_mem.*) emit no code.
// =============================================================================

namespace seeml::update {

using ResolveFn = std::function<std::expected<uint64_t, std::string>(
    const seecpp::sir::Value*)>;

/// Lowers `ops` in order. Lowering over an explicit op list (rather than a
/// whole block) lets the driver emit the evaluation program from the primal
/// prefix of the training block — the ops present before autodiff appended
/// the backward pass. `quant_scales` maps frozen weights stored as int8 to
/// their dequantization scale; GEMMs over them lower to the q8 opcodes.
[[nodiscard]] std::expected<std::vector<UpdateInstruction>, std::string>
LowerOps(const std::vector<seecpp::sir::Operation*>& ops,
         const ResolveFn& resolve,
         const std::unordered_map<const seecpp::sir::Value*, float>&
             quant_scales);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_BACKEND_INSTRUCTION_LOWERING_H_
