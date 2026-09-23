#ifndef SEEML_COMPILER_BACKEND_TRAINER_INSTRUCTION_LOWERING_H_
#define SEEML_COMPILER_BACKEND_TRAINER_INSTRUCTION_LOWERING_H_

#include <cstdint>
#include <expected>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "compiler/frontend/representation/sir.h"
#include "source/plan/update_types.h"

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
    const seeml::sir::Value*)>;

/// Lowers `ops` in order. Lowering over an explicit op list (rather than a
/// whole block) lets the driver emit the evaluation program from the primal
/// prefix of the training block — the ops present before autodiff appended
/// the backward pass. `quant_scales` maps frozen weights stored as int8 to
/// their dequantization scale; GEMMs over them lower to the q8 opcodes.
/// `quant_column_scales` (plan v17) maps such a weight to the rodata ref of
/// its per-output-column scale vector: when present the q8 GEMM carries
/// kFlagQ8ColScale and that ref in in[3] instead of the per-tensor scale.
[[nodiscard]] std::expected<std::vector<UpdateInstruction>, std::string>
LowerOps(const std::vector<seeml::sir::Operation*>& ops,
         const ResolveFn& resolve,
         const std::unordered_map<const seeml::sir::Value*, float>&
             quant_scales,
    const std::unordered_set<const seeml::sir::Value*>& bf16_weights = {},
    const std::unordered_map<const seeml::sir::Value*, uint64_t>&
        quant_column_scales = {});

}  // namespace seeml::update

#endif  // SEEML_COMPILER_BACKEND_TRAINER_INSTRUCTION_LOWERING_H_
