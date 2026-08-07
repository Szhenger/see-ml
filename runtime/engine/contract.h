#ifndef SEEML_RUNTIME_ENGINE_CONTRACT_H_
#define SEEML_RUNTIME_ENGINE_CONTRACT_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>

#include "runtime/feeder/dataset.h"
#include "source/plan/update_types.h"

// =============================================================================
// Subsystem usage contracts — the engine's verification layer, mirroring
// compiler/driver/contract.h. Each runtime subsystem guards its own
// invariants (the validator's per-instruction sweep, the custodian's
// checkpoint binding, the feeder's format checks); these contracts check
// the *seams*: that the engine hands each subsystem what it requires and
// that what the compiler's driver promised at emit time still holds at
// load time. One verifier per boundary the engine crosses:
//
//   VerifyPlanContract      at load      — the header's section table, arena
//                           relations, and I/O slots are self-consistent
//                           with the plan blob
//   VerifyExecutorContract  after decode — every instruction of every
//                           program is bounds-proven (validator/) and every
//                           emit entry targets the arena, so Execute() may
//                           dispatch blindly
//   VerifyFeederContract    before train — the dataset's geometry matches
//                           the compiled plan's batch, input, and label
//                           slots
//   WellFormedDiagnostic    on any error — the message is attributable to a
//                           unit registered in runtime/diagnostics/
//
// Contract violations mean a corrupt or foreign artifact, or a misused
// subsystem — the engine reports them under its own unit.
// =============================================================================

namespace seeml::update_rt {

/// True iff `message` reads "<unit>: <nonempty message>" for a unit
/// registered in one of the four runtime diagnostics process modules.
[[nodiscard]] bool WellFormedDiagnostic(std::string_view message);

/// The plan header is self-consistent with a blob of `plan_size` bytes:
/// no section size overflows, every section lies inside the blob, the
/// arena dominates its persistent segment and image, and the input, label,
/// and loss I/O slots lie inside the mutable arena.
[[nodiscard]] std::expected<void, std::string> VerifyPlanContract(
    const seeml::update::PlanHeader& header, uint64_t plan_size);

/// The decoded programs are fit for blind dispatch: every operand ref of
/// every instruction passes the validator against its address space, and
/// every emit entry's delta range lies inside the arena.
[[nodiscard]] std::expected<void, std::string> VerifyExecutorContract(
    std::span<const seeml::update::UpdateInstruction> train,
    std::span<const seeml::update::UpdateInstruction> merge,
    std::span<const seeml::update::UpdateInstruction> eval,
    std::span<const seeml::update::EmitEntry> emit_table,
    const seeml::update::PlanHeader& header);

/// The dataset can feed this plan: batch x input_dim matches the input
/// slot, the label kind and per-batch width match the label slot, and
/// class labels (when used) all index inside the softmax width. For token
/// plans (header.input_kind == 1) the corpus must be a token corpus whose
/// record length equals the plan's seq_len, and every token id must index
/// inside BOTH the narrowest embedding table (`vocab_bound`) and — via the
/// derived next-token labels — the narrowest softmax width.
[[nodiscard]] std::expected<void, std::string> VerifyFeederContract(
    const seeml::update::PlanHeader& header, Dataset& data,
    uint64_t num_classes, uint64_t vocab_bound);

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_ENGINE_CONTRACT_H_
