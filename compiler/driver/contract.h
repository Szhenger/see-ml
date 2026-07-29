#ifndef SEEML_COMPILER_DRIVER_CONTRACT_H_
#define SEEML_COMPILER_DRIVER_CONTRACT_H_

#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>

#include "compiler/analysis/algebra/lora_grafter.h"
#include "compiler/analysis/algebra/merge_builder.h"
#include "compiler/driver/update_compiler.h"
#include "compiler/frontend/parser/graph_build.h"
#include "compiler/frontend/representation/sir.h"

// =============================================================================
// Subsystem usage contracts — the driver's verification layer. Each subsystem
// already guards its own invariants (Block::verify, the PassManager gate,
// ValidateGemmTiling); these contracts check the *seams*: that the driver
// handed each subsystem what it required and received what the next one
// assumes. One verifier per boundary the driver crosses:
//
//   VerifyFrontendContract   after frontend/  — the forward SIR and its
//                            graph-build state are fit for analysis
//   VerifyAnalysisContract   after analysis/  — grafts, gradients, and the
//                            merge program are fit for code generation
//   VerifyGeneratedPlan      after backend/   — the assembled plan is
//                            internally consistent with what was compiled
//   WellFormedDiagnostic     on any error     — the message is attributable
//                            to a unit registered in diagnostics/
//
// Contract violations are driver bugs, not user errors — they mean a
// subsystem was misused, so the driver reports them under its own unit.
// =============================================================================

namespace seeml::update {

/// True iff `message` reads "<unit>: <nonempty message>" for a unit
/// registered in one of the six diagnostics process modules. Every error
/// that escapes the driver must satisfy this, or the process that failed
/// cannot be delimited.
[[nodiscard]] bool WellFormedDiagnostic(std::string_view message);

/// The frontend was used correctly: the graph build carries an input and an
/// output, the forward SIR satisfies its SSA invariants, and every recorded
/// weight source is a constant SMF tensor materialized as sc_mem.weight.
[[nodiscard]] std::expected<void, std::string> VerifyFrontendContract(
    const seeml::sir::Block& block, const GraphBuild& build);

/// The analysis was used correctly: at least one adapter was grafted, each
/// adapter's A/B are sc_mem.param declarations with a gradient reaching
/// them (exactly two per adapter), the training block still satisfies its
/// invariants, and the merge program covers every adapter and verifies.
[[nodiscard]] std::expected<void, std::string> VerifyAnalysisContract(
    const seeml::sir::Block& block, std::span<const GraftedAdapter> adapters,
    const std::unordered_map<seeml::sir::Value*, seeml::sir::Value*>&
        param_grads,
    const MergeProgram& merge);

/// The backend was used correctly: the plan is non-empty, all three programs
/// were lowered (with the eval program a prefix of training, and at least
/// one merge instruction per adapter), the arena contains its persistent
/// segment, frozen weights were packed to rodata, and the debug hooks cover
/// every adapter and trainable parameter.
[[nodiscard]] std::expected<void, std::string> VerifyGeneratedPlan(
    const CompiledUpdate& compiled, size_t num_adapters);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_DRIVER_CONTRACT_H_
