#ifndef SEEML_COMPILER_BACKEND_ARENA_BINDER_H_
#define SEEML_COMPILER_BACKEND_ARENA_BINDER_H_

#include <cstdint>
#include <expected>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "compiler/frontend/forward_builder.h"
#include "compiler/frontend/sir.h"
#include "source/update_types.h"

// =============================================================================
// Segmented arena binding — the memory planner of the update compiler.
//
// Assigns every SIR value a fixed offset in the runtime's single arena:
//
//   [ PERSISTENT | IO | TRANSIENT ]        + RODATA (in the plan blob)
//
//   PERSISTENT  trainable adapters + optimizer state, checkpointed
//   IO          batch input + label slots, rewritten every step
//   TRANSIENT   liveness-scanned workspace, offsets reused across values
//   RODATA      packed frozen weights (f32, or per-tensor symmetric int8
//               for weights selected by SelectQuantizedWeights)
//
// Everything here runs at compile time; the runtime just does base + offset.
// =============================================================================

namespace seeml::update {

/// Arena/section alignment shared by the binder and plan assembly.
inline constexpr uint64_t kArenaAlign = 64;
inline uint64_t AlignUp(uint64_t v) {
  return (v + kArenaAlign - 1) & ~(kArenaAlign - 1);
}

/// Deterministic initialization recipe for one persistent parameter.
struct ParamInit {
  const seecpp::sir::Value* value;
  uint64_t offset;  // arena offset
  uint64_t bytes;
  std::string init;  // "randn" | "zeros"
  float std = 0.0f;
  uint64_t seed = 0;
};

struct ArenaBinding {
  // value -> ref word (MakeArenaRef / MakeRodataRef encoded).
  std::unordered_map<const seecpp::sir::Value*, uint64_t> refs;
  uint64_t persistent_size = 0;
  uint64_t io_end = 0;      // end of [persistent | io] prefix
  uint64_t arena_size = 0;  // total (after transient scan, both programs)
  std::vector<ParamInit> params;  // in allocation order
  std::vector<uint8_t> rodata;    // packed frozen weights
};

/// Selects frozen weights that can be stored as per-tensor symmetric int8:
/// every use must be the B operand of a GEMM with a q8 variant (the
/// student/teacher forward MatMuls and the dX backward). Bias vectors,
/// LayerNorm affine parameters, and anything feeding another op stay f32.
/// Returns weight value -> dequantization scale.
std::unordered_map<const seecpp::sir::Value*, float> SelectQuantizedWeights(
    seecpp::sir::Block& block, const GraphBuild& build);

/// Liveness-driven linear-scan allocation for transient values starting at
/// `base`. Values in `already_bound` are skipped; values in `pinned` are
/// never reclaimed (loss slot, parameter gradients, merged deltas).
/// Returns the high-water mark; assignments are appended to `refs_out`.
uint64_t LinearScanTransients(
    seecpp::sir::Block& block, uint64_t base,
    const std::unordered_map<const seecpp::sir::Value*, uint64_t>&
        already_bound,
    const std::unordered_set<const seecpp::sir::Value*>& pinned,
    std::unordered_map<const seecpp::sir::Value*, uint64_t>& refs_out);

/// Binds the training block: persistent params, IO slots, rodata packing
/// (quantizing weights in `quant_scales`), then the transient scan.
[[nodiscard]] std::expected<ArenaBinding, std::string> BindArena(
    seecpp::sir::Block& train_block, const GraphBuild& build,
    const std::unordered_set<const seecpp::sir::Value*>& pinned,
    const std::unordered_map<const seecpp::sir::Value*, float>& quant_scales);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_BACKEND_ARENA_BINDER_H_
