#ifndef SEEML_COMPILER_BACKEND_ALLOCATION_ARENA_BINDER_H_
#define SEEML_COMPILER_BACKEND_ALLOCATION_ARENA_BINDER_H_

#include <cstdint>
#include <expected>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "compiler/frontend/computation/graph_build.h"
#include "compiler/frontend/representation/sir.h"
#include "source/plan/update_types.h"

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
//               for weights the quantization review selected —
//               compiler/analysis/statistics/quantization.h)
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
  const seeml::sir::Value* value;
  uint64_t offset;  // arena offset
  uint64_t bytes;
  std::string init;  // "randn" | "zeros"
  float std = 0.0f;
  uint64_t seed = 0;
};

/// One frozen weight's place in the rodata section: where it goes, how many
/// bytes it takes there, and how its f32 source is stored.
struct RodataPack {
  enum class Storage : uint8_t { kF32, kInt8, kBf16 };
  const SmfTensor* source = nullptr;  // the f32 payload in the loaded model
  uint64_t offset = 0;                // within the rodata section
  uint64_t bytes = 0;                 // at that offset
  Storage storage = Storage::kF32;
  float scale = 0.0f;                 // kInt8: the per-tensor dequant scale
  // kInt8 (plan v17): one scale per output column of W [K, M], stored as
  // f32 at `scales_offset` right after the int8 levels; the levels are
  // quantized per column. Empty = the per-tensor form above.
  std::vector<float> column_scales{};
  uint64_t scales_offset = 0;
};

struct ArenaBinding {
  // value -> ref word (MakeArenaRef / MakeRodataRef encoded).
  std::unordered_map<const seeml::sir::Value*, uint64_t> refs;
  uint64_t persistent_size = 0;
  uint64_t io_end = 0;      // end of [persistent | io] prefix
  uint64_t arena_size = 0;  // total (after transient scan, both programs)
  std::vector<ParamInit> params;  // in allocation order
  // The rodata section as a LAYOUT, in offset order — its bytes do not
  // exist until PackRodata writes them straight into the plan blob (E2,
  // #81). Materializing the section here first kept a third copy of the
  // weights alive through assembly (model, this, the plan) and grew it by
  // repeated zero-filling resize.
  std::vector<RodataPack> rodata_packs;
  uint64_t rodata_size = 0;
  // v17: int8 weight -> rodata ref of its per-column scale vector.
  std::unordered_map<const seeml::sir::Value*, uint64_t> quant_column_scales{};
};

/// Writes one pack's bytes to `dst` (`pack.bytes` long; gaps between packs
/// are the caller's to zero). Byte-for-byte what BindArena used to
/// materialize: the raw f32 payload, per-tensor symmetric int8
/// (round-half-away of value/scale, clamped to +-127), or bfloat16
/// (round-to-nearest-even). Chunked over ParallelFor with a fixed grain, so
/// the bytes never depend on the worker count.
void PackRodata(const RodataPack& pack, uint8_t* dst);

/// Liveness-driven linear-scan allocation for transient values starting at
/// `base`. Values in `already_bound` are skipped; values in `pinned` are
/// never reclaimed (loss slot, parameter gradients, merged deltas).
/// Returns the high-water mark; assignments are appended to `refs_out`.
uint64_t LinearScanTransients(
    seeml::sir::Block& block, uint64_t base,
    const std::unordered_map<const seeml::sir::Value*, uint64_t>&
        already_bound,
    const std::unordered_set<const seeml::sir::Value*>& pinned,
    std::unordered_map<const seeml::sir::Value*, uint64_t>& refs_out);

/// Binds the training block: persistent params, IO slots, the rodata layout
/// (int8 for weights in `quant_scales`), then the transient scan.
/// `bf16_weights`: frozen weights packed as bfloat16 rodata (half of f32,
/// round-to-nearest-even; roadmap 2c), disjoint from `quant_scales`.
[[nodiscard]] std::expected<ArenaBinding, std::string> BindArena(
    seeml::sir::Block& train_block, const GraphBuild& build,
    const std::unordered_set<const seeml::sir::Value*>& pinned,
    const std::unordered_map<const seeml::sir::Value*, float>& quant_scales,
    const std::unordered_set<const seeml::sir::Value*>& bf16_weights = {});

}  // namespace seeml::update

#endif  // SEEML_COMPILER_BACKEND_ALLOCATION_ARENA_BINDER_H_
