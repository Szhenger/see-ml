#ifndef SEEML_COMPILER_ANALYSIS_ALGEBRA_ATTENTION_TILING_H_
#define SEEML_COMPILER_ANALYSIS_ALGEBRA_ATTENTION_TILING_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>

#include "compiler/frontend/representation/sir.h"
#include "source/plan/config.h"

// =============================================================================
// AttentionTiling — E11 (#94): decides, per compiled update, whether the
// attention ops keep the probability cache (the cached family: P
// [B*H*S, S] alive from forward to backward, dP and dS at the same size)
// or run tiled (four floats per query row, every probability recomputed).
// The two families compute identical bits — the tiled kernels reproduce the
// cached kernels' expressions in their orders — so the choice is a memory
// question alone: kAuto tiles every attention op when the probability
// caches of all of them together exceed `probs_cache_budget_bytes`, at
// roughly twice the attention arithmetic, and keeps the cache otherwise.
//
// Two sites take part, on purpose. The driver's step-0 memory gate decides
// first, from the SMF-level footprint estimate, because the gate must price
// the choice before any SIR exists; it hands that verdict in as
// `gate_tiled`. This pass then re-checks kAuto against the EXACT cache
// shapes the SIR carries and tiles if either says so — tiling only ever
// lowers memory, so overriding a "cached" verdict the estimate got wrong is
// always safe. The decision the pass returns is the one the plan carries
// and the compile report records.
//
// The rewrite is in place, before the primal snapshot and before autodiff:
// each `sc_high.attention` gains the attribute "tiled" and its second
// result — the cache the backward reads — becomes the [B*H*S, 4] stats
// row. The eval program, the VJP rule and the lowering read the attribute.
// =============================================================================

namespace seeml::update {

struct AttilingDecision {
  size_t attention_ops = 0;
  uint64_t probs_cache_bytes = 0;  // what the cached family would hold
  bool tiled = false;
};

class AttentionTiling {
 public:
  AttentionTiling(AttentionKind kind, bool gate_tiled,
                  uint64_t probs_cache_budget)
      : kind_(kind), gate_tiled_(gate_tiled), budget_(probs_cache_budget) {}
  [[nodiscard]] std::expected<AttilingDecision, std::string> Run(
      seeml::sir::Block& block);

 private:
  AttentionKind kind_;
  bool gate_tiled_;
  uint64_t budget_;
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_ALGEBRA_ATTENTION_TILING_H_
