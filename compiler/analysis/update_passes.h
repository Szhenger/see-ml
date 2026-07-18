#ifndef SEEML_COMPILER_ANALYSIS_UPDATE_PASSES_H_
#define SEEML_COMPILER_ANALYSIS_UPDATE_PASSES_H_

#include <expected>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "compiler/frontend/sir.h"
#include "source/update_types.h"

// =============================================================================
// The four SIR-to-SIR passes of the update compiler:
//
//   1. LoraGrafter          — rewrites C = X@W into C' = X@W + (α/r)·(X@A)@B,
//                             injecting randomized trainable adapters A, B.
//   2. TrainableAutodiff    — reverse-mode AD pruned to the trainable set:
//                             adjoints are synthesized only along paths that
//                             reach a LoRA parameter; the frozen base model
//                             generates no backward computation.
//   3. OptimizerSynthesizer — appends the optimizer step (SGD / AdamW) as
//                             ordinary SIR ops, declaring persistent moment
//                             state so one program execution = one full
//                             training step (fwd + bwd + update).
//   4. MergeBuilder         — builds the separate merge program that
//                             materializes each adapter's weight delta,
//                             Δ = (α/r)·A@B; commit adds Δ to the pristine
//                             f32 weights inside the model file itself, so
//                             the merge never needs the frozen base (which
//                             may live in rodata as quantized int8).
//
// Op dialect used (matching sir.h's mnemonic prefixes):
//   sc_mem.weight  frozen base/teacher weight (rodata); attrs: smf_offset
//   sc_mem.param   persistent trainable/state value; attrs: trainable, init
//                  ("randn"|"zeros"), std, seed
//   sc_high.*      differentiable forward ops
//   sc_low.*       synthesized adjoint / optimizer / merge ops
// =============================================================================

namespace seeml::update {

/// One grafted LoRA adapter and the frozen weight it updates.
struct GraftedAdapter {
  seecpp::sir::Value* frozen_weight = nullptr;  // W  [K, M] (rodata)
  seecpp::sir::Value* A = nullptr;              // A  [K, r] (randn init)
  seecpp::sir::Value* B = nullptr;              // B  [r, M] (zeros init)
  float scale = 1.0f;                           // α / r
};

class LoraGrafter {
 public:
  explicit LoraGrafter(LoRASpec spec) : spec_(std::move(spec)) {}

  /// Grafts adapters onto every eligible MatMul: the second operand must be a
  /// frozen sc_mem.weight whose name passes the target filters, and the op
  /// must not belong to the teacher subgraph (ids prefixed "t::").
  /// Because B is zero-initialized, step 0 of the update is exactly the
  /// source model — the update starts from the identity.
  [[nodiscard]] std::expected<std::vector<GraftedAdapter>, std::string> Run(
      seecpp::sir::Block& block);

 private:
  LoRASpec spec_;
};

class TrainableAutodiff {
 public:
  /// Weaves the backward pass into `block`. `loss` must be a scalar value in
  /// the block; `trainables` defines the set of parameters that need
  /// gradients. Returns the map param -> synthesized gradient value.
  [[nodiscard]] std::expected<
      std::unordered_map<seecpp::sir::Value*, seecpp::sir::Value*>, std::string>
  Run(seecpp::sir::Block& block, seecpp::sir::Value* loss,
      const std::vector<seecpp::sir::Value*>& trainables);
};

class OptimizerSynthesizer {
 public:
  explicit OptimizerSynthesizer(OptimizerSpec spec) : spec_(spec) {}

  [[nodiscard]] std::expected<void, std::string> Run(
      seecpp::sir::Block& block,
      const std::unordered_map<seecpp::sir::Value*, seecpp::sir::Value*>&
          param_grads);

 private:
  OptimizerSpec spec_;
};

/// The merge program lives in its own block. Its A/B operands alias storage
/// owned by the training block (the persistent segment), so the builder
/// returns an alias map that lowering uses to resolve those mirrors to the
/// training program's already-bound offsets.
struct MergeProgram {
  std::unique_ptr<seecpp::sir::Block> block;
  // mirror value in the merge block -> original value in the training block
  std::unordered_map<seecpp::sir::Value*, seecpp::sir::Value*> aliases;
  // delta output value -> the adapter it belongs to (for the emit table)
  std::vector<std::pair<seecpp::sir::Value*, const GraftedAdapter*>> outputs;
};

class MergeBuilder {
 public:
  [[nodiscard]] std::expected<MergeProgram, std::string> Run(
      const std::vector<GraftedAdapter>& adapters);
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_UPDATE_PASSES_H_
