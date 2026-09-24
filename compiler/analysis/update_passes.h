#ifndef SEEML_COMPILER_ANALYSIS_UPDATE_PASSES_H_
#define SEEML_COMPILER_ANALYSIS_UPDATE_PASSES_H_

// =============================================================================
// The analysis subsystem, partitioned by discipline. This façade re-exports
// the whole pipeline for consumers (the backend driver, tests); each group
// lives in its own directory so a stack trace or a diff lands in exactly
// one concern:
//
// Each directory is named for the field of mathematics that studies what it
// does (docs/next-project/seeai.md §4):
//
//   pass_manager   runs SIR passes under the Block::verify invariant gate
//   algebra/       value-preserving rewrites: LoraGrafter (C = X@W  ->
//                  C' = X@W + (α/r)·(X@A)@B), MergeBuilder (fused
//                  Δ = (α/r)·A@B via sc_low.gemm_acc), GemmEpilogueFuser /
//                  GemmAddendFuser / ElementwiseChainFuser (chains folded
//                  where the use-lists prove no other reader), RopeTable,
//                  ConvLowering (conv2d -> im2col-GEMM form)
//   calculus/      TrainableAutodiff (reverse-mode AD pruned to the
//                  trainable set)
//   statistics/    decisions from measured numbers: SelectQuantizedWeights
//                  (which frozen weights pack as int8 rodata, at what
//                  scale), AttentionTiling (the attention family from the
//                  step-0 footprint)
//   topology/      DeadCodeElimination (reachability: proves the emitted
//                  programs carry no unreferenced compute)
//   optimization/  OptimizerSynthesizer (SGD / AdamW step synthesis)
//
// Op dialect used (matching sir.h's mnemonic prefixes):
//   sc_mem.weight  frozen base/teacher weight (rodata); attrs: smf_offset
//   sc_mem.param   persistent trainable/state value; attrs: trainable, init
//                  ("randn"|"zeros"), std, seed
//   sc_high.*      differentiable forward ops
//   sc_low.*       synthesized adjoint / optimizer / merge ops
// =============================================================================

#include "compiler/analysis/algebra/epilogue_fuser.h" // IWYU pragma: export
#include "compiler/analysis/algebra/addend_fuser.h"   // IWYU pragma: export
#include "compiler/analysis/statistics/attention_tiling.h"  // IWYU pragma: export
#include "compiler/analysis/algebra/chain_fuser.h"    // IWYU pragma: export
#include "compiler/analysis/algebra/rope_table.h"     // IWYU pragma: export
#include "compiler/analysis/algebra/lora_grafter.h"   // IWYU pragma: export
#include "compiler/analysis/algebra/merge_builder.h"  // IWYU pragma: export
#include "compiler/analysis/calculus/autodiff.h"      // IWYU pragma: export
#include "compiler/analysis/optimization/optimizer.h"     // IWYU pragma: export
#include "compiler/analysis/statistics/quantization.h"  // IWYU pragma: export
#include "compiler/analysis/algebra/conv_lowering.h"  // IWYU pragma: export
#include "compiler/analysis/topology/dce.h"            // IWYU pragma: export
#include "compiler/analysis/pass_manager.h"   // IWYU pragma: export

#endif  // SEEML_COMPILER_ANALYSIS_UPDATE_PASSES_H_
