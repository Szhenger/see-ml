#ifndef SEEML_COMPILER_ANALYSIS_UPDATE_PASSES_H_
#define SEEML_COMPILER_ANALYSIS_UPDATE_PASSES_H_

// =============================================================================
// The analysis subsystem, partitioned by discipline. This façade re-exports
// the whole pipeline for consumers (the backend driver, tests); each group
// lives in its own directory so a stack trace or a diff lands in exactly
// one concern:
//
//   updater/   pass management + structural lowering:
//              PassManager (runs SIR passes under the Block::verify
//              invariant gate), ConvLowering (conv2d -> im2col-GEMM form),
//              DeadCodeElimination (the optimization phase's sweep: proves
//              the emitted programs carry no unreferenced compute)
//   algebra/   the adapter linear algebra and kernel fusion:
//              LoraGrafter (C = X@W  ->  C' = X@W + (α/r)·(X@A)@B),
//              MergeBuilder (fused Δ = (α/r)·A@B via sc_low.gemm_acc)
//   calculus/  the SGD machinery:
//              TrainableAutodiff (reverse-mode AD pruned to the trainable
//              set), OptimizerSynthesizer (SGD / AdamW step synthesis)
//   reviewer/  model preprocessing that configures the backend:
//              SelectQuantizedWeights (which frozen weights pack as int8
//              rodata, and at what per-tensor scale)
//
// Op dialect used (matching sir.h's mnemonic prefixes):
//   sc_mem.weight  frozen base/teacher weight (rodata); attrs: smf_offset
//   sc_mem.param   persistent trainable/state value; attrs: trainable, init
//                  ("randn"|"zeros"), std, seed
//   sc_high.*      differentiable forward ops
//   sc_low.*       synthesized adjoint / optimizer / merge ops
// =============================================================================

#include "compiler/analysis/algebra/lora_grafter.h"   // IWYU pragma: export
#include "compiler/analysis/algebra/merge_builder.h"  // IWYU pragma: export
#include "compiler/analysis/calculus/autodiff.h"      // IWYU pragma: export
#include "compiler/analysis/calculus/optimizer.h"     // IWYU pragma: export
#include "compiler/analysis/reviewer/quantization.h"  // IWYU pragma: export
#include "compiler/analysis/updater/conv_lowering.h"  // IWYU pragma: export
#include "compiler/analysis/updater/dce.h"            // IWYU pragma: export
#include "compiler/analysis/updater/pass_manager.h"   // IWYU pragma: export

#endif  // SEEML_COMPILER_ANALYSIS_UPDATE_PASSES_H_
