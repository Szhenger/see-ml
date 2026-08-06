#ifndef SEEML_SOURCE_PLAN_CONFIG_H_
#define SEEML_SOURCE_PLAN_CONFIG_H_

#include <cstdint>
#include <string>
#include <vector>

// =============================================================================
// config/ discipline of the plan ABI: the compilation request. What the user
// hands the update compiler — loss selection, LoRA and optimizer
// specifications, batch and budget — before any byte of the plan exists.
// The runtime never reads these; their compiled consequences land in the
// PlanHeader (schema.h).
// =============================================================================

namespace seeml::update {

enum class LossKind : uint8_t {
  kSoftmaxXEnt = 0,  // supervised: cross-entropy over class labels
  kMse = 1,          // supervised: mean squared error over dense targets
  kKLDistill = 2,    // teacher-driven: KL(teacher || student) on logits
  kXEntPlusKL = 3,   // composite: (1-w)*xent + w*kl
};

enum class OptimizerKind : uint8_t { kSgd = 0, kAdamW = 1 };

// Learning-rate schedule applied by the runtime on top of OptimizerSpec::lr.
// Warmup ramps linearly from 0 over `warmup_steps`; cosine then decays to
// lr * min_lr_factor across the plan's default_steps horizon.
enum class LrSchedule : uint32_t { kConstant = 0, kCosineWithWarmup = 1 };

struct LoRASpec {
  int64_t rank = 8;
  float alpha = 16.0f;
  uint64_t seed = 42;
  // Substring filters matched against the frozen weight's tensor name.
  // Empty => every eligible student MatMul is adapted.
  std::vector<std::string> target_filters;
};

struct OptimizerSpec {
  OptimizerKind kind = OptimizerKind::kAdamW;
  float lr = 1e-3f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  float weight_decay = 0.01f;
  // Per-tensor L2 gradient clipping applied before each optimizer step;
  // 0 disables (no clip instructions are emitted).
  float clip_norm = 0.0f;
  LrSchedule lr_schedule = LrSchedule::kConstant;
  uint64_t warmup_steps = 0;
  float min_lr_factor = 0.0f;  // cosine floor as a fraction of lr
};

struct UpdateConfig {
  int64_t batch = 32;
  LossKind loss = LossKind::kSoftmaxXEnt;
  float distill_weight = 0.5f;  // weight on the KL term for kXEntPlusKL
  float temperature = 2.0f;     // distillation softmax temperature
  LoRASpec lora;
  OptimizerSpec optimizer;
  uint64_t default_steps = 1000;
  // Quantize frozen base/teacher weights that feed MatMuls to per-tensor
  // symmetric int8 in the plan's rodata (QLoRA-style). Adapters, gradients,
  // and all activations stay f32; the source .smf on disk is untouched.
  bool quantize_base = false;
  // Fuse GEMM -> AddBias -> activation chains into flagged GEMM epilogues
  // wherever the SIR use-lists prove no other reader of the intermediates
  // (teacher subgraphs, unadapted layers). Bitwise-neutral by construction;
  // exposed so a fused and an unfused compilation of the same model can be
  // compared bit-for-bit.
  bool fuse_epilogues = true;
  // Test hook: when false, the plan contains forward+backward only (no
  // parameter mutation), enabling finite-difference gradient verification.
  bool emit_optimizer = true;
  // Fail-fast memory gate (ingressor resource analyzer): compilation is
  // rejected up front when the statically proven lower bound of the training
  // footprint exceeds this many bytes. 0 = detect the host's physical memory.
  uint64_t memory_budget_bytes = 0;
};

}  // namespace seeml::update

#endif  // SEEML_SOURCE_PLAN_CONFIG_H_
