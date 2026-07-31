#include "runtime/engine/contract.h"

#include "runtime/diagnostics/executing/error.h"
#include "runtime/diagnostics/feeding/error.h"
#include "runtime/diagnostics/persisting/error.h"
#include "runtime/diagnostics/validating/error.h"
#include "runtime/validator/plan_validator.h"

namespace seeml::update_rt {

namespace up = seeml::update;

namespace {

/// Every unit registered across the four runtime diagnostics process
/// modules. A message that cannot be attributed to one of these escaped
/// the partition.
constexpr std::string_view kRegisteredUnits[] = {
    diag::feeding::kDataset,       diag::feeding::kBatchPipeline,
    diag::validating::kPlanValidator, diag::executing::kEngine,
    diag::persisting::kDurableIo,  diag::persisting::kCheckpoint,
};

}  // namespace

bool WellFormedDiagnostic(std::string_view message) {
  for (std::string_view unit : kRegisteredUnits) {
    if (message.size() > unit.size() + 2 && message.starts_with(unit) &&
        message.substr(unit.size(), 2) == ": ")
      return true;
  }
  return false;
}

std::expected<void, std::string> VerifyPlanContract(
    const up::PlanHeader& header, uint64_t plan_size) {
  uint64_t train_bytes = 0, merge_bytes = 0, eval_bytes = 0, emit_bytes = 0;
  if (!MulOk(header.train_instr_count, sizeof(up::UpdateInstruction),
             &train_bytes) ||
      !MulOk(header.merge_instr_count, sizeof(up::UpdateInstruction),
             &merge_bytes) ||
      !MulOk(header.eval_instr_count, sizeof(up::UpdateInstruction),
             &eval_bytes) ||
      !MulOk(header.emit_count, sizeof(up::EmitEntry), &emit_bytes))
    return diag::executing::Error("plan section size overflows");
  if (!RangeOk(header.train_instr_offset, train_bytes, plan_size) ||
      !RangeOk(header.merge_instr_offset, merge_bytes, plan_size) ||
      !RangeOk(header.eval_instr_offset, eval_bytes, plan_size) ||
      !RangeOk(header.rodata_offset, header.rodata_size, plan_size) ||
      !RangeOk(header.persist_init_offset, header.persist_init_size,
               plan_size) ||
      !RangeOk(header.emit_table_offset, emit_bytes, plan_size))
    return diag::executing::Error("plan section out of bounds");

  // The arena is the target of the persistent image, the checkpoints, and
  // every arena ref — its size must dominate all of them.
  if (header.arena_size > UINT64_MAX - 63)
    return diag::executing::Error("arena size overflows");
  if (header.persist_init_size > header.arena_size ||
      header.persistent_size > header.arena_size)
    return diag::executing::Error("persistent segment exceeds arena");

  // Execution ceil-divides by the batch (Evaluate's pass count) and the
  // feeder stages batch-sized copies; zero would reach an integer division
  // by zero that no later guard checks — batch = 0 with input_floats = 0
  // passes every slot check below.
  if (header.batch == 0)
    return diag::executing::Error("plan batch must be nonzero");

  // The header's I/O slots are written by the data feeder each step and
  // read through typed pointers, so they must be in bounds and element-
  // aligned (labels are i32 or f32 — 4 bytes either way).
  uint64_t input_bytes = 0;
  if (up::IsRodataRef(header.input_ref) ||
      up::RefOffset(header.input_ref) % sizeof(float) != 0 ||
      !MulOk(header.input_floats, sizeof(float), &input_bytes) ||
      !RangeOk(up::RefOffset(header.input_ref), input_bytes,
               header.arena_size))
    return diag::executing::Error("plan input slot out of bounds");
  if (header.label_kind != 0 &&
      (up::IsRodataRef(header.label_ref) ||
       up::RefOffset(header.label_ref) % sizeof(float) != 0 ||
       !RangeOk(up::RefOffset(header.label_ref), header.label_bytes,
                header.arena_size)))
    return diag::executing::Error("plan label slot out of bounds");
  if (up::IsRodataRef(header.loss_ref) ||
      up::RefOffset(header.loss_ref) % sizeof(float) != 0 ||
      !RangeOk(up::RefOffset(header.loss_ref), sizeof(float),
               header.arena_size))
    return diag::executing::Error("plan loss slot out of bounds");
  return {};
}

std::expected<void, std::string> VerifyExecutorContract(
    std::span<const up::UpdateInstruction> train,
    std::span<const up::UpdateInstruction> merge,
    std::span<const up::UpdateInstruction> eval,
    std::span<const up::EmitEntry> emit_table, const up::PlanHeader& header) {
  for (const auto* program : {&train, &merge, &eval})
    for (const up::UpdateInstruction& ins : *program)
      if (auto r = ValidateInstruction(ins, header.arena_size,
                                       header.rodata_size);
          !r)
        return r;

  // The emit table's arena side is fixed at compile time; its file side is
  // validated against the actual model at commit. Commit reads the delta
  // through a float*, so the arena offset must be element-aligned too.
  for (const up::EmitEntry& e : emit_table) {
    if (!RangeOk(e.arena_offset, e.byte_size, header.arena_size))
      return diag::executing::Error("emit entry outside the arena");
    if (e.arena_offset % sizeof(float) != 0 ||
        e.byte_size % sizeof(float) != 0)
      return diag::executing::Error("emit entry misaligned");
  }
  return {};
}

std::expected<void, std::string> VerifyFeederContract(
    const up::PlanHeader& header, Dataset& data, uint64_t num_classes) {
  uint64_t expected_floats = 0;
  if (!MulOk(header.batch, data.input_dim(), &expected_floats) ||
      expected_floats != header.input_floats)
    return diag::executing::Error(
        "dataset input width does not match the compiled plan");
  if (header.label_kind != 0) {
    if (data.label_kind() != header.label_kind)
      return diag::executing::Error(
          "dataset label kind does not match the compiled plan");
    // Same kind is not enough: batch staging copies the dataset's per-sample
    // width into the plan's fixed label slot, so the widths must agree too.
    uint64_t batch_label_bytes = 0;
    if (!MulOk(header.batch, data.label_bytes_per_sample(),
               &batch_label_bytes) ||
        batch_label_bytes != header.label_bytes)
      return diag::executing::Error(
          "dataset label width does not match the compiled plan");
  }
  if (header.label_kind == 1)
    if (auto r = data.ValidateClassLabels(num_classes); !r)
      return std::unexpected(r.error());
  return {};
}

}  // namespace seeml::update_rt
