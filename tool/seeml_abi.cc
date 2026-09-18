// =============================================================================
// seeml-abi — prints the ABI manifest: every constant, enum value and packed
// layout of the four byte formats the two planes exchange (P6, #86).
//
//   seeml-abi            > tool/seeml/abi.json
//
// The Python plane reads and writes SMF, SDS, SEEU and SEKP; until this
// tool, each of those contracts was two hand-maintained copies and a
// developer remembering to edit both. The manifest is emitted FROM the C++
// definitions — sizeof, offsetof, the enumerators themselves — committed as
// tool/seeml/abi.json, and checked twice: a Python test asserts that
// tool/seeml/formats.py agrees with it field for field, and CI regenerates
// it and fails on a diff. So a format change that forgets the other plane
// cannot merge. No arguments; deterministic output (no host facts).
// =============================================================================

#include <bit>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>

#include "runtime/custodian/checkpoint_format.h"
#include "runtime/feeder/dataset.h"
#include "source/identity/version.h"
#include "source/language/model_format.h"
#include "source/plan/opcode_names.h"
#include "source/plan/update_types.h"
#include "tool/probe_trace.h"

namespace {

namespace up = seeml::update;
namespace rt = seeml::update_rt;

bool g_first = true;
void Field(const char* name, size_t offset, size_t size) {
  std::printf("%s\n        {\"name\": \"%s\", \"offset\": %zu, \"size\": %zu}",
              g_first ? "" : ",", name, offset, size);
  g_first = false;
}

#define SEEML_FIELD(T, f) Field(#f, offsetof(T, f), sizeof(T::f))

// Every field of a packed struct, and — the point — a proof that the list
// is complete: the fields' sizes must add up to the struct's.
#define PLAN_HEADER_FIELDS(X)                                                  \
  X(magic) X(version) X(arena_size) X(persistent_size) X(input_ref)           \
  X(input_floats) X(label_ref) X(label_bytes) X(label_kind) X(optimizer_kind) \
  X(loss_ref) X(train_instr_offset) X(train_instr_count)                      \
  X(merge_instr_offset) X(merge_instr_count) X(rodata_offset) X(rodata_size)  \
  X(persist_init_offset) X(persist_init_size) X(emit_table_offset)            \
  X(emit_count) X(lr) X(beta1) X(beta2) X(eps) X(weight_decay)                \
  X(gemm_tile_k) X(batch) X(default_steps) X(eval_instr_offset)               \
  X(eval_instr_count) X(source_model_hash) X(plan_hash) X(lr_schedule)        \
  X(gemm_tile_n) X(warmup_steps) X(min_lr_factor) X(clip_norm) X(input_kind)  \
  X(grad_accum_steps) X(seq_len) X(step_instr_offset) X(step_instr_count)
#define INSTRUCTION_FIELDS(X) X(opcode) X(flags) X(imm) X(in) X(out)
#define EMIT_ENTRY_FIELDS(X) X(smf_data_offset) X(byte_size) X(arena_offset)
#define CKPT_FIELDS(X) \
  X(magic) X(version) X(plan_hash) X(step) X(persistent_size) X(payload_hash)

#define SIZE_OF(T, f) +sizeof(T::f)
#define SIZE_OF_PlanHeader(f) SIZE_OF(up::PlanHeader, f)
#define SIZE_OF_UpdateInstruction(f) SIZE_OF(up::UpdateInstruction, f)
#define SIZE_OF_EmitEntry(f) SIZE_OF(up::EmitEntry, f)
#define SIZE_OF_CkptHeader(f) SIZE_OF(rt::CkptHeader, f)
static_assert(0 PLAN_HEADER_FIELDS(SIZE_OF_PlanHeader) == sizeof(up::PlanHeader),
              "PlanHeader: the manifest's field list is missing a field");
static_assert(0 INSTRUCTION_FIELDS(SIZE_OF_UpdateInstruction) ==
                  sizeof(up::UpdateInstruction),
              "UpdateInstruction: the manifest's field list is incomplete");
static_assert(0 EMIT_ENTRY_FIELDS(SIZE_OF_EmitEntry) == sizeof(up::EmitEntry),
              "EmitEntry: the manifest's field list is incomplete");
static_assert(0 CKPT_FIELDS(SIZE_OF_CkptHeader) == sizeof(rt::CkptHeader),
              "CkptHeader: the manifest's field list is incomplete");

template <typename Emit>
void Struct(const char* name, size_t size, Emit&& emit, bool last = false) {
  std::printf("      \"%s\": {\"size\": %zu, \"fields\": [", name, size);
  g_first = true;
  emit();
  std::printf("\n      ]}%s\n", last ? "" : ",");
}

struct Named {
  const char* name;
  unsigned value;
};

void Enum(const char* key, const Named* values, size_t count, bool last) {
  std::printf("    \"%s\": {", key);
  for (size_t i = 0; i < count; ++i)
    std::printf("%s\"%s\": %u", i ? ", " : "", values[i].name,
                values[i].value);
  std::printf("}%s\n", last ? "" : ",");
}

}  // namespace

int main(int argc, char**) {
  if (argc != 1) {
    std::fprintf(stderr, "seeml-abi: takes no arguments\n");
    return 2;
  }
  using K = up::SmfOpKind;
  const Named smf_ops[] = {
      {"matmul", unsigned(K::kMatMul)},       {"add_bias", unsigned(K::kAddBias)},
      {"relu", unsigned(K::kRelu)},           {"gelu", unsigned(K::kGelu)},
      {"silu", unsigned(K::kSilu)},           {"mul", unsigned(K::kMul)},
      {"layer_norm", unsigned(K::kLayerNorm)}, {"add", unsigned(K::kAdd)},
      {"rms_norm", unsigned(K::kRmsNorm)},    {"rope", unsigned(K::kRope)},
      {"attention", unsigned(K::kAttention)}, {"embedding", unsigned(K::kEmbedding)},
  };
  static_assert(sizeof(smf_ops) / sizeof(smf_ops[0]) == up::kSmfOpKindMax + 1,
                "an SMF op kind is missing from the manifest");
  using F = up::FusedStage;
  const Named fused[] = {{"end", unsigned(F::kEnd)},   {"add", unsigned(F::kAdd)},
                         {"mul", unsigned(F::kMul)},   {"scale", unsigned(F::kScale)},
                         {"relu", unsigned(F::kRelu)}, {"gelu", unsigned(F::kGelu)},
                         {"silu", unsigned(F::kSilu)}};

  std::printf("{\n  \"schema\": 1,\n  \"generator\": \"tool/seeml_abi.cc\",\n");
  std::printf("  \"smf\": {\n    \"magic\": %u, \"version\": %u, "
              "\"min_version\": %u, \"default_rope_base\": %g,\n",
              up::kSmfMagic, up::kSmfVersion, up::kSmfMinVersion,
              static_cast<double>(up::kSmfDefaultRopeBase));
  Enum("op_kinds", smf_ops, sizeof(smf_ops) / sizeof(smf_ops[0]), true);
  std::printf("  },\n  \"sds\": {\"magic\": %u, \"version\": %u, "
              "\"min_version\": %u, \"header_bytes\": %llu, "
              "\"label_kind_max\": %u},\n",
              rt::kSdsMagic, rt::kSdsVersion, rt::kSdsMinVersion,
              static_cast<unsigned long long>(rt::kSdsHeaderBytes),
              rt::kSdsLabelKindMax);

  std::printf("  \"seeu\": {\n    \"magic\": %u, \"version\": %u, "
              "\"oldest_readable\": %u,\n    \"rodata_bit\": %d, "
              "\"source_bit\": %d, "
              "\"rodata_alignment\": %llu, \"gemm_panel_floats\": %llu,\n",
              up::kSeeuMagic, up::kSeeuVersion, up::kSeeuOldestReadable,
              std::countr_zero(up::kRodataBit), std::countr_zero(up::kSourceBit),
              static_cast<unsigned long long>(up::kSeeuRodataAlignment),
              static_cast<unsigned long long>(up::kDefaultGemmPanelFloats));
  std::printf("    \"flags\": {\"epilogue_bias\": %u, \"epilogue_act_shift\": "
              "%u, \"epilogue_act_mask\": %u, \"gemm_addend\": %u, "
              "\"q8_col_scale\": %u},\n",
              unsigned(up::kFlagEpilogueBias),
              unsigned(up::kFlagEpilogueActShift),
              unsigned(up::kFlagEpilogueActMask),
              unsigned(up::kFlagGemmAddend),
              unsigned(up::kFlagQ8ColScale));
  Enum("fused_stages", fused, sizeof(fused) / sizeof(fused[0]), false);
  std::printf("    \"fused_stage\": {\"kind_mask\": %u, \"arg_shift\": %u, "
              "\"arg_mask\": %u, \"run_is_right\": %u, \"max_stages\": %zu},\n",
              unsigned(up::kFusedStageKindMask),
              unsigned(up::kFusedStageArgShift),
              unsigned(up::kFusedStageArgMask),
              unsigned(up::kFusedStageRunIsRight), up::kFusedMapMaxStages);
  std::printf("    \"opcodes\": {");
  for (unsigned op = 0; op <= up::kOpCodeMax; ++op)
    std::printf("%s\"%s\": %u", op ? ", " : "",
                up::OpCodeName(static_cast<uint16_t>(op)), op);
  std::printf("},\n    \"structs\": {\n");
#define X(f) SEEML_FIELD(up::PlanHeader, f);
  Struct("PlanHeader", sizeof(up::PlanHeader), [] { PLAN_HEADER_FIELDS(X) });
#undef X
#define X(f) SEEML_FIELD(up::UpdateInstruction, f);
  Struct("UpdateInstruction", sizeof(up::UpdateInstruction),
         [] { INSTRUCTION_FIELDS(X) });
#undef X
#define X(f) SEEML_FIELD(up::EmitEntry, f);
  Struct("EmitEntry", sizeof(up::EmitEntry), [] { EMIT_ENTRY_FIELDS(X) }, true);
#undef X
  std::printf("    }\n  },\n");

  std::printf("  \"sekp\": {\n    \"magic\": %u, \"version\": %u, "
              "\"oldest_readable\": %u,\n    \"flags\": {\"has_val_initial\": "
              "%u, \"has_accuracy\": %u, \"has_best_payload\": %u, "
              "\"has_best\": %u},\n    \"structs\": {\n",
              rt::kCkptMagic, rt::kCkptVersion, rt::kCkptOldestReadable,
              rt::kCkptHasValInitial, rt::kCkptHasAccuracy,
              rt::kCkptHasBestPayload, rt::kCkptHasBest);
#define X(f) SEEML_FIELD(rt::CkptHeader, f);
  Struct("CkptHeader", sizeof(rt::CkptHeader), [] { CKPT_FIELDS(X) });
#undef X
  Struct("CkptHeaderV4Tail", sizeof(rt::CkptHeaderV4Tail), [] {
    SEEML_FIELD(rt::CkptHeaderV4Tail, horizon_steps);
  });
#define CKPT_V5_FIELDS(X)                                                  \
  X(shuffle_origin) X(train_samples) X(val_samples) X(best_step)           \
  X(best_payload_hash) X(flags) X(val_initial_loss_bits)                   \
  X(val_initial_accuracy_bits) X(best_loss_bits) X(best_accuracy_bits)     \
  X(stale_evals)
#define SIZE_OF_V5(f) +sizeof(rt::CkptHeaderV5Tail::f)
  static_assert(0 CKPT_V5_FIELDS(SIZE_OF_V5) == sizeof(rt::CkptHeaderV5Tail),
                "a CkptHeaderV5Tail field is missing from the manifest");
#undef SIZE_OF_V5
#define X(f) SEEML_FIELD(rt::CkptHeaderV5Tail, f);
  Struct("CkptHeaderV5Tail", sizeof(rt::CkptHeaderV5Tail),
         [] { CKPT_V5_FIELDS(X) }, true);
#undef X
  std::printf("    }\n  },\n");

  std::printf("  \"probe_trace\": {\"magic\": %u, \"version\": %u}\n}\n",
              seeml::tool::kProbeTraceMagic, seeml::tool::kProbeTraceVersion);
  return std::ferror(stdout) ? 1 : 0;
}
