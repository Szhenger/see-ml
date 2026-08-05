#include "runtime/engine/update_engine.h"

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "runtime/custodian/checkpoint.h"
#include "runtime/custodian/durable_io.h"
#include "runtime/diagnostics/executing/error.h"
#include "runtime/engine/contract.h"
#include "runtime/executor/update_kernels.h"
#include "runtime/feeder/batch_pipeline.h"
#include "runtime/validator/plan_validator.h"
#include "source/identity/hash.h"
#include "source/parallel/parallel_for.h"

namespace seeml::update_rt {

namespace up = seeml::update;
namespace k = kernels;

namespace {

float BitsToF32(uint64_t bits) {
  return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

}  // namespace

UpdateEngine::~UpdateEngine() {
  std::free(arena_);
}

const float* UpdateEngine::ReadPtr(uint64_t ref) const {
  const uint64_t offset = up::RefOffset(ref);
  if (up::IsRodataRef(ref))
    return reinterpret_cast<const float*>(rodata_ + offset);
  return reinterpret_cast<const float*>(arena_ + offset);
}

const int8_t* UpdateEngine::ReadPtrQ8(uint64_t ref) const {
  // Validation pinned q8 sources to rodata; see ValidateInstruction.
  return reinterpret_cast<const int8_t*>(rodata_ + up::RefOffset(ref));
}

float* UpdateEngine::WritePtr(uint64_t ref) {
  // Lowering never emits a rodata destination; the frozen weights are
  // physically unwritable from the instruction stream by construction.
  return reinterpret_cast<float*>(arena_ + up::RefOffset(ref));
}

std::expected<void, std::string> UpdateEngine::LoadFromMemory(
    const uint8_t* plan, size_t size) {
  if (auto r = Initialize(plan, size); !r) return r;
  // The committed plan is borrowed; a previously file-owned plan is no
  // longer referenced by plan_/rodata_ and can be released.
  owned_plan_.clear();
  owned_plan_.shrink_to_fit();
  return {};
}

std::expected<void, std::string> UpdateEngine::LoadFromFile(
    const std::string& path) {
  auto bytes = ReadFileBytes(path);
  if (!bytes) return std::unexpected(bytes.error());
  // Initialize against the candidate buffer first: replacing owned_plan_ up
  // front would free the buffer a still-loaded previous plan's rodata_
  // points into even when the candidate is rejected. The move below keeps
  // the heap buffer (and the just-committed plan_/rodata_) stable.
  if (auto r = Initialize(bytes->data(), bytes->size()); !r) return r;
  owned_plan_ = std::move(*bytes);
  return {};
}

std::expected<void, std::string> UpdateEngine::Initialize(const uint8_t* plan,
                                                          size_t plan_size) {
  // Everything below validates into locals; engine members are only
  // assigned in the commit block at the end, after every contract has
  // passed. A failed re-Load must leave the previous plan fully usable —
  // not a half-overwritten mixture of two plans that the `if (!arena_)`
  // guards would happily execute.
  up::PlanHeader header{};
  if (plan_size < sizeof(header))
    return diag::executing::Error("plan smaller than its header");
  std::memcpy(&header, plan, sizeof(header));
  if (header.magic != up::kSeeuMagic)
    return diag::executing::Error("bad plan magic");
  // Version negotiation (schema.h): additive format bumps stay readable,
  // semantic breaks raise the floor, unknown-future formats are rejected.
  // Checked on the freshly decoded local header — the header_ member still
  // holds whatever plan was loaded before this call.
  if (header.version < up::kSeeuOldestReadable ||
      header.version > up::kSeeuVersion)
    return diag::executing::Error(
        "unsupported plan version " + std::to_string(header.version) +
        " (this runtime reads v" + std::to_string(up::kSeeuOldestReadable) +
        "..v" + std::to_string(up::kSeeuVersion) + ")");

  // Integrity: the plan hashes over itself with the hash field zeroed.
  // A flipped bit anywhere — header, instructions, frozen weights — fails
  // here instead of surfacing as silent numerical garbage on-device.
  {
    constexpr size_t kHashAt = offsetof(up::PlanHeader, plan_hash);
    if (up::PlanSelfHash(plan, plan_size, kHashAt) != header.plan_hash)
      return diag::executing::Error(
          "plan hash mismatch — the .seeu blob is corrupt");
  }

  // Plan boundary: the header must be self-consistent with the blob (and
  // its I/O slots with the arena) before any section is decoded. The
  // contract also proves every section size overflow-free, so the byte
  // counts below are plain multiplies.
  if (auto ok = VerifyPlanContract(header, plan_size); !ok)
    return std::unexpected(ok.error());
  const uint64_t train_bytes =
      header.train_instr_count * sizeof(up::UpdateInstruction);
  const uint64_t merge_bytes =
      header.merge_instr_count * sizeof(up::UpdateInstruction);
  const uint64_t eval_bytes =
      header.eval_instr_count * sizeof(up::UpdateInstruction);
  const uint64_t emit_bytes = header.emit_count * sizeof(up::EmitEntry);

  // Decode the instruction streams once; per-step execution touches only the
  // decoded vectors and the arena.
  std::vector<up::UpdateInstruction> train(header.train_instr_count);
  std::memcpy(train.data(), plan + header.train_instr_offset, train_bytes);
  std::vector<up::UpdateInstruction> merge(header.merge_instr_count);
  std::memcpy(merge.data(), plan + header.merge_instr_offset, merge_bytes);
  std::vector<up::UpdateInstruction> eval(header.eval_instr_count);
  std::memcpy(eval.data(), plan + header.eval_instr_offset, eval_bytes);
  std::vector<up::EmitEntry> emit(header.emit_count);
  std::memcpy(emit.data(), plan + header.emit_table_offset, emit_bytes);

  // Executor boundary: every operand ref of every instruction is
  // bounds-proven by the validator and every emit entry targets the arena —
  // after this, Execute() dispatches the programs blindly.
  if (auto ok = VerifyExecutorContract(train, merge, eval, emit, header); !ok)
    return std::unexpected(ok.error());

  // Softmax class-width discipline (the kernels index rows of this width
  // with raw dataset labels). Both executable label-consuming programs —
  // train and eval — must agree on the forward width, and a class-label
  // plan with no softmax at all would leave labels entirely unvalidated,
  // so it is rejected. Scanned on the freshly decoded local streams and
  // the local header — the members still describe whatever plan was loaded
  // before this call, and nothing may be committed until every contract
  // has passed.
  uint64_t fwd_classes = 0;
  for (const auto* program : {&train, &eval})
    for (const up::UpdateInstruction& ins : *program)
      if (static_cast<up::OpCode>(ins.opcode) ==
          up::OpCode::kSoftmaxXEntFwd) {
        if (fwd_classes != 0 && fwd_classes != ins.out[1])
          return diag::executing::Error(
              "plan softmax class widths disagree across programs");
        fwd_classes = ins.out[1];
      }
  if (header.label_kind == 1 && fwd_classes == 0)
    return diag::executing::Error(
        "class-label plan carries no softmax cross-entropy — its labels "
        "could never be validated");

  // The accuracy metric reads the probabilities the eval program's softmax
  // already materializes (its in[3] write); no extra compute or arena space
  // is spent. Plans without a class-label softmax report loss only.
  // Scanned on the freshly decoded local stream — like everything here,
  // committed to members only after every contract has passed.
  uint64_t eval_probs_ref = up::kNullRef;
  uint64_t eval_softmax_rows = 0;
  uint64_t eval_softmax_cols = 0;
  for (const up::UpdateInstruction& ins : eval)
    if (static_cast<up::OpCode>(ins.opcode) == up::OpCode::kSoftmaxXEntFwd) {
      eval_probs_ref = ins.in[3];
      eval_softmax_rows = ins.out[0];
      eval_softmax_cols = ins.out[1];
    }

  // Class count for validating class-index labels at Train() time. The raw
  // dataset labels feed every softmax kernel in every program, so the check
  // must hold for the narrowest width anywhere — not just the train
  // program's last softmax, which would leave a narrower eval softmax
  // indexing past its validated rows.
  uint64_t num_classes = 0;
  for (const auto* program : {&train, &merge, &eval})
    for (const up::UpdateInstruction& ins : *program) {
      const auto op = static_cast<up::OpCode>(ins.opcode);
      if (op != up::OpCode::kSoftmaxXEntFwd &&
          op != up::OpCode::kSoftmaxXEntBwd)
        continue;
      const uint64_t classes = ins.out[1];
      if (classes != 0 && (num_classes == 0 || classes < num_classes))
        num_classes = classes;
    }

  // The single allocation of the update: the pre-planned arena. Its size was
  // known at compile time — the device's resource contract.
  const size_t arena_bytes = (header.arena_size + 63) & ~size_t{63};
  uint8_t* arena = static_cast<uint8_t*>(std::aligned_alloc(64, arena_bytes));
  if (!arena) return diag::executing::Error("arena allocation failed");
  std::memset(arena, 0, arena_bytes);
  std::memcpy(arena, plan + header.persist_init_offset,
              header.persist_init_size);

  // Commit — nothing below can fail.
  std::free(arena_);
  arena_ = arena;
  header_ = header;
  train_program_ = std::move(train);
  merge_program_ = std::move(merge);
  eval_program_ = std::move(eval);
  emit_table_ = std::move(emit);
  num_classes_ = num_classes;
  eval_probs_ref_ = eval_probs_ref;
  eval_softmax_rows_ = eval_softmax_rows;
  eval_softmax_cols_ = eval_softmax_cols;
  plan_ = plan;
  plan_size_ = plan_size;
  rodata_ = plan + header.rodata_offset;
  step_ = 0;
  merged_ = false;
  return {};
}

void UpdateEngine::Execute(const std::vector<up::UpdateInstruction>& program) {
  for (const up::UpdateInstruction& ins : program) {
    switch (static_cast<up::OpCode>(ins.opcode)) {
      case up::OpCode::kNop:
        break;
      case up::OpCode::kGemmNN:
        // v5 epilogue flags: bias ref rides the otherwise-free in[3]; the
        // validator proved the flag/slot combination before dispatch.
        k::GemmNN(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), WritePtr(ins.in[2]),
                  ins.out[0], ins.out[1], ins.out[2],
                  ins.flags & up::kFlagEpilogueBias ? ReadPtr(ins.in[3])
                                                    : nullptr,
                  up::EpilogueActOf(ins.flags));
        break;
      case up::OpCode::kGemmNT:
        k::GemmNT(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), WritePtr(ins.in[2]),
                  ins.out[0], ins.out[1], ins.out[2]);
        break;
      case up::OpCode::kGemmTN:
        k::GemmTN(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), WritePtr(ins.in[2]),
                  ins.out[0], ins.out[1], ins.out[2]);
        break;
      case up::OpCode::kGemmAccNN:
        k::GemmAccNN(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]),
                     WritePtr(ins.in[2]), ins.out[0], ins.out[1], ins.out[2],
                     BitsToF32(ins.in[3]));
        break;
      case up::OpCode::kGemmNNQ8:
        k::GemmNNQ8(ReadPtr(ins.in[0]), ReadPtrQ8(ins.in[1]),
                    WritePtr(ins.in[2]), ins.out[0], ins.out[1], ins.out[2],
                    BitsToF32(ins.in[3]), up::EpilogueActOf(ins.flags));
        break;
      case up::OpCode::kGemmNTQ8:
        k::GemmNTQ8(ReadPtr(ins.in[0]), ReadPtrQ8(ins.in[1]),
                    WritePtr(ins.in[2]), ins.out[0], ins.out[1], ins.out[2],
                    BitsToF32(ins.in[3]));
        break;
      case up::OpCode::kAddEW:
        k::AddEW(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), WritePtr(ins.in[2]),
                 ins.out[0]);
        break;
      case up::OpCode::kMulEW:
        k::MulEW(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), WritePtr(ins.in[2]),
                 ins.out[0]);
        break;
      case up::OpCode::kAddBias:
        k::AddBias(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), WritePtr(ins.in[2]),
                   ins.out[0], ins.out[1]);
        break;
      case up::OpCode::kReluFwd:
        k::ReluFwd(ReadPtr(ins.in[0]), WritePtr(ins.in[1]), ins.out[0]);
        break;
      case up::OpCode::kReluBwd:
        k::ReluBwd(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), WritePtr(ins.in[2]),
                   ins.out[0]);
        break;
      case up::OpCode::kGeluFwd:
        k::GeluFwd(ReadPtr(ins.in[0]), WritePtr(ins.in[1]), ins.out[0]);
        break;
      case up::OpCode::kGeluBwd:
        k::GeluBwd(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), WritePtr(ins.in[2]),
                   ins.out[0]);
        break;
      case up::OpCode::kSiluFwd:
        k::SiluFwd(ReadPtr(ins.in[0]), WritePtr(ins.in[1]), ins.out[0]);
        break;
      case up::OpCode::kSiluBwd:
        k::SiluBwd(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), WritePtr(ins.in[2]),
                   ins.out[0]);
        break;
      case up::OpCode::kLayerNormFwd:
        k::LayerNormFwd(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]),
                        ReadPtr(ins.in[2]), WritePtr(ins.in[3]),
                        WritePtr(ins.out[1]), WritePtr(ins.out[2]),
                        ins.out[0] >> 32, ins.out[0] & 0xFFFFFFFFu);
        break;
      case up::OpCode::kLayerNormBwd:
        k::LayerNormBwd(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]),
                        ReadPtr(ins.in[2]), ReadPtr(ins.out[0]),
                        ReadPtr(ins.out[1]), WritePtr(ins.in[3]),
                        ins.out[2] >> 32, ins.out[2] & 0xFFFFFFFFu);
        break;
      case up::OpCode::kClipNorm:
        k::ClipNorm(WritePtr(ins.in[0]), ins.out[0], BitsToF32(ins.in[1]));
        break;
      case up::OpCode::kScale:
        k::Scale(ReadPtr(ins.in[0]), WritePtr(ins.in[1]), BitsToF32(ins.in[2]),
                 ins.out[0]);
        break;
      case up::OpCode::kReduceRows:
        k::ReduceRows(ReadPtr(ins.in[0]), WritePtr(ins.in[1]), ins.out[0],
                      ins.out[1]);
        break;
      case up::OpCode::kSoftmaxXEntFwd:
        k::SoftmaxXEntFwd(ReadPtr(ins.in[0]),
                          reinterpret_cast<const int32_t*>(ReadPtr(ins.in[1])),
                          WritePtr(ins.in[2]), WritePtr(ins.in[3]), ins.out[0],
                          ins.out[1]);
        break;
      case up::OpCode::kSoftmaxXEntBwd:
        k::SoftmaxXEntBwd(ReadPtr(ins.in[0]),
                          reinterpret_cast<const int32_t*>(ReadPtr(ins.in[1])),
                          ReadPtr(ins.in[2]), WritePtr(ins.in[3]), ins.out[0],
                          ins.out[1]);
        break;
      case up::OpCode::kMseFwd:
        k::MseFwd(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), WritePtr(ins.in[2]),
                  ins.out[0]);
        break;
      case up::OpCode::kMseBwd:
        k::MseBwd(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), ReadPtr(ins.in[2]),
                  WritePtr(ins.in[3]), ins.out[0]);
        break;
      case up::OpCode::kKLDistillFwd:
        k::KLDistillFwd(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]),
                        WritePtr(ins.in[2]), WritePtr(ins.in[3]),
                        WritePtr(ins.out[0]), ins.out[1] >> 32,
                        ins.out[1] & 0xFFFFFFFFu, BitsToF32(ins.out[2]));
        break;
      case up::OpCode::kKLDistillBwd:
        k::KLDistillBwd(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]),
                        ReadPtr(ins.in[2]), WritePtr(ins.in[3]),
                        ins.out[0] >> 32, ins.out[0] & 0xFFFFFFFFu,
                        BitsToF32(ins.out[1]));
        break;
      case up::OpCode::kSgdStep:
        k::SgdStep(WritePtr(ins.in[0]), ReadPtr(ins.in[1]), ins.out[0],
                   EffectiveLr(), header_.weight_decay);
        break;
      case up::OpCode::kAdamWStep:
        k::AdamWStep(WritePtr(ins.in[0]), ReadPtr(ins.in[1]),
                     WritePtr(ins.in[2]), WritePtr(ins.in[3]), ins.out[0],
                     EffectiveLr(), header_.beta1, header_.beta2, header_.eps,
                     header_.weight_decay, step_);
        break;
      case up::OpCode::kFill:
        k::Fill(WritePtr(ins.in[0]), BitsToF32(ins.in[1]), ins.out[0]);
        break;
      case up::OpCode::kCopy:
        k::Copy(ReadPtr(ins.in[0]), WritePtr(ins.in[1]), ins.out[0]);
        break;
    }
  }
}

float UpdateEngine::LossValue() const {
  return *ReadPtr(header_.loss_ref);
}

float UpdateEngine::EffectiveLr() const {
  const float base = header_.lr;
  if (static_cast<up::LrSchedule>(header_.lr_schedule) !=
      up::LrSchedule::kCosineWithWarmup)
    return base;
  // Linear warmup over warmup_steps, cosine decay to lr*min_lr_factor across
  // the plan's default_steps horizon; clamped at the floor beyond it.
  if (header_.warmup_steps > 0 && step_ <= header_.warmup_steps)
    return base * static_cast<float>(step_) /
           static_cast<float>(header_.warmup_steps);
  const uint64_t horizon = header_.default_steps > header_.warmup_steps
                               ? header_.default_steps - header_.warmup_steps
                               : 0;
  const float floor = base * header_.min_lr_factor;
  if (horizon == 0 || step_ - header_.warmup_steps >= horizon) return floor;
  const float t = static_cast<float>(step_ - header_.warmup_steps) /
                  static_cast<float>(horizon);
  constexpr float kPi = 3.14159265358979323846f;
  return floor + (base - floor) * 0.5f * (1.0f + std::cos(kPi * t));
}

void UpdateEngine::ExecuteTrainOnce() {
  if (step_ == 0) step_ = 1;  // AdamW bias correction is 1-indexed
  Execute(train_program_);
}

std::expected<void, std::string> UpdateEngine::ValidateDataset(
    Dataset& data) const {
  // Feeder boundary: the corpus must match the compiled plan's geometry.
  return VerifyFeederContract(header_, data, num_classes_);
}

std::expected<float, std::string> UpdateEngine::Evaluate(Dataset& data) {
  auto m = EvaluateMetrics(data);
  if (!m) return std::unexpected(m.error());
  return m->loss;
}

std::expected<EvalMetrics, std::string> UpdateEngine::EvaluateMetrics(
    Dataset& data) {
  if (!arena_) return diag::executing::Error("no plan loaded");
  if (eval_program_.empty())
    return diag::executing::Error("plan carries no eval program");
  if (auto r = ValidateDataset(data); !r) return std::unexpected(r.error());

  // Every evaluation of a given set must score the same sample sequence:
  // the validation gate compares a pre- and post-training Evaluate, and a
  // cursor left mid-set by the previous pass would make the two passes
  // weight different samples. Rewind alone is not enough for a shuffled
  // set — a pass whose final partial batch wraps advances the permutation
  // epoch, so the next pass would iterate a different order and
  // double-count a different wrapped tail. Snapshot-and-restore pins the
  // epoch too: the restore replays the permutation deterministically, so
  // every pass over this set scores the identical multiset.
  const Dataset::ServingPos entry_pos = data.SaveServingPos();
  data.Rewind();

  float* input_slot = WritePtr(header_.input_ref);
  uint8_t* label_slot =
      header_.label_kind == 0
          ? nullptr
          : reinterpret_cast<uint8_t*>(WritePtr(header_.label_ref));

  // Accuracy needs class labels staged in the label slot and a softmax in
  // the eval program whose row count covers the batch.
  const bool track_accuracy =
      header_.label_kind == 1 && label_slot != nullptr &&
      eval_probs_ref_ != up::kNullRef && eval_softmax_cols_ > 0 &&
      eval_softmax_rows_ >= header_.batch;

  // One pass over the set in compiled-batch chunks (final partial batch
  // wraps — the fixed-shape contract admits no ragged batch). The feeder
  // stages batch b+1 while batch b evaluates, exactly as TrainImpl
  // pipelines; the batch sequence is identical to the serial one, and the
  // eval program never mutates the dataset. Scoped so the feeder joins
  // before the caller sees the dataset single-threaded again.
  const uint64_t batches =
      std::max<uint64_t>(1, (data.num_samples() + header_.batch - 1) /
                                header_.batch);
  double total = 0.0;
  uint64_t correct = 0, counted = 0;
  {
    BatchPipeline feeder(data, header_.batch, header_.input_floats,
                         header_.label_kind == 0 ? 0 : header_.label_bytes);
    for (uint64_t b = 0; b < batches; ++b) {
      feeder.NextBatch(input_slot, label_slot);
      Execute(eval_program_);
      total += LossValue();
      if (track_accuracy) {
        // Rows past the dataset's tail in the final batch are wrapped
        // duplicates — the loss's fixed-shape mean cannot exclude them, but
        // accuracy can and does: each real sample is judged exactly once.
        // NextBatch copied the staged batch into the slots before Execute,
        // so label_slot still holds THIS batch while the feeder stages the
        // next one internally.
        const uint64_t served = b * header_.batch;
        const uint64_t real = std::min<uint64_t>(
            header_.batch, data.num_samples() > served
                               ? data.num_samples() - served
                               : 0);
        const float* probs = ReadPtr(eval_probs_ref_);
        const auto* labels = reinterpret_cast<const int32_t*>(label_slot);
        for (uint64_t r = 0; r < real; ++r) {
          const float* row = probs + r * eval_softmax_cols_;
          uint64_t argmax = 0;
          for (uint64_t c = 1; c < eval_softmax_cols_; ++c)
            if (row[c] > row[argmax]) argmax = c;
          if (labels[r] >= 0 && static_cast<uint64_t>(labels[r]) == argmax)
            ++correct;
        }
        counted += real;
      }
    }
  }
  data.RestoreServingPos(entry_pos);
  EvalMetrics m;
  m.loss = static_cast<float>(total / static_cast<double>(batches));
  if (track_accuracy && counted > 0) {
    m.accuracy = static_cast<float>(static_cast<double>(correct) /
                                    static_cast<double>(counted));
    m.has_accuracy = true;
  }
  return m;
}

std::expected<TrainReport, std::string> UpdateEngine::Train(
    Dataset& data, uint64_t steps, const TrainOptions& options) {
  auto report = TrainImpl(data, steps, options);
  // The diagnostics contract at the engine's main boundary: every error
  // must name a registered unit, or the failing process cannot be
  // delimited (mirrors the driver's Compile boundary).
  if (!report && !WellFormedDiagnostic(report.error()))
    return diag::executing::Error(
        "unattributed diagnostic escaped a subsystem: " + report.error());
  return report;
}

std::expected<TrainReport, std::string> UpdateEngine::TrainImpl(
    Dataset& data, uint64_t steps, const TrainOptions& options) {
  if (!arena_) return diag::executing::Error("no plan loaded");
  if (steps == 0) steps = header_.default_steps;
  if (steps == 0)
    return diag::executing::Error(
        "no steps requested and the plan has no default");
  if (auto r = ValidateDataset(data); !r) return std::unexpected(r.error());
  if (options.validation)
    if (auto r = ValidateDataset(*options.validation); !r)
      return std::unexpected(r.error());

  if (options.resume && !options.checkpoint_path.empty()) {
    if (auto r = LoadCheckpoint(options.checkpoint_path); !r)
      std::fprintf(stderr, "seeml-update: no checkpoint resumed (%s)\n",
                   r.error().c_str());
  }

  TrainReport report;
  if (options.validation) {
    auto v = EvaluateMetrics(*options.validation);
    if (!v) return std::unexpected(v.error());
    report.has_validation = true;
    report.val_initial_loss = v->loss;
    report.has_val_accuracy = v->has_accuracy;
    report.val_initial_accuracy = v->accuracy;
  }

  float* input_slot = WritePtr(header_.input_ref);
  uint8_t* label_slot =
      header_.label_kind == 0
          ? nullptr
          : reinterpret_cast<uint8_t*>(WritePtr(header_.label_ref));

  const uint64_t window = std::max<uint64_t>(1, std::min<uint64_t>(20, steps / 5));
  double first_sum = 0.0, last_sum = 0.0;
  uint64_t first_n = 0, last_n = 0;
  if (options.record_loss_curve) report.loss_curve.reserve(steps);

  const uint64_t start = step_;
  uint64_t executed = 0;
  // Training is about to move the adapter parameters, so any previously
  // materialized merge deltas are stale: commit must be preceded by a fresh
  // RunMerge. Cleared before the first step, not after the loop — the error
  // exits inside the loop (non-finite loss, checkpoint failure) leave
  // mutated parameters behind too.
  merged_ = false;
  {
    // The feeder thread stages batch s+1 (shuffle gather + epoch reshuffles)
    // while step s computes; the batch sequence is exactly the serial one, so
    // pipelining never changes what is trained on. Scoped to the loop: the
    // destructor joins the feeder on every exit path — including the error
    // returns below — and the surrounding Evaluate() calls see the dataset
    // single-threaded again.
    BatchPipeline feeder(data, header_.batch, header_.input_floats,
                         header_.label_kind == 0 ? 0 : header_.label_bytes);
    for (uint64_t s = start; s < start + steps; ++s) {
      if (options.should_stop && options.should_stop()) {
        report.stopped_early = true;
        break;
      }
      step_ = s + 1;  // 1-indexed timestep for AdamW bias correction
      feeder.NextBatch(input_slot, label_slot);
      Execute(train_program_);
      ++executed;

      const float loss = LossValue();
      // A non-finite loss means the parameters (and any AdamW moments) are
      // already poisoned; continuing can only burn energy. Fail the update —
      // the source model on disk is untouched by construction.
      if (!std::isfinite(loss))
        return diag::executing::Error(
            "loss became non-finite at step " +
            std::to_string(step_) + " — aborting the update");
      if (options.record_loss_curve) report.loss_curve.push_back(loss);
      if (s - start < window) {
        first_sum += loss;
        ++first_n;
      }
      if (s - start >= steps - window) {
        last_sum += loss;
        ++last_n;
      }
      if (options.log_every && (s - start) % options.log_every == 0)
        std::fprintf(stderr, "seeml-update: step %llu  loss %.6f\n",
                     static_cast<unsigned long long>(step_), loss);
      if (options.checkpoint_every && !options.checkpoint_path.empty() &&
          step_ % options.checkpoint_every == 0) {
        if (auto r = SaveCheckpoint(options.checkpoint_path); !r)
          return std::unexpected(r.error());
      }
    }
  }

  report.steps = executed;
  report.initial_avg_loss =
      first_n ? static_cast<float>(first_sum / first_n) : 0.0f;
  report.final_avg_loss =
      last_n ? static_cast<float>(last_sum / last_n) : report.initial_avg_loss;

  if (options.validation) {
    auto v = EvaluateMetrics(*options.validation);
    if (!v) return std::unexpected(v.error());
    report.val_final_loss = v->loss;
    report.val_final_accuracy = v->accuracy;
  }
  return report;
}

std::expected<void, std::string> UpdateEngine::RunMerge() {
  if (!arena_) return diag::executing::Error("no plan loaded");
  Execute(merge_program_);
  merged_ = true;
  return {};
}

std::expected<void, std::string> UpdateEngine::CommitToModel(
    const std::string& source_model_path, const std::string& out_path) const {
  if (!merged_)
    return diag::executing::Error("RunMerge() must precede commit");

  // Serialize committers: without the lock, two updates targeting the same
  // output race at the atomic rename and the loser's work silently
  // disappears. The lock dies with the process, so a crash cannot wedge
  // future commits.
  auto lock = CommitLock::Acquire(out_path);
  if (!lock) return std::unexpected(lock.error());

  // Copy-on-write sidecar first, then verify the identity of the *copy* —
  // the exact bytes about to be patched — so no window exists between the
  // check and the patch. The whole path streams in bounded chunks: commit
  // memory is O(chunk), never O(model).
  auto edit = DurableFileEdit::Begin(source_model_path, out_path);
  if (!edit) return std::unexpected(edit.error());

  // Identity check before any byte moves: the emit table's offsets are only
  // meaningful inside the exact file the plan was compiled from. A same-sized
  // different model would otherwise be silently corrupted.
  if (header_.source_model_hash != 0) {
    auto hash = HashFileContent(edit->sidecar_path());
    if (!hash) return std::unexpected(hash.error());
    if (*hash != header_.source_model_hash)
      return diag::executing::Error(
          "source model does not match the plan's "
          "source_model_hash — refusing to patch '" +
          source_model_path + "'");
  }

  // Apply every adapter delta to its f32 weight range: W' = W + Δ. The
  // file's pristine weights are the base, so quantized plans commit no
  // quantization error. Read-modify-write in bounded chunks; the buffer is
  // float-typed, so alignment holds regardless of the file offset.
  constexpr uint64_t kPatchChunkFloats = 1u << 16;  // 256 KiB per chunk
  std::vector<float> buf;
  for (const up::EmitEntry& e : emit_table_) {
    if (!RangeOk(e.smf_data_offset, e.byte_size, edit->size()) ||
        e.byte_size % sizeof(float) != 0)
      return diag::executing::Error(
          "emit entry exceeds the source model file — plan and "
          "model are out of sync");
    const auto* delta =
        reinterpret_cast<const float*>(arena_ + e.arena_offset);
    const uint64_t count = e.byte_size / sizeof(float);
    for (uint64_t i = 0; i < count; i += kPatchChunkFloats) {
      const uint64_t n = std::min<uint64_t>(kPatchChunkFloats, count - i);
      buf.resize(n);
      const uint64_t at = e.smf_data_offset + i * sizeof(float);
      if (auto r = edit->ReadAt(at, reinterpret_cast<uint8_t*>(buf.data()),
                                n * sizeof(float));
          !r)
        return std::unexpected(r.error());
      for (uint64_t j = 0; j < n; ++j) buf[j] += delta[i + j];
      if (auto r = edit->WriteAt(at, reinterpret_cast<const uint8_t*>(
                                         buf.data()),
                                 n * sizeof(float));
          !r)
        return std::unexpected(r.error());
    }
  }

  // Transactional and durable: fsync'd sidecar, atomic rename, dir fsync.
  return edit->Commit();
}

std::expected<void, std::string> UpdateEngine::SaveCheckpoint(
    const std::string& path) const {
  return SaveCheckpointFile(path, header_.plan_hash, step_, arena_,
                            header_.persistent_size);
}

std::expected<void, std::string> UpdateEngine::LoadCheckpoint(
    const std::string& path) {
  auto step = LoadCheckpointFile(path, header_.plan_hash,
                                 header_.persistent_size, arena_);
  if (!step) return std::unexpected(step.error());
  step_ = *step;
  // The restored persistent segment replaces the adapter state any earlier
  // RunMerge materialized deltas from; committing those would patch deltas
  // that no longer match the parameters.
  merged_ = false;
  return {};
}

}  // namespace seeml::update_rt
