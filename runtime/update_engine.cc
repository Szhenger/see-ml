#include "runtime/update_engine.h"

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "runtime/update_kernels.h"

namespace seeml::update_rt {

namespace up = seeml::update;
namespace k = kernels;

namespace {

float BitsToF32(uint64_t bits) {
  return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

// --- Overflow-safe validation of file-supplied offsets and sizes -------------

bool MulOk(uint64_t a, uint64_t b, uint64_t* out) {
  if (b != 0 && a > UINT64_MAX / b) return false;
  *out = a * b;
  return true;
}

bool RangeOk(uint64_t off, uint64_t bytes, uint64_t size) {
  return off <= size && bytes <= size - off;
}

// Checks that every ref operand of `ins` lies fully inside its address space
// (arena or rodata), with byte extents derived from the instruction's dims the
// same way the kernels derive their loop bounds, and that writes only target
// the mutable arena. Rejects unknown opcodes, which Execute() would silently
// skip.
std::expected<void, std::string> ValidateInstruction(
    const up::UpdateInstruction& ins, uint64_t arena_size,
    uint64_t rodata_size) {
  auto ref_ok = [&](uint64_t ref, uint64_t elems, bool write) {
    if (ref == up::kNullRef) return false;
    if (write && up::IsRodataRef(ref)) return false;
    uint64_t bytes = 0;
    if (!MulOk(elems, sizeof(float), &bytes)) return false;  // i32 == f32 width
    const uint64_t space = up::IsRodataRef(ref) ? rodata_size : arena_size;
    return RangeOk(up::RefOffset(ref), bytes, space);
  };
  auto fail = [&] {
    return std::unexpected("UpdateEngine: instruction operand out of bounds "
                           "(opcode " +
                           std::to_string(ins.opcode) + ")");
  };

  const uint64_t d0 = ins.out[0], d1 = ins.out[1], d2 = ins.out[2];
  uint64_t mk = 0, kn = 0, mn = 0, nc = 0;
  switch (static_cast<up::OpCode>(ins.opcode)) {
    case up::OpCode::kNop:
      return {};
    case up::OpCode::kGemmNN:
    case up::OpCode::kGemmNT:
    case up::OpCode::kGemmTN:
    case up::OpCode::kGemmAccNN:
      // Every layout variant reads M*K (A) and K*N (B), writes M*N (C).
      if (!MulOk(d0, d2, &mk) || !MulOk(d2, d1, &kn) || !MulOk(d0, d1, &mn))
        return fail();
      if (!ref_ok(ins.in[0], mk, false) || !ref_ok(ins.in[1], kn, false) ||
          !ref_ok(ins.in[2], mn, true))
        return fail();
      return {};
    case up::OpCode::kAddEW:
    case up::OpCode::kReluBwd:
      if (!ref_ok(ins.in[0], d0, false) || !ref_ok(ins.in[1], d0, false) ||
          !ref_ok(ins.in[2], d0, true))
        return fail();
      return {};
    case up::OpCode::kAddBias:
      if (!MulOk(d0, d1, &mn)) return fail();
      if (!ref_ok(ins.in[0], mn, false) || !ref_ok(ins.in[1], d1, false) ||
          !ref_ok(ins.in[2], mn, true))
        return fail();
      return {};
    case up::OpCode::kReluFwd:
    case up::OpCode::kScale:
    case up::OpCode::kCopy:
      if (!ref_ok(ins.in[0], d0, false) || !ref_ok(ins.in[1], d0, true))
        return fail();
      return {};
    case up::OpCode::kReduceRows:
      if (!MulOk(d0, d1, &mn)) return fail();
      if (!ref_ok(ins.in[0], mn, false) || !ref_ok(ins.in[1], d1, true))
        return fail();
      return {};
    case up::OpCode::kSoftmaxXEntFwd:
      if (!MulOk(d0, d1, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], d0, false) ||
          !ref_ok(ins.in[2], 1, true) || !ref_ok(ins.in[3], nc, true))
        return fail();
      return {};
    case up::OpCode::kSoftmaxXEntBwd:
      if (!MulOk(d0, d1, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], d0, false) ||
          !ref_ok(ins.in[2], 1, false) || !ref_ok(ins.in[3], nc, true))
        return fail();
      return {};
    case up::OpCode::kMseFwd:
      if (!ref_ok(ins.in[0], d0, false) || !ref_ok(ins.in[1], d0, false) ||
          !ref_ok(ins.in[2], 1, true))
        return fail();
      return {};
    case up::OpCode::kMseBwd:
      if (!ref_ok(ins.in[0], d0, false) || !ref_ok(ins.in[1], d0, false) ||
          !ref_ok(ins.in[2], 1, false) || !ref_ok(ins.in[3], d0, true))
        return fail();
      return {};
    case up::OpCode::kKLDistillFwd:
      if (!MulOk(d1 >> 32, d1 & 0xFFFFFFFFu, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], nc, false) ||
          !ref_ok(ins.in[2], 1, true) || !ref_ok(ins.in[3], nc, true) ||
          !ref_ok(ins.out[0], nc, true))
        return fail();
      return {};
    case up::OpCode::kKLDistillBwd:
      if (!MulOk(d0 >> 32, d0 & 0xFFFFFFFFu, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], nc, false) ||
          !ref_ok(ins.in[2], 1, false) || !ref_ok(ins.in[3], nc, true))
        return fail();
      return {};
    case up::OpCode::kSgdStep:
      if (!ref_ok(ins.in[0], d0, true) || !ref_ok(ins.in[1], d0, false))
        return fail();
      return {};
    case up::OpCode::kAdamWStep:
      if (!ref_ok(ins.in[0], d0, true) || !ref_ok(ins.in[1], d0, false) ||
          !ref_ok(ins.in[2], d0, true) || !ref_ok(ins.in[3], d0, true))
        return fail();
      return {};
    case up::OpCode::kFill:
      if (!ref_ok(ins.in[0], d0, true)) return fail();
      return {};
  }
  return std::unexpected("UpdateEngine: unknown opcode " +
                         std::to_string(ins.opcode));
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

float* UpdateEngine::WritePtr(uint64_t ref) {
  // Lowering never emits a rodata destination; the frozen weights are
  // physically unwritable from the instruction stream by construction.
  return reinterpret_cast<float*>(arena_ + up::RefOffset(ref));
}

std::expected<void, std::string> UpdateEngine::LoadFromMemory(
    const uint8_t* plan, size_t size) {
  plan_ = plan;
  plan_size_ = size;
  return Initialize();
}

std::expected<void, std::string> UpdateEngine::LoadFromFile(
    const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return std::unexpected("UpdateEngine: cannot open '" + path + "'");
  owned_plan_.assign((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
  plan_ = owned_plan_.data();
  plan_size_ = owned_plan_.size();
  return Initialize();
}

std::expected<void, std::string> UpdateEngine::Initialize() {
  if (plan_size_ < sizeof(up::PlanHeader))
    return std::unexpected("UpdateEngine: plan smaller than its header");
  std::memcpy(&header_, plan_, sizeof(header_));
  if (header_.magic != up::kSeeuMagic)
    return std::unexpected("UpdateEngine: bad plan magic");
  if (header_.version != up::kSeeuVersion)
    return std::unexpected("UpdateEngine: unsupported plan version");

  uint64_t train_bytes = 0, merge_bytes = 0, emit_bytes = 0;
  if (!MulOk(header_.train_instr_count, sizeof(up::UpdateInstruction),
             &train_bytes) ||
      !MulOk(header_.merge_instr_count, sizeof(up::UpdateInstruction),
             &merge_bytes) ||
      !MulOk(header_.emit_count, sizeof(up::EmitEntry), &emit_bytes))
    return std::unexpected("UpdateEngine: plan section size overflows");
  if (!RangeOk(header_.train_instr_offset, train_bytes, plan_size_) ||
      !RangeOk(header_.merge_instr_offset, merge_bytes, plan_size_) ||
      !RangeOk(header_.rodata_offset, header_.rodata_size, plan_size_) ||
      !RangeOk(header_.persist_init_offset, header_.persist_init_size,
               plan_size_) ||
      !RangeOk(header_.emit_table_offset, emit_bytes, plan_size_))
    return std::unexpected("UpdateEngine: plan section out of bounds");

  // The arena is the target of the persistent image, the checkpoints, and
  // every arena ref below — its size must dominate all of them.
  if (header_.arena_size > UINT64_MAX - 63)
    return std::unexpected("UpdateEngine: arena size overflows");
  if (header_.persist_init_size > header_.arena_size ||
      header_.persistent_size > header_.arena_size)
    return std::unexpected("UpdateEngine: persistent segment exceeds arena");

  // Decode the instruction streams once; per-step execution touches only the
  // decoded vectors and the arena.
  train_program_.resize(header_.train_instr_count);
  std::memcpy(train_program_.data(), plan_ + header_.train_instr_offset,
              train_bytes);
  merge_program_.resize(header_.merge_instr_count);
  std::memcpy(merge_program_.data(), plan_ + header_.merge_instr_offset,
              merge_bytes);
  emit_table_.resize(header_.emit_count);
  std::memcpy(emit_table_.data(), plan_ + header_.emit_table_offset,
              emit_bytes);

  // Validate every operand ref of every instruction against the address
  // space it targets — after this, Execute() can trust the programs blindly.
  for (const up::UpdateInstruction& ins : train_program_)
    if (auto r = ValidateInstruction(ins, header_.arena_size,
                                     header_.rodata_size);
        !r)
      return r;
  for (const up::UpdateInstruction& ins : merge_program_)
    if (auto r = ValidateInstruction(ins, header_.arena_size,
                                     header_.rodata_size);
        !r)
      return r;

  // The emit table's arena side is fixed at compile time; its file side is
  // validated against the actual model in CommitToModel().
  for (const up::EmitEntry& e : emit_table_)
    if (!RangeOk(e.arena_offset, e.byte_size, header_.arena_size))
      return std::unexpected("UpdateEngine: emit entry outside the arena");

  // The header's I/O slots are written by the data feeder each step.
  uint64_t input_bytes = 0;
  if (up::IsRodataRef(header_.input_ref) ||
      !MulOk(header_.input_floats, sizeof(float), &input_bytes) ||
      !RangeOk(up::RefOffset(header_.input_ref), input_bytes,
               header_.arena_size))
    return std::unexpected("UpdateEngine: plan input slot out of bounds");
  if (header_.label_kind != 0 &&
      (up::IsRodataRef(header_.label_ref) ||
       !RangeOk(up::RefOffset(header_.label_ref), header_.label_bytes,
                header_.arena_size)))
    return std::unexpected("UpdateEngine: plan label slot out of bounds");
  if (up::IsRodataRef(header_.loss_ref) ||
      !RangeOk(up::RefOffset(header_.loss_ref), sizeof(float),
               header_.arena_size))
    return std::unexpected("UpdateEngine: plan loss slot out of bounds");

  // Class count for validating class-index labels at Train() time (the
  // softmax kernels index rows of this width with raw dataset labels).
  num_classes_ = 0;
  for (const up::UpdateInstruction& ins : train_program_)
    if (static_cast<up::OpCode>(ins.opcode) == up::OpCode::kSoftmaxXEntFwd)
      num_classes_ = ins.out[1];

  rodata_ = plan_ + header_.rodata_offset;

  // The single allocation of the update: the pre-planned arena. Its size was
  // known at compile time — the device's resource contract.
  std::free(arena_);
  const size_t arena_bytes = (header_.arena_size + 63) & ~size_t{63};
  arena_ = static_cast<uint8_t*>(std::aligned_alloc(64, arena_bytes));
  if (!arena_) return std::unexpected("UpdateEngine: arena allocation failed");
  std::memset(arena_, 0, arena_bytes);
  std::memcpy(arena_, plan_ + header_.persist_init_offset,
              header_.persist_init_size);
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
        k::GemmNN(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), WritePtr(ins.in[2]),
                  ins.out[0], ins.out[1], ins.out[2]);
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
      case up::OpCode::kAddEW:
        k::AddEW(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), WritePtr(ins.in[2]),
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
                   header_.lr, header_.weight_decay);
        break;
      case up::OpCode::kAdamWStep:
        k::AdamWStep(WritePtr(ins.in[0]), ReadPtr(ins.in[1]),
                     WritePtr(ins.in[2]), WritePtr(ins.in[3]), ins.out[0],
                     header_.lr, header_.beta1, header_.beta2, header_.eps,
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

void UpdateEngine::ExecuteTrainOnce() {
  if (step_ == 0) step_ = 1;  // AdamW bias correction is 1-indexed
  Execute(train_program_);
}

std::expected<TrainReport, std::string> UpdateEngine::Train(
    Dataset& data, uint64_t steps, const TrainOptions& options) {
  if (!arena_) return std::unexpected("UpdateEngine: no plan loaded");
  if (steps == 0) steps = header_.default_steps;
  if (steps == 0)
    return std::unexpected(
        "UpdateEngine: no steps requested and the plan has no default");

  uint64_t expected_floats = 0;
  if (!MulOk(header_.batch, data.input_dim(), &expected_floats) ||
      expected_floats != header_.input_floats)
    return std::unexpected(
        "UpdateEngine: dataset input width does not match the compiled plan");
  if (header_.label_kind != 0) {
    if (data.label_kind() != header_.label_kind)
      return std::unexpected(
          "UpdateEngine: dataset label kind does not match the compiled plan");
    // Same kind is not enough: FillBatch copies the dataset's per-sample
    // width into the plan's fixed label slot, so the widths must agree too.
    uint64_t batch_label_bytes = 0;
    if (!MulOk(header_.batch, data.label_bytes_per_sample(),
               &batch_label_bytes) ||
        batch_label_bytes != header_.label_bytes)
      return std::unexpected(
          "UpdateEngine: dataset label width does not match the compiled plan");
  }
  if (header_.label_kind == 1)
    if (auto r = data.ValidateClassLabels(num_classes_); !r)
      return std::unexpected(r.error());

  if (options.resume && !options.checkpoint_path.empty()) {
    if (auto r = LoadCheckpoint(options.checkpoint_path); !r)
      std::fprintf(stderr, "seeml-update: no checkpoint resumed (%s)\n",
                   r.error().c_str());
  }

  float* input_slot = WritePtr(header_.input_ref);
  uint8_t* label_slot =
      header_.label_kind == 0
          ? nullptr
          : reinterpret_cast<uint8_t*>(WritePtr(header_.label_ref));

  const uint64_t window = std::max<uint64_t>(1, std::min<uint64_t>(20, steps / 5));
  double first_sum = 0.0, last_sum = 0.0;
  uint64_t first_n = 0, last_n = 0;

  TrainReport report;
  const uint64_t start = step_;
  for (uint64_t s = start; s < start + steps; ++s) {
    step_ = s + 1;  // 1-indexed timestep for AdamW bias correction
    data.FillBatch(header_.batch, input_slot, label_slot);
    Execute(train_program_);

    const float loss = LossValue();
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

  report.steps = steps;
  report.initial_avg_loss = static_cast<float>(first_sum / first_n);
  report.final_avg_loss = static_cast<float>(last_sum / last_n);
  return report;
}

std::expected<void, std::string> UpdateEngine::RunMerge() {
  if (!arena_) return std::unexpected("UpdateEngine: no plan loaded");
  Execute(merge_program_);
  merged_ = true;
  return {};
}

std::expected<void, std::string> UpdateEngine::CommitToModel(
    const std::string& source_model_path, const std::string& out_path) const {
  if (!merged_)
    return std::unexpected("UpdateEngine: RunMerge() must precede commit");

  std::ifstream src(source_model_path, std::ios::binary);
  if (!src)
    return std::unexpected("UpdateEngine: cannot open source model '" +
                           source_model_path + "'");
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(src)),
                             std::istreambuf_iterator<char>());

  // Patch every merged weight's byte range — the delta application.
  for (const up::EmitEntry& e : emit_table_) {
    if (!RangeOk(e.smf_data_offset, e.byte_size, bytes.size()))
      return std::unexpected(
          "UpdateEngine: emit entry exceeds the source model file — plan and "
          "model are out of sync");
    std::memcpy(bytes.data() + e.smf_data_offset, arena_ + e.arena_offset,
                e.byte_size);
  }

  // Transactional commit: write sidecar, then atomic rename.
  const std::string tmp = out_path + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out)
      return std::unexpected("UpdateEngine: cannot write '" + tmp + "'");
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out) return std::unexpected("UpdateEngine: short write to '" + tmp + "'");
  }
  if (std::rename(tmp.c_str(), out_path.c_str()) != 0)
    return std::unexpected("UpdateEngine: atomic rename to '" + out_path +
                           "' failed");
  return {};
}

std::expected<void, std::string> UpdateEngine::SaveCheckpoint(
    const std::string& path) const {
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) return std::unexpected("UpdateEngine: cannot write checkpoint");
    const uint32_t magic = 0x504B4553;  // "SEKP"
    f.write(reinterpret_cast<const char*>(&magic), 4);
    f.write(reinterpret_cast<const char*>(&step_), 8);
    f.write(reinterpret_cast<const char*>(&header_.persistent_size), 8);
    f.write(reinterpret_cast<const char*>(arena_),
            static_cast<std::streamsize>(header_.persistent_size));
    if (!f) return std::unexpected("UpdateEngine: short checkpoint write");
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0)
    return std::unexpected("UpdateEngine: checkpoint rename failed");
  return {};
}

std::expected<void, std::string> UpdateEngine::LoadCheckpoint(
    const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return std::unexpected("UpdateEngine: no checkpoint at '" + path + "'");
  uint32_t magic = 0;
  uint64_t step = 0, size = 0;
  f.read(reinterpret_cast<char*>(&magic), 4);
  f.read(reinterpret_cast<char*>(&step), 8);
  f.read(reinterpret_cast<char*>(&size), 8);
  if (!f || magic != 0x504B4553 || size != header_.persistent_size)
    return std::unexpected("UpdateEngine: checkpoint incompatible with plan");
  f.read(reinterpret_cast<char*>(arena_), static_cast<std::streamsize>(size));
  if (!f) return std::unexpected("UpdateEngine: truncated checkpoint");
  step_ = step;
  return {};
}

}  // namespace seeml::update_rt
