#include "source/update_runtime/update_engine.h"

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "source/update_runtime/update_kernels.h"

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

  auto section_ok = [&](uint64_t off, uint64_t bytes) {
    return off + bytes <= plan_size_;
  };
  const uint64_t train_bytes =
      header_.train_instr_count * sizeof(up::UpdateInstruction);
  const uint64_t merge_bytes =
      header_.merge_instr_count * sizeof(up::UpdateInstruction);
  const uint64_t emit_bytes = header_.emit_count * sizeof(up::EmitEntry);
  if (!section_ok(header_.train_instr_offset, train_bytes) ||
      !section_ok(header_.merge_instr_offset, merge_bytes) ||
      !section_ok(header_.rodata_offset, header_.rodata_size) ||
      !section_ok(header_.persist_init_offset, header_.persist_init_size) ||
      !section_ok(header_.emit_table_offset, emit_bytes))
    return std::unexpected("UpdateEngine: plan section out of bounds");

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

  const uint64_t expected_floats = header_.batch * data.input_dim();
  if (expected_floats != header_.input_floats)
    return std::unexpected(
        "UpdateEngine: dataset input width does not match the compiled plan");
  if (header_.label_kind != 0 && data.label_kind() != header_.label_kind)
    return std::unexpected(
        "UpdateEngine: dataset label kind does not match the compiled plan");

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
    if (e.smf_data_offset + e.byte_size > bytes.size())
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
