#include "runtime/update_engine.h"

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

#include "runtime/update_kernels.h"
#include "source/hash.h"

namespace seeml::update_rt {

namespace up = seeml::update;
namespace k = kernels;

namespace {

float BitsToF32(uint64_t bits) {
  return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

// --- Durable file replacement -------------------------------------------------
// Write-tmp + rename alone is atomic but NOT durable: after a power cut the
// rename may survive while the data does not, leaving a truncated model on
// exactly the class of device this runtime targets. POSIX durability needs
// fsync on the file before the rename and fsync on the directory after it.
std::expected<void, std::string> WriteFileDurable(const std::string& path,
                                                  const uint8_t* data,
                                                  size_t size) {
  const std::string tmp = path + ".tmp";
#ifndef _WIN32
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0)
    return std::unexpected("UpdateEngine: cannot write '" + tmp + "'");
  size_t written = 0;
  while (written < size) {
    const ssize_t n = ::write(fd, data + written, size - written);
    if (n < 0) {
      ::close(fd);
      return std::unexpected("UpdateEngine: short write to '" + tmp + "'");
    }
    written += static_cast<size_t>(n);
  }
  if (::fsync(fd) != 0) {
    ::close(fd);
    return std::unexpected("UpdateEngine: fsync of '" + tmp + "' failed");
  }
  ::close(fd);
  if (std::rename(tmp.c_str(), path.c_str()) != 0)
    return std::unexpected("UpdateEngine: atomic rename to '" + path +
                           "' failed");
  // Persist the rename itself.
  const size_t slash = path.find_last_of('/');
  const std::string dir = slash == std::string::npos
                              ? std::string(".")
                              : path.substr(0, slash == 0 ? 1 : slash);
  const int dfd = ::open(dir.c_str(), O_RDONLY);
  if (dfd >= 0) {
    ::fsync(dfd);  // best effort: some filesystems reject directory fsync
    ::close(dfd);
  }
  return {};
#else
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out)
      return std::unexpected("UpdateEngine: cannot write '" + tmp + "'");
    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(size));
    if (!out)
      return std::unexpected("UpdateEngine: short write to '" + tmp + "'");
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0)
    return std::unexpected("UpdateEngine: atomic rename to '" + path +
                           "' failed");
  return {};
#endif
}

std::expected<std::vector<uint8_t>, std::string> ReadFileBytes(
    const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return std::unexpected("UpdateEngine: cannot open '" + path + "'");
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
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
  // elem_bytes: f32/i32 operands are 4 bytes; quantized weights are 1.
  auto ref_ok_w = [&](uint64_t ref, uint64_t elems, bool write,
                      uint64_t elem_bytes) {
    if (ref == up::kNullRef) return false;
    if (write && up::IsRodataRef(ref)) return false;
    uint64_t bytes = 0;
    if (!MulOk(elems, elem_bytes, &bytes)) return false;
    const uint64_t space = up::IsRodataRef(ref) ? rodata_size : arena_size;
    return RangeOk(up::RefOffset(ref), bytes, space);
  };
  auto ref_ok = [&](uint64_t ref, uint64_t elems, bool write) {
    return ref_ok_w(ref, elems, write, sizeof(float));
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
    case up::OpCode::kGemmNNQ8:
    case up::OpCode::kGemmNTQ8: {
      // Every layout variant reads M*K (A) and K*N (B), writes M*N (C).
      const bool q8 = static_cast<up::OpCode>(ins.opcode) ==
                          up::OpCode::kGemmNNQ8 ||
                      static_cast<up::OpCode>(ins.opcode) ==
                          up::OpCode::kGemmNTQ8;
      if (!MulOk(d0, d2, &mk) || !MulOk(d2, d1, &kn) || !MulOk(d0, d1, &mn))
        return fail();
      // Quantized B must live in rodata: only the compiler's own int8
      // packing produces it, and it is 1 byte per element.
      if (q8 && !up::IsRodataRef(ins.in[1])) return fail();
      if (!ref_ok(ins.in[0], mk, false) ||
          !ref_ok_w(ins.in[1], kn, false, q8 ? 1 : sizeof(float)) ||
          !ref_ok(ins.in[2], mn, true))
        return fail();
      return {};
    }
    case up::OpCode::kAddEW:
    case up::OpCode::kMulEW:
    case up::OpCode::kReluBwd:
    case up::OpCode::kGeluBwd:
    case up::OpCode::kSiluBwd:
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
    case up::OpCode::kGeluFwd:
    case up::OpCode::kSiluFwd:
    case up::OpCode::kScale:
    case up::OpCode::kCopy:
      if (!ref_ok(ins.in[0], d0, false) || !ref_ok(ins.in[1], d0, true))
        return fail();
      return {};
    case up::OpCode::kLayerNormFwd: {
      const uint64_t rows = d0 >> 32, cols = d0 & 0xFFFFFFFFu;
      if (!MulOk(rows, cols, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], cols, false) ||
          !ref_ok(ins.in[2], cols, false) || !ref_ok(ins.in[3], nc, true) ||
          !ref_ok(ins.out[1], rows, true) || !ref_ok(ins.out[2], rows, true))
        return fail();
      return {};
    }
    case up::OpCode::kLayerNormBwd: {
      const uint64_t rows = d2 >> 32, cols = d2 & 0xFFFFFFFFu;
      if (!MulOk(rows, cols, &nc)) return fail();
      if (!ref_ok(ins.in[0], nc, false) || !ref_ok(ins.in[1], nc, false) ||
          !ref_ok(ins.in[2], cols, false) || !ref_ok(ins.in[3], nc, true) ||
          !ref_ok(ins.out[0], rows, false) || !ref_ok(ins.out[1], rows, false))
        return fail();
      return {};
    }
    case up::OpCode::kClipNorm:
      if (!ref_ok(ins.in[0], d0, true)) return fail();
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

  // Integrity: the plan hashes over itself with the hash field zeroed.
  // A flipped bit anywhere — header, instructions, frozen weights — fails
  // here instead of surfacing as silent numerical garbage on-device.
  {
    uint64_t state = up::kFnvOffsetBasis;
    constexpr size_t kHashAt = offsetof(up::PlanHeader, plan_hash);
    constexpr uint8_t kZero[sizeof(uint64_t)] = {};
    state = up::Fnv1a64(plan_, kHashAt, state);
    state = up::Fnv1a64(kZero, sizeof(kZero), state);
    state = up::Fnv1a64(plan_ + kHashAt + sizeof(uint64_t),
                        plan_size_ - kHashAt - sizeof(uint64_t), state);
    if (state != header_.plan_hash)
      return std::unexpected(
          "UpdateEngine: plan hash mismatch — the .seeu blob is corrupt");
  }

  uint64_t train_bytes = 0, merge_bytes = 0, eval_bytes = 0, emit_bytes = 0;
  if (!MulOk(header_.train_instr_count, sizeof(up::UpdateInstruction),
             &train_bytes) ||
      !MulOk(header_.merge_instr_count, sizeof(up::UpdateInstruction),
             &merge_bytes) ||
      !MulOk(header_.eval_instr_count, sizeof(up::UpdateInstruction),
             &eval_bytes) ||
      !MulOk(header_.emit_count, sizeof(up::EmitEntry), &emit_bytes))
    return std::unexpected("UpdateEngine: plan section size overflows");
  if (!RangeOk(header_.train_instr_offset, train_bytes, plan_size_) ||
      !RangeOk(header_.merge_instr_offset, merge_bytes, plan_size_) ||
      !RangeOk(header_.eval_instr_offset, eval_bytes, plan_size_) ||
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
  eval_program_.resize(header_.eval_instr_count);
  std::memcpy(eval_program_.data(), plan_ + header_.eval_instr_offset,
              eval_bytes);
  emit_table_.resize(header_.emit_count);
  std::memcpy(emit_table_.data(), plan_ + header_.emit_table_offset,
              emit_bytes);

  // Validate every operand ref of every instruction against the address
  // space it targets — after this, Execute() can trust the programs blindly.
  for (const auto* program : {&train_program_, &merge_program_, &eval_program_})
    for (const up::UpdateInstruction& ins : *program)
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
      case up::OpCode::kGemmNNQ8:
        k::GemmNNQ8(ReadPtr(ins.in[0]), ReadPtrQ8(ins.in[1]),
                    WritePtr(ins.in[2]), ins.out[0], ins.out[1], ins.out[2],
                    BitsToF32(ins.in[3]));
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
  return {};
}

std::expected<float, std::string> UpdateEngine::Evaluate(Dataset& data) {
  if (!arena_) return std::unexpected("UpdateEngine: no plan loaded");
  if (eval_program_.empty())
    return std::unexpected("UpdateEngine: plan carries no eval program");
  if (auto r = ValidateDataset(data); !r) return std::unexpected(r.error());

  float* input_slot = WritePtr(header_.input_ref);
  uint8_t* label_slot =
      header_.label_kind == 0
          ? nullptr
          : reinterpret_cast<uint8_t*>(WritePtr(header_.label_ref));

  // One pass over the set in compiled-batch chunks (final partial batch
  // wraps — the fixed-shape contract admits no ragged batch).
  const uint64_t batches =
      std::max<uint64_t>(1, (data.num_samples() + header_.batch - 1) /
                                header_.batch);
  double total = 0.0;
  for (uint64_t b = 0; b < batches; ++b) {
    data.FillBatch(header_.batch, input_slot, label_slot);
    Execute(eval_program_);
    total += LossValue();
  }
  return static_cast<float>(total / static_cast<double>(batches));
}

std::expected<TrainReport, std::string> UpdateEngine::Train(
    Dataset& data, uint64_t steps, const TrainOptions& options) {
  if (!arena_) return std::unexpected("UpdateEngine: no plan loaded");
  if (steps == 0) steps = header_.default_steps;
  if (steps == 0)
    return std::unexpected(
        "UpdateEngine: no steps requested and the plan has no default");
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
    auto v = Evaluate(*options.validation);
    if (!v) return std::unexpected(v.error());
    report.has_validation = true;
    report.val_initial_loss = *v;
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
  for (uint64_t s = start; s < start + steps; ++s) {
    if (options.should_stop && options.should_stop()) {
      report.stopped_early = true;
      break;
    }
    step_ = s + 1;  // 1-indexed timestep for AdamW bias correction
    data.FillBatch(header_.batch, input_slot, label_slot);
    Execute(train_program_);
    ++executed;

    const float loss = LossValue();
    // A non-finite loss means the parameters (and any AdamW moments) are
    // already poisoned; continuing can only burn energy. Fail the update —
    // the source model on disk is untouched by construction.
    if (!std::isfinite(loss))
      return std::unexpected(
          "UpdateEngine: loss became non-finite at step " +
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

  report.steps = executed;
  report.initial_avg_loss =
      first_n ? static_cast<float>(first_sum / first_n) : 0.0f;
  report.final_avg_loss =
      last_n ? static_cast<float>(last_sum / last_n) : report.initial_avg_loss;

  if (options.validation) {
    auto v = Evaluate(*options.validation);
    if (!v) return std::unexpected(v.error());
    report.val_final_loss = *v;
  }
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

  auto bytes = ReadFileBytes(source_model_path);
  if (!bytes) return std::unexpected(bytes.error());

  // Identity check before any byte moves: the emit table's offsets are only
  // meaningful inside the exact file the plan was compiled from. A same-sized
  // different model would otherwise be silently corrupted.
  if (header_.source_model_hash != 0 &&
      up::Fnv1a64(bytes->data(), bytes->size()) != header_.source_model_hash)
    return std::unexpected(
        "UpdateEngine: source model does not match the plan's "
        "source_model_hash — refusing to patch '" +
        source_model_path + "'");

  // Apply every adapter delta to its f32 weight range: W' = W + Δ. The
  // file's pristine weights are the base, so quantized plans commit no
  // quantization error.
  for (const up::EmitEntry& e : emit_table_) {
    if (!RangeOk(e.smf_data_offset, e.byte_size, bytes->size()) ||
        e.byte_size % sizeof(float) != 0)
      return std::unexpected(
          "UpdateEngine: emit entry exceeds the source model file — plan and "
          "model are out of sync");
    const auto* delta =
        reinterpret_cast<const float*>(arena_ + e.arena_offset);
    // The SMF data section is 64-aligned, but go through memcpy anyway: the
    // commit path has no business assuming the container's alignment.
    for (uint64_t i = 0; i < e.byte_size / sizeof(float); ++i) {
      float w;
      uint8_t* at = bytes->data() + e.smf_data_offset + i * sizeof(float);
      std::memcpy(&w, at, sizeof(float));
      w += delta[i];
      std::memcpy(at, &w, sizeof(float));
    }
  }

  // Transactional and durable: fsync'd sidecar, atomic rename, dir fsync.
  return WriteFileDurable(out_path, bytes->data(), bytes->size());
}

namespace {

// Checkpoint container: everything little-endian, hash-bound to its plan.
//   u32 magic "SEKP"; u32 version
//   u64 plan_hash        must equal the plan's PlanHeader::plan_hash
//   u64 step             1-indexed AdamW timestep at save
//   u64 persistent_size  byte length of the payload
//   u64 payload_hash     Fnv1a64 of the payload
//   payload              the arena's persistent segment
inline constexpr uint32_t kCkptMagic = 0x504B4553;  // "SEKP"
inline constexpr uint32_t kCkptVersion = 2;

#pragma pack(push, 1)
struct CkptHeader {
  uint32_t magic = kCkptMagic;
  uint32_t version = kCkptVersion;
  uint64_t plan_hash = 0;
  uint64_t step = 0;
  uint64_t persistent_size = 0;
  uint64_t payload_hash = 0;
};
#pragma pack(pop)

}  // namespace

std::expected<void, std::string> UpdateEngine::SaveCheckpoint(
    const std::string& path) const {
  CkptHeader h;
  h.plan_hash = header_.plan_hash;
  h.step = step_;
  h.persistent_size = header_.persistent_size;
  h.payload_hash = up::Fnv1a64(arena_, header_.persistent_size);

  std::vector<uint8_t> blob(sizeof(h) + header_.persistent_size);
  std::memcpy(blob.data(), &h, sizeof(h));
  std::memcpy(blob.data() + sizeof(h), arena_, header_.persistent_size);
  // Durable: a checkpoint that can vanish in a power cut is not a checkpoint.
  return WriteFileDurable(path, blob.data(), blob.size());
}

std::expected<void, std::string> UpdateEngine::LoadCheckpoint(
    const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return std::unexpected("UpdateEngine: no checkpoint at '" + path + "'");
  CkptHeader h;
  f.read(reinterpret_cast<char*>(&h), sizeof(h));
  if (!f || h.magic != kCkptMagic || h.version != kCkptVersion)
    return std::unexpected("UpdateEngine: not a v2 checkpoint: '" + path + "'");
  // Binding: a checkpoint carries adapter and optimizer state laid out by
  // one specific plan. Resuming it under any other plan is silent corruption.
  if (h.plan_hash != header_.plan_hash)
    return std::unexpected(
        "UpdateEngine: checkpoint belongs to a different plan");
  if (h.persistent_size != header_.persistent_size)
    return std::unexpected("UpdateEngine: checkpoint incompatible with plan");
  std::vector<uint8_t> payload(h.persistent_size);
  f.read(reinterpret_cast<char*>(payload.data()),
         static_cast<std::streamsize>(payload.size()));
  if (!f) return std::unexpected("UpdateEngine: truncated checkpoint");
  if (up::Fnv1a64(payload.data(), payload.size()) != h.payload_hash)
    return std::unexpected("UpdateEngine: checkpoint payload is corrupt");
  std::memcpy(arena_, payload.data(), payload.size());
  step_ = h.step;
  return {};
}

}  // namespace seeml::update_rt
