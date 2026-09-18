#include <cstdint>
#include <cstdlib>
#include <bit>

#include "runtime/executor/backend.h"
#include "runtime/executor/update_kernels.h"

// =============================================================================
// CpuBackend — the reference executor: the portable kernel library behind
// the ExecutorBackend seam. The dispatch switch below is the engine's
// pre-seam switch moved verbatim (same call sites, same argument order);
// the only edits are the per-step scalars (learning rate, AdamW
// hyperparameters, timestep), which now arrive in StepParams instead of
// being read off the engine. Nothing is deferred: Execute() runs the kernel
// to completion and Flush() is a no-op, so the arena is always coherent.
// =============================================================================

namespace seeml::update_rt {

namespace up = seeml::update;
namespace k = kernels;

namespace {

float BitsToF32(uint64_t bits) {
  return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

// The KL temperature word packs (loss_scale bits << 32) | T bits. A zero
// high word is a pre-v8 plan: no scale was ever written, and 1.0 selects
// the pre-change behavior (schema.h, the additive-field rule).
constexpr const char* kNonFiniteNorm =
    "gradient norm is not finite — the step was refused with the parameters "
    "and optimizer state untouched";

float KlTemperatureOf(uint64_t word) { return BitsToF32(word & 0xFFFFFFFFu); }
float KlLossScaleOf(uint64_t word) {
  const uint64_t hi = word >> 32;
  return hi == 0 ? 1.0f : BitsToF32(hi);
}


class CpuBackend final : public ExecutorBackend {
 public:
  const char* name() const override { return "cpu"; }
  std::string device() const override { return "host CPU"; }

  std::expected<void, std::string> Bind(uint8_t* arena, uint64_t,
                                        const uint8_t* rodata, uint64_t,
                                        uint64_t) override {
    arena_ = arena;
    rodata_ = rodata;
    return {};
  }
  std::expected<void, std::string> BindSource(const uint8_t* data,
                                              uint64_t) override {
    source_ = data;
    return {};
  }

  void Configure(const k::KernelPolicy& policy) override { policy_ = policy; }

  std::expected<void, std::string> Execute(const up::UpdateInstruction& ins,
                                           const StepParams& params) override;

  std::expected<void, std::string> Flush() override { return {}; }

 private:
  const float* ReadPtr(uint64_t ref) const {
    const uint64_t offset = up::RefOffset(ref);
    if (up::IsRodataRef(ref))
      return reinterpret_cast<const float*>(rodata_ + offset);
    if (up::IsSourceRef(ref))  // v17: validated against the bound file
      return reinterpret_cast<const float*>(source_ + offset);
    return reinterpret_cast<const float*>(arena_ + offset);
  }
  const int8_t* ReadPtrQ8(uint64_t ref) const {
    // Validation pinned q8 sources to rodata; see ValidateInstruction.
    return reinterpret_cast<const int8_t*>(rodata_ + up::RefOffset(ref));
  }
  const uint16_t* ReadPtrBF16(uint64_t ref) const {  // likewise pinned
    return reinterpret_cast<const uint16_t*>(rodata_ + up::RefOffset(ref));
  }
  float* WritePtr(uint64_t ref) {
    // Lowering never emits a rodata destination; the frozen weights are
    // physically unwritable from the instruction stream by construction.
    return reinterpret_cast<float*>(arena_ + up::RefOffset(ref));
  }

  uint8_t* arena_ = nullptr;
  const uint8_t* rodata_ = nullptr;
  const uint8_t* source_ = nullptr;  // v17: the bound source model file
  k::KernelPolicy policy_;  // the plan header's tiles; defaults until told
};

std::expected<void, std::string> CpuBackend::Execute(
    const up::UpdateInstruction& ins, const StepParams& params) {
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
                up::EpilogueActOf(ins.flags), policy_.gemm_tiles,
                ins.flags & up::kFlagGemmAddend ? ReadPtr(ins.in[3])
                                                : nullptr);
      break;
    case up::OpCode::kGemmNT:
      k::GemmNT(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), WritePtr(ins.in[2]),
                ins.out[0], ins.out[1], ins.out[2], policy_.gemm_tiles,
                ins.flags & up::kFlagGemmAddend ? ReadPtr(ins.in[3])
                                                : nullptr);
      break;
    case up::OpCode::kGemmTN:
      k::GemmTN(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), WritePtr(ins.in[2]),
                ins.out[0], ins.out[1], ins.out[2], policy_.gemm_tiles,
                ins.flags & up::kFlagGemmAddend ? ReadPtr(ins.in[3])
                                                : nullptr);
      break;
    case up::OpCode::kGemmAccNN:
      k::GemmAccNN(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]),
                   WritePtr(ins.in[2]), ins.out[0], ins.out[1], ins.out[2],
                   BitsToF32(ins.in[3]), policy_.gemm_tiles);
      break;
    case up::OpCode::kGemmNNQ8:
      k::GemmNNQ8(ReadPtr(ins.in[0]), ReadPtrQ8(ins.in[1]),
                  WritePtr(ins.in[2]), ins.out[0], ins.out[1], ins.out[2],
                  BitsToF32(ins.in[3]), up::EpilogueActOf(ins.flags),
                  policy_.gemm_tiles);
      break;
    case up::OpCode::kGemmNTQ8:
      k::GemmNTQ8(ReadPtr(ins.in[0]), ReadPtrQ8(ins.in[1]),
                  WritePtr(ins.in[2]), ins.out[0], ins.out[1], ins.out[2],
                  BitsToF32(ins.in[3]), policy_.gemm_tiles);
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
                      ins.out[0] >> 32, ins.out[0] & 0xFFFFFFFFu,
                      up::NormEpsOf(ins.imm));
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
                      ins.out[1] & 0xFFFFFFFFu, KlTemperatureOf(ins.out[2]),
                      KlLossScaleOf(ins.out[2]));
      break;
    case up::OpCode::kKLDistillBwd:
      k::KLDistillBwd(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]),
                      ReadPtr(ins.in[2]), WritePtr(ins.in[3]),
                      ins.out[0] >> 32, ins.out[0] & 0xFFFFFFFFu,
                      KlTemperatureOf(ins.out[1]),
                      KlLossScaleOf(ins.out[1]));
      break;
    // v12: out[1] is the fused clip threshold (0 = none). The one kernel
    // that can refuse: a non-finite gradient norm aborts the step with
    // the parameters and the moments untouched.
    case up::OpCode::kSgdStep:
      if (!k::SgdStep(WritePtr(ins.in[0]), ReadPtr(ins.in[1]), ins.out[0],
                      params.lr, params.weight_decay, BitsToF32(ins.out[1])))
        return std::unexpected(kNonFiniteNorm);
      break;
    case up::OpCode::kAdamWStep:
      if (!k::AdamWStep(WritePtr(ins.in[0]), ReadPtr(ins.in[1]),
                        WritePtr(ins.in[2]), WritePtr(ins.in[3]), ins.out[0],
                        params.lr, params.beta1, params.beta2, params.eps,
                        params.weight_decay, params.step,
                        BitsToF32(ins.out[1])))
        return std::unexpected(kNonFiniteNorm);
      break;
    case up::OpCode::kFill:
      k::Fill(WritePtr(ins.in[0]), BitsToF32(ins.in[1]), ins.out[0]);
      break;
    case up::OpCode::kCopy:
      k::Copy(ReadPtr(ins.in[0]), WritePtr(ins.in[1]), ins.out[0]);
      break;
    case up::OpCode::kAccumulate:
      k::Accumulate(WritePtr(ins.in[0]), ReadPtr(ins.in[1]), ins.out[0]);
      break;
    case up::OpCode::kGemmNNBF16:
      k::GemmNNBF16(ReadPtr(ins.in[0]), ReadPtrBF16(ins.in[1]),
                    WritePtr(ins.in[2]), ins.out[0], ins.out[1], ins.out[2],
                    ins.flags & up::kFlagEpilogueBias ? ReadPtr(ins.in[3])
                                                      : nullptr,
                    up::EpilogueActOf(ins.flags), policy_.gemm_tiles);
      break;
    case up::OpCode::kGemmNTBF16:
      k::GemmNTBF16(ReadPtr(ins.in[0]), ReadPtrBF16(ins.in[1]),
                    WritePtr(ins.in[2]), ins.out[0], ins.out[1], ins.out[2],
                    policy_.gemm_tiles);
      break;
    case up::OpCode::kRmsNormFwd:
      k::RmsNormFwd(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]),
                    WritePtr(ins.in[2]), WritePtr(ins.in[3]),
                    ins.out[0] >> 32, ins.out[0] & 0xFFFFFFFFu,
                    up::NormEpsOf(ins.imm));
      break;
    case up::OpCode::kRmsNormBwd:
      k::RmsNormBwd(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]),
                    ReadPtr(ins.in[2]), ReadPtr(ins.out[0]),
                    WritePtr(ins.in[3]), ins.out[1] >> 32,
                    ins.out[1] & 0xFFFFFFFFu);
      break;
    // v12: in[2] names a kRopeTable result, or is kNullRef.
    case up::OpCode::kRopeFwd:
      k::RopeFwd(ReadPtr(ins.in[0]), WritePtr(ins.in[1]), ins.out[0] >> 32,
                 ins.out[0] & 0xFFFFFFFFu, ins.out[1] >> 32,
                 ins.out[1] & 0xFFFFFFFFu, BitsToF32(ins.out[2]),
                 ins.in[2] == up::kNullRef ? nullptr : ReadPtr(ins.in[2]));
      break;
    case up::OpCode::kRopeBwd:
      k::RopeBwd(ReadPtr(ins.in[0]), WritePtr(ins.in[1]), ins.out[0] >> 32,
                 ins.out[0] & 0xFFFFFFFFu, ins.out[1] >> 32,
                 ins.out[1] & 0xFFFFFFFFu, BitsToF32(ins.out[2]),
                 ins.in[2] == up::kNullRef ? nullptr : ReadPtr(ins.in[2]));
      break;
    case up::OpCode::kFusedMap: {
      // in[1..2] are kNullRef when no stage names them; never dereferenced.
      const float* others[3] = {
          nullptr,
          ins.in[1] == up::kNullRef ? nullptr : ReadPtr(ins.in[1]),
          ins.in[2] == up::kNullRef ? nullptr : ReadPtr(ins.in[2])};
      const float imm[2] = {BitsToF32(ins.out[2]), BitsToF32(ins.out[2] >> 32)};
      k::FusedMap(ReadPtr(ins.in[0]), others, WritePtr(ins.in[3]), ins.out[0],
                  ins.out[1], imm);
      break;
    }
    case up::OpCode::kRopeTable:
      k::RopeTable(WritePtr(ins.in[0]), ins.out[0] >> 32,
                   ins.out[0] & 0xFFFFFFFFu, BitsToF32(ins.out[1]));
      break;
    case up::OpCode::kAttnFwd:
      k::AttnFwd(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), ReadPtr(ins.in[2]),
                 WritePtr(ins.in[3]), WritePtr(ins.out[0]),
                 ins.out[1] >> 32, ins.out[1] & 0xFFFFFFFFu,
                 ins.out[2] >> 32, ins.out[2] & 0xFFFFFFFFu);
      break;
    case up::OpCode::kAttnDP:
      k::AttnDP(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), WritePtr(ins.in[2]),
                ins.out[0] >> 32, ins.out[0] & 0xFFFFFFFFu,
                ins.out[1] >> 32, ins.out[1] & 0xFFFFFFFFu);
      break;
    case up::OpCode::kAttnDV:
      k::AttnDV(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), WritePtr(ins.in[2]),
                ins.out[0] >> 32, ins.out[0] & 0xFFFFFFFFu,
                ins.out[1] >> 32, ins.out[1] & 0xFFFFFFFFu);
      break;
    case up::OpCode::kSoftmaxRowsBwd:
      k::SoftmaxRowsBwd(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]),
                        WritePtr(ins.in[2]), ins.out[0] >> 32,
                        ins.out[0] & 0xFFFFFFFFu);
      break;
    case up::OpCode::kAttnDQ:
      k::AttnDQ(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), WritePtr(ins.in[2]),
                ins.out[0] >> 32, ins.out[0] & 0xFFFFFFFFu,
                ins.out[1] >> 32, ins.out[1] & 0xFFFFFFFFu);
      break;
    case up::OpCode::kAttnDK:
      k::AttnDK(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]), WritePtr(ins.in[2]),
                ins.out[0] >> 32, ins.out[0] & 0xFFFFFFFFu,
                ins.out[1] >> 32, ins.out[1] & 0xFFFFFFFFu);
      break;
    case up::OpCode::kAttnFwdTiled:
      k::AttnFwdTiled(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]),
                      ReadPtr(ins.in[2]), WritePtr(ins.in[3]),
                      WritePtr(ins.out[0]), ins.out[1] >> 32,
                      ins.out[1] & 0xFFFFFFFFu, ins.out[2] >> 32,
                      ins.out[2] & 0xFFFFFFFFu);
      break;
    case up::OpCode::kAttnDQTiled:
    case up::OpCode::kAttnDKTiled:
    case up::OpCode::kAttnDVTiled: {
      const auto g = up::UnpackAttnGeometry(ins.out[2]);
      const auto op = static_cast<up::OpCode>(ins.opcode);
      if (op == up::OpCode::kAttnDQTiled)
        k::AttnDQTiled(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]),
                       ReadPtr(ins.in[2]), ReadPtr(ins.in[3]),
                       WritePtr(ins.out[0]), WritePtr(ins.out[1]), g.B, g.S,
                       g.H, g.d);
      else if (op == up::OpCode::kAttnDKTiled)
        k::AttnDKTiled(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]),
                       ReadPtr(ins.in[2]), ReadPtr(ins.in[3]),
                       ReadPtr(ins.out[0]), WritePtr(ins.out[1]), g.B, g.S,
                       g.H, g.d);
      else
        k::AttnDVTiled(ReadPtr(ins.in[0]), ReadPtr(ins.in[1]),
                       ReadPtr(ins.in[3]), ReadPtr(ins.out[0]),
                       WritePtr(ins.out[1]), g.B, g.S, g.H, g.d);
      break;
    }
    case up::OpCode::kEmbedFwd:
      k::EmbedFwd(reinterpret_cast<const int32_t*>(ReadPtr(ins.in[0])),
                  ReadPtr(ins.in[1]), WritePtr(ins.in[2]), ins.out[0],
                  ins.out[1] & 0xFFFFFFFFu);
      break;
  }
  return {};
}

}  // namespace

std::unique_ptr<ExecutorBackend> CreateCpuBackend() {
  return std::make_unique<CpuBackend>();
}

}  // namespace seeml::update_rt
