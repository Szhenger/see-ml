#include "runtime/executor/metal_backend.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach.h>
#include <mach/mach_vm.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <map>
#include <cstdint>
#include <cstring>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "runtime/executor/kernel_policy.h"
#include "runtime/executor/metal_kernels.h"
#include "runtime/validator/plan_validator.h"
#include "source/parallel/parallel_for.h"

// =============================================================================
// MetalBackend (roadmap Project 5, G1b-2/3/4): the Apple-GPU executor
// behind the ExecutorBackend seam.
//
// Residency (G1b-2): the engine's page-aligned arena is wrapped ONCE as a
// shared MTLBuffer (newBufferWithBytesNoCopy) — the GPU and the CPU read
// and write the same pages, so checkpoints, the loss slot and the merge
// deltas need no copies. The frozen weights are wrapped zero-copy too when
// the plan blob is page-aligned (the .incbin embedding) and page-padded
// (the compiler pads it); otherwise — a file-loaded plan in a heap vector
// — they are copied into one shared buffer at Bind, once, and device()
// says so.
//
// Batching (G1b-3): consecutive GPU-resident instructions are encoded into
// one command buffer through one serial compute encoder (Metal orders
// dispatches within it and tracks their memory hazards). A CPU-resident
// instruction — the loss kernels, the embedding gather — runs inline, but
// only after any pending GPU instruction it depends on has completed: the
// validator's operand extents (DescribeInstruction) decide, so dependency
// tracking is never looser than the bounds proof. The engine flushes at
// the end of every range, which makes each program (or timed phase) the
// batching unit. A command buffer that ends in any state but Completed
// aborts the update with an executor diagnostic — never a silent fallback
// to stale arena contents.
//
// Coverage (G1b-4): the GEMM family (simdgroup-MMA tiles with fused
// epilogues and the int8 variants), the transformer family, elementwise,
// normalization, RoPE, ClipNorm (the CPU's chunk geometry, partials
// combined in chunk order) and both optimizer steps run on the GPU; the
// three loss families and the embedding gather stay on the CPU (once per
// step, small). Determinism is per-backend: bitwise run-to-run on the same
// device, tolerance against the CPU (docs/roadmap.md).
// =============================================================================

namespace seeml::update_rt {

namespace up = seeml::update;

namespace {

// Mirrors `struct KArgs` in metal_kernels.h, field for field.
struct KArgs {
  uint64_t off[6] = {0, 0, 0, 0, 0, 0};
  uint32_t space = 0;
  uint32_t m = 0, n = 0, k = 0;
  uint32_t rows = 0, cols = 0;
  uint32_t B = 0, S = 0, H = 0, D = 0;
  uint32_t flags = 0;
  uint32_t step = 0;
  uint32_t pad0 = 0, pad1 = 0;
  float f[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  // GEMM family: leading dimensions, batch geometry, per-operand batch
  // strides (elements; [2i] per outer batch, [2i+1] per inner) for A, B, C.
  uint32_t lda = 0, ldb = 0, ldc = 0;
  uint32_t batch = 1, batch_h = 1;
  uint32_t pad2 = 0, pad3 = 0, pad4 = 0;
  uint64_t bs[6] = {0, 0, 0, 0, 0, 0};
};
static_assert(sizeof(KArgs) == 216, "KArgs must match the MSL layout");

enum Pipe : int {
  kPGemmNN, kPGemmNT, kPGemmTN, kPGemmAcc, kPGemmNNQ8, kPGemmNTQ8,
  kPGemmNNBF16, kPGemmNTBF16,
  // The skinny-shape variants, each in the same order (pipe + offset):
  // _small (K <= 16: one thread per output), _rows (N <= 16: one simdgroup
  // per row), _cols (M <= 16, wide N: one thread per column), _colsg
  // (M <= 16, narrower N: one simdgroup per column).
  kPGemmNNSmall, kPGemmNTSmall, kPGemmTNSmall, kPGemmAccSmall,
  kPGemmNNQ8Small, kPGemmNTQ8Small, kPGemmNNBF16Small, kPGemmNTBF16Small,
  kPGemmNNRows, kPGemmNTRows, kPGemmTNRows, kPGemmAccRows,
  kPGemmNNQ8Rows, kPGemmNTQ8Rows, kPGemmNNBF16Rows, kPGemmNTBF16Rows,
  kPGemmNNCols, kPGemmNTCols, kPGemmTNCols, kPGemmAccCols,
  kPGemmNNQ8Cols, kPGemmNTQ8Cols, kPGemmNNBF16Cols, kPGemmNTBF16Cols,
  kPGemmNNColsG, kPGemmNTColsG, kPGemmTNColsG, kPGemmAccColsG,
  kPGemmNNQ8ColsG, kPGemmNTQ8ColsG, kPGemmNNBF16ColsG, kPGemmNTBF16ColsG,
  kPGemmSplitKFin,
  kPAddEW, kPMulEW, kPAddBias, kPReluFwd, kPReluBwd, kPGeluFwd, kPGeluBwd,
  kPSiluFwd, kPSiluBwd, kPScale, kPFusedMap, kPFill, kPCopy, kPAccumulate, kPSgd, kPAdamW,
  kPReduceRows, kPLnFwd, kPLnBwd, kPRmsFwd, kPRmsBwd,
  kPClipPartials, kPClipFinish, kPClipApply, kPSgdClip, kPAdamWClip,
  kPRopeFwd, kPRopeBwd, kPAttnSoftmax, kPSoftmaxRowsBwd,
  kPipeCount
};
const char* const kPipeNames[kPipeCount] = {
    "k_gemm_nn", "k_gemm_nt", "k_gemm_tn", "k_gemm_acc", "k_gemm_nn_q8",
    "k_gemm_nt_q8", "k_gemm_nn_bf16", "k_gemm_nt_bf16",
    "k_gemm_nn_small", "k_gemm_nt_small", "k_gemm_tn_small",
    "k_gemm_acc_small", "k_gemm_nn_q8_small", "k_gemm_nt_q8_small",
    "k_gemm_nn_bf16_small", "k_gemm_nt_bf16_small",
    "k_gemm_nn_rows", "k_gemm_nt_rows", "k_gemm_tn_rows", "k_gemm_acc_rows",
    "k_gemm_nn_q8_rows", "k_gemm_nt_q8_rows", "k_gemm_nn_bf16_rows",
    "k_gemm_nt_bf16_rows",
    "k_gemm_nn_cols", "k_gemm_nt_cols", "k_gemm_tn_cols", "k_gemm_acc_cols",
    "k_gemm_nn_q8_cols", "k_gemm_nt_q8_cols", "k_gemm_nn_bf16_cols",
    "k_gemm_nt_bf16_cols",
    "k_gemm_nn_colsg", "k_gemm_nt_colsg", "k_gemm_tn_colsg",
    "k_gemm_acc_colsg", "k_gemm_nn_q8_colsg", "k_gemm_nt_q8_colsg",
    "k_gemm_nn_bf16_colsg", "k_gemm_nt_bf16_colsg",
    "k_gemm_splitk_fin", "k_add_ew", "k_mul_ew", "k_add_bias", "k_relu_fwd",
    "k_relu_bwd", "k_gelu_fwd", "k_gelu_bwd", "k_silu_fwd", "k_silu_bwd",
    "k_scale", "k_fused_map", "k_fill", "k_copy", "k_accumulate", "k_sgd", "k_adamw",
    "k_reduce_rows",
    "k_layernorm_fwd", "k_layernorm_bwd", "k_rmsnorm_fwd", "k_rmsnorm_bwd",
    "k_clip_partials", "k_clip_finish", "k_clip_apply", "k_sgd_clip",
    "k_adamw_clip", "k_rope_fwd",
    "k_rope_bwd", "k_attn_softmax", "k_softmax_rows_bwd"};

constexpr uint32_t kElementwiseGroup = 256;
constexpr uint32_t kGemmTile = 64;
constexpr uint32_t kGemmThreads = 128;  // 4 simdgroups of 32
constexpr int kGemmSmall = kPGemmNNSmall - kPGemmNN;
constexpr int kGemmRows = kPGemmNNRows - kPGemmNN;
constexpr int kGemmCols = kPGemmNNCols - kPGemmNN;
constexpr int kGemmColsG = kPGemmNNColsG - kPGemmNN;
// M <= 16: below this N a simdgroup per column (lanes striding K) beats a
// thread per column; above it the per-thread form's coalesced B reads win.
constexpr uint32_t kGemmColsPerThreadN = 4096;
// A GEMM with any dimension at or below this takes a skinny kernel rather
// than 64x64 tiles: an adapter's rank-8 factors, a K=8 product.
constexpr uint32_t kGemmSkinnyDim = 16;
// Split-K: a tiled GEMM with fewer output tiles than this splits K across
// threadgroup z (partials summed in a fixed order) so ten GPU cores stay
// fed; splits never exceed kGemmMaxSplits nor leave a split below
// kGemmMinSplitK.
constexpr uint32_t kGemmTargetTiles = 256;
constexpr uint32_t kGemmMaxSplits = 16;
constexpr uint32_t kGemmMinSplitK = 512;

uint32_t SplitsFor(uint32_t tiles, uint32_t k) {
  uint32_t splits = (kGemmTargetTiles + tiles - 1) / tiles;
  splits = std::max(splits, k / 4096);  // a very long K splits regardless
  splits = std::min(splits, kGemmMaxSplits);
  splits = std::min(splits, std::max(1u, k / kGemmMinSplitK));
  return std::max(splits, 1u);
}
constexpr size_t kScratchFloats = up::kMaxParallelChunks + 1;

std::string NsError(NSError* err, const char* what) {
  std::string s = what;
  if (err && err.localizedDescription)
    s += std::string(": ") + err.localizedDescription.UTF8String;
  return s;
}

float BitsToF32(uint64_t bits) {
  return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

struct Extent {
  uint64_t off, bytes;
};

// Whether every page of [ptr, ptr + len) is mapped writable. A no-copy
// MTLBuffer over read-only pages (an .incbin plan in __TEXT,__const) leaves
// them unreadable from the CPU afterwards — the embedding gather then dies
// with a protection fault — so zero-copy residency is offered only to
// writable memory; everything else is copied once at Bind.
bool PagesWritable(const void* ptr, uint64_t len) {
  mach_vm_address_t addr = reinterpret_cast<mach_vm_address_t>(ptr);
  const mach_vm_address_t end = addr + len;
  while (addr < end) {
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    mach_vm_address_t probe = addr;
    if (mach_vm_region(mach_task_self(), &probe, &size, VM_REGION_BASIC_INFO_64,
                       reinterpret_cast<vm_region_info_t>(&info), &count,
                       &object) != KERN_SUCCESS)
      return false;
    if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
    if (probe > addr) return false;  // a hole inside the range
    if (!(info.protection & VM_PROT_WRITE)) return false;
    addr = probe + size;
  }
  return true;
}
bool Overlaps(const Extent& a, const Extent& b) {
  return a.off < b.off + b.bytes && b.off < a.off + a.bytes;
}

class MetalBackend final : public ExecutorBackend {
 public:
  static std::expected<std::unique_ptr<ExecutorBackend>, std::string> Create();

  ~MetalBackend() override {
    (void)Flush();
    if (profile_) PrintProfile();
  }

  const char* name() const override { return "metal"; }
  std::string device() const override {
    std::string s = device_ ? device_.name.UTF8String : "no device";
    s += rodata_buf_ == nil ? "" : (rodata_zero_copy_ ? ", rodata zero-copy"
                                                        : ", rodata copied");
    return s;
  }

  std::expected<void, std::string> Bind(uint8_t* arena, uint64_t arena_bytes,
                                        const uint8_t* rodata,
                                        uint64_t rodata_bytes,
                                        uint64_t rodata_mapped_bytes) override;
  std::expected<void, std::string> Execute(const up::UpdateInstruction& ins,
                                           const StepParams& params) override;
  std::expected<void, std::string> Flush() override;

 private:
  const InstructionExtents* ExtentsOf(const up::UpdateInstruction& ins);
  bool HazardWithPending(const InstructionExtents& ex) const;
  void RecordPending(const InstructionExtents& ex);
  std::expected<void, std::string> EnsureEncoder();
  void Dispatch(Pipe pipe, const KArgs& a, uint32_t threads_x, uint32_t group,
                bool scratch = false);
  void DispatchGemm(Pipe pipe, KArgs a, bool at, bool bt, uint32_t b_elem);
  void EncodeAttentionGemm(Pipe pipe, bool at, bool bt, uint64_t a_ref,
                           uint64_t b_ref, uint64_t c_ref, uint32_t B,
                           uint32_t S, uint32_t H, uint32_t d,
                           uint32_t a_kind, uint32_t b_kind, uint32_t c_kind,
                           uint32_t m, uint32_t n, uint32_t k, float alpha);
  std::expected<void, std::string> Encode(const up::UpdateInstruction& ins,
                                          const StepParams& params);
  void PrintProfile() const;

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLComputePipelineState> pipes_[kPipeCount];
  id<MTLBuffer> scratch_ = nil;
  id<MTLBuffer> splitk_ws_ = nil;  // split-K partials, grown on demand
  uint64_t splitk_ws_bytes_ = 0;
  id<MTLBuffer> arena_buf_ = nil;
  id<MTLBuffer> rodata_buf_ = nil;
  bool rodata_zero_copy_ = false;
  uint8_t* arena_ = nullptr;
  uint64_t arena_bytes_ = 0;
  const uint8_t* rodata_ = nullptr;
  uint64_t rodata_bytes_ = 0;
  std::unique_ptr<ExecutorBackend> cpu_;
  id<MTLCommandBuffer> cmd_ = nil;
  id<MTLComputeCommandEncoder> enc_ = nil;
  std::vector<Extent> pending_reads_, pending_writes_;
  // Keyed on the instruction's 64 bytes, not its address: a caller that
  // reuses one UpdateInstruction object with different operands (the
  // hand-built test pattern) must never see stale extents.
  std::unordered_map<std::string, InstructionExtents> extents_;
  // SEEML_METAL_PROFILE=1: one command buffer per GPU instruction, its
  // GPU time (GPUStartTime..GPUEndTime) accumulated per kernel and shape,
  // CPU-resident instructions by wall clock; a table on stderr at
  // destruction. Diagnostic only — it serializes the stream.
  const bool profile_ = std::getenv("SEEML_METAL_PROFILE") != nullptr &&
                        *std::getenv("SEEML_METAL_PROFILE") != '\0' &&
                        *std::getenv("SEEML_METAL_PROFILE") != '0';
  struct ProfRow { double seconds = 0; uint64_t count = 0; };
  std::map<std::string, ProfRow> prof_;
  std::string prof_pipes_;  // every kernel the current instruction dispatched
  double last_gpu_seconds_ = 0;
  void NoteDispatch(Pipe pipe) {
    if (!profile_) return;
    if (!prof_pipes_.empty()) prof_pipes_ += '+';
    prof_pipes_ += kPipeNames[pipe];
  }
};

// Dimensions AND the element indices built from them ride 32-bit in the
// kernels. Plain counts (GEMM M/N/K, elementwise counts, AddBias/ReduceRows
// rows x cols) and every index product (M·K, K·N, M·N; B·H·S·S for the
// probability matrix; B·S·H·d for activations) must fit; the transformer
// family's packed dim words are two 32-bit halves by construction, their
// products are not. A wider plan — one that could not exist on the devices
// this targets — takes the CPU path for that instruction.
bool DimsFit32(up::OpCode op, const up::UpdateInstruction& ins) {
  constexpr uint64_t kMax = 0xFFFFFFFFu;
  auto hi = [](uint64_t v) { return v >> 32; };
  auto lo = [](uint64_t v) { return v & kMax; };
  switch (op) {
    case up::OpCode::kGemmNN:
    case up::OpCode::kGemmNT:
    case up::OpCode::kGemmTN:
    case up::OpCode::kGemmAccNN:
    case up::OpCode::kGemmNNQ8:
    case up::OpCode::kGemmNTQ8:
    case up::OpCode::kGemmNNBF16:
    case up::OpCode::kGemmNTBF16:
      return ins.out[0] <= kMax && ins.out[1] <= kMax && ins.out[2] <= kMax &&
             ins.out[0] * ins.out[1] <= kMax && ins.out[0] * ins.out[2] <= kMax &&
             ins.out[2] * ins.out[1] <= kMax;
    case up::OpCode::kRopeFwd:
    case up::OpCode::kRopeBwd:
      return hi(ins.out[0]) * lo(ins.out[0]) * hi(ins.out[1]) * lo(ins.out[1]) <=
             kMax;
    case up::OpCode::kAttnFwd: {
      const uint64_t B = hi(ins.out[1]), S = lo(ins.out[1]);
      const uint64_t H = hi(ins.out[2]), d = lo(ins.out[2]);
      return B * H * S * S <= kMax && B * S * H * d <= kMax;
    }
    case up::OpCode::kAttnDP:
    case up::OpCode::kAttnDV:
    case up::OpCode::kAttnDQ:
    case up::OpCode::kAttnDK: {
      const uint64_t B = hi(ins.out[0]), S = lo(ins.out[0]);
      const uint64_t H = hi(ins.out[1]), d = lo(ins.out[1]);
      return B * H * S * S <= kMax && B * S * H * d <= kMax;
    }
    case up::OpCode::kSoftmaxRowsBwd:
    case up::OpCode::kLayerNormFwd:
    case up::OpCode::kRmsNormFwd:
      return hi(ins.out[0]) * lo(ins.out[0]) <= kMax;
    case up::OpCode::kLayerNormBwd:
      return hi(ins.out[2]) * lo(ins.out[2]) <= kMax;
    case up::OpCode::kRmsNormBwd:
      return hi(ins.out[1]) * lo(ins.out[1]) <= kMax;
    case up::OpCode::kAddBias:
    case up::OpCode::kReduceRows:
      return ins.out[0] <= kMax && ins.out[1] <= kMax &&
             ins.out[0] * ins.out[1] <= kMax;
    case up::OpCode::kAddEW:
    case up::OpCode::kMulEW:
    case up::OpCode::kReluFwd:
    case up::OpCode::kReluBwd:
    case up::OpCode::kGeluFwd:
    case up::OpCode::kGeluBwd:
    case up::OpCode::kSiluFwd:
    case up::OpCode::kSiluBwd:
    case up::OpCode::kScale:
    case up::OpCode::kFill:
    case up::OpCode::kCopy:
    case up::OpCode::kAccumulate:
    case up::OpCode::kFusedMap:
    case up::OpCode::kSgdStep:
    case up::OpCode::kAdamWStep:
    case up::OpCode::kClipNorm:
      return ins.out[0] <= kMax;
    default:
      return true;
  }
}

bool IsGpuOpcode(up::OpCode op) {
  switch (op) {
    case up::OpCode::kNop:
    case up::OpCode::kSoftmaxXEntFwd:
    case up::OpCode::kSoftmaxXEntBwd:
    case up::OpCode::kMseFwd:
    case up::OpCode::kMseBwd:
    case up::OpCode::kKLDistillFwd:
    case up::OpCode::kKLDistillBwd:
    case up::OpCode::kEmbedFwd:
    // The RoPE angle table (v12) is a CPU-side convenience the GPU rotation
    // kernels do not read — one thread per (b, s, h) unit recomputes its
    // angles, as before — so the table is built on the CPU, where the
    // rotations that do read it run.
    case up::OpCode::kRopeTable:
      return false;
    default:
      return true;
  }
}

std::expected<std::unique_ptr<ExecutorBackend>, std::string>
MetalBackend::Create() {
  @autoreleasepool {
    auto be = std::unique_ptr<MetalBackend>(new MetalBackend());
    be->device_ = MTLCreateSystemDefaultDevice();
    if (!be->device_) return std::unexpected("no Metal device available");
    be->queue_ = [be->device_ newCommandQueue];
    if (!be->queue_) return std::unexpected("cannot create a command queue");

    MTLCompileOptions* options = [MTLCompileOptions new];
    // Precise math, both halves of it: `mathMode` governs algebraic
    // fast-math (reassociation, no NaN/inf), `mathFloatingPointFunctions`
    // governs which transcendental variants the unqualified tanh/exp/sin/
    // cos/sqrt calls resolve to — and the fast tanh returns NaN above
    // ~44, which through the GELU means NaN for any pre-activation past
    // ~10.25. The kernel library mirrors the CPU expressions and is
    // compared against them at tolerance; both knobs must be precise.
    // (The pre-15 `fastMathEnabled = NO` selects both at once.)
#if defined(__MAC_15_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
      options.mathMode = MTLMathModeSafe;
      options.mathFloatingPointFunctions =
          MTLMathFloatingPointFunctionsPrecise;
    } else
#endif
    {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      options.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:kMetalKernelSource];
    id<MTLLibrary> lib = [be->device_ newLibraryWithSource:src
                                                   options:options
                                                     error:&err];
    if (!lib)
      return std::unexpected(NsError(err, "kernel library compilation failed"));
    for (int i = 0; i < kPipeCount; ++i) {
      id<MTLFunction> fn = [lib
          newFunctionWithName:[NSString stringWithUTF8String:kPipeNames[i]]];
      if (!fn)
        return std::unexpected(std::string("kernel '") + kPipeNames[i] +
                               "' missing from the library");
      be->pipes_[i] = [be->device_ newComputePipelineStateWithFunction:fn
                                                                  error:&err];
      if (!be->pipes_[i])
        return std::unexpected(NsError(err, "pipeline creation failed"));
    }
    // The GEMM tiles assume 32-wide simdgroups (four per 128-thread group)
    // and threadgroups of at least 256 for the elementwise kernels.
    if (be->pipes_[kPGemmNN].threadExecutionWidth != 32)
      return std::unexpected("unsupported SIMD width " +
                             std::to_string(
                                 be->pipes_[kPGemmNN].threadExecutionWidth) +
                             " (the GEMM tiles need 32)");
    for (int i = 0; i < kPipeCount; ++i)
      if (be->pipes_[i].maxTotalThreadsPerThreadgroup < kElementwiseGroup)
        return std::unexpected(
            "device threadgroup limit below " +
            std::to_string(kElementwiseGroup) + " for kernel " + kPipeNames[i]);
    be->scratch_ =
        [be->device_ newBufferWithLength:kScratchFloats * sizeof(float)
                                 options:MTLResourceStorageModeShared];
    if (!be->scratch_) return std::unexpected("scratch allocation failed");
    be->cpu_ = CreateCpuBackend();
    return std::unique_ptr<ExecutorBackend>(std::move(be));
  }
}

std::expected<void, std::string> MetalBackend::Bind(
    uint8_t* arena, uint64_t arena_bytes, const uint8_t* rodata,
    uint64_t rodata_bytes, uint64_t rodata_mapped_bytes) {
  @autoreleasepool {
    // Nothing may still be in flight against the previous binding.
    if (auto r = Flush(); !r) return r;
    const uint64_t page = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
    if (page == 0 || reinterpret_cast<uintptr_t>(arena) % page != 0 ||
        arena_bytes % page != 0)
      return std::unexpected("arena is not page-aligned (" +
                             std::to_string(page) + "-byte pages)");
    id<MTLBuffer> arena_buf =
        [device_ newBufferWithBytesNoCopy:arena
                                   length:arena_bytes
                                  options:MTLResourceStorageModeShared
                              deallocator:nil];
    if (!arena_buf) return std::unexpected("cannot wrap the arena zero-copy");
    id<MTLBuffer> rodata_buf = nil;
    bool zero_copy = false;
    if (rodata_bytes > 0) {
      const uint64_t wrapped = (rodata_bytes + page - 1) & ~(page - 1);
      if (reinterpret_cast<uintptr_t>(rodata) % page == 0 &&
          wrapped <= rodata_mapped_bytes && PagesWritable(rodata, wrapped)) {
        rodata_buf = [device_
            newBufferWithBytesNoCopy:const_cast<uint8_t*>(rodata)
                              length:wrapped
                             options:MTLResourceStorageModeShared
                         deallocator:nil];
        zero_copy = rodata_buf != nil;
      }
      if (!rodata_buf) {
        // Documented cost of a heap-resident plan: one rodata-sized copy,
        // made once per Bind (device() reports it).
        rodata_buf = [device_ newBufferWithBytes:rodata
                                          length:rodata_bytes
                                         options:MTLResourceStorageModeShared];
        if (!rodata_buf)
          return std::unexpected("cannot allocate the rodata buffer");
      }
    }
    if (auto r = cpu_->Bind(arena, arena_bytes, rodata, rodata_bytes,
                            rodata_mapped_bytes);
        !r)
      return r;
    arena_buf_ = arena_buf;
    rodata_buf_ = rodata_buf;
    rodata_zero_copy_ = zero_copy;
    arena_ = arena;
    arena_bytes_ = arena_bytes;
    rodata_ = rodata;
    rodata_bytes_ = rodata_bytes;
    extents_.clear();
    pending_reads_.clear();
    pending_writes_.clear();
    return {};
  }
}

const InstructionExtents* MetalBackend::ExtentsOf(
    const up::UpdateInstruction& ins) {
  std::string key(reinterpret_cast<const char*>(&ins), sizeof(ins));
  auto it = extents_.find(key);
  if (it != extents_.end()) return &it->second;
  auto ex = DescribeInstruction(ins, arena_bytes_, rodata_bytes_,
                                up::kSeeuVersion);
  if (!ex) return nullptr;
  return &extents_.emplace(std::move(key), *ex).first->second;
}

bool MetalBackend::HazardWithPending(const InstructionExtents& ex) const {
  for (size_t i = 0; i < ex.count; ++i) {
    const OperandExtent& e = ex.ranges[i];
    if (e.rodata) continue;  // never written by anyone
    const Extent mine{e.off, e.bytes};
    for (const Extent& w : pending_writes_)
      if (Overlaps(mine, w)) return true;
    if (e.write)
      for (const Extent& r : pending_reads_)
        if (Overlaps(mine, r)) return true;
  }
  return false;
}

void MetalBackend::RecordPending(const InstructionExtents& ex) {
  for (size_t i = 0; i < ex.count; ++i) {
    const OperandExtent& e = ex.ranges[i];
    if (e.rodata) continue;
    (e.write ? pending_writes_ : pending_reads_).push_back({e.off, e.bytes});
  }
}

std::expected<void, std::string> MetalBackend::EnsureEncoder() {
  if (enc_) return {};
  cmd_ = [queue_ commandBuffer];
  enc_ = cmd_ ? [cmd_ computeCommandEncoder] : nil;  // serial dispatch type
  if (!cmd_ || !enc_) {
    cmd_ = nil;
    enc_ = nil;
    return std::unexpected("cannot create a GPU command encoder");
  }
  [enc_ setBuffer:arena_buf_ offset:0 atIndex:0];
  [enc_ setBuffer:(rodata_buf_ ? rodata_buf_ : arena_buf_) offset:0 atIndex:1];
  [enc_ setBuffer:scratch_ offset:0 atIndex:3];
  // Every tiled GEMM kernel declares the split-K workspace at index 4 and
  // reads it only when it splits; the binding must still exist (an unbound
  // argument is a contract violation the debug layer aborts on), so the
  // scratch buffer stands in until a workspace is allocated.
  [enc_ setBuffer:(splitk_ws_ ? splitk_ws_ : scratch_) offset:0 atIndex:4];
  return {};
}

void MetalBackend::Dispatch(Pipe pipe, const KArgs& a, uint32_t threads_x,
                            uint32_t group, bool) {
  NoteDispatch(pipe);
  [enc_ setComputePipelineState:pipes_[pipe]];
  [enc_ setBytes:&a length:sizeof(a) atIndex:2];
  // 64-bit: DimsFit32 admits counts up to 2^32 - 1, and the round-up must
  // not wrap to a near-empty grid.
  const uint64_t groups = (uint64_t{threads_x} + group - 1) / group;
  [enc_ dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
}

// `at`/`bt`: operand stored transposed; `b_elem`: B's element size. A
// caller that leaves lda/ldb/ldc zero gets the dense strides. Sets the
// vector-load flag bits (16: A, 32: B) when the operand's byte offset and
// leading dimension keep every 4-element vector 4*elem-aligned, picks a
// skinny kernel for skinny shapes, and splits K for an unbatched GEMM
// with too few tiles.
void MetalBackend::DispatchGemm(Pipe pipe, KArgs a, bool at, bool bt,
                                uint32_t b_elem) {
  if (a.lda == 0) a.lda = at ? a.m : a.k;
  if (a.ldb == 0) a.ldb = bt ? a.k : a.n;
  if (a.ldc == 0) a.ldc = a.n;
  if (a.off[0] % 16 == 0 && a.lda % 4 == 0 && a.bs[0] % 4 == 0 &&
      a.bs[1] % 4 == 0)
    a.flags |= 16u;
  if (a.off[1] % (4 * b_elem) == 0 && a.ldb % 4 == 0 && a.bs[2] % 4 == 0 &&
      a.bs[3] % 4 == 0)
    a.flags |= 32u;
  if (a.k <= kGemmSkinnyDim) {
    Dispatch(static_cast<Pipe>(pipe + kGemmSmall), a, a.batch * a.m * a.n,
             kElementwiseGroup);
    return;
  }
  if (a.n <= kGemmSkinnyDim) {
    Dispatch(static_cast<Pipe>(pipe + kGemmRows), a, a.batch * a.m * 32u,
             kElementwiseGroup);
    return;
  }
  if (a.m <= kGemmSkinnyDim) {
    if (a.n >= kGemmColsPerThreadN)
      Dispatch(static_cast<Pipe>(pipe + kGemmCols), a, a.batch * a.n,
               kElementwiseGroup);
    else
      Dispatch(static_cast<Pipe>(pipe + kGemmColsG), a, a.batch * a.n * 32u,
               kElementwiseGroup);
    return;
  }
  const uint32_t gx = (a.n + kGemmTile - 1) / kGemmTile;
  const uint32_t gy = (a.m + kGemmTile - 1) / kGemmTile;
  const uint32_t splits = a.batch > 1 ? 1 : SplitsFor(gx * gy, a.k);
  a.pad0 = splits;
  if (splits > 1) {
    const uint64_t need = uint64_t{splits} * a.m * a.n * sizeof(float);
    if (need > splitk_ws_bytes_) {
      // Grow geometrically; an encoder that still references the old
      // buffer keeps it alive until its command buffer completes.
      const uint64_t bytes = std::max(need, std::max<uint64_t>(
                                                splitk_ws_bytes_ * 2, 1 << 20));
      splitk_ws_ = [device_ newBufferWithLength:bytes
                                        options:MTLResourceStorageModePrivate];
      splitk_ws_bytes_ = splitk_ws_ ? bytes : 0;
      if (!splitk_ws_) {
        // No workspace: run unsplit (slower, never wrong).
        a.pad0 = 1;
      }
    }
    if (a.pad0 > 1) [enc_ setBuffer:splitk_ws_ offset:0 atIndex:4];
  }
  NoteDispatch(pipe);
  [enc_ setComputePipelineState:pipes_[pipe]];
  [enc_ setBytes:&a length:sizeof(a) atIndex:2];
  [enc_ dispatchThreadgroups:MTLSizeMake(gx, gy, a.pad0 > 1 ? a.pad0 : a.batch)
       threadsPerThreadgroup:MTLSizeMake(kGemmThreads, 1, 1)];
  if (a.pad0 > 1) {
    // The finish kernel: flags bit 6 selects the accumulate form.
    if (pipe == kPGemmAcc) a.flags |= 64u;
    Dispatch(kPGemmSplitKFin, a, a.m * a.n, kElementwiseGroup);
  }
}

std::expected<void, std::string> MetalBackend::Encode(
    const up::UpdateInstruction& ins, const StepParams& params) {
  KArgs a;
  auto ref = [&](int slot, uint64_t r) {
    a.off[slot] = up::RefOffset(r);
    if (up::IsRodataRef(r)) a.space |= 1u << slot;
  };
  auto u32 = [](uint64_t v) { return static_cast<uint32_t>(v); };
  auto hi = [](uint64_t v) { return static_cast<uint32_t>(v >> 32); };
  auto lo = [](uint64_t v) { return static_cast<uint32_t>(v & 0xFFFFFFFFu); };
  // The clip factor min(1, max/||g||) for the gradient in ref slot
  // `g_slot` of `args`, left in scratch[256] — kClipNorm's first two
  // dispatches, shared with the fused-clip steps (v12), whose `word` is
  // zero when they carry no clip. Runs on a copy: the clip kernels read g
  // through slot 0 and their own f[0]. The partial sums follow the CPU's
  // chunk geometry (a pure function of n), so they combine in the order
  // the reference combines them.
  auto EncodeClipScale = [&](const KArgs& args, uint64_t word,
                             int g_slot = 1) {
    if (word == 0) return false;
    KArgs c;
    c.off[0] = args.off[g_slot];
    c.space = (args.space >> g_slot) & 1u;
    c.n = args.n;
    c.f[0] = BitsToF32(word);
    c.k = static_cast<uint32_t>(
        up::ParallelChunkGrain(c.n, kernels::kGrainCheap));
    c.m = static_cast<uint32_t>(
        up::ParallelChunkCount(c.n, kernels::kGrainCheap));
    Dispatch(kPClipPartials, c, c.m, kElementwiseGroup, true);
    Dispatch(kPClipFinish, c, 1, 1, true);
    return true;
  };
  // The GEMM addend (v14): C = D + C, in place, one thread per element —
  // the two dispatches the GEMM + kAddEW pair always was on this backend.
  auto EncodeAddend = [&](uint32_t count) {
    if (!(ins.flags & up::kFlagGemmAddend)) return;
    KArgs sum;
    sum.off[0] = up::RefOffset(ins.in[3]);
    if (up::IsRodataRef(ins.in[3])) sum.space |= 1u;
    sum.off[1] = sum.off[2] = up::RefOffset(ins.in[2]);
    sum.n = count;
    Dispatch(kPAddEW, sum, sum.n, kElementwiseGroup);
  };
  const auto op = static_cast<up::OpCode>(ins.opcode);
  switch (op) {
    case up::OpCode::kGemmNN:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      if (ins.flags & up::kFlagEpilogueBias) ref(3, ins.in[3]);
      a.m = u32(ins.out[0]); a.n = u32(ins.out[1]); a.k = u32(ins.out[2]);
      a.f[0] = 1.0f;
      // flags: bit 3 = bias present, low bits = EpilogueAct (1..3)
      a.flags = static_cast<uint32_t>(up::EpilogueActOf(ins.flags)) |
                ((ins.flags & up::kFlagEpilogueBias) ? 8u : 0u);
      DispatchGemm(kPGemmNN, a, false, false, 4);
      EncodeAddend(a.m * a.n);
      return {};
    case up::OpCode::kGemmNT:
    case up::OpCode::kGemmTN:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      a.m = u32(ins.out[0]); a.n = u32(ins.out[1]); a.k = u32(ins.out[2]);
      a.f[0] = 1.0f;
      if (op == up::OpCode::kGemmNT)
        DispatchGemm(kPGemmNT, a, false, true, 4);
      else
        DispatchGemm(kPGemmTN, a, true, false, 4);
      EncodeAddend(a.m * a.n);
      return {};
    case up::OpCode::kGemmAccNN:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      a.m = u32(ins.out[0]); a.n = u32(ins.out[1]); a.k = u32(ins.out[2]);
      a.f[0] = BitsToF32(ins.in[3]);
      DispatchGemm(kPGemmAcc, a, false, false, 4);
      return {};
    case up::OpCode::kGemmNNBF16:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      if (ins.flags & up::kFlagEpilogueBias) ref(3, ins.in[3]);
      a.m = u32(ins.out[0]); a.n = u32(ins.out[1]); a.k = u32(ins.out[2]);
      a.f[0] = 1.0f;
      a.flags = static_cast<uint32_t>(up::EpilogueActOf(ins.flags)) |
                ((ins.flags & up::kFlagEpilogueBias) ? 8u : 0u);
      DispatchGemm(kPGemmNNBF16, a, false, false, 2);
      return {};
    case up::OpCode::kGemmNTBF16:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      a.m = u32(ins.out[0]); a.n = u32(ins.out[1]); a.k = u32(ins.out[2]);
      a.f[0] = 1.0f;
      DispatchGemm(kPGemmNTBF16, a, false, true, 2);
      return {};
    case up::OpCode::kGemmNNQ8:
    case up::OpCode::kGemmNTQ8:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      a.m = u32(ins.out[0]); a.n = u32(ins.out[1]); a.k = u32(ins.out[2]);
      a.f[0] = BitsToF32(ins.in[3]);  // dequant scale
      a.flags = static_cast<uint32_t>(up::EpilogueActOf(ins.flags));
      if (op == up::OpCode::kGemmNNQ8)
        DispatchGemm(kPGemmNNQ8, a, false, false, 1);
      else
        DispatchGemm(kPGemmNTQ8, a, false, true, 1);
      return {};
    case up::OpCode::kAddEW:
    case up::OpCode::kMulEW:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      a.n = u32(ins.out[0]);
      Dispatch(op == up::OpCode::kAddEW ? kPAddEW : kPMulEW, a, a.n,
               kElementwiseGroup);
      return {};
    case up::OpCode::kAddBias:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      a.rows = u32(ins.out[0]); a.cols = u32(ins.out[1]);
      Dispatch(kPAddBias, a, a.rows * a.cols, kElementwiseGroup);
      return {};
    case up::OpCode::kReluFwd:
    case up::OpCode::kGeluFwd:
    case up::OpCode::kSiluFwd:
      ref(0, ins.in[0]); ref(1, ins.in[1]);
      a.n = u32(ins.out[0]);
      Dispatch(op == up::OpCode::kReluFwd   ? kPReluFwd
               : op == up::OpCode::kGeluFwd ? kPGeluFwd
                                            : kPSiluFwd,
               a, a.n, kElementwiseGroup);
      return {};
    case up::OpCode::kReluBwd:
    case up::OpCode::kGeluBwd:
    case up::OpCode::kSiluBwd:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      a.n = u32(ins.out[0]);
      Dispatch(op == up::OpCode::kReluBwd   ? kPReluBwd
               : op == up::OpCode::kGeluBwd ? kPGeluBwd
                                            : kPSiluBwd,
               a, a.n, kElementwiseGroup);
      return {};
    case up::OpCode::kScale:
      ref(0, ins.in[0]); ref(1, ins.in[1]);
      a.f[0] = BitsToF32(ins.in[2]);
      a.n = u32(ins.out[0]);
      Dispatch(kPScale, a, a.n, kElementwiseGroup);
      return {};
    case up::OpCode::kFill:
      ref(0, ins.in[0]);
      a.f[0] = BitsToF32(ins.in[1]);
      a.n = u32(ins.out[0]);
      Dispatch(kPFill, a, a.n, kElementwiseGroup);
      return {};
    case up::OpCode::kCopy:
      ref(0, ins.in[0]); ref(1, ins.in[1]);
      a.n = u32(ins.out[0]);
      Dispatch(kPCopy, a, a.n, kElementwiseGroup);
      return {};
    case up::OpCode::kAccumulate:
      ref(0, ins.in[0]); ref(1, ins.in[1]);
      a.n = u32(ins.out[0]);
      Dispatch(kPAccumulate, a, a.n, kElementwiseGroup);
      return {};
    case up::OpCode::kFusedMap:
      ref(0, ins.in[0]); ref(3, ins.in[3]);
      if (ins.in[1] != up::kNullRef) ref(1, ins.in[1]);
      if (ins.in[2] != up::kNullRef) ref(2, ins.in[2]);
      a.n = u32(ins.out[0]);
      a.flags = u32(ins.out[1]);
      a.f[0] = BitsToF32(ins.out[2]);
      a.f[1] = BitsToF32(ins.out[2] >> 32);
      Dispatch(kPFusedMap, a, a.n, kElementwiseGroup);
      return {};
    case up::OpCode::kSgdStep: {
      ref(0, ins.in[0]); ref(1, ins.in[1]);
      a.n = u32(ins.out[0]);
      const bool clip = EncodeClipScale(a, ins.out[1]);
      a.f[0] = params.lr; a.f[1] = params.weight_decay;
      Dispatch(clip ? kPSgdClip : kPSgd, a, a.n, kElementwiseGroup, clip);
      return {};
    }
    case up::OpCode::kAdamWStep: {
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]); ref(3, ins.in[3]);
      a.n = u32(ins.out[0]);
      const bool clip = EncodeClipScale(a, ins.out[1]);
      // The per-step factors exactly as the CPU kernel hoists them.
      const float step = static_cast<float>(params.step);
      a.f[0] = params.lr; a.f[1] = params.beta1; a.f[2] = params.beta2;
      a.f[3] = params.eps; a.f[4] = params.weight_decay;
      a.f[5] = 1.0f / (1.0f - std::pow(params.beta1, step));
      a.f[6] = 1.0f / (1.0f - std::pow(params.beta2, step));
      Dispatch(clip ? kPAdamWClip : kPAdamW, a, a.n, kElementwiseGroup, clip);
      return {};
    }
    case up::OpCode::kReduceRows:
      ref(0, ins.in[0]); ref(1, ins.in[1]);
      a.rows = u32(ins.out[0]); a.cols = u32(ins.out[1]);
      Dispatch(kPReduceRows, a, a.cols, kElementwiseGroup);
      return {};
    case up::OpCode::kLayerNormFwd:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]); ref(3, ins.in[3]);
      ref(4, ins.out[1]); ref(5, ins.out[2]);
      a.rows = hi(ins.out[0]); a.cols = lo(ins.out[0]);
      Dispatch(kPLnFwd, a, a.rows * 32u, kElementwiseGroup);
      return {};
    case up::OpCode::kLayerNormBwd:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]); ref(3, ins.in[3]);
      ref(4, ins.out[0]); ref(5, ins.out[1]);
      a.rows = hi(ins.out[2]); a.cols = lo(ins.out[2]);
      Dispatch(kPLnBwd, a, a.rows * 32u, kElementwiseGroup);
      return {};
    case up::OpCode::kRmsNormFwd:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]); ref(3, ins.in[3]);
      a.rows = hi(ins.out[0]); a.cols = lo(ins.out[0]);
      Dispatch(kPRmsFwd, a, a.rows * 32u, kElementwiseGroup);
      return {};
    case up::OpCode::kRmsNormBwd:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]); ref(3, ins.in[3]);
      ref(4, ins.out[0]);
      a.rows = hi(ins.out[1]); a.cols = lo(ins.out[1]);
      Dispatch(kPRmsBwd, a, a.rows * 32u, kElementwiseGroup);
      return {};
    case up::OpCode::kClipNorm: {
      ref(0, ins.in[0]);
      a.n = u32(ins.out[0]);
      EncodeClipScale(a, ins.in[1], /*slot=*/0);
      Dispatch(kPClipApply, a, a.n, kElementwiseGroup, true);
      return {};
    }
    case up::OpCode::kRopeFwd:
    case up::OpCode::kRopeBwd: {
      ref(0, ins.in[0]); ref(1, ins.in[1]);
      a.B = hi(ins.out[0]); a.S = lo(ins.out[0]);
      a.H = hi(ins.out[1]); a.D = lo(ins.out[1]);
      const float base = BitsToF32(ins.out[2]);
      a.f[0] = std::pow(base, -2.0f / static_cast<float>(a.D));
      Dispatch(op == up::OpCode::kRopeFwd ? kPRopeFwd : kPRopeBwd, a,
               a.B * a.S * a.H, kElementwiseGroup);
      return {};
    }
    // The attention family (docs/runtime.md): each primitive is one or
    // more batched GEMMs over the [B*S, H*d] activations plus a row
    // softmax kernel — the same tile machinery as the projections.
    // Operand kinds: 0 = activation [B*S, H*d] (per-batch S*D, per-head
    // d), 1 = probability matrix [B*H*S, S] (per-batch H*S*S, per-head
    // S*S).
    case up::OpCode::kAttnFwd: {
      const uint32_t B = hi(ins.out[1]), S = lo(ins.out[1]);
      const uint32_t H = hi(ins.out[2]), d = lo(ins.out[2]);
      const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
      // scores = (Q K^T) / sqrt(d) into the P cache, then a causal softmax
      // in place, then O = P V.
      EncodeAttentionGemm(kPGemmNT, false, true, ins.in[0], ins.in[1],
                          ins.out[0], B, S, H, d, 0, 0, 1, S, S, d,
                          inv_sqrt_d);
      ref(0, ins.out[0]);
      a.rows = B * H * S; a.cols = S;
      Dispatch(kPAttnSoftmax, a, a.rows * 32u, kElementwiseGroup);
      EncodeAttentionGemm(kPGemmNN, false, false, ins.out[0], ins.in[2],
                          ins.in[3], B, S, H, d, 1, 0, 0, S, d, S, 1.0f);
      return {};
    }
    case up::OpCode::kAttnDP: {  // dP = dO V^T
      const uint32_t B = hi(ins.out[0]), S = lo(ins.out[0]);
      const uint32_t H = hi(ins.out[1]), d = lo(ins.out[1]);
      EncodeAttentionGemm(kPGemmNT, false, true, ins.in[0], ins.in[1],
                          ins.in[2], B, S, H, d, 0, 0, 1, S, S, d, 1.0f);
      return {};
    }
    case up::OpCode::kAttnDV: {  // dV = P^T dO
      const uint32_t B = hi(ins.out[0]), S = lo(ins.out[0]);
      const uint32_t H = hi(ins.out[1]), d = lo(ins.out[1]);
      EncodeAttentionGemm(kPGemmTN, true, false, ins.in[0], ins.in[1],
                          ins.in[2], B, S, H, d, 1, 0, 0, S, d, S, 1.0f);
      return {};
    }
    case up::OpCode::kAttnDQ: {  // dQ = (dS K) / sqrt(d)
      const uint32_t B = hi(ins.out[0]), S = lo(ins.out[0]);
      const uint32_t H = hi(ins.out[1]), d = lo(ins.out[1]);
      EncodeAttentionGemm(kPGemmNN, false, false, ins.in[0], ins.in[1],
                          ins.in[2], B, S, H, d, 1, 0, 0, S, d, S,
                          1.0f / std::sqrt(static_cast<float>(d)));
      return {};
    }
    case up::OpCode::kAttnDK: {  // dK = (dS^T Q) / sqrt(d)
      const uint32_t B = hi(ins.out[0]), S = lo(ins.out[0]);
      const uint32_t H = hi(ins.out[1]), d = lo(ins.out[1]);
      EncodeAttentionGemm(kPGemmTN, true, false, ins.in[0], ins.in[1],
                          ins.in[2], B, S, H, d, 1, 0, 0, S, d, S,
                          1.0f / std::sqrt(static_cast<float>(d)));
      return {};
    }
    case up::OpCode::kSoftmaxRowsBwd:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      a.rows = hi(ins.out[0]); a.cols = lo(ins.out[0]);
      Dispatch(kPSoftmaxRowsBwd, a, a.rows * 32u, kElementwiseGroup);
      return {};
    default:
      return std::unexpected("opcode " + std::to_string(ins.opcode) +
                             " has no GPU kernel");
  }
}

void MetalBackend::EncodeAttentionGemm(
    Pipe pipe, bool at, bool bt, uint64_t a_ref, uint64_t b_ref,
    uint64_t c_ref, uint32_t B, uint32_t S, uint32_t H, uint32_t d,
    uint32_t a_kind, uint32_t b_kind, uint32_t c_kind, uint32_t m,
    uint32_t n, uint32_t k, float alpha) {
  KArgs a;
  const uint64_t D = uint64_t{H} * d;
  auto set = [&](int slot, uint64_t r, uint32_t kind, uint32_t& ld) {
    a.off[slot] = up::RefOffset(r);
    if (up::IsRodataRef(r)) a.space |= 1u << slot;
    if (kind == 0) {
      a.bs[2 * slot] = uint64_t{S} * D;
      a.bs[2 * slot + 1] = d;
      ld = static_cast<uint32_t>(D);
    } else {
      a.bs[2 * slot] = uint64_t{H} * S * S;
      a.bs[2 * slot + 1] = uint64_t{S} * S;
      ld = S;
    }
  };
  set(0, a_ref, a_kind, a.lda);
  set(1, b_ref, b_kind, a.ldb);
  set(2, c_ref, c_kind, a.ldc);
  a.m = m; a.n = n; a.k = k;
  a.batch = B * H; a.batch_h = H;
  a.f[0] = alpha;
  DispatchGemm(pipe, a, at, bt, 4);
}

std::expected<void, std::string> MetalBackend::Execute(
    const up::UpdateInstruction& ins, const StepParams& params) {
  @autoreleasepool {
    if (!arena_buf_) return std::unexpected("no plan bound");
    const auto op = static_cast<up::OpCode>(ins.opcode);
    const InstructionExtents* ex = ExtentsOf(ins);
    if (!ex && op != up::OpCode::kNop)
      return std::unexpected("instruction " + std::to_string(ins.opcode) +
                             " failed re-validation at dispatch");
    if (!IsGpuOpcode(op) || !DimsFit32(op, ins)) {
      // CPU-resident: wait for any pending GPU work it depends on.
      if (ex && HazardWithPending(*ex))
        if (auto r = Flush(); !r) return r;
      if (!profile_) return cpu_->Execute(ins, params);
      const auto t0 = std::chrono::steady_clock::now();
      auto r = cpu_->Execute(ins, params);
      ProfRow& row = prof_["cpu opcode " + std::to_string(ins.opcode)];
      row.seconds += std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
      row.count++;
      return r;
    }
    if (auto r = EnsureEncoder(); !r) return r;
    prof_pipes_.clear();
    if (auto r = Encode(ins, params); !r) return r;
    RecordPending(*ex);
    if (profile_) {
      if (auto r = Flush(); !r) return r;
      auto dim = [](uint64_t v) {
        return v >> 32 ? std::to_string(v >> 32) + "/" +
                             std::to_string(v & 0xFFFFFFFFu)
                       : std::to_string(v);
      };
      ProfRow& row = prof_[prof_pipes_ + " " + dim(ins.out[0]) + "x" +
                           dim(ins.out[1]) + "x" + dim(ins.out[2])];
      row.seconds += last_gpu_seconds_;
      row.count++;
    }
    return {};
  }
}

std::expected<void, std::string> MetalBackend::Flush() {
  @autoreleasepool {
    if (!enc_) return {};
    [enc_ endEncoding];
    [cmd_ commit];
    [cmd_ waitUntilCompleted];
    const MTLCommandBufferStatus status = cmd_.status;
    NSError* err = cmd_.error;
    if (profile_) last_gpu_seconds_ = cmd_.GPUEndTime - cmd_.GPUStartTime;
    enc_ = nil;
    cmd_ = nil;
    pending_reads_.clear();
    pending_writes_.clear();
    // Anything short of Completed means the arena holds a mixture of
    // finished and unfinished work: refuse to continue on it.
    if (status != MTLCommandBufferStatusCompleted)
      return std::unexpected(NsError(err, "GPU command buffer did not complete"));
    return {};
  }
}

void MetalBackend::PrintProfile() const {
  std::vector<std::pair<std::string, ProfRow>> rows(prof_.begin(), prof_.end());
  std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
    return a.second.seconds > b.second.seconds;
  });
  double total = 0;
  for (const auto& r : rows) total += r.second.seconds;
  std::fprintf(stderr, "metal profile: %.3f s over %zu keys\n", total,
               rows.size());
  for (const auto& r : rows)
    std::fprintf(stderr, "  %8.3f s  %5.1f%%  %7llu x %8.1f us  %s\n",
                 r.second.seconds,
                 total > 0 ? 100.0 * r.second.seconds / total : 0.0,
                 (unsigned long long)r.second.count,
                 1e6 * r.second.seconds / (double)r.second.count,
                 r.first.c_str());
}

}  // namespace

bool MetalBackendAvailable() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device != nil;
  }
}

std::expected<std::unique_ptr<ExecutorBackend>, std::string>
CreateMetalBackend() {
  return MetalBackend::Create();
}

}  // namespace seeml::update_rt
