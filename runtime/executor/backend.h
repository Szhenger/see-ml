#ifndef SEEML_RUNTIME_EXECUTOR_BACKEND_H_
#define SEEML_RUNTIME_EXECUTOR_BACKEND_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "runtime/executor/kernel_policy.h"
#include "source/plan/instruction.h"

// =============================================================================
// ExecutorBackend — the seam between the engine's instruction stream and the
// kernels that execute it (roadmap Project 5, G1b-1).
//
// The engine decodes, validates and sequences the plan; a backend executes
// one instruction at a time against the two address spaces the plan
// addresses (the mutable arena and the read-only rodata), and may defer the
// work — a GPU backend batches consecutive instructions into one command
// buffer — until Flush(), after which every byte of the arena is coherent
// for CPU reads. The CPU backend is the existing kernel library behind the
// same call sites, in the same argument order: zero behavioral change, the
// bitwise-deterministic reference every other backend is measured against.
//
// Determinism doctrine (docs/roadmap.md, Project 5): a backend must be
// bitwise-reproducible against itself at any thread or threadgroup count;
// different backends compare at tolerance (FMA contraction differs), and
// the backend name is recorded wherever results are compared.
// =============================================================================

namespace seeml::update_rt {

enum class BackendKind : uint8_t {
  kCpu = 0,    // the portable kernel library (default; builds anywhere)
  kMetal = 1,  // Apple GPU; an error where no device or no Metal build exists
  kAuto = 2,   // Metal when a device is present, else CPU (with a note)
};

const char* BackendKindName(BackendKind kind);
/// "cpu" | "metal" | "auto" (case-sensitive); nullopt for anything else.
std::optional<BackendKind> ParseBackendKind(std::string_view text);

/// The arena alignment every backend may rely on. Page alignment (16 KiB on
/// Apple Silicon, a multiple of every 4 KiB host page) lets a GPU backend
/// wrap the arena as a zero-copy shared buffer; on the CPU path it changes
/// the allocation's address, never a result bit.
inline constexpr size_t kArenaAlignment = 16384;

/// Per-execution scalars the optimizer instructions read from the engine:
/// the scheduled learning rate at this step and the header's AdamW
/// hyperparameters. Every other operand rides in the instruction word.
struct StepParams {
  float lr = 0.0f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  float weight_decay = 0.0f;
  uint64_t step = 0;  // 1-indexed AdamW timestep
};

class ExecutorBackend {
 public:
  virtual ~ExecutorBackend() = default;

  virtual const char* name() const = 0;   // "cpu" | "metal"
  virtual std::string device() const = 0;  // human-readable device label

  /// Binds the address spaces for the plan about to execute. `arena` is
  /// kArenaAlignment-aligned and `arena_bytes` a multiple of it; both
  /// pointers stay valid until the next Bind or the backend's destruction.
  /// `rodata_mapped_bytes` is how much readable memory follows `rodata`
  /// inside the plan blob (>= rodata_bytes): a GPU backend may wrap the
  /// frozen weights zero-copy only if whole pages of the blob cover them.
  /// A backend that cannot bind (no device, a buffer wrap refused) reports
  /// why and leaves nothing half-bound: the PREVIOUS binding, if any,
  /// stays intact and executable — the engine relies on it to keep a
  /// loaded plan usable after a refused re-Load.
  [[nodiscard]] virtual std::expected<void, std::string> Bind(
      uint8_t* arena, uint64_t arena_bytes, const uint8_t* rodata,
      uint64_t rodata_bytes, uint64_t rodata_mapped_bytes) = 0;

  /// Binds the third, read-only address space (plan v17): the source model
  /// file, mapped by the engine, that the eval program's source refs read.
  /// `data` stays valid until the next BindSource or the backend's
  /// destruction; nullptr unbinds. A backend binds it wherever it binds
  /// rodata (a GPU backend wraps the mapping as a buffer).
  [[nodiscard]] virtual std::expected<void, std::string> BindSource(
      const uint8_t* data, uint64_t bytes) = 0;

  /// Tells the backend how the loaded plan wants its kernels run beyond
  /// what the instruction stream says — today the CPU GEMM tile geometry
  /// the compiler wrote into the header (v11). Called after every
  /// successful Bind, before any Execute. A backend whose kernels do not
  /// take the policy (the GPU: its own tiles) ignores it; results never
  /// depend on it on any backend.
  virtual void Configure(const kernels::KernelPolicy& /*policy*/) {}

  /// Executes (or enqueues) one validated instruction. The engine's
  /// validator proved every operand before dispatch; the backend trusts the
  /// instruction blindly, exactly as the pre-seam switch did.
  [[nodiscard]] virtual std::expected<void, std::string> Execute(
      const seeml::update::UpdateInstruction& ins,
      const StepParams& params) = 0;

  /// Completes all deferred work. After a successful Flush the arena's
  /// bytes are what the instruction sequence so far computed, readable by
  /// the CPU (loss slot, eval probabilities, checkpoints, merge deltas).
  [[nodiscard]] virtual std::expected<void, std::string> Flush() = 0;
};

/// The reference backend: the portable kernel library, thread-count
/// invariant by construction (kernel_policy.h).
std::unique_ptr<ExecutorBackend> CreateCpuBackend();

/// A resolved backend choice. `resolved` names what was actually built
/// (kAuto never survives resolution); `note` explains a fallback the
/// caller should log, and is empty otherwise.
struct BackendSelection {
  std::unique_ptr<ExecutorBackend> backend;
  BackendKind resolved = BackendKind::kCpu;
  std::string note;
};

/// Resolves `requested`: kCpu always succeeds; kMetal fails loudly where no
/// device or no Metal build exists; kAuto degrades to the CPU with a note —
/// exactly the behavior a Metal-less Mac (or a CI runner) needs.
[[nodiscard]] std::expected<BackendSelection, std::string> CreateBackend(
    BackendKind requested);

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_EXECUTOR_BACKEND_H_
