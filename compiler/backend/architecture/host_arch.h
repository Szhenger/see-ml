#ifndef SEEML_COMPILER_BACKEND_ARCHITECTURE_HOST_ARCH_H_
#define SEEML_COMPILER_BACKEND_ARCHITECTURE_HOST_ARCH_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

// =============================================================================
// Host architecture analysis — the backend's view of the machine the update
// program will run on. Detection reads the ISA and SIMD capability from the
// compilation target and the cache/core geometry from the OS, and the
// derived tiling hints tell allocation, selection and the driver what microarchitecture details
// matter for efficient code generation (and give the tuner its starting
// point). Everything derived is a pure function of the reported info, so
// hints are reproducible for a given host description.
// =============================================================================

namespace seeml::update {

struct HostArchInfo {
  std::string_view isa = "unknown";  // "arm64" | "x86_64" | "unknown"
  size_t simd_width_f32 = 4;         // f32 lanes per vector register
  bool has_fma = false;
  uint64_t l1d_bytes = 0;            // 0 = undetectable
  uint64_t l2_bytes = 0;             // 0 = undetectable
  size_t physical_cores = 1;
  size_t cache_line_bytes = 64;
  std::string cpu_model = "unknown";  // the OS's CPU brand string, trimmed
};

/// The host identity a measured kernel-policy table is keyed on
/// (tool/autotune.py writes it, the compiler looks it up): ISA, CPU model,
/// physical cores, L1d and L2 bytes, SIMD width — every quantity that
/// changes which GEMM tiling runs fastest, and nothing that does not (no
/// OS version, no hostname). A pure function of the description, so the
/// bench that measured and the compiler that consumes compute the same
/// string on the same machine. Format, one line, no newline:
///   "<isa>;<cpu_model>;cores=<n>;l1d=<bytes>;l2=<bytes>;simd=<lanes>"
std::string HostKey(const HostArchInfo& arch);

/// ISA and SIMD width come from the compilation target (this compiler runs
/// on the device it compiles for — the AOT plan is host=target); cache and
/// core geometry come from sysctl/sysconf, with zeros when undetectable.
HostArchInfo DetectHostArch();

/// Cache-blocking geometry for the CPU NN GEMM core (runtime/executor/
/// gemm.cc): each kc x nc tile of B is packed into an L1-resident panel
/// and swept by a four-row register block of C.
struct GemmTiling {
  size_t mc = 0;  // rows of A per L2-resident band
  size_t kc = 0;  // K tile: rows of B per packed panel (the header's tile_k)
  size_t nc = 0;  // N tile: the vectorized sweep width (the header's tile_n)

  bool operator==(const GemmTiling&) const = default;
};

/// The analytic tiling for `arch`: the packed panel (kc x nc floats) sized
/// to at most half of L1 and at most the runtime panel's capacity
/// (kDefaultGemmPanelFloats, source/plan/schema.h — one constant for both
/// planes); nc the largest power of two not exceeding sqrt(2 * panel), so
/// the sweep is a long vector loop and kc is never starved; mc so an
/// mc x kc band of A fits half of L2. Unknown cache sizes fall back to
/// 32 KiB L1 / 512 KiB L2. All dimensions are multiples of the SIMD width.
///
/// History, because the mistake is instructive (#90): the first model
/// defined nc as a 4-vector REGISTER width for a packed BLIS microkernel
/// the runtime did not have; the runtime consumed it as its N cache tile
/// and ran 1.3-3.3x slower than its own defaults while the nightly bench,
/// built without the emitted flags, measured a configuration no package
/// shipped. Nothing emits a tiling any more — the geometry travels in the
/// plan header (v11), decided from a measured table or left to the runtime
/// defaults — and this model is now derived against the kernel that exists
/// (E1, #80). It remains a HYPOTHESIS: one arm of the offline tuner's
/// sweep (`analytic_gemm_tiles` in bench.json), never a decision.
GemmTiling SuggestGemmTiling(const HostArchInfo& arch);

/// Checks a tiling against the contract SuggestGemmTiling documents for
/// `arch`: every dimension nonzero and a multiple of the SIMD width, the
/// kc x nc panel of B within half of L1, and the mc x kc panel of A within
/// half of L2 (cache halves are only enforced when the size was detected).
/// Errors are formed by diagnostics/architecting — a hint that lies about
/// fitting the cache hierarchy silently costs every training step, so
/// consumers should gate handwritten or deserialized tilings through this.
[[nodiscard]] std::expected<void, std::string> ValidateGemmTiling(
    const GemmTiling& tiling, const HostArchInfo& arch);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_BACKEND_ARCHITECTURE_HOST_ARCH_H_
