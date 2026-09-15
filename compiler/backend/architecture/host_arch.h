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
// derived tiling hints tell the trainer what microarchitecture details
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

/// Cache-blocking geometry for a GEMM microkernel, BLIS-style: a kc-deep
/// panel of B stays resident in L1 across the mc rows of A it multiplies,
/// and the mc x kc panel of A stays resident in L2.
struct GemmTiling {
  size_t mc = 0;  // rows of A per L2-resident panel
  size_t kc = 0;  // shared depth per L1-resident panel
  size_t nc = 0;  // columns of B per register-blocked sweep

  bool operator==(const GemmTiling&) const = default;
};

/// Derives blocking from the cache geometry: kc sized so a kc x nc f32
/// panel fills at most half of L1; mc sized so an mc x kc panel fills at
/// most half of L2; nc a small multiple of the SIMD width. Unknown cache
/// sizes fall back to 32 KiB L1 / 512 KiB L2. All dimensions are rounded
/// to the SIMD width and clamped to sane minima, so the result is always
/// usable geometry.
///
/// This is a HYPOTHESIS, not a decision: the model assumes a packed BLIS
/// microkernel the CPU runtime does not have (its cores are unpacked loop
/// nests whose fast configuration is a 64 KiB B panel — the kernel
/// defaults), and measured 1.3–3.3x slower than those defaults when it was
/// emitted as the package tiling (#90). Nothing emits it any more; it is
/// one arm of the offline tuner's sweep (tool/autotune.py reads it out of
/// bench.json as `analytic_gemm_tiles`) and the GPU kernel emitter's
/// clamp source, so that when a packed microkernel lands (E1, #80) the
/// arm is already measured against the table.
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
