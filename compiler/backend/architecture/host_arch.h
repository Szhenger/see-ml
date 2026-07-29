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
};

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
GemmTiling SuggestGemmTiling(const HostArchInfo& arch);

/// Checks a tiling against the contract SuggestGemmTiling documents for
/// `arch`: every dimension nonzero and a multiple of the SIMD width, the
/// kc x nc panel of B within half of L1, and the mc x kc panel of A within
/// half of L2 (cache halves are only enforced when the size was detected).
/// Errors are formed by diagnostics/architecting — a hint that lies about
/// fitting the cache hierarchy silently costs every training step, so
/// consumers should gate handwritten or deserialized tilings through this.
/// The autotuner's exploratory candidates intentionally probe beyond the
/// cache-fit half of the contract and are not gated.
[[nodiscard]] std::expected<void, std::string> ValidateGemmTiling(
    const GemmTiling& tiling, const HostArchInfo& arch);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_BACKEND_ARCHITECTURE_HOST_ARCH_H_
