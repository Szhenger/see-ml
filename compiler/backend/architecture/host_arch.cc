#include "compiler/backend/architecture/host_arch.h"

#include <algorithm>
#include <thread>

#include "compiler/diagnostics/architecting/error.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#else
#include <unistd.h>
#endif

namespace seeml::update {

namespace {

#if defined(__APPLE__)
uint64_t SysctlU64(const char* name) {
  uint64_t v = 0;
  size_t len = sizeof(v);
  if (sysctlbyname(name, &v, &len, nullptr, 0) != 0) return 0;
  return v;
}
#endif

/// Rounds `v` down to a multiple of `unit`, but never below `unit`.
size_t RoundToUnit(size_t v, size_t unit) {
  return std::max(unit, v - v % unit);
}

/// Whether half of `cache_bytes` can hold even the minimal conforming
/// (simd x simd) f32 panel. Below this, the half-cache contract and the
/// SIMD-multiple contract are jointly unsatisfiable, so the detected value
/// is treated as garbage: SuggestGemmTiling falls back to defaults and
/// ValidateGemmTiling skips the unsatisfiable check.
bool CacheHalfFeasible(uint64_t cache_bytes, size_t simd) {
  return cache_bytes / 2 >= uint64_t{simd} * simd * sizeof(float);
}

/// Largest multiple of `unit` such that (dim x other) f32 fits in `budget`
/// bytes; callers guarantee feasibility (>= unit) via CacheHalfFeasible.
size_t FitDim(uint64_t budget_bytes, size_t other, size_t unit) {
  return RoundToUnit(
      static_cast<size_t>(budget_bytes / (other * sizeof(float))), unit);
}

}  // namespace

HostArchInfo DetectHostArch() {
  HostArchInfo info;

  // ISA and SIMD width are compile-time facts of the target: the update
  // compiler runs on the device it compiles for.
#if defined(__aarch64__) || defined(_M_ARM64)
  info.isa = "arm64";
  info.simd_width_f32 = 4;  // NEON: 128-bit vectors
  info.has_fma = true;      // FMLA is baseline AArch64
#elif defined(__x86_64__) || defined(_M_X64)
  info.isa = "x86_64";
#if defined(__AVX512F__)
  info.simd_width_f32 = 16;
#elif defined(__AVX2__) || defined(__AVX__)
  info.simd_width_f32 = 8;
#else
  info.simd_width_f32 = 4;  // SSE baseline for x86_64
#endif
#if defined(__FMA__)
  info.has_fma = true;
#endif
#endif

#if defined(__APPLE__)
  info.l1d_bytes = SysctlU64("hw.l1dcachesize");
  info.l2_bytes = SysctlU64("hw.l2cachesize");
  if (uint64_t cores = SysctlU64("hw.physicalcpu"); cores > 0)
    info.physical_cores = static_cast<size_t>(cores);
  if (uint64_t line = SysctlU64("hw.cachelinesize"); line > 0)
    info.cache_line_bytes = static_cast<size_t>(line);
#else
#if defined(_SC_LEVEL1_DCACHE_SIZE)
  if (long l1 = sysconf(_SC_LEVEL1_DCACHE_SIZE); l1 > 0)
    info.l1d_bytes = static_cast<uint64_t>(l1);
#endif
#if defined(_SC_LEVEL2_CACHE_SIZE)
  if (long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE); l2 > 0)
    info.l2_bytes = static_cast<uint64_t>(l2);
#endif
#if defined(_SC_LEVEL1_DCACHE_LINESIZE)
  if (long line = sysconf(_SC_LEVEL1_DCACHE_LINESIZE); line > 0)
    info.cache_line_bytes = static_cast<size_t>(line);
#endif
  if (long n = sysconf(_SC_NPROCESSORS_ONLN); n > 0)
    info.physical_cores = static_cast<size_t>(n);
#endif

  if (info.physical_cores == 0) {
    const unsigned hc = std::thread::hardware_concurrency();
    info.physical_cores = hc > 0 ? hc : 1;
  }
  return info;
}

GemmTiling SuggestGemmTiling(const HostArchInfo& arch) {
  namespace architecting = seeml::diag::architecting;
  const size_t simd = std::max<size_t>(arch.simd_width_f32, 4);
  if (arch.l1d_bytes == 0)
    architecting::DetectionFallback(
        architecting::kHostArch,
        "L1d size undetected; assuming 32 KiB for GEMM tiling");
  else if (!CacheHalfFeasible(arch.l1d_bytes, simd))
    architecting::DetectionFallback(
        architecting::kHostArch,
        "detected L1d cannot hold a SIMD panel; assuming 32 KiB for GEMM "
        "tiling");
  if (arch.l2_bytes == 0)
    architecting::DetectionFallback(
        architecting::kHostArch,
        "L2 size undetected; assuming 512 KiB for GEMM tiling");
  else if (!CacheHalfFeasible(arch.l2_bytes, simd))
    architecting::DetectionFallback(
        architecting::kHostArch,
        "detected L2 cannot hold a SIMD panel; assuming 512 KiB for GEMM "
        "tiling");
  const bool l1_usable = CacheHalfFeasible(arch.l1d_bytes, simd);
  const bool l2_usable = CacheHalfFeasible(arch.l2_bytes, simd);
  const uint64_t l1 = l1_usable ? arch.l1d_bytes : 32u << 10;
  const uint64_t l2 = l2_usable ? arch.l2_bytes : 512u << 10;

  // Every dimension is fitted downward so the result always satisfies
  // ValidateGemmTiling against the same arch: nc shrinks below the 4-vector
  // ideal when half of L1 cannot hold a (simd x nc) panel, and kc respects
  // both halves (a kc so large that no conforming mc panel fits half of L2
  // would fail the L2 check no matter the mc).
  GemmTiling t;
  // nc: a small register-blocked sweep — 4 vectors of C columns.
  t.nc = std::min(4 * simd, FitDim(l1 / 2, simd, simd));
  // kc: kc x nc f32 panel of B in at most half of L1.
  t.kc = std::min(FitDim(l1 / 2, t.nc, simd), FitDim(l2 / 2, simd, simd));
  // mc: mc x kc f32 panel of A in at most half of L2.
  t.mc = FitDim(l2 / 2, t.kc, simd);
  return t;
}

std::expected<void, std::string> ValidateGemmTiling(const GemmTiling& tiling,
                                                    const HostArchInfo& arch) {
  namespace architecting = seeml::diag::architecting;
  const size_t simd = std::max<size_t>(arch.simd_width_f32, 4);

  const struct { const char* name; size_t v; } dims[] = {
      {"mc", tiling.mc}, {"kc", tiling.kc}, {"nc", tiling.nc}};
  for (const auto& d : dims) {
    if (d.v == 0)
      return architecting::Error(architecting::kHostArch,
                                 std::string(d.name) + " must be nonzero");
    if (d.v % simd != 0)
      return architecting::Error(
          architecting::kHostArch,
          std::string(d.name) + "=" + std::to_string(d.v) +
              " is not a multiple of the SIMD width (" + std::to_string(simd) +
              " f32 lanes)");
  }

  // The cache halves are only a contract when the geometry was detected
  // and can hold at least a minimal SIMD panel — an all-defaults
  // HostArchInfo must accept the fallback tiling, and a degenerate detected
  // cache makes this check jointly unsatisfiable with the SIMD-multiple
  // rule above (SuggestGemmTiling ignores such a value the same way).
  if (CacheHalfFeasible(arch.l1d_bytes, simd) &&
      tiling.kc * tiling.nc * sizeof(float) > arch.l1d_bytes / 2)
    return architecting::Error(
        architecting::kHostArch,
        "kc x nc panel (" + std::to_string(tiling.kc) + " x " +
            std::to_string(tiling.nc) + " f32) exceeds half of L1d (" +
            std::to_string(arch.l1d_bytes) + " B)");
  if (CacheHalfFeasible(arch.l2_bytes, simd) &&
      tiling.mc * tiling.kc * sizeof(float) > arch.l2_bytes / 2)
    return architecting::Error(
        architecting::kHostArch,
        "mc x kc panel (" + std::to_string(tiling.mc) + " x " +
            std::to_string(tiling.kc) + " f32) exceeds half of L2 (" +
            std::to_string(arch.l2_bytes) + " B)");
  return {};
}

}  // namespace seeml::update
