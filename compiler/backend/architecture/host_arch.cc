#include "compiler/backend/architecture/host_arch.h"

#include <algorithm>
#include <thread>

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
  const size_t simd = std::max<size_t>(arch.simd_width_f32, 4);
  const uint64_t l1 = arch.l1d_bytes > 0 ? arch.l1d_bytes : 32u << 10;
  const uint64_t l2 = arch.l2_bytes > 0 ? arch.l2_bytes : 512u << 10;

  GemmTiling t;
  // nc: a small register-blocked sweep — 4 vectors of C columns.
  t.nc = 4 * simd;
  // kc: kc x nc f32 panel of B in at most half of L1.
  t.kc = RoundToUnit(
      static_cast<size_t>(l1 / 2 / (t.nc * sizeof(float))), simd);
  // mc: mc x kc f32 panel of A in at most half of L2.
  t.mc = RoundToUnit(
      static_cast<size_t>(l2 / 2 / (t.kc * sizeof(float))), simd);
  return t;
}

}  // namespace seeml::update
