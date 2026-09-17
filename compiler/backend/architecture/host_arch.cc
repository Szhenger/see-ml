#include "compiler/backend/architecture/host_arch.h"

#include "source/plan/schema.h"  // kDefaultGemmPanelFloats

#include <algorithm>
#include <cctype>
#include <string>
#include <thread>

#include "compiler/diagnostics/architecting/error.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#else
#include <unistd.h>

#include <fstream>
#include <set>
#include <utility>
#endif

namespace seeml::update {

namespace {

/// Whitespace-trimmed with internal runs collapsed to one space: the brand
/// string is a table key, so two probes of one machine must agree byte for
/// byte, and a padded "Intel(R) Xeon(R)  CPU" must not fork the table.
std::string CanonicalModel(std::string_view raw) {
  std::string out;
  bool pending_space = false;
  for (const char c : raw) {
    if (std::isspace(static_cast<unsigned char>(c))) {
      pending_space = !out.empty();
      continue;
    }
    if (pending_space) out += ' ';
    pending_space = false;
    out += c;
  }
  return out.empty() ? std::string("unknown") : out;
}

#if defined(__APPLE__)
uint64_t SysctlU64(const char* name) {
  uint64_t v = 0;
  size_t len = sizeof(v);
  if (sysctlbyname(name, &v, &len, nullptr, 0) != 0) return 0;
  return v;
}

std::string SysctlString(const char* name) {
  char buf[256] = {};
  size_t len = sizeof(buf) - 1;
  if (sysctlbyname(name, buf, &len, nullptr, 0) != 0) return "";
  return std::string(buf, len > 0 && buf[len - 1] == '\0' ? len - 1 : len);
}
#else
/// The first "model name" line of /proc/cpuinfo (x86); arm64 kernels
/// print no such line, and the key then says "unknown" — cores and caches
/// still distinguish the host class.
std::string ProcCpuinfoModel() {
  std::ifstream f("/proc/cpuinfo");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("model name", 0) != 0) continue;
    const size_t colon = line.find(':');
    if (colon == std::string::npos) continue;
    return line.substr(colon + 1);
  }
  return "";
}
/// _SC_NPROCESSORS_ONLN counts *logical* processors (hardware threads); the
/// physical_cores field means physical cores, as the macOS path's
/// hw.physicalcpu query reports. Count unique (package, core) pairs from the
/// sysfs topology; 0 means unreadable and the caller falls back.
size_t CountPhysicalCoresSysfs() {
  std::set<std::pair<long, long>> cores;
  for (int cpu = 0;; ++cpu) {
    const std::string base =
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    std::ifstream core_f(base + "core_id");
    long core = -1;
    if (!(core_f >> core)) break;
    long pkg = -1;
    std::ifstream pkg_f(base + "physical_package_id");
    pkg_f >> pkg;
    cores.emplace(pkg, core);
  }
  return cores.size();
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

/// Whether an `a` x `b` f32 panel exceeds `cap_bytes`, computed without the
/// multiplication that could wrap: tiling dims arrive from deserialized or
/// handwritten configs, and a wrapped product would pass the very check
/// that exists to reject it. Requires b > 0 (dims are pre-checked nonzero).
bool PanelExceedsBytes(size_t a, size_t b, uint64_t cap_bytes) {
  const uint64_t cap_elems = cap_bytes / sizeof(float);
  return static_cast<uint64_t>(a) > cap_elems / b;
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
  info.cpu_model = CanonicalModel(SysctlString("machdep.cpu.brand_string"));
  info.l1d_bytes = SysctlU64("hw.l1dcachesize");
  info.l2_bytes = SysctlU64("hw.l2cachesize");
  if (uint64_t cores = SysctlU64("hw.physicalcpu"); cores > 0)
    info.physical_cores = static_cast<size_t>(cores);
  if (uint64_t line = SysctlU64("hw.cachelinesize"); line > 0)
    info.cache_line_bytes = static_cast<size_t>(line);
#else
  info.cpu_model = CanonicalModel(ProcCpuinfoModel());
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
  if (const size_t phys = CountPhysicalCoresSysfs(); phys > 0)
    info.physical_cores = phys;
  else if (long n = sysconf(_SC_NPROCESSORS_ONLN); n > 0)
    info.physical_cores = static_cast<size_t>(n);
#endif

  if (info.physical_cores == 0) {
    const unsigned hc = std::thread::hardware_concurrency();
    info.physical_cores = hc > 0 ? hc : 1;
  }
  return info;
}

std::string HostKey(const HostArchInfo& arch) {
  std::string key;
  key.append(arch.isa).append(";");
  // The model may carry anything the firmware wrote; ';' is this key's
  // field separator and a newline would break a one-line record.
  for (const char c : CanonicalModel(arch.cpu_model))
    key += (c == ';' || c == '\n' || c == '\r') ? '_' : c;
  key.append(";cores=").append(std::to_string(arch.physical_cores));
  key.append(";l1d=").append(std::to_string(arch.l1d_bytes));
  key.append(";l2=").append(std::to_string(arch.l2_bytes));
  key.append(";simd=").append(std::to_string(arch.simd_width_f32));
  return key;
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

  // Derived for the kernel that exists (E7, #90 — re-derived after E1,
  // #80, landed it): the CPU NN core copies each kc x nc tile of B into a
  // packed panel of kDefaultGemmPanelFloats and sweeps it with a four-row
  // register block of C. So:
  //   the panel is what must stay in L1 — kc * nc floats, at most half of
  //     L1 and at most the panel's capacity;
  //   nc is the vectorized sweep, a LONG unit-stride loop (the first model
  //     made it a 4-vector register width, which the runtime read as its
  //     N tile and ran 1.3-3.3x slower on): the largest power of two with
  //     nc <= sqrt(2 * panel), which keeps kc within a factor of two of nc
  //     — measured flat from 16x512 to 128x128, and best at the long-K
  //     shapes (a 49k-vocabulary head) when kc is not starved;
  //   kc takes the rest of the panel, a multiple of the quad and of the
  //     SIMD width.
  // Every dimension is fitted downward, so the result always satisfies
  // ValidateGemmTiling against the same arch.
  const uint64_t panel_floats =
      std::min<uint64_t>(l1 / 2 / sizeof(float), kDefaultGemmPanelFloats);
  GemmTiling t;
  size_t nc = simd;
  while (uint64_t{2} * nc * 2 * nc <= 2 * panel_floats) nc *= 2;
  t.nc = RoundToUnit(nc, simd);
  t.kc = std::min(FitDim(panel_floats * sizeof(float), t.nc, simd),
                  FitDim(l2 / 2, simd, simd));
  // mc: mc x kc f32 rows of A in at most half of L2 — the band of rows one
  // task sweeps the panel across.
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
  // rule above (SuggestGemmTiling ignores such a value the same way). The
  // panel comparison itself is overflow-safe: the dims arrive from
  // deserialized or handwritten configs.
  if (CacheHalfFeasible(arch.l1d_bytes, simd) &&
      PanelExceedsBytes(tiling.kc, tiling.nc, arch.l1d_bytes / 2))
    return architecting::Error(
        architecting::kHostArch,
        "kc x nc panel (" + std::to_string(tiling.kc) + " x " +
            std::to_string(tiling.nc) + " f32) exceeds half of L1d (" +
            std::to_string(arch.l1d_bytes) + " B)");
  if (CacheHalfFeasible(arch.l2_bytes, simd) &&
      PanelExceedsBytes(tiling.mc, tiling.kc, arch.l2_bytes / 2))
    return architecting::Error(
        architecting::kHostArch,
        "mc x kc panel (" + std::to_string(tiling.mc) + " x " +
            std::to_string(tiling.kc) + " f32) exceeds half of L2 (" +
            std::to_string(arch.l2_bytes) + " B)");
  return {};
}

}  // namespace seeml::update
