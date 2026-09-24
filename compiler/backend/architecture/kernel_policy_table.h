#ifndef SEEML_COMPILER_BACKEND_ARCHITECTURE_KERNEL_POLICY_TABLE_H_
#define SEEML_COMPILER_BACKEND_ARCHITECTURE_KERNEL_POLICY_TABLE_H_

#include <cstdint>
#include <expected>
#include <map>
#include <optional>
#include <string>
#include <string_view>

// =============================================================================
// The kernel-policy table — the compiler's side of the tuning seam.
//
// Tuning is measurement, and measurement is not the compiler's job: the
// offline tuner (tool/autotune.py, Python plane) sweeps kernel-policy arms
// through seeml-bench on the target host and writes what it measured into
// a JSON table keyed on the host's identity (HostKey, host_arch.h). This
// module reads that table back — strictly, with a purpose-built JSON reader
// (the compiler has no third-party dependencies) — and resolves the policy
// a compile should write into the plan header (schema.h, v11):
//
//   --gemm-tiles K,N      an explicit geometry, measured or not ("flag")
//   --kernel-policy FILE  the table; the entry for this host, or for
//                         --target-host KEY when cross-compiling ("table")
//   neither               the runtime's compiled-in defaults ("default")
//
// The table only ever picks among bitwise-equivalent schedules — every
// tiling the kernels accept computes the same bits — so determinism is
// untouched, and a missing entry is a note (the defaults apply), while a
// table that does not parse or names a geometry the kernels reject is a
// hard error: a bad policy silently costs every training step.
//
// Table schema (version 1; unknown keys are ignored so the tuner can
// record its measurements beside the decision):
//   {"schema": 1,
//    "hosts": {"<host key>": {"cpu": {"gemm_tile_k": K, "gemm_tile_n": N},
//                             ...provenance...}}}
// =============================================================================

namespace seeml::update {

/// One host's CPU policy: the GEMM tile geometry (both nonzero; the K tile a
/// multiple of the kernel's 4-wide unroll).
struct KernelPolicyEntry {
  uint32_t gemm_tile_k = 0;
  uint32_t gemm_tile_n = 0;
  bool operator==(const KernelPolicyEntry&) const = default;
};

struct KernelPolicyTable {
  std::map<std::string, KernelPolicyEntry> hosts;  // by HostKey

  /// The entry for `host_key`, or null.
  const KernelPolicyEntry* Find(std::string_view host_key) const;
};

/// Parses a table from its JSON text. Errors name the KernelPolicy unit and
/// say where in the text the reader stopped.
[[nodiscard]] std::expected<KernelPolicyTable, std::string>
ParseKernelPolicyTable(std::string_view json);

/// Reads and parses the table at `path`.
[[nodiscard]] std::expected<KernelPolicyTable, std::string>
LoadKernelPolicyTable(const std::string& path);

/// Parses the "--gemm-tiles K,N" flag value: two positive integers, K a
/// multiple of 4.
[[nodiscard]] std::expected<KernelPolicyEntry, std::string> ParseGemmTilesFlag(
    std::string_view text);

struct KernelPolicyRequest {
  std::optional<KernelPolicyEntry> explicit_tiles;  // --gemm-tiles
  std::optional<std::string> table_path;            // --kernel-policy
  std::optional<std::string> target_host;           // --target-host
};

struct KernelPolicyChoice {
  /// The tiles to write into the plan header; both zero for "default"
  /// (the runtime's compiled-in geometry).
  KernelPolicyEntry tiles;
  std::string source;    // "flag" | "table" | "default"
  std::string host_key;  // the key looked up (or that would have been)
  std::string note;      // a fallback worth logging, else empty
};

/// Resolves the request in the precedence the header comment lists. A
/// --target-host without a table is a usage error; a table without an
/// entry for the host is a note and the defaults.
[[nodiscard]] std::expected<KernelPolicyChoice, std::string>
ResolveKernelPolicy(const KernelPolicyRequest& request);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_BACKEND_ARCHITECTURE_KERNEL_POLICY_TABLE_H_
