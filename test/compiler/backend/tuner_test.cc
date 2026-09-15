// =============================================================================
// Backend partition tests: architecture/ (host detection sanity, the host
// key a kernel-policy table is keyed on, the analytic tiling hypothesis),
// tuner/ (the kernel-policy table reader and the policy resolution the
// compiler and the bench share), and the trainer/ GPU kernel emitter.
// =============================================================================

#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "compiler/backend/architecture/host_arch.h"
#include "compiler/backend/trainer/kernel_emitter.h"
#include "compiler/backend/tuner/kernel_policy_table.h"
#include "test/framework/seetest.h"

namespace {

using namespace seeml::update;

// =============================================================================
// architecture/
// =============================================================================

TEST(HostArch, DetectionYieldsUsableGeometry) {
  const HostArchInfo arch = DetectHostArch();
  EXPECT_TRUE(arch.isa == "arm64" || arch.isa == "x86_64" ||
              arch.isa == "unknown");
  EXPECT_GE(arch.simd_width_f32, 4u);
  EXPECT_GE(arch.physical_cores, 1u);
  EXPECT_GE(arch.cache_line_bytes, 32u);
  EXPECT_FALSE(arch.cpu_model.empty());
}

TEST(HostArch, HostKeyIsPureCanonicalAndOneLine) {
  HostArchInfo arch;
  arch.isa = "arm64";
  arch.cpu_model = "  Apple   M5 ";
  arch.physical_cores = 10;
  arch.l1d_bytes = 131072;
  arch.l2_bytes = 16777216;
  arch.simd_width_f32 = 4;
  EXPECT_EQ(HostKey(arch),
            "arm64;Apple M5;cores=10;l1d=131072;l2=16777216;simd=4");
  EXPECT_EQ(HostKey(arch), HostKey(arch));  // pure
  // The separator cannot come from the brand string, and line breaks are
  // whitespace — collapsed like any other run of it.
  arch.cpu_model = "Odd;Vendor\nCPU";
  EXPECT_EQ(HostKey(arch), "arm64;Odd_Vendor CPU;cores=10;l1d=131072;"
                           "l2=16777216;simd=4");
  // An empty brand string reads "unknown" rather than an empty field.
  arch.cpu_model = "";
  EXPECT_EQ(HostKey(arch).find("arm64;unknown;"), 0u);
  // The detected host's key is the one the bench writes and the compiler
  // reads: same function, same machine, same string.
  EXPECT_EQ(HostKey(DetectHostArch()), HostKey(DetectHostArch()));
}

TEST(HostArch, TilingIsPureAndSimdAligned) {
  HostArchInfo arch;
  arch.simd_width_f32 = 8;
  arch.l1d_bytes = 64u << 10;
  arch.l2_bytes = 1u << 20;

  const GemmTiling a = SuggestGemmTiling(arch);
  const GemmTiling b = SuggestGemmTiling(arch);
  EXPECT_TRUE(a == b);  // pure function of the host description

  EXPECT_GT(a.mc, 0u);
  EXPECT_GT(a.kc, 0u);
  EXPECT_EQ(a.nc, 4u * 8u);
  EXPECT_EQ(a.mc % 8, 0u);
  EXPECT_EQ(a.kc % 8, 0u);

  // kc x nc panel of B must fit in half of L1, mc x kc panel of A in half
  // of L2 — the contract the heuristic documents.
  EXPECT_LE(a.kc * a.nc * sizeof(float), arch.l1d_bytes / 2);
  EXPECT_LE(a.mc * a.kc * sizeof(float), arch.l2_bytes / 2);
}

TEST(HostArch, UnknownCachesFallBackToUsableTiling) {
  HostArchInfo arch;  // no cache info at all
  const GemmTiling t = SuggestGemmTiling(arch);
  EXPECT_GT(t.mc, 0u);
  EXPECT_GT(t.kc, 0u);
  EXPECT_GT(t.nc, 0u);
}

// =============================================================================
// tuner/ — the kernel-policy table
// =============================================================================

constexpr const char* kTable = R"({
  "schema": 1,
  "tool": "tool/autotune.py",
  "hosts": {
    "arm64;Apple M5;cores=10;l1d=131072;l2=16777216;simd=4": {
      "cpu": {"gemm_tile_k": 128, "gemm_tile_n": 512},
      "tuned": {"date": "2026-09-15", "arms": [{"gemm_tile_k": 64,
                "gemm_tile_n": 256, "speedup": 1.0, "label": "default"}],
                "note": "quotes \"inside\" and a é escape"}
    },
    "x86_64;Intel(R) Xeon(R);cores=8;l1d=32768;l2=1048576;simd=8": {
      "cpu": {"gemm_tile_k": 64, "gemm_tile_n": 256}
    },
    "gpu-only-host": {"metal": {"some": "future policy"}}
  }
})";

TEST(KernelPolicyTable, ParsesEntriesAndIgnoresProvenance) {
  auto table = ParseKernelPolicyTable(kTable);
  ASSERT_TRUE(table.has_value());
  EXPECT_EQ(table->hosts.size(), 2u);  // the GPU-only host has no cpu entry
  const KernelPolicyEntry* m5 =
      table->Find("arm64;Apple M5;cores=10;l1d=131072;l2=16777216;simd=4");
  ASSERT_TRUE(m5 != nullptr);
  EXPECT_EQ(m5->gemm_tile_k, 128u);
  EXPECT_EQ(m5->gemm_tile_n, 512u);
  EXPECT_TRUE(table->Find("nobody") == nullptr);
}

TEST(KernelPolicyTable, RejectsWhatItCannotTrust) {
  // Every rejection names the KernelPolicy unit — the diagnostics contract.
  auto expect_error = [](std::string_view json, std::string_view needle) {
    auto r = ParseKernelPolicyTable(json);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().find("KernelPolicy: "), 0u);
    EXPECT_TRUE(r.error().find(needle) != std::string::npos);
  };
  expect_error("", "not valid JSON");
  expect_error("{\"schema\": 1, \"hosts\": {}} trailing", "trailing");
  expect_error("[1, 2]", "must be a JSON object");
  expect_error("{\"schema\": 2, \"hosts\": {}}", "\"schema\" must be 1");
  expect_error("{\"schema\": 1}", "\"hosts\" must be an object");
  expect_error("{\"schema\": 1, \"hosts\": {\"h\": 3}}", "must be an object");
  expect_error("{\"schema\": 1, \"hosts\": {\"h\": {\"cpu\": {\"gemm_tile_k\": 64}}}}",
               "gemm_tile_n must be a number");
  expect_error("{\"schema\": 1, \"hosts\": {\"h\": {\"cpu\": {\"gemm_tile_k\": 6, "
               "\"gemm_tile_n\": 16}}}}",
               "not a multiple of the kernel's 4-wide unroll");
  expect_error("{\"schema\": 1, \"hosts\": {\"h\": {\"cpu\": {\"gemm_tile_k\": 64, "
               "\"gemm_tile_n\": 0}}}}",
               "must be positive");
  expect_error("{\"schema\": 1, \"hosts\": {\"h\": {\"cpu\": {\"gemm_tile_k\": 64.5, "
               "\"gemm_tile_n\": 16}}}}",
               "whole number");
  expect_error("{\"schema\": 1, \"hosts\": {\"h\": {\"cpu\": {\"gemm_tile_k\": 64, "
               "\"gemm_tile_n\": 16,}}}}",
               "expected a string key");
  expect_error("{\"schema\": 1, \"schema\": 1, \"hosts\": {}}", "duplicate key");
  expect_error("{\"schema\": 1, \"hosts\": {\"a\\x\": {}}}", "unknown escape");
}

TEST(KernelPolicyTable, LoadsFromDiskAndNamesThePath) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("seeml_kernel_policy_test_" + std::to_string(::getpid()) + ".json");
  {
    std::ofstream out(path);
    out << kTable;
  }
  auto table = LoadKernelPolicyTable(path.string());
  ASSERT_TRUE(table.has_value());
  EXPECT_EQ(table->hosts.size(), 2u);
  {
    std::ofstream out(path);
    out << "{\"schema\": 1, \"hosts\": ";  // truncated
  }
  auto bad = LoadKernelPolicyTable(path.string());
  ASSERT_FALSE(bad.has_value());
  EXPECT_TRUE(bad.error().find(path.string()) != std::string::npos);
  std::filesystem::remove(path);
  auto missing = LoadKernelPolicyTable(path.string());
  ASSERT_FALSE(missing.has_value());
  EXPECT_TRUE(missing.error().find("cannot read") != std::string::npos);
}

TEST(KernelPolicyTable, GemmTilesFlagIsStrict) {
  auto ok = ParseGemmTilesFlag("128,512");
  ASSERT_TRUE(ok.has_value());
  EXPECT_EQ(ok->gemm_tile_k, 128u);
  EXPECT_EQ(ok->gemm_tile_n, 512u);
  for (const char* bad : {"", "64", "64,", ",64", "64x256", "6,16", "0,16",
                          "64,0", "64,16,4", "1e2,16", "-64,16",
                          "99999999999,16"})
    EXPECT_FALSE(ParseGemmTilesFlag(bad).has_value());
}

TEST(KernelPolicyTable, ResolutionPrecedenceIsFlagTableDefault) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("seeml_kernel_policy_resolve_" + std::to_string(::getpid()) + ".json");
  const std::string here = HostKey(DetectHostArch());
  {
    std::ofstream out(path);
    out << "{\"schema\": 1, \"hosts\": {\"" << here
        << "\": {\"cpu\": {\"gemm_tile_k\": 32, \"gemm_tile_n\": 64}}, "
           "\"elsewhere\": {\"cpu\": {\"gemm_tile_k\": 16, \"gemm_tile_n\": 32}}}}";
  }
  // Nothing asked: the defaults, with the host named.
  auto d = ResolveKernelPolicy({});
  ASSERT_TRUE(d.has_value());
  EXPECT_EQ(d->source, "default");
  EXPECT_EQ(d->tiles.gemm_tile_k, 0u);
  EXPECT_EQ(d->tiles.gemm_tile_n, 0u);
  EXPECT_EQ(d->host_key, here);
  EXPECT_TRUE(d->note.empty());
  // The table's entry for this host.
  KernelPolicyRequest req;
  req.table_path = path.string();
  auto t = ResolveKernelPolicy(req);
  ASSERT_TRUE(t.has_value());
  EXPECT_EQ(t->source, "table");
  EXPECT_EQ(t->tiles.gemm_tile_k, 32u);
  EXPECT_EQ(t->tiles.gemm_tile_n, 64u);
  // Cross-compiling: another host's entry by key.
  req.target_host = "elsewhere";
  auto x = ResolveKernelPolicy(req);
  ASSERT_TRUE(x.has_value());
  EXPECT_EQ(x->source, "table");
  EXPECT_EQ(x->tiles.gemm_tile_k, 16u);
  EXPECT_EQ(x->host_key, "elsewhere");
  // A host the table does not know: a note and the defaults, not an error.
  req.target_host = "nowhere";
  auto n = ResolveKernelPolicy(req);
  ASSERT_TRUE(n.has_value());
  EXPECT_EQ(n->source, "default");
  EXPECT_FALSE(n->note.empty());
  EXPECT_EQ(n->tiles.gemm_tile_k, 0u);
  // An explicit geometry beats the table.
  req.explicit_tiles = KernelPolicyEntry{.gemm_tile_k = 8, .gemm_tile_n = 16};
  req.target_host.reset();
  auto f = ResolveKernelPolicy(req);
  ASSERT_TRUE(f.has_value());
  EXPECT_EQ(f->source, "flag");
  EXPECT_EQ(f->tiles.gemm_tile_k, 8u);
  // --target-host without a table is a usage error; a bad explicit tile too.
  KernelPolicyRequest orphan;
  orphan.target_host = "elsewhere";
  EXPECT_FALSE(ResolveKernelPolicy(orphan).has_value());
  KernelPolicyRequest bad;
  bad.explicit_tiles = KernelPolicyEntry{.gemm_tile_k = 6, .gemm_tile_n = 16};
  EXPECT_FALSE(ResolveKernelPolicy(bad).has_value());
  std::filesystem::remove(path);
}

// =============================================================================
// trainer/ — GPU kernel emitter
// =============================================================================

TEST(KernelEmitter, GpuTilingClampsHostTiling) {
  const GpuTiling t = GpuTilingFromHost({.mc = 512, .kc = 96, .nc = 16});
  EXPECT_EQ(t.tile_m, 32u);  // clamped down from 512
  EXPECT_EQ(t.tile_k, 32u);
  EXPECT_EQ(t.tile_n, 16u);

  const GpuTiling tiny = GpuTilingFromHost({.mc = 4, .kc = 4, .nc = 4});
  EXPECT_EQ(tiny.tile_m, 8u);  // clamped up to the SIMD-group minimum
}

}  // namespace
