// =============================================================================
// NativeEmitter tests: the packaged update program's files exist, the raw
// plan round-trips through the runtime loader, and the generated TUs carry
// the embedded plan symbols the driver links against.
// =============================================================================

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "compiler/backend/packaging/native_emitter.h"
#include "compiler/driver/update_compiler.h"
#include "runtime/dispatcher/update_engine.h"
#include "source/language/model_format.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"
#include "test/support/scoped_temp_dir.h"

namespace {

using namespace seeml::update;
using seeml::testing::BaseConfig;
using seeml::testing::MakeMlp;
using seeml::testing::RepoRoot;
using seeml::testing::ScopedTempDir;

std::string ReadText(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
}

TEST(NativeEmitter, EmitsCompletePackage) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(6, 10, 3, 1);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(BaseConfig(4)).Compile(model));

  const std::string out_dir = (dir.path() / "pkg").string();
  std::filesystem::create_directories(out_dir);
  // A stub the packer left from an earlier pack of this directory must not
  // outlive a fresh decimal emission: build.sh would prefer it.
  std::ofstream(out_dir + "/update_plan_embedded.S") << "stale";
  ASSERT_OK_AND_ASSIGN(EmitPaths paths,
                       EmitNativePackage(compiled.plan, out_dir, RepoRoot()));
  EXPECT_FALSE(std::filesystem::exists(out_dir + "/update_plan_embedded.S"));

  for (const std::string& p :
       {paths.plan_file, paths.embedded_tu, paths.main_tu,
        paths.build_script}) {
    EXPECT_TRUE(std::filesystem::exists(p));
    EXPECT_GT(std::filesystem::file_size(p), 0u);
  }

  // The raw .seeu artifact is directly loadable by the update runtime.
  seeml::update_rt::UpdateEngine engine;
  EXPECT_OK(engine.LoadFromFile(paths.plan_file));

  // The embedded TU defines the aligned byte-array symbols the generated
  // driver expects to link against.
  const std::string embedded = ReadText(paths.embedded_tu);
  EXPECT_STR_CONTAINS(embedded, "kSeemlUpdatePlan");
  EXPECT_STR_CONTAINS(embedded, "kSeemlUpdatePlanSize");
  EXPECT_STR_CONTAINS(embedded, "aligned(64)");

  const std::string main_tu = ReadText(paths.main_tu);
  EXPECT_STR_CONTAINS(main_tu, "UpdateEngine");

  // The package is self-contained: the runtime sources are vendored in and
  // the build script compiles them from the package directory, never from
  // the repository checkout.
  for (const char* rel :
       {"runtime/dispatcher/update_engine.cc", "runtime/dispatcher/contract.cc",
        "runtime/executor/update_kernels.h", "runtime/executor/gemm.cc",
        "runtime/executor/elementwise.cc", "runtime/executor/activation.cc",
        "runtime/executor/normalization.cc", "runtime/executor/loss.cc",
        "runtime/executor/optimizer.cc", "runtime/executor/attention.cc",
        "runtime/pipeline/dataset.cc",
        "runtime/pipeline/batch_pipeline.cc", "runtime/storage/durable_io.cc",
        "runtime/verifier/plan_validator.cc",
        "runtime/storage/checkpoint.cc",
        "runtime/diagnostics/diagnostic.h",
        "runtime/diagnostics/executing/error.h", "source/plan/update_types.h",
        "source/plan/config.h", "source/plan/instruction.h",
        "source/plan/schema.h", "source/identity/hash.h",
        "source/parallel/parallel_for.h",
        "source/parallel/parallel_for.cc"}) {
    EXPECT_TRUE(std::filesystem::exists(
        std::filesystem::path(out_dir) / rel));
  }
  const std::string script = ReadText(paths.build_script);
  EXPECT_STR_CONTAINS(script, "dispatcher/update_engine");
  // The vendored runtime is threaded: the script must compile the parallel
  // substrate and link with -pthread.
  EXPECT_STR_CONTAINS(script, "source/parallel/parallel_for.cc");
  EXPECT_STR_CONTAINS(script, "pipeline/batch_pipeline");
  EXPECT_STR_CONTAINS(script, "-pthread");

  // The GEMM tile geometry travels in the plan header (v11), decided by
  // the driver from the kernel-policy table or the defaults — the script
  // bakes no host-derived tiling (#90: the analytic one it used to emit
  // ran 1.3–3.3x slower than the kernel defaults). The SEEML_TILE_FLAGS
  // hook stays, empty by default, to change the compiled-in default.
  EXPECT_STR_CONTAINS(script, "TILE_FLAGS=\"${SEEML_TILE_FLAGS-}\"");
  EXPECT_TRUE(script.find("${SEEML_TILE_FLAGS--D") == std::string::npos);
  EXPECT_STR_CONTAINS(script, "$TILE_FLAGS");

  // The script consumes whichever embedded-plan TU the package carries:
  // the packer's .incbin stub when present (assembled without the C++
  // flags — it is preprocessed assembly), else the decimal TU emitted here.
  EXPECT_STR_CONTAINS(script, "if [ -f update_plan_embedded.S ]; then");
  EXPECT_STR_CONTAINS(script,
                      "$CXX -c update_plan_embedded.S -o update_plan_embedded.o");
  EXPECT_STR_CONTAINS(
      script, "$CXX $FLAGS -c update_plan_embedded.cc -o update_plan_embedded.o");
  EXPECT_STR_CONTAINS(script, "update_main.o update_plan_embedded.o");

  // The build script is marked executable.
  const auto perms = std::filesystem::status(paths.build_script).permissions();
  EXPECT_TRUE((perms & std::filesystem::perms::owner_exec) !=
              std::filesystem::perms::none);
}

TEST(NativeEmitter, SkipsTheDecimalTUOnRequest) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(6, 10, 3, 4);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(BaseConfig(4)).Compile(model));
  const std::string out_dir = (dir.path() / "pkg").string();

  // A previous emission into the same directory left a decimal TU behind;
  // the stub-bound package must not ship it.
  std::filesystem::create_directories(out_dir);
  std::ofstream(out_dir + "/update_plan_embedded.cc") << "stale";

  EmitOptions options;
  options.embed_plan_tu = false;
  ASSERT_OK_AND_ASSIGN(
      EmitPaths paths,
      EmitNativePackage(compiled.plan, out_dir, RepoRoot(), options));
  EXPECT_TRUE(paths.embedded_tu.empty());
  EXPECT_FALSE(std::filesystem::exists(out_dir + "/update_plan_embedded.cc"));

  // Everything the packer needs is still there: the plan exactly as the
  // default emission writes it, the driver, the vendored runtime, and a
  // build.sh that will assemble the stub the packer adds.
  EXPECT_TRUE(std::filesystem::exists(paths.plan_file));
  EXPECT_TRUE(std::filesystem::exists(paths.main_tu));
  EXPECT_TRUE(std::filesystem::exists(
      std::filesystem::path(out_dir) / "runtime/dispatcher/update_engine.cc"));
  const std::string plan_bytes = ReadText(paths.plan_file);
  EXPECT_EQ(plan_bytes.size(), compiled.plan.size());
  EXPECT_TRUE(std::equal(compiled.plan.begin(), compiled.plan.end(),
                         plan_bytes.begin(), plan_bytes.end(),
                         [](uint8_t a, char b) {
                           return a == static_cast<uint8_t>(b);
                         }));
  const std::string script = ReadText(paths.build_script);
  EXPECT_STR_CONTAINS(script, "update_plan_embedded.S");
  EXPECT_STR_CONTAINS(script, "run tool/pack_update.py");
}

TEST(NativeEmitter, CreatesMissingOutputDir) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(6, 10, 3, 2);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(BaseConfig(4)).Compile(model));
  // The emitter creates nested output directories on demand.
  const std::string nested = (dir.path() / "a" / "b").string();
  EXPECT_OK(EmitNativePackage(compiled.plan, nested, RepoRoot()));
}

TEST(NativeEmitter, RejectsOutputDirBlockedByFile) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(6, 10, 3, 3);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(BaseConfig(4)).Compile(model));
  // A regular file where a directory component must go.
  std::ofstream(dir.File("blocker")) << "not a directory";
  const std::string bogus = (dir.path() / "blocker" / "pkg").string();
  EXPECT_ERROR(EmitNativePackage(compiled.plan, bogus, RepoRoot()));
}

TEST(NativeEmitter, TheStreamedDecimalTUIsCharacterIdenticalAcrossWindows) {
  // The TU is rendered in 16 MiB windows of 256 KiB chunks and written as
  // it goes (E2, #81). The line breaks are keyed off the GLOBAL byte index,
  // so a blob that is not a multiple of anything — chunk, window, or the
  // 24-byte line — must come out exactly as a serial renderer writes it.
  // (The emitter embeds bytes; it does not interpret them.)
  const size_t n = (16u << 20) + (256u << 10) + 12345;
  std::vector<uint8_t> blob(n);
  uint64_t state = 0x9E3779B97F4A7C15ull;
  for (uint8_t& b : blob) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    b = static_cast<uint8_t>(state >> 56);
  }
  ScopedTempDir dir;
  ASSERT_OK_AND_ASSIGN(EmitPaths paths,
                       EmitNativePackage(blob, dir.path(), RepoRoot()));
  std::ifstream f(paths.embedded_tu, std::ios::binary);
  const std::string got((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
  const size_t body = got.find("= {\n");
  ASSERT_TRUE(body != std::string::npos);
  std::string want;
  want.reserve(n * 4);
  for (size_t i = 0; i < n; ++i) {
    want += std::to_string(static_cast<unsigned>(blob[i]));
    want += ',';
    if ((i + 1) % 24 == 0) want += '\n';
  }
  want += "\n};\nconst size_t kSeemlUpdatePlanSize = sizeof(kSeemlUpdatePlan);\n";
  EXPECT_TRUE(got.compare(body + 4, std::string::npos, want) == 0);
}

}  // namespace
