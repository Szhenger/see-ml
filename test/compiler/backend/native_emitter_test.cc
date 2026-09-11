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

#include "compiler/backend/trainer/native_emitter.h"
#include "compiler/driver/update_compiler.h"
#include "runtime/engine/update_engine.h"
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
  ASSERT_OK_AND_ASSIGN(EmitPaths paths,
                       EmitNativePackage(compiled.plan, out_dir, RepoRoot()));

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
       {"runtime/engine/update_engine.cc", "runtime/engine/contract.cc",
        "runtime/executor/update_kernels.h", "runtime/executor/gemm.cc",
        "runtime/executor/elementwise.cc", "runtime/executor/activation.cc",
        "runtime/executor/normalization.cc", "runtime/executor/loss.cc",
        "runtime/executor/optimizer.cc", "runtime/executor/attention.cc",
        "runtime/feeder/dataset.cc",
        "runtime/feeder/batch_pipeline.cc", "runtime/custodian/durable_io.cc",
        "runtime/validator/plan_validator.cc",
        "runtime/custodian/checkpoint.cc",
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
  EXPECT_STR_CONTAINS(script, "engine/update_engine");
  // The vendored runtime is threaded: the script must compile the parallel
  // substrate and link with -pthread.
  EXPECT_STR_CONTAINS(script, "source/parallel/parallel_for.cc");
  EXPECT_STR_CONTAINS(script, "feeder/batch_pipeline");
  EXPECT_STR_CONTAINS(script, "-pthread");

  // The architecture analysis reaches the delivered program: the script
  // bakes the host-derived GEMM tiling as build-line defines, overridable
  // (or clearable) through SEEML_TILE_FLAGS for cross-compilation. The
  // suggested tiling always validates on the suggesting host, so the
  // fallback no-define form must not appear here.
  EXPECT_STR_CONTAINS(script, "-DSEEML_GEMM_TILE_K=");
  EXPECT_STR_CONTAINS(script, "-DSEEML_GEMM_TILE_N=");
  EXPECT_STR_CONTAINS(script, "SEEML_TILE_FLAGS");
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
      std::filesystem::path(out_dir) / "runtime/engine/update_engine.cc"));
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

}  // namespace
