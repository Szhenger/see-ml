// =============================================================================
// NativeEmitter tests: the packaged update program's files exist, the raw
// plan round-trips through the runtime loader, and the generated TUs carry
// the embedded plan symbols the driver links against.
// =============================================================================

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "compiler/backend/native_emitter.h"
#include "compiler/backend/update_compiler.h"
#include "runtime/update_engine.h"
#include "source/smf.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"
#include "test/support/scoped_temp_dir.h"

namespace {

using namespace seeml::update;
using seeml::testing::BaseConfig;
using seeml::testing::MakeMlp;
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
                       EmitNativePackage(compiled.plan, out_dir,
                                         dir.path().string()));

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

  // The build script is marked executable.
  const auto perms = std::filesystem::status(paths.build_script).permissions();
  EXPECT_TRUE((perms & std::filesystem::perms::owner_exec) !=
              std::filesystem::perms::none);
}

TEST(NativeEmitter, CreatesMissingOutputDir) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(6, 10, 3, 2);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(BaseConfig(4)).Compile(model));
  // The emitter creates nested output directories on demand.
  const std::string nested = (dir.path() / "a" / "b").string();
  EXPECT_OK(EmitNativePackage(compiled.plan, nested, dir.path().string()));
}

TEST(NativeEmitter, RejectsOutputDirBlockedByFile) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(6, 10, 3, 3);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(BaseConfig(4)).Compile(model));
  // A regular file where a directory component must go.
  std::ofstream(dir.File("blocker")) << "not a directory";
  const std::string bogus = (dir.path() / "blocker" / "pkg").string();
  EXPECT_ERROR(EmitNativePackage(compiled.plan, bogus, dir.path().string()));
}

}  // namespace
