// =============================================================================
// Driver tests: the subsystem usage contracts the driver enforces at each
// phase boundary (frontend / analysis / backend), the diagnostics contract
// at the driver's error boundary, and a full compile proving a correct
// pipeline crosses every gate silently.
// =============================================================================

#include <string>
#include <unordered_map>
#include <vector>

#include "compiler/driver/contract.h"
#include "compiler/driver/update_compiler.h"
#include "compiler/frontend/representation/sir.h"
#include "source/model_format.h"
#include "source/update_types.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"

namespace {

using namespace seeml::update;
namespace sir = seeml::sir;
using seeml::testing::BaseConfig;
using seeml::testing::MakeMlp;

// =============================================================================
// The diagnostics contract
// =============================================================================

TEST(DriverContract, RecognizesEveryRegisteredUnit) {
  EXPECT_TRUE(WellFormedDiagnostic("SMF: truncated file 'm.smf'"));
  EXPECT_TRUE(WellFormedDiagnostic("Ingressor: model is too big"));
  EXPECT_TRUE(WellFormedDiagnostic("Parser: MatMul 'mm0' needs 2 inputs"));
  EXPECT_TRUE(WellFormedDiagnostic(
      "PassManager: SIR invariants violated after pass 'x': y"));
  EXPECT_TRUE(WellFormedDiagnostic("TrainableAutodiff: empty trainable set"));
  EXPECT_TRUE(WellFormedDiagnostic("HostArch: kc must be nonzero"));
  EXPECT_TRUE(WellFormedDiagnostic("UpdateCompiler: unbound value 'v'"));
  EXPECT_TRUE(WellFormedDiagnostic("NativeEmitter: cannot write '/out'"));
}

TEST(DriverContract, RejectsUnattributableDiagnostics) {
  EXPECT_FALSE(WellFormedDiagnostic("something went wrong"));
  EXPECT_FALSE(WellFormedDiagnostic("Frobnicator: unknown unit"));
  EXPECT_FALSE(WellFormedDiagnostic("Parser"));       // no message at all
  EXPECT_FALSE(WellFormedDiagnostic("Parser: "));     // empty message
  EXPECT_FALSE(WellFormedDiagnostic("Parser:x"));     // malformed separator
  EXPECT_FALSE(WellFormedDiagnostic(""));
}

// =============================================================================
// The frontend boundary
// =============================================================================

TEST(DriverContract, FrontendContractAcceptsAWellFormedBuild) {
  sir::Block block;
  sir::Value* in = block.addArgument(sir::DataType::F32, sir::Shape{2, 2});
  sir::Operation* w = block.appendOp("sc_mem.weight");
  sir::Value* wv = w->addResult("w", sir::DataType::F32, sir::Shape{2, 2});

  SmfTensor tensor;
  tensor.is_const = true;
  tensor.dims = {2, 2};

  GraphBuild build;
  build.input = in;
  build.output = wv;
  build.weight_sources[wv] = &tensor;
  EXPECT_TRUE(VerifyFrontendContract(block, build).has_value());
}

TEST(DriverContract, FrontendContractCatchesMisuse) {
  sir::Block block;
  GraphBuild empty;  // no input/output was ever built
  const auto r_empty = VerifyFrontendContract(block, empty);
  EXPECT_FALSE(r_empty.has_value());
  EXPECT_TRUE(WellFormedDiagnostic(r_empty.error()));
  EXPECT_NE(r_empty.error().find("frontend contract"), std::string::npos);

  // A weight source whose value was not declared as sc_mem.weight.
  sir::Value* in = block.addArgument(sir::DataType::F32, sir::Shape{2, 2});
  sir::Operation* relu = block.appendOp("sc_high.relu");
  relu->addOperand(in);
  sir::Value* rv = relu->addResult("r", sir::DataType::F32, sir::Shape{2, 2});

  SmfTensor tensor;
  tensor.is_const = true;
  tensor.dims = {2, 2};

  GraphBuild build;
  build.input = in;
  build.output = rv;
  build.weight_sources[rv] = &tensor;
  const auto r_bad = VerifyFrontendContract(block, build);
  EXPECT_FALSE(r_bad.has_value());
  EXPECT_NE(r_bad.error().find("not an sc_mem.weight declaration"),
            std::string::npos);
}

// =============================================================================
// The analysis boundary
// =============================================================================

TEST(DriverContract, AnalysisContractDemandsGrafts) {
  sir::Block block;
  std::unordered_map<sir::Value*, sir::Value*> no_grads;
  const auto r = VerifyAnalysisContract(block, {}, no_grads, MergeProgram{});
  EXPECT_FALSE(r.has_value());
  EXPECT_TRUE(WellFormedDiagnostic(r.error()));
  EXPECT_NE(r.error().find("no adapters were grafted"), std::string::npos);
}

// =============================================================================
// The backend boundary
// =============================================================================

TEST(DriverContract, PlanContractRejectsInconsistentPlans) {
  SmfModel model = MakeMlp(6, 10, 3, 1);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(BaseConfig(4)).Compile(model));
  const size_t n = compiled.adapters.size();
  EXPECT_TRUE(VerifyGeneratedPlan(compiled, n).has_value());

  CompiledUpdate empty = compiled;
  empty.plan.clear();
  EXPECT_FALSE(VerifyGeneratedPlan(empty, n).has_value());

  CompiledUpdate inverted = compiled;
  inverted.eval_instruction_count = inverted.train_instruction_count + 1;
  const auto r_inv = VerifyGeneratedPlan(inverted, n);
  EXPECT_FALSE(r_inv.has_value());
  EXPECT_TRUE(WellFormedDiagnostic(r_inv.error()));

  CompiledUpdate no_arena = compiled;
  no_arena.arena_size = 0;
  EXPECT_FALSE(VerifyGeneratedPlan(no_arena, n).has_value());

  CompiledUpdate hookless = compiled;
  hookless.adapters.pop_back();
  EXPECT_FALSE(VerifyGeneratedPlan(hookless, n).has_value());
}

// =============================================================================
// The driver end to end
// =============================================================================

TEST(Driver, FullCompileCrossesEveryContractSilently) {
  SmfModel model = MakeMlp(6, 10, 3, 7);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(BaseConfig(4)).Compile(model));
  EXPECT_FALSE(compiled.plan.empty());
  EXPECT_GT(compiled.adapters.size(), 0u);
  // The contracts the driver already enforced re-verify externally.
  EXPECT_TRUE(
      VerifyGeneratedPlan(compiled, compiled.adapters.size()).has_value());
}

TEST(Driver, ErrorsCrossingTheBoundaryAreWellFormed) {
  SmfModel model = MakeMlp(6, 10, 3, 3);
  UpdateConfig config = BaseConfig(4);
  config.loss = LossKind::kKLDistill;  // requires a teacher we withhold
  const auto r = UpdateCompiler(config).Compile(model);
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(WellFormedDiagnostic(r.error()));
}

}  // namespace
