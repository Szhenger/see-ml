// =============================================================================
// Diagnostics partition tests: the shared one-line diagnostic form, each
// process module's unit registry and message shapes (tokenizing / parsing /
// passing / updating / architecting / generating), and the architecting
// tiling-contract validator those shapes serve.
// =============================================================================

#include <expected>
#include <string>

#include "compiler/backend/architecture/host_arch.h"
#include "compiler/diagnostics/diagnostic.h"
#include "compiler/diagnostics/generating/error.h"
#include "compiler/diagnostics/parsing/error.h"
#include "compiler/diagnostics/passing/error.h"
#include "compiler/diagnostics/tokenizing/error.h"
#include "compiler/diagnostics/updating/error.h"
#include "test/framework/seetest.h"

namespace {

using namespace seeml::update;
namespace diag = seeml::diag;

std::string Msg(std::unexpected<std::string> u) { return std::move(u).error(); }

// =============================================================================
// The shared diagnostic form
// =============================================================================

TEST(Diagnostic, FormsTheCanonicalUnitPrefixedLine) {
  EXPECT_EQ(Msg(diag::Fail("Unit", "message")), "Unit: message");
  // Builders slot into any std::expected<T, std::string> return.
  const std::expected<int, std::string> e = diag::Fail("Unit", "message");
  EXPECT_FALSE(e.has_value());
  EXPECT_EQ(e.error(), "Unit: message");
}

// =============================================================================
// tokenizing/ — the SMF byte stream
// =============================================================================

TEST(Tokenizing, ShapesFileAndTensorErrors) {
  EXPECT_EQ(Msg(diag::tokenizing::FileError("cannot open", "/m.smf")),
            "SMF: cannot open '/m.smf'");
  EXPECT_EQ(Msg(diag::tokenizing::TensorError("w0", "has invalid dims")),
            "SMF: tensor 'w0' has invalid dims");
  EXPECT_EQ(Msg(diag::tokenizing::Error("unknown op kind 9")),
            "SMF: unknown op kind 9");
  EXPECT_EQ(Msg(diag::tokenizing::IngressError("model is too big")),
            "Ingressor: model is too big");
}

// =============================================================================
// parsing/ — SMF graph -> forward SIR
// =============================================================================

TEST(Parsing, SpeaksAsTheParserAndNamesTheOp) {
  EXPECT_EQ(Msg(diag::parsing::OpError("MatMul", "mm0", "needs 2 inputs")),
            "Parser: MatMul 'mm0' needs 2 inputs");
  EXPECT_EQ(Msg(diag::parsing::Error("batch must be at least 1, got 0")),
            "Parser: batch must be at least 1, got 0");
}

// =============================================================================
// passing/ — pass orchestration and lowering legality
// =============================================================================

TEST(Passing, AttributesCorruptionToTheOffendingPass) {
  EXPECT_EQ(Msg(diag::passing::InvariantsViolated("lora-graft",
                                                  "duplicate value id 7")),
            "PassManager: SIR invariants violated after pass 'lora-graft': "
            "duplicate value id 7");
  EXPECT_EQ(Msg(diag::passing::LoweringError(
                "conv0", "lacks stride/pad geometry attributes")),
            "ConvLowering: 'conv0' lacks stride/pad geometry attributes");
}

// =============================================================================
// updating/ — the analytic methods
// =============================================================================

TEST(Updating, RoutesEveryAnalyticUnit) {
  EXPECT_EQ(Msg(diag::updating::Error(diag::updating::kAutodiff,
                                      "empty trainable set")),
            "TrainableAutodiff: empty trainable set");
  EXPECT_EQ(Msg(diag::updating::Error(diag::updating::kLoraGrafter,
                                      "rank must be positive")),
            "LoraGrafter: rank must be positive");
  EXPECT_EQ(Msg(diag::updating::Error(diag::updating::kMergeBuilder,
                                      "no adapters to merge")),
            "MergeBuilder: no adapters to merge");
}

// =============================================================================
// generating/ — code generation
// =============================================================================

TEST(Generating, ShapesCodegenAndEmissionErrors) {
  EXPECT_EQ(Msg(diag::generating::Error(diag::generating::kInstructionLowering,
                                        "cannot lower 'sc_high.tanh'")),
            "InstructionLowering: cannot lower 'sc_high.tanh'");
  EXPECT_EQ(Msg(diag::generating::FileError(diag::generating::kNativeEmitter,
                                            "cannot write", "/out/plan")),
            "NativeEmitter: cannot write '/out/plan'");
  EXPECT_EQ(Msg(diag::generating::Error(diag::generating::kArenaBinder,
                                        "frozen weight without SMF backing "
                                        "data")),
            "ArenaBinder: frozen weight without SMF backing data");
}

// =============================================================================
// architecting/ — the tiling contract the process enforces
// =============================================================================

TEST(Architecting, AcceptsTheHeuristicHint) {
  HostArchInfo arch;
  arch.simd_width_f32 = 8;
  arch.l1d_bytes = 64u << 10;
  arch.l2_bytes = 1u << 20;
  EXPECT_TRUE(ValidateGemmTiling(SuggestGemmTiling(arch), arch).has_value());

  // The real machine's hint honors the real machine's contract.
  const HostArchInfo host = DetectHostArch();
  EXPECT_TRUE(ValidateGemmTiling(SuggestGemmTiling(host), host).has_value());

  // Unknown caches: only the structural half of the contract applies, so the
  // fallback tiling must self-validate.
  const HostArchInfo unknown;
  EXPECT_TRUE(
      ValidateGemmTiling(SuggestGemmTiling(unknown), unknown).has_value());
}

TEST(Architecting, RejectsContractViolationsAsHostArch) {
  HostArchInfo arch;
  arch.simd_width_f32 = 4;
  arch.l1d_bytes = 32u << 10;
  arch.l2_bytes = 512u << 10;
  const GemmTiling good = SuggestGemmTiling(arch);

  GemmTiling zero = good;
  zero.kc = 0;
  const auto r_zero = ValidateGemmTiling(zero, arch);
  EXPECT_FALSE(r_zero.has_value());
  EXPECT_EQ(r_zero.error().find("HostArch: "), 0u);
  EXPECT_NE(r_zero.error().find("kc must be nonzero"), std::string::npos);

  GemmTiling misaligned = good;
  misaligned.nc = good.nc + 1;  // no longer a SIMD multiple
  EXPECT_FALSE(ValidateGemmTiling(misaligned, arch).has_value());

  GemmTiling fat = good;
  fat.kc = good.kc * 64;  // kc x nc panel blows through half of L1
  const auto r_fat = ValidateGemmTiling(fat, arch);
  EXPECT_FALSE(r_fat.has_value());
  EXPECT_NE(r_fat.error().find("exceeds half of L1d"), std::string::npos);
}

}  // namespace
