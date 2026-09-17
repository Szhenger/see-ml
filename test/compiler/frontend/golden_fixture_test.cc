// =============================================================================
// The cross-plane golden fixtures: files the Python exporter wrote
// (test/fixtures/golden/make_golden.py), committed, and read here value for
// value. The Python suite byte-compares the same files against a fresh run
// of the exporter, so the two planes are held to one set of bytes — the
// writer cannot drift from the reader without one of the two suites failing.
// =============================================================================

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "compiler/frontend/ingressor/model_reader.h"
#include "runtime/feeder/dataset.h"
#include "source/language/model_format.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"

namespace {

using namespace seeml::update;
using seeml::testing::RepoRoot;
using seeml::update_rt::Dataset;

std::string Golden(const char* name) {
  return RepoRoot() + "/test/fixtures/golden/" + name;
}

std::vector<float> Floats(const SmfTensor& t) {
  std::vector<float> out(t.byte_size / sizeof(float));
  std::memcpy(out.data(), t.data.data(), out.size() * sizeof(float));
  return out;
}

TEST(GoldenFixture, TheExportersModelLoadsValueForValue) {
  ASSERT_OK_AND_ASSIGN(SmfModel model, LoadSmf(Golden("mlp.smf")));
  EXPECT_EQ(model.input_name, "x");
  EXPECT_EQ(model.output_name, "zb0");
  ASSERT_EQ(model.tensors.size(), 3u);
  ASSERT_EQ(model.ops.size(), 2u);
  EXPECT_TRUE(model.ops[0].kind == SmfOpKind::kMatMul);
  EXPECT_TRUE(model.ops[1].kind == SmfOpKind::kAddBias);
  EXPECT_EQ(model.ops[1].output, "zb0");

  const SmfTensor* w = model.FindTensor("w0");
  ASSERT_NE(w, nullptr);
  ASSERT_EQ(w->dims.size(), 2u);
  EXPECT_EQ(w->dims[0], 4);
  EXPECT_EQ(w->dims[1], 3);
  EXPECT_EQ(w->data_offset % 64, 0u);
  const std::vector<float> wv = Floats(*w);
  ASSERT_EQ(wv.size(), 12u);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 3; ++j)
      EXPECT_EQ(wv[i * 3 + j], static_cast<float>(i * 3 + j - 5) / 8.0f);

  const SmfTensor* b = model.FindTensor("b0");
  ASSERT_NE(b, nullptr);
  const std::vector<float> bv = Floats(*b);
  ASSERT_EQ(bv.size(), 3u);
  EXPECT_EQ(bv[0], 0.5f);
  EXPECT_EQ(bv[1], -0.25f);
  EXPECT_EQ(bv[2], 0.125f);
}

TEST(GoldenFixture, TheExportersFeatureCorpusServesItsRows) {
  ASSERT_OK_AND_ASSIGN(Dataset data, Dataset::LoadFromFile(Golden("class.sds")));
  EXPECT_EQ(data.num_samples(), 5u);
  EXPECT_EQ(data.input_dim(), 4u);
  EXPECT_EQ(data.label_kind(), 1u);
  EXPECT_EQ(data.input_kind(), 0u);
  ASSERT_OK(data.ValidateClassLabels(3));

  std::vector<float> x(20);
  std::vector<int32_t> y(5);
  data.FillBatch(5, x.data(), reinterpret_cast<uint8_t*>(y.data()));
  for (int n = 0; n < 5; ++n) {
    EXPECT_EQ(y[n], n % 3);
    for (int d = 0; d < 4; ++d)
      EXPECT_EQ(x[n * 4 + d], static_cast<float>(n * 4 + d) / 16.0f);
  }
}

TEST(GoldenFixture, TheExportersTokenCorpusServesShiftedLabels) {
  ASSERT_OK_AND_ASSIGN(Dataset data,
                       Dataset::LoadFromFile(Golden("tokens.sds")));
  EXPECT_EQ(data.num_samples(), 3u);
  EXPECT_EQ(data.input_dim(), 4u);
  EXPECT_EQ(data.input_kind(), 1u);

  // One record = 4 token rows; the slot carries i32 ids, the labels are the
  // record shifted by one.
  std::vector<float> slot(12);
  std::vector<int32_t> labels(12);
  data.FillBatch(12, slot.data(), reinterpret_cast<uint8_t*>(labels.data()));
  std::vector<int32_t> ids(12);
  std::memcpy(ids.data(), slot.data(), sizeof(int32_t) * ids.size());
  for (int r = 0; r < 3; ++r)
    for (int k = 0; k < 4; ++k) {
      EXPECT_EQ(ids[r * 4 + k], (r * 5 + k) % 7);
      EXPECT_EQ(labels[r * 4 + k], (r * 5 + k + 1) % 7);
    }
}

}  // namespace
