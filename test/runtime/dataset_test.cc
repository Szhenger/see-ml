// =============================================================================
// SDS dataset tests: in-memory construction, batch feeding with wraparound,
// class-label validation, file round-trips, and rejection of malformed files.
// =============================================================================

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "runtime/dataset.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"
#include "test/support/scoped_temp_dir.h"

namespace {

using seeml::update_rt::Dataset;
using seeml::testing::AsBytes;
using seeml::testing::ScopedTempDir;

std::vector<uint8_t> ClassLabels(const std::vector<int32_t>& labels) {
  std::vector<uint8_t> bytes(labels.size() * sizeof(int32_t));
  std::memcpy(bytes.data(), labels.data(), bytes.size());
  return bytes;
}

/// A 3-sample, input_dim=2 dataset with class labels {0, 1, 2}.
std::expected<Dataset, std::string> TinyClassDataset() {
  return Dataset::FromMemory({0, 1, 10, 11, 20, 21}, ClassLabels({0, 1, 2}),
                             /*num_samples=*/3, /*input_dim=*/2,
                             /*label_kind=*/1, /*label_dim=*/0);
}

void WriteBytes(const std::string& path, const std::vector<uint8_t>& bytes) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
}

/// Hand-writes an SDS header (+ optional payload) for corruption tests.
std::vector<uint8_t> SdsHeader(uint32_t magic, uint32_t version,
                               uint64_t num_samples, uint64_t input_dim,
                               uint32_t label_kind, uint64_t label_dim) {
  std::vector<uint8_t> b;
  auto put = [&](const void* src, size_t n) {
    const auto* p = static_cast<const uint8_t*>(src);
    b.insert(b.end(), p, p + n);
  };
  const uint32_t pad = 0;
  put(&magic, 4);
  put(&version, 4);
  put(&num_samples, 8);
  put(&input_dim, 8);
  put(&label_kind, 4);
  put(&pad, 4);
  put(&label_dim, 8);
  return b;
}

TEST(Dataset, FromMemoryValidatesBufferSizes) {
  EXPECT_OK(TinyClassDataset());

  // Zero samples.
  EXPECT_ERROR_CONTAINS(
      Dataset::FromMemory({}, {}, 0, 2, 0, 0), "zero samples");
  // Input buffer too small for num_samples * input_dim.
  EXPECT_ERROR_CONTAINS(
      Dataset::FromMemory({1, 2, 3}, ClassLabels({0, 1}), 2, 2, 1, 0),
      "input buffer size mismatch");
  // Label buffer disagrees with label_kind.
  EXPECT_ERROR_CONTAINS(
      Dataset::FromMemory({1, 2, 3, 4}, ClassLabels({0}), 2, 2, 1, 0),
      "label buffer size mismatch");
}

TEST(Dataset, LabelBytesPerSample) {
  ASSERT_OK_AND_ASSIGN(Dataset none,
                       Dataset::FromMemory({1, 2}, {}, 2, 1, 0, 0));
  EXPECT_EQ(none.label_bytes_per_sample(), 0u);

  ASSERT_OK_AND_ASSIGN(Dataset klass, TinyClassDataset());
  EXPECT_EQ(klass.label_bytes_per_sample(), sizeof(int32_t));

  ASSERT_OK_AND_ASSIGN(
      Dataset dense,
      Dataset::FromMemory({1, 2}, AsBytes({5, 6, 7, 8, 9, 10}), 2, 1, 2, 3));
  EXPECT_EQ(dense.label_bytes_per_sample(), 3 * sizeof(float));
}

TEST(Dataset, FillBatchIsSequentialWithWraparound) {
  ASSERT_OK_AND_ASSIGN(Dataset data, TinyClassDataset());
  std::vector<float> inputs(2 * 2);
  std::vector<int32_t> labels(2);

  data.FillBatch(2, inputs.data(),
                 reinterpret_cast<uint8_t*>(labels.data()));
  EXPECT_NEAR(inputs[0], 0.0f, 0.0);
  EXPECT_NEAR(inputs[2], 10.0f, 0.0);
  EXPECT_EQ(labels[0], 0);
  EXPECT_EQ(labels[1], 1);

  // Second batch wraps: samples 2, then 0 again.
  data.FillBatch(2, inputs.data(),
                 reinterpret_cast<uint8_t*>(labels.data()));
  EXPECT_NEAR(inputs[0], 20.0f, 0.0);
  EXPECT_NEAR(inputs[2], 0.0f, 0.0);
  EXPECT_EQ(labels[0], 2);
  EXPECT_EQ(labels[1], 0);
}

TEST(Dataset, FillBatchToleratesNullLabelSlot) {
  ASSERT_OK_AND_ASSIGN(
      Dataset data, Dataset::FromMemory({1, 2, 3}, {}, 3, 1, 0, 0));
  std::vector<float> inputs(2);
  data.FillBatch(2, inputs.data(), nullptr);  // kind 0: no labels to copy
  EXPECT_NEAR(inputs[0], 1.0f, 0.0);
  EXPECT_NEAR(inputs[1], 2.0f, 0.0);
}

TEST(Dataset, ValidateClassLabels) {
  ASSERT_OK_AND_ASSIGN(Dataset data, TinyClassDataset());  // labels {0, 1, 2}
  EXPECT_OK(data.ValidateClassLabels(3));
  EXPECT_ERROR_CONTAINS(data.ValidateClassLabels(2), "outside [0, 2)");
  // num_classes == 0 disables the check.
  EXPECT_OK(data.ValidateClassLabels(0));

  ASSERT_OK_AND_ASSIGN(
      Dataset negative,
      Dataset::FromMemory({1, 2}, ClassLabels({-1, 0}), 2, 1, 1, 0));
  EXPECT_ERROR(negative.ValidateClassLabels(4));

  // Non-class label kinds are exempt.
  ASSERT_OK_AND_ASSIGN(Dataset none,
                       Dataset::FromMemory({1, 2}, {}, 2, 1, 0, 0));
  EXPECT_OK(none.ValidateClassLabels(1));
}

TEST(Dataset, FileRoundTripClassLabels) {
  ScopedTempDir dir;
  ASSERT_OK_AND_ASSIGN(Dataset data, TinyClassDataset());
  const std::string path = dir.File("data.sds");
  ASSERT_OK(data.SaveToFile(path));

  ASSERT_OK_AND_ASSIGN(Dataset loaded, Dataset::LoadFromFile(path));
  EXPECT_EQ(loaded.num_samples(), 3u);
  EXPECT_EQ(loaded.input_dim(), 2u);
  EXPECT_EQ(loaded.label_kind(), 1u);

  // Batches served from the loaded copy are identical to the original's.
  std::vector<float> want_in(6), got_in(6);
  std::vector<int32_t> want_lab(3), got_lab(3);
  data.FillBatch(3, want_in.data(),
                 reinterpret_cast<uint8_t*>(want_lab.data()));
  loaded.FillBatch(3, got_in.data(),
                   reinterpret_cast<uint8_t*>(got_lab.data()));
  EXPECT_TRUE(want_in == got_in);
  EXPECT_TRUE(want_lab == got_lab);
}

TEST(Dataset, FileRoundTripDenseLabels) {
  ScopedTempDir dir;
  ASSERT_OK_AND_ASSIGN(
      Dataset data,
      Dataset::FromMemory({1, 2}, AsBytes({0.5f, 1.5f, 2.5f, 3.5f}), 2, 1, 2,
                          2));
  const std::string path = dir.File("dense.sds");
  ASSERT_OK(data.SaveToFile(path));

  ASSERT_OK_AND_ASSIGN(Dataset loaded, Dataset::LoadFromFile(path));
  EXPECT_EQ(loaded.label_kind(), 2u);
  EXPECT_EQ(loaded.label_bytes_per_sample(), 2 * sizeof(float));

  std::vector<float> in(2);
  std::vector<float> lab(4);
  loaded.FillBatch(2, in.data(), reinterpret_cast<uint8_t*>(lab.data()));
  EXPECT_NEAR(lab[0], 0.5f, 0.0);
  EXPECT_NEAR(lab[3], 3.5f, 0.0);
}

TEST(Dataset, LoadRejectsMalformedFiles) {
  ScopedTempDir dir;
  const std::string path = dir.File("bad.sds");

  EXPECT_ERROR_CONTAINS(Dataset::LoadFromFile(dir.File("missing.sds")),
                        "cannot open");

  WriteBytes(path, {'N', 'O', 'P', 'E'});
  EXPECT_ERROR_CONTAINS(Dataset::LoadFromFile(path), "bad magic");

  WriteBytes(path, SdsHeader(seeml::update_rt::kSdsMagic, 9, 1, 1, 0, 0));
  EXPECT_ERROR_CONTAINS(Dataset::LoadFromFile(path), "unsupported version");

  // Magic + version but the header stops there.
  std::vector<uint8_t> truncated =
      SdsHeader(seeml::update_rt::kSdsMagic, 1, 1, 1, 0, 0);
  truncated.resize(12);
  WriteBytes(path, truncated);
  EXPECT_ERROR_CONTAINS(Dataset::LoadFromFile(path), "truncated header");

  WriteBytes(path, SdsHeader(seeml::update_rt::kSdsMagic, 1, 0, 1, 0, 0));
  EXPECT_ERROR_CONTAINS(Dataset::LoadFromFile(path), "empty dataset");

  WriteBytes(path, SdsHeader(seeml::update_rt::kSdsMagic, 1, 1, 1, 3, 0));
  EXPECT_ERROR_CONTAINS(Dataset::LoadFromFile(path), "unknown label kind");

  WriteBytes(path, SdsHeader(seeml::update_rt::kSdsMagic, 1, 1, 1, 2, 0));
  EXPECT_ERROR_CONTAINS(Dataset::LoadFromFile(path), "label dim out of range");

  // Header claims a million samples; no payload follows. The loader must
  // reject before allocating.
  WriteBytes(path,
             SdsHeader(seeml::update_rt::kSdsMagic, 1, 1'000'000, 16, 0, 0));
  EXPECT_ERROR_CONTAINS(Dataset::LoadFromFile(path),
                        "exceeds file size");
}

// =============================================================================
// Shuffling and validation splits
// =============================================================================

/// 64 samples whose first feature is the sample index (an id marker) and
/// whose class label is index % 2 — lets tests verify permutation coverage
/// and that labels travel with their sample.
std::expected<Dataset, std::string> MarkerDataset() {
  const uint64_t n = 64;
  const int64_t in_dim = 4;
  std::vector<float> inputs(n * in_dim);
  std::vector<int32_t> labels(n);
  for (uint64_t i = 0; i < n; ++i) {
    labels[i] = static_cast<int32_t>(i % 2);
    for (int64_t d = 0; d < in_dim; ++d)
      inputs[i * in_dim + d] = static_cast<float>(i);
  }
  return Dataset::FromMemory(std::move(inputs), ClassLabels(labels), n, in_dim,
                             /*label_kind=*/1, /*label_dim=*/0);
}

TEST(DatasetShuffle, IsSeededDeterministicAndCoversEachEpoch) {
  ASSERT_OK_AND_ASSIGN(Dataset a, MarkerDataset());
  ASSERT_OK_AND_ASSIGN(Dataset b, MarkerDataset());
  a.EnableShuffle(9);
  b.EnableShuffle(9);

  const int64_t in_dim = 4;
  std::vector<float> xa(8 * in_dim), xb(8 * in_dim);
  std::vector<int32_t> la(8), lb(8);
  std::vector<int> seen(64, 0);
  bool any_permuted = false;
  for (int batch = 0; batch < 8; ++batch) {
    a.FillBatch(8, xa.data(), reinterpret_cast<uint8_t*>(la.data()));
    b.FillBatch(8, xb.data(), reinterpret_cast<uint8_t*>(lb.data()));
    // Same seed => identical batch sequences.
    EXPECT_TRUE(xa == xb);
    for (int i = 0; i < 8; ++i) {
      const auto id = static_cast<uint64_t>(xa[i * in_dim]);
      ASSERT_LT(id, 64u);
      ++seen[id];
      if (id != static_cast<uint64_t>(batch * 8 + i)) any_permuted = true;
      // Labels must travel with their sample.
      EXPECT_EQ(la[i], static_cast<int32_t>(id % 2));
    }
  }
  // One epoch is a permutation, not sampling with replacement.
  for (int c : seen) EXPECT_EQ(c, 1);
  EXPECT_TRUE(any_permuted);

  // Epoch 2 re-shuffles, still deterministically.
  a.FillBatch(8, xa.data(), reinterpret_cast<uint8_t*>(la.data()));
  b.FillBatch(8, xb.data(), reinterpret_cast<uint8_t*>(lb.data()));
  EXPECT_TRUE(xa == xb);
}

TEST(DatasetSplit, HoldsOutTheTailDeterministically) {
  ASSERT_OK_AND_ASSIGN(Dataset data, MarkerDataset());
  ASSERT_OK_AND_ASSIGN(Dataset val, data.SplitValidation(0.25));
  EXPECT_EQ(data.num_samples(), 48u);
  EXPECT_EQ(val.num_samples(), 16u);

  // The held-out set is exactly the tail, in order.
  std::vector<float> x(4);
  std::vector<int32_t> l(1);
  val.FillBatch(1, x.data(), reinterpret_cast<uint8_t*>(l.data()));
  EXPECT_EQ(static_cast<uint64_t>(x[0]), 48u);
}

TEST(DatasetSplit, RejectsDegenerateRequests) {
  ASSERT_OK_AND_ASSIGN(Dataset data, MarkerDataset());
  EXPECT_ERROR_CONTAINS(data.SplitValidation(0.0), "fraction");
  EXPECT_ERROR_CONTAINS(data.SplitValidation(1.0), "fraction");

  // Splitting after shuffling would leak training samples into the held-out
  // set (the permutation has already mixed the tail): refused.
  data.EnableShuffle(1);
  EXPECT_ERROR_CONTAINS(data.SplitValidation(0.25), "before enabling shuffle");
}

}  // namespace
