// =============================================================================
// SMF container tests: serialization round-trips, layout invariants, and
// rejection of malformed files (the binary-format hardening surface).
// =============================================================================

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "source/smf.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"
#include "test/support/scoped_temp_dir.h"

namespace {

using namespace seeml::update;
using seeml::testing::AsBytes;
using seeml::testing::MakeMlp;
using seeml::testing::ScopedTempDir;

std::vector<uint8_t> ReadAll(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

void WriteAll(const std::string& path, const std::vector<uint8_t>& bytes) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
}

TEST(Smf, FindTensor) {
  SmfModel model = MakeMlp(4, 8, 2, 1);
  const SmfTensor* w1 = model.FindTensor("w1");
  ASSERT_NE(w1, nullptr);
  EXPECT_EQ(w1->name, "w1");
  EXPECT_EQ(model.FindTensor("nonexistent"), nullptr);
}

TEST(Smf, SaveLoadRoundTrip) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(4, 8, 2, 1);
  const std::string path = dir.File("model.smf");
  ASSERT_OK(SaveSmf(path, model));

  ASSERT_OK_AND_ASSIGN(SmfModel loaded, LoadSmf(path));
  EXPECT_EQ(loaded.input_name, model.input_name);
  EXPECT_EQ(loaded.output_name, model.output_name);
  ASSERT_EQ(loaded.tensors.size(), model.tensors.size());
  ASSERT_EQ(loaded.ops.size(), model.ops.size());

  for (size_t i = 0; i < model.tensors.size(); ++i) {
    const SmfTensor& want = model.tensors[i];
    const SmfTensor& got = loaded.tensors[i];
    EXPECT_EQ(got.name, want.name);
    EXPECT_TRUE(got.dims == want.dims);
    EXPECT_EQ(got.is_const, want.is_const);
    // Constant tensors carry their data through the round trip bit-exactly.
    EXPECT_TRUE(got.data == want.data);
  }
  // The dynamic batch dim of the non-const input survives.
  const SmfTensor* x = loaded.FindTensor("x");
  ASSERT_NE(x, nullptr);
  ASSERT_EQ(x->dims.size(), 2u);
  EXPECT_EQ(x->dims[0], -1);

  for (size_t i = 0; i < model.ops.size(); ++i) {
    EXPECT_EQ(loaded.ops[i].kind, model.ops[i].kind);
    EXPECT_EQ(loaded.ops[i].name, model.ops[i].name);
    EXPECT_TRUE(loaded.ops[i].inputs == model.ops[i].inputs);
    EXPECT_EQ(loaded.ops[i].output, model.ops[i].output);
  }
}

TEST(Smf, SaveComputesAlignedNonOverlappingOffsets) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(4, 8, 2, 2);
  const std::string path = dir.File("model.smf");
  ASSERT_OK(SaveSmf(path, model));

  // SaveSmf rewrites data_offset/byte_size on the model itself.
  uint64_t prev_end = 0;
  for (const SmfTensor& t : model.tensors) {
    if (!t.is_const) continue;
    EXPECT_EQ(t.data_offset % 64, 0u);
    EXPECT_GE(t.data_offset, prev_end);
    EXPECT_EQ(t.byte_size, t.data.size());
    prev_end = t.data_offset + t.byte_size;
  }
  EXPECT_EQ(ReadAll(path).size() % 64, 0u);
}

TEST(Smf, LoadRejectsMissingFile) {
  ScopedTempDir dir;
  EXPECT_ERROR_CONTAINS(LoadSmf(dir.File("nope.smf")), "cannot open");
}

TEST(Smf, LoadRejectsBadMagic) {
  ScopedTempDir dir;
  const std::string path = dir.File("bad.smf");
  WriteAll(path, {'N', 'O', 'P', 'E', 0, 0, 0, 0});
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "bad magic");
}

TEST(Smf, LoadRejectsBadVersion) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(4, 8, 2, 3);
  const std::string path = dir.File("model.smf");
  ASSERT_OK(SaveSmf(path, model));

  std::vector<uint8_t> bytes = ReadAll(path);
  bytes[4] = 99;  // u32 version lives at offset 4
  WriteAll(path, bytes);
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "unsupported version");
}

TEST(Smf, LoadRejectsTruncatedMetadata) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(4, 8, 2, 4);
  const std::string path = dir.File("model.smf");
  ASSERT_OK(SaveSmf(path, model));

  std::vector<uint8_t> bytes = ReadAll(path);
  bytes.resize(20);  // cuts into the tensor table
  WriteAll(path, bytes);
  EXPECT_ERROR(LoadSmf(path));
}

TEST(Smf, LoadRejectsDataRangeBeyondFile) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(4, 8, 2, 5);
  const std::string path = dir.File("model.smf");
  ASSERT_OK(SaveSmf(path, model));

  // Cut into the data section: some tensor's [offset, offset+size) now
  // exceeds the file.
  std::vector<uint8_t> bytes = ReadAll(path);
  bytes.resize(model.FindTensor("w1")->data_offset + 4);
  WriteAll(path, bytes);
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "exceeds file size");
}

TEST(Smf, LoadRejectsByteSizeDimsDisagreement) {
  ScopedTempDir dir;
  // dims say 2x2 = 16 bytes, but the blob holds 3 floats. SaveSmf writes
  // byte_size = data.size() = 12, which the loader must reject.
  SmfModel model;
  model.input_name = "x";
  model.output_name = "y";
  model.tensors.push_back({.name = "x", .dims = {-1, 2}, .is_const = false});
  model.tensors.push_back({.name = "w",
                           .dims = {2, 2},
                           .is_const = true,
                           .data = AsBytes({1.0f, 2.0f, 3.0f})});
  model.ops.push_back({SmfOpKind::kMatMul, "mm", {"x", "w"}, "y"});

  const std::string path = dir.File("model.smf");
  ASSERT_OK(SaveSmf(path, model));
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "byte size disagrees");
}

TEST(Smf, LoadRejectsInvalidDims) {
  ScopedTempDir dir;
  const std::string path = dir.File("model.smf");

  // A zero dim is never valid.
  SmfModel zero_dim;
  zero_dim.input_name = "x";
  zero_dim.output_name = "y";
  zero_dim.tensors.push_back(
      {.name = "w", .dims = {0}, .is_const = true, .data = AsBytes({1.0f})});
  ASSERT_OK(SaveSmf(path, zero_dim));
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "invalid dims");

  // A dynamic (-1) dim is only allowed on non-constant tensors.
  SmfModel dynamic_const;
  dynamic_const.input_name = "x";
  dynamic_const.output_name = "y";
  dynamic_const.tensors.push_back(
      {.name = "w", .dims = {-1}, .is_const = true, .data = AsBytes({1.0f})});
  ASSERT_OK(SaveSmf(path, dynamic_const));
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "invalid dims");

  // Rank 0 is rejected.
  SmfModel rank0;
  rank0.input_name = "x";
  rank0.output_name = "y";
  rank0.tensors.push_back(
      {.name = "w", .dims = {}, .is_const = true, .data = AsBytes({1.0f})});
  ASSERT_OK(SaveSmf(path, rank0));
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "invalid dims");
}

TEST(Smf, LoadRejectsOverlappingDataRanges) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(4, 8, 2, 1);
  const std::string path = dir.File("model.smf");
  ASSERT_OK(SaveSmf(path, model));
  ASSERT_OK_AND_ASSIGN(SmfModel saved, LoadSmf(path));
  const SmfTensor* w1 = saved.FindTensor("w1");
  const SmfTensor* b1 = saved.FindTensor("b1");
  ASSERT_NE(w1, nullptr);
  ASSERT_NE(b1, nullptr);

  // Patch b1's metadata offset field to alias w1's blob. Both ranges stay
  // individually inside the file, so only the disjointness check can fire.
  // The field is the first occurrence of b1's offset below the data section.
  std::vector<uint8_t> bytes = ReadAll(path);
  uint64_t data_start = bytes.size();
  for (const auto& t : saved.tensors)
    if (t.is_const) data_start = std::min(data_start, t.data_offset);
  const auto* needle = reinterpret_cast<const uint8_t*>(&b1->data_offset);
  auto it = std::search(bytes.begin(),
                        bytes.begin() + static_cast<ptrdiff_t>(data_start),
                        needle, needle + sizeof(uint64_t));
  ASSERT_TRUE(it != bytes.begin() + static_cast<ptrdiff_t>(data_start));
  std::memcpy(&*it, &w1->data_offset, sizeof(uint64_t));
  WriteAll(path, bytes);

  EXPECT_ERROR_CONTAINS(LoadSmf(path), "overlap");
}

TEST(Smf, LoadRejectsUnknownOpKind) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(4, 8, 2, 1);
  const std::string path = dir.File("model.smf");
  ASSERT_OK(SaveSmf(path, model));

  // An op record is: kind u8, then the u16-length-prefixed name. Locate op
  // "mm1" by its length-prefixed name; the byte before the prefix is the kind.
  std::vector<uint8_t> bytes = ReadAll(path);
  const uint8_t needle[] = {3, 0, 'm', 'm', '1'};
  auto it = std::search(bytes.begin(), bytes.end(), needle,
                        needle + sizeof(needle));
  ASSERT_TRUE(it != bytes.end());
  *(it - 1) = 200;
  WriteAll(path, bytes);

  EXPECT_ERROR_CONTAINS(LoadSmf(path), "unknown kind");
}

TEST(Smf, SaveRejectsOversizedMetadata) {
  ScopedTempDir dir;
  const std::string path = dir.File("model.smf");

  // The formats carry u16 string lengths and u8 rank/input counts; a silent
  // cast would desynchronize every record after the oversized field.
  SmfModel long_name = MakeMlp(4, 8, 2, 1);
  long_name.tensors[1].name.assign(70000, 'n');
  EXPECT_ERROR_CONTAINS(SaveSmf(path, long_name), "65535");

  SmfModel deep_rank = MakeMlp(4, 8, 2, 1);
  deep_rank.tensors[1].dims.assign(300, 1);
  EXPECT_ERROR_CONTAINS(SaveSmf(path, deep_rank), "rank exceeds 255");

  SmfModel wide_op = MakeMlp(4, 8, 2, 1);
  wide_op.ops[0].inputs.assign(300, "x");
  EXPECT_ERROR_CONTAINS(SaveSmf(path, wide_op), "more than 255 inputs");
}

TEST(Smf, SaveRejectsConstantTensorWithoutData) {
  ScopedTempDir dir;
  SmfModel model;
  model.input_name = "x";
  model.output_name = "y";
  model.tensors.push_back({.name = "w", .dims = {2, 2}, .is_const = true});
  EXPECT_ERROR_CONTAINS(SaveSmf(dir.File("model.smf"), model),
                        "has no data to serialize");
}

TEST(Smf, SaveIsDeterministic) {
  ScopedTempDir dir;
  SmfModel a = MakeMlp(4, 8, 2, 6);
  SmfModel b = MakeMlp(4, 8, 2, 6);
  const std::string pa = dir.File("a.smf");
  const std::string pb = dir.File("b.smf");
  ASSERT_OK(SaveSmf(pa, a));
  ASSERT_OK(SaveSmf(pb, b));
  EXPECT_TRUE(ReadAll(pa) == ReadAll(pb));
}

}  // namespace
