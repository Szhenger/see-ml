// =============================================================================
// feeder/batch_pipeline unit tests: the pipelining efficiency mechanism must
// be invisible to training — the staged batch sequence is exactly the serial
// FillBatch sequence (shuffled or not, across epoch wraparound), labels are
// optional, and the feeder thread joins on early exits.
// =============================================================================

#include <cstdint>
#include <cstring>
#include <vector>

#include "runtime/feeder/batch_pipeline.h"
#include "runtime/feeder/dataset.h"
#include "test/framework/seetest.h"

namespace {

using namespace seeml::update_rt;

constexpr uint64_t kSamples = 10;
constexpr int64_t kDim = 3;
constexpr uint64_t kBatch = 4;
constexpr uint64_t kInputFloats = kBatch * kDim;
constexpr uint64_t kLabelBytes = kBatch * sizeof(int32_t);

/// Sample i is [i*3, i*3+1, i*3+2] with class label i — every byte of a
/// staged batch identifies exactly which sample it came from.
Dataset MakeCorpus() {
  std::vector<float> inputs(kSamples * kDim);
  std::vector<uint8_t> labels(kSamples * sizeof(int32_t));
  auto* lab = reinterpret_cast<int32_t*>(labels.data());
  for (uint64_t i = 0; i < kSamples; ++i) {
    for (int64_t d = 0; d < kDim; ++d)
      inputs[i * kDim + d] = static_cast<float>(i * kDim + d);
    lab[i] = static_cast<int32_t>(i);
  }
  auto ds = Dataset::FromMemory(std::move(inputs), std::move(labels),
                                kSamples, kDim, /*label_kind=*/1,
                                /*label_dim=*/0);
  if (!ds) std::abort();
  return std::move(*ds);
}

TEST(BatchPipeline, StagesExactlyTheSerialSequence) {
  Dataset serial = MakeCorpus();
  Dataset piped = MakeCorpus();

  BatchPipeline pipeline(piped, kBatch, kInputFloats, kLabelBytes);
  std::vector<float> want_x(kInputFloats), got_x(kInputFloats);
  std::vector<uint8_t> want_l(kLabelBytes), got_l(kLabelBytes);

  // Seven batches of four over ten samples: crosses the wraparound twice.
  for (int step = 0; step < 7; ++step) {
    serial.FillBatch(kBatch, want_x.data(), want_l.data());
    pipeline.NextBatch(got_x.data(), got_l.data());
    EXPECT_TRUE(want_x == got_x);
    EXPECT_TRUE(want_l == got_l);
  }
}

TEST(BatchPipeline, StagesExactlyTheSerialShuffledSequence) {
  Dataset serial = MakeCorpus();
  Dataset piped = MakeCorpus();
  serial.EnableShuffle(42);
  piped.EnableShuffle(42);

  BatchPipeline pipeline(piped, kBatch, kInputFloats, kLabelBytes);
  std::vector<float> want_x(kInputFloats), got_x(kInputFloats);
  std::vector<uint8_t> want_l(kLabelBytes), got_l(kLabelBytes);

  // Enough batches to trigger the per-epoch reshuffle inside the feeder.
  for (int step = 0; step < 9; ++step) {
    serial.FillBatch(kBatch, want_x.data(), want_l.data());
    pipeline.NextBatch(got_x.data(), got_l.data());
    EXPECT_TRUE(want_x == got_x);
    EXPECT_TRUE(want_l == got_l);
  }
}

TEST(BatchPipeline, ServesUnlabeledCorpora) {
  std::vector<float> inputs(kSamples * kDim, 1.5f);
  auto ds = Dataset::FromMemory(std::move(inputs), {}, kSamples, kDim,
                                /*label_kind=*/0, /*label_dim=*/0);
  ASSERT_TRUE(ds.has_value());

  BatchPipeline pipeline(*ds, kBatch, kInputFloats, /*label_bytes=*/0);
  std::vector<float> x(kInputFloats, 0.0f);
  pipeline.NextBatch(x.data(), /*label_slot=*/nullptr);
  for (float v : x) EXPECT_NEAR(v, 1.5f, 0.0f);
}

TEST(BatchPipeline, JoinsTheFeederOnEarlyExit) {
  // The destructor must join the staging thread even when the consumer
  // abandons the run mid-stream (the engine's error paths do exactly this).
  Dataset data = MakeCorpus();
  {
    BatchPipeline pipeline(data, kBatch, kInputFloats, kLabelBytes);
    std::vector<float> x(kInputFloats);
    std::vector<uint8_t> l(kLabelBytes);
    pipeline.NextBatch(x.data(), l.data());
    // Dropped with a staged batch in flight.
  }
  // Reaching here without a hang or crash is the assertion; the dataset is
  // usable again, single-threaded.
  std::vector<float> x(kInputFloats);
  std::vector<uint8_t> l(kLabelBytes);
  data.FillBatch(kBatch, x.data(), l.data());
}

}  // namespace
