#include "runtime/feeder/batch_pipeline.h"

#include <cstring>

#include "source/parallel/parallel_for.h"

namespace seeml::update_rt {

BatchPipeline::BatchPipeline(Dataset& data, uint64_t batch,
                             uint64_t input_floats, uint64_t label_bytes)
    : data_(data), batch_(batch) {
  staged_inputs_.resize(input_floats);
  staged_labels_.resize(label_bytes);
  threaded_ = seeml::update::ParallelThreadCount() > 1;
  if (!threaded_) return;
  // Stage the first batch immediately: step 1's data cost is already paid
  // by the time the engine asks for it.
  feeder_ = std::thread([this] { FeederLoop(); });
}

BatchPipeline::~BatchPipeline() {
  if (!threaded_) return;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  feeder_.join();
  // Un-consume a staged batch the engine never took (`full_` still set —
  // NextBatch would have cleared it). Without this, whether the feeder had
  // already advanced the dataset past the last consumed batch would depend
  // on how the teardown raced the staging thread, and a later consumer of
  // the same dataset (the regression gate's post-training evaluation, a
  // resumed run) would serve a scheduling-dependent batch sequence. The
  // restore makes pipelining invisible: at any thread count, the cursor
  // reads exactly "the batches the engine consumed".
  if (full_) data_.RestoreServingPos(stage_start_);
}

void BatchPipeline::FeederLoop() {
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return !full_ || stop_; });
      if (stop_) return;
    }
    // Unlocked on purpose: the consumer never reads the staging buffers
    // while `full_` is false, so the fill races with nothing. The dataset
    // position is snapshotted first so the destructor can un-consume this
    // batch if the engine never takes it (read only after join).
    stage_start_ = data_.SaveServingPos();
    data_.FillBatch(batch_, staged_inputs_.data(), StagedLabelPtr());
    {
      std::lock_guard<std::mutex> lock(mutex_);
      full_ = true;
    }
    cv_.notify_all();
  }
}

void BatchPipeline::NextBatch(float* input_slot, uint8_t* label_slot) {
  if (!threaded_) {
    data_.FillBatch(batch_, input_slot, label_slot);
    return;
  }
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return full_; });
  }
  std::memcpy(input_slot, staged_inputs_.data(),
              staged_inputs_.size() * sizeof(float));
  if (label_slot && !staged_labels_.empty())
    std::memcpy(label_slot, staged_labels_.data(), staged_labels_.size());
  {
    std::lock_guard<std::mutex> lock(mutex_);
    full_ = false;
  }
  cv_.notify_all();
}

}  // namespace seeml::update_rt
