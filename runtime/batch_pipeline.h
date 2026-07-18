#ifndef SEEML_RUNTIME_BATCH_PIPELINE_H_
#define SEEML_RUNTIME_BATCH_PIPELINE_H_

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "runtime/dataset.h"

// =============================================================================
// BatchPipeline — overlaps data feeding with training compute.
//
// A feeder thread fills the NEXT batch (shuffle gather + copies, and the
// per-epoch reshuffle when it lands on an epoch boundary) into a staging
// buffer while the engine executes the CURRENT step; NextBatch() then moves
// the staged batch into the arena's I/O slots with two bulk copies. The
// batch sequence is exactly the serial FillBatch sequence — pipelining
// changes when batches are materialized, never which batches or in which
// order — so training results are independent of the overlap.
//
// Concurrency contract: the Dataset is touched only by the feeder thread
// between construction and destruction; staging buffers are handed back and
// forth under one mutex (feeder writes while `full` is false, consumer
// reads while it is true). The destructor joins on every path, so an early
// engine exit (non-finite loss, checkpoint failure, cancellation) cannot
// leak the thread. Because the feeder stays one batch ahead, an interrupted
// run leaves the dataset cursor one batch past the last consumed batch.
//
// When the configured thread count is 1 (SEEML_THREADS=1 — the bare-metal
// serial contract) no thread is created and NextBatch() fills the slots
// directly.
// =============================================================================

namespace seeml::update_rt {

class BatchPipeline {
 public:
  /// `data` must outlive the pipeline. `input_floats` / `label_bytes` are
  /// the per-batch slot sizes from the plan header (label_bytes may be 0).
  BatchPipeline(Dataset& data, uint64_t batch, uint64_t input_floats,
                uint64_t label_bytes);
  ~BatchPipeline();

  BatchPipeline(const BatchPipeline&) = delete;
  BatchPipeline& operator=(const BatchPipeline&) = delete;

  /// Blocks until the staged batch is ready, copies it into the plan's I/O
  /// slots (`label_slot` may be null when label_bytes == 0), and wakes the
  /// feeder to stage the next one.
  void NextBatch(float* input_slot, uint8_t* label_slot);

 private:
  void FeederLoop();
  uint8_t* StagedLabelPtr() {
    return staged_labels_.empty() ? nullptr : staged_labels_.data();
  }

  Dataset& data_;
  const uint64_t batch_;
  std::vector<float> staged_inputs_;
  std::vector<uint8_t> staged_labels_;

  std::mutex mutex_;
  std::condition_variable cv_;
  bool full_ = false;   // guarded by mutex_: staged batch ready to consume
  bool stop_ = false;   // guarded by mutex_
  bool threaded_ = false;
  std::thread feeder_;
};

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_BATCH_PIPELINE_H_
