#include "runtime/dataset.h"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace seeml::update_rt {

namespace {

// Overflow-checked u64 multiply for validating file-supplied sizes.
bool MulU64(uint64_t a, uint64_t b, uint64_t* out) {
  if (b != 0 && a > UINT64_MAX / b) return false;
  *out = a * b;
  return true;
}

constexpr uint64_t kSdsHeaderBytes = 40;

}  // namespace

uint64_t Dataset::label_bytes_per_sample() const {
  switch (label_kind_) {
    case 1: return sizeof(int32_t);
    case 2: return label_dim_ * sizeof(float);
    default: return 0;
  }
}

std::expected<Dataset, std::string> Dataset::FromMemory(
    std::vector<float> inputs, std::vector<uint8_t> labels,
    uint64_t num_samples, uint64_t input_dim, uint32_t label_kind,
    uint64_t label_dim) {
  Dataset d;
  d.num_samples_ = num_samples;
  d.input_dim_ = input_dim;
  d.label_kind_ = label_kind;
  d.label_dim_ = label_dim;
  d.inputs_ = std::move(inputs);
  d.labels_ = std::move(labels);
  if (num_samples == 0) return std::unexpected("Dataset: zero samples");
  uint64_t want_inputs = 0, want_labels = 0;
  if (!MulU64(num_samples, input_dim, &want_inputs) ||
      d.inputs_.size() != want_inputs)
    return std::unexpected("Dataset: input buffer size mismatch");
  if (!MulU64(num_samples, d.label_bytes_per_sample(), &want_labels) ||
      d.labels_.size() != want_labels)
    return std::unexpected("Dataset: label buffer size mismatch");
  return d;
}

std::expected<Dataset, std::string> Dataset::LoadFromFile(
    const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return std::unexpected("Dataset: cannot open '" + path + "'");

  // The file size bounds every allocation below: a corrupt header cannot ask
  // for more sample data than the file actually holds.
  f.seekg(0, std::ios::end);
  const std::streamoff end_pos = f.tellg();
  if (end_pos < 0) return std::unexpected("Dataset: cannot stat '" + path + "'");
  const uint64_t file_size = static_cast<uint64_t>(end_pos);
  f.seekg(0);

  auto read = [&](void* dst, size_t n) {
    f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
    return static_cast<bool>(f);
  };

  uint32_t magic = 0, version = 0, label_kind = 0, pad = 0;
  uint64_t num_samples = 0, input_dim = 0, label_dim = 0;
  if (!read(&magic, 4) || magic != kSdsMagic)
    return std::unexpected("Dataset: bad magic in '" + path + "'");
  if (!read(&version, 4) || version != 1)
    return std::unexpected("Dataset: unsupported version");
  if (!read(&num_samples, 8) || !read(&input_dim, 8) || !read(&label_kind, 4) ||
      !read(&pad, 4) || !read(&label_dim, 8))
    return std::unexpected("Dataset: truncated header");

  Dataset d;
  d.num_samples_ = num_samples;
  d.input_dim_ = input_dim;
  d.label_kind_ = label_kind;
  d.label_dim_ = label_dim;
  if (num_samples == 0 || input_dim == 0)
    return std::unexpected("Dataset: empty dataset");
  if (label_kind > 2)
    return std::unexpected("Dataset: unknown label kind");
  if (label_kind == 2 &&
      (label_dim == 0 || label_dim > UINT64_MAX / sizeof(float)))
    return std::unexpected("Dataset: label dim out of range");

  // Overflow-safe sizing, cross-checked against the actual file size before
  // any allocation.
  const uint64_t lbytes = d.label_bytes_per_sample();
  uint64_t input_bytes = 0, payload = 0;
  if (!MulU64(input_dim, sizeof(float), &input_bytes) ||
      input_bytes > UINT64_MAX - lbytes ||
      !MulU64(num_samples, input_bytes + lbytes, &payload) ||
      payload > file_size - kSdsHeaderBytes)
    return std::unexpected("Dataset: sample section exceeds file size");

  d.inputs_.resize(num_samples * input_dim);
  d.labels_.resize(num_samples * lbytes);

  if (lbytes == 0) {
    // No labels: the sample section is one contiguous f32 block — read it
    // straight into place, a single bulk transfer instead of one syscall
    // per sample.
    if (!read(d.inputs_.data(), num_samples * input_bytes))
      return std::unexpected("Dataset: truncated inputs");
    return d;
  }

  // Interleaved records: stream fixed chunks of whole records through a
  // bounded buffer and deinterleave in memory. Bulk reads amortize the
  // stream overhead; the buffer bound keeps peak memory flat no matter how
  // large the corpus is.
  const uint64_t record = input_bytes + lbytes;
  constexpr uint64_t kChunkBudget = 256 * 1024;
  const uint64_t per_chunk =
      record >= kChunkBudget ? 1 : kChunkBudget / record;
  std::vector<uint8_t> chunk(per_chunk * record);
  for (uint64_t i = 0; i < num_samples;) {
    const uint64_t n = std::min(per_chunk, num_samples - i);
    if (!read(chunk.data(), n * record))
      return std::unexpected("Dataset: truncated inputs");
    for (uint64_t s = 0; s < n; ++s) {
      const uint8_t* rec = chunk.data() + s * record;
      std::memcpy(d.inputs_.data() + (i + s) * input_dim, rec, input_bytes);
      std::memcpy(d.labels_.data() + (i + s) * lbytes, rec + input_bytes,
                  lbytes);
    }
    i += n;
  }
  return d;
}

std::expected<void, std::string> Dataset::ValidateClassLabels(
    uint64_t num_classes) const {
  if (label_kind_ != 1 || num_classes == 0) return {};
  const auto* labels = reinterpret_cast<const int32_t*>(labels_.data());
  for (uint64_t i = 0; i < num_samples_; ++i)
    if (labels[i] < 0 || static_cast<uint64_t>(labels[i]) >= num_classes)
      return std::unexpected(
          "Dataset: class label " + std::to_string(labels[i]) + " at sample " +
          std::to_string(i) + " outside [0, " + std::to_string(num_classes) +
          ")");
  return {};
}

std::expected<void, std::string> Dataset::SaveToFile(
    const std::string& path) const {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return std::unexpected("Dataset: cannot write '" + path + "'");
  auto write = [&](const void* src, size_t n) {
    f.write(reinterpret_cast<const char*>(src),
            static_cast<std::streamsize>(n));
  };
  const uint32_t version = 1, pad = 0;
  write(&kSdsMagic, 4);
  write(&version, 4);
  write(&num_samples_, 8);
  write(&input_dim_, 8);
  write(&label_kind_, 4);
  write(&pad, 4);
  write(&label_dim_, 8);
  const uint64_t lbytes = label_bytes_per_sample();
  if (lbytes == 0) {
    // Unlabeled: the sample section is exactly the input block.
    write(inputs_.data(), num_samples_ * input_dim_ * sizeof(float));
  } else {
    // Interleave whole records through a bounded staging chunk and write in
    // bulk — the mirror of LoadFromFile's chunked reader.
    const uint64_t input_bytes = input_dim_ * sizeof(float);
    const uint64_t record = input_bytes + lbytes;
    constexpr uint64_t kChunkBudget = 256 * 1024;
    const uint64_t per_chunk =
        record >= kChunkBudget ? 1 : kChunkBudget / record;
    std::vector<uint8_t> chunk(per_chunk * record);
    for (uint64_t i = 0; i < num_samples_;) {
      const uint64_t n = std::min(per_chunk, num_samples_ - i);
      for (uint64_t s = 0; s < n; ++s) {
        uint8_t* rec = chunk.data() + s * record;
        std::memcpy(rec, inputs_.data() + (i + s) * input_dim_, input_bytes);
        std::memcpy(rec + input_bytes, labels_.data() + (i + s) * lbytes,
                    lbytes);
      }
      write(chunk.data(), n * record);
      i += n;
    }
  }
  if (!f) return std::unexpected("Dataset: short write to '" + path + "'");
  return {};
}

void Dataset::FillBatch(uint64_t batch, float* input_slot,
                        uint8_t* label_slot) {
  const uint64_t lbytes = label_bytes_per_sample();

  if (order_.empty()) {
    // Sequential serving: samples are contiguous in memory, so copy whole
    // runs (up to the wraparound point) instead of one sample at a time.
    uint64_t b = 0;
    while (b < batch) {
      const uint64_t run = std::min(batch - b, num_samples_ - cursor_);
      std::memcpy(input_slot + b * input_dim_,
                  inputs_.data() + cursor_ * input_dim_,
                  run * input_dim_ * sizeof(float));
      if (label_slot && lbytes)
        std::memcpy(label_slot + b * lbytes, labels_.data() + cursor_ * lbytes,
                    run * lbytes);
      b += run;
      cursor_ += run;
      if (cursor_ == num_samples_) cursor_ = 0;
    }
    return;
  }

  for (uint64_t b = 0; b < batch; ++b) {
    const uint64_t i = order_[cursor_];
    std::memcpy(input_slot + b * input_dim_, inputs_.data() + i * input_dim_,
                input_dim_ * sizeof(float));
    if (label_slot && lbytes)
      std::memcpy(label_slot + b * lbytes, labels_.data() + i * lbytes, lbytes);
    cursor_ = cursor_ + 1;
    if (cursor_ == num_samples_) {
      cursor_ = 0;
      Reshuffle();  // fresh permutation every epoch
    }
  }
}

namespace {

// splitmix64: tiny, deterministic, and high-quality enough for shuffling.
// Avoids dragging <random> (and its per-platform distribution differences)
// into the device runtime — the permutation must be reproducible everywhere.
uint64_t SplitMix64(uint64_t* state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

}  // namespace

void Dataset::EnableShuffle(uint64_t seed) {
  // Fold the seed through one splitmix step so seed 0 is a valid choice
  // (shuffle_state_ == 0 means "shuffling off").
  shuffle_state_ = seed;
  shuffle_state_ = SplitMix64(&shuffle_state_) | 1ULL;
  order_.resize(num_samples_);
  Reshuffle();
  cursor_ = 0;
}

void Dataset::Reshuffle() {
  for (uint64_t i = 0; i < num_samples_; ++i) order_[i] = i;
  // Fisher–Yates with an unbiased-enough bound (num_samples << 2^64).
  for (uint64_t i = num_samples_ - 1; i > 0; --i) {
    const uint64_t j = SplitMix64(&shuffle_state_) % (i + 1);
    std::swap(order_[i], order_[j]);
  }
}

std::expected<Dataset, std::string> Dataset::SplitValidation(double fraction) {
  if (!(fraction > 0.0) || fraction >= 1.0)
    return std::unexpected("Dataset: validation fraction must be in (0, 1)");
  if (num_samples_ < 2)
    return std::unexpected("Dataset: too few samples to split");
  if (!order_.empty())
    return std::unexpected("Dataset: split before enabling shuffle");

  uint64_t val_n = static_cast<uint64_t>(
      static_cast<double>(num_samples_) * fraction);
  if (val_n == 0) val_n = 1;
  if (val_n >= num_samples_) val_n = num_samples_ - 1;
  const uint64_t train_n = num_samples_ - val_n;
  const uint64_t lbytes = label_bytes_per_sample();

  Dataset val;
  val.num_samples_ = val_n;
  val.input_dim_ = input_dim_;
  val.label_kind_ = label_kind_;
  val.label_dim_ = label_dim_;
  val.inputs_.assign(inputs_.begin() + static_cast<ptrdiff_t>(train_n * input_dim_),
                     inputs_.end());
  val.labels_.assign(labels_.begin() + static_cast<ptrdiff_t>(train_n * lbytes),
                     labels_.end());

  inputs_.resize(train_n * input_dim_);
  labels_.resize(train_n * lbytes);
  num_samples_ = train_n;
  cursor_ = 0;
  return val;
}

}  // namespace seeml::update_rt
