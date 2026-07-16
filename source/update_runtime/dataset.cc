#include "source/update_runtime/dataset.h"

#include <cstring>
#include <fstream>

namespace seeml::update_rt {

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
  if (d.inputs_.size() != num_samples * input_dim)
    return std::unexpected("Dataset: input buffer size mismatch");
  if (d.labels_.size() != num_samples * d.label_bytes_per_sample())
    return std::unexpected("Dataset: label buffer size mismatch");
  return d;
}

std::expected<Dataset, std::string> Dataset::LoadFromFile(
    const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return std::unexpected("Dataset: cannot open '" + path + "'");

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

  const uint64_t lbytes = d.label_bytes_per_sample();
  d.inputs_.resize(num_samples * input_dim);
  d.labels_.resize(num_samples * lbytes);
  for (uint64_t i = 0; i < num_samples; ++i) {
    if (!read(d.inputs_.data() + i * input_dim, input_dim * sizeof(float)))
      return std::unexpected("Dataset: truncated inputs");
    if (lbytes && !read(d.labels_.data() + i * lbytes, lbytes))
      return std::unexpected("Dataset: truncated labels");
  }
  return d;
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
  for (uint64_t i = 0; i < num_samples_; ++i) {
    write(inputs_.data() + i * input_dim_, input_dim_ * sizeof(float));
    if (lbytes) write(labels_.data() + i * lbytes, lbytes);
  }
  if (!f) return std::unexpected("Dataset: short write to '" + path + "'");
  return {};
}

void Dataset::FillBatch(uint64_t batch, float* input_slot,
                        uint8_t* label_slot) {
  const uint64_t lbytes = label_bytes_per_sample();
  for (uint64_t b = 0; b < batch; ++b) {
    const uint64_t i = cursor_;
    std::memcpy(input_slot + b * input_dim_, inputs_.data() + i * input_dim_,
                input_dim_ * sizeof(float));
    if (label_slot && lbytes)
      std::memcpy(label_slot + b * lbytes, labels_.data() + i * lbytes, lbytes);
    cursor_ = (cursor_ + 1) % num_samples_;
  }
}

}  // namespace seeml::update_rt
