#include "compiler/frontend/ingressor/model_writer.h"

#include <cstring>
#include <fstream>
#include <vector>

#include "source/hash.h"

namespace seeml::update {

namespace {

constexpr uint64_t kDataAlignment = 64;

uint64_t AlignUp(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

// --- Little-endian primitive writer into an in-memory buffer -----------------

struct Writer {
  std::vector<uint8_t> buf;

  template <typename T>
  void Write(T v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    buf.insert(buf.end(), p, p + sizeof(T));
  }

  void WriteStr(const std::string& s) {
    Write<uint16_t>(static_cast<uint16_t>(s.size()));
    buf.insert(buf.end(), s.begin(), s.end());
  }
};

}  // namespace

std::expected<void, std::string> SaveSmf(const std::string& path,
                                         SmfModel& model) {
  // Pass 1: serialize the header/metadata with zeroed data offsets to learn
  // the metadata section size, then lay out the 64-aligned data section.
  auto serialize_meta = [&](Writer& w) {
    w.Write<uint32_t>(kSmfMagic);
    w.Write<uint32_t>(kSmfVersion);
    w.Write<uint32_t>(static_cast<uint32_t>(model.tensors.size()));
    w.Write<uint32_t>(static_cast<uint32_t>(model.ops.size()));
    w.WriteStr(model.input_name);
    w.WriteStr(model.output_name);
    for (const auto& t : model.tensors) {
      w.WriteStr(t.name);
      w.Write<uint8_t>(static_cast<uint8_t>(t.dims.size()));
      w.Write<uint8_t>(t.is_const ? 1 : 0);
      for (int64_t d : t.dims) w.Write<int64_t>(d);
      w.Write<uint64_t>(t.data_offset);
      w.Write<uint64_t>(t.byte_size);
    }
    for (const auto& op : model.ops) {
      w.Write<uint8_t>(static_cast<uint8_t>(op.kind));
      w.WriteStr(op.name);
      w.Write<uint8_t>(static_cast<uint8_t>(op.inputs.size()));
      for (const auto& in : op.inputs) w.WriteStr(in);
      w.WriteStr(op.output);
    }
  };

  Writer probe;
  serialize_meta(probe);

  uint64_t cursor = AlignUp(probe.buf.size(), kDataAlignment);
  for (auto& t : model.tensors) {
    if (!t.is_const) continue;
    if (t.data.empty())
      return std::unexpected("SMF: constant tensor '" + t.name +
                             "' has no data to serialize");
    t.byte_size = t.data.size();
    t.data_offset = cursor;
    cursor = AlignUp(cursor + t.byte_size, kDataAlignment);
  }

  // Pass 2: re-serialize with the final offsets, append the data section.
  // The probe told us the exact final size — reserve it up front so neither
  // the metadata append nor the data-section resize ever reallocates.
  Writer w;
  w.buf.reserve(cursor);
  serialize_meta(w);
  w.buf.resize(cursor, 0);
  for (const auto& t : model.tensors) {
    if (!t.is_const) continue;
    std::memcpy(w.buf.data() + t.data_offset, t.data.data(), t.byte_size);
  }

  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return std::unexpected("SMF: cannot write '" + path + "'");
  f.write(reinterpret_cast<const char*>(w.buf.data()),
          static_cast<std::streamsize>(w.buf.size()));
  if (!f) return std::unexpected("SMF: short write to '" + path + "'");
  // The model now corresponds to the saved bytes: bind its identity so a
  // plan compiled from this in-memory model patches this exact file.
  model.content_hash = ContentHash64(w.buf.data(), w.buf.size());
  return {};
}

}  // namespace seeml::update
