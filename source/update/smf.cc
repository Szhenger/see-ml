#include "source/update/smf.h"

#include <cstring>
#include <fstream>

namespace seeml::update {

namespace {

constexpr uint64_t kDataAlignment = 64;

uint64_t AlignUp(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

// --- Little-endian primitive readers over an in-memory buffer ----------------

struct Reader {
  const uint8_t* data;
  size_t size;
  size_t pos = 0;
  bool ok = true;

  template <typename T>
  T Read() {
    T v{};
    if (pos + sizeof(T) > size) {
      ok = false;
      return v;
    }
    std::memcpy(&v, data + pos, sizeof(T));
    pos += sizeof(T);
    return v;
  }

  std::string ReadStr() {
    uint16_t len = Read<uint16_t>();
    if (!ok || pos + len > size) {
      ok = false;
      return {};
    }
    std::string s(reinterpret_cast<const char*>(data + pos), len);
    pos += len;
    return s;
  }
};

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

const SmfTensor* SmfModel::FindTensor(std::string_view name) const {
  for (const auto& t : tensors)
    if (t.name == name) return &t;
  return nullptr;
}

std::expected<SmfModel, std::string> LoadSmf(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return std::unexpected("SMF: cannot open '" + path + "'");
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());

  Reader r{bytes.data(), bytes.size()};
  if (r.Read<uint32_t>() != kSmfMagic)
    return std::unexpected("SMF: bad magic in '" + path + "'");
  if (r.Read<uint32_t>() != kSmfVersion)
    return std::unexpected("SMF: unsupported version in '" + path + "'");

  const uint32_t num_tensors = r.Read<uint32_t>();
  const uint32_t num_ops = r.Read<uint32_t>();

  SmfModel model;
  model.input_name = r.ReadStr();
  model.output_name = r.ReadStr();

  for (uint32_t i = 0; i < num_tensors && r.ok; ++i) {
    SmfTensor t;
    t.name = r.ReadStr();
    const uint8_t rank = r.Read<uint8_t>();
    const uint8_t flags = r.Read<uint8_t>();
    t.is_const = (flags & 1) != 0;
    for (uint8_t d = 0; d < rank; ++d) t.dims.push_back(r.Read<int64_t>());
    t.data_offset = r.Read<uint64_t>();
    t.byte_size = r.Read<uint64_t>();
    if (t.is_const) {
      if (t.data_offset + t.byte_size > bytes.size())
        return std::unexpected("SMF: tensor '" + t.name +
                               "' data range exceeds file size");
      t.data.assign(bytes.begin() + static_cast<ptrdiff_t>(t.data_offset),
                    bytes.begin() +
                        static_cast<ptrdiff_t>(t.data_offset + t.byte_size));
    }
    model.tensors.push_back(std::move(t));
  }

  for (uint32_t i = 0; i < num_ops && r.ok; ++i) {
    SmfOp op;
    op.kind = static_cast<SmfOpKind>(r.Read<uint8_t>());
    op.name = r.ReadStr();
    const uint8_t n_in = r.Read<uint8_t>();
    for (uint8_t k = 0; k < n_in; ++k) op.inputs.push_back(r.ReadStr());
    op.output = r.ReadStr();
    model.ops.push_back(std::move(op));
  }

  if (!r.ok) return std::unexpected("SMF: truncated file '" + path + "'");
  return model;
}

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
  Writer w;
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
  return {};
}

}  // namespace seeml::update
