#include "source/smf.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <utility>

namespace seeml::update {

namespace {

constexpr uint64_t kDataAlignment = 64;

uint64_t AlignUp(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

// Overflow-checked u64 multiply for validating file-supplied sizes.
bool MulU64(uint64_t a, uint64_t b, uint64_t* out) {
  if (b != 0 && a > UINT64_MAX / b) return false;
  *out = a * b;
  return true;
}

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

    // Dims must be strictly positive — except a dynamic (-1) dim on non-const
    // tensors, which the compiler binds to the compiled batch size — with a
    // volume that cannot overflow the signed shape math downstream
    // (sir::Shape::volume / byteSize).
    uint64_t volume = 1;
    bool dims_ok = !t.dims.empty();
    for (int64_t dim : t.dims) {
      if (dim == -1 && !t.is_const) continue;
      if (dim <= 0 || !MulU64(volume, static_cast<uint64_t>(dim), &volume) ||
          volume > static_cast<uint64_t>(INT64_MAX) / sizeof(float)) {
        dims_ok = false;
        break;
      }
    }
    if (!dims_ok)
      return std::unexpected("SMF: tensor '" + t.name + "' has invalid dims");

    if (t.is_const) {
      // The instruction stream sizes reads from dims while rodata packing and
      // emit-table patching size from byte_size — they must agree exactly.
      uint64_t expected_bytes = 0;
      if (!MulU64(volume, sizeof(float), &expected_bytes) ||
          t.byte_size != expected_bytes)
        return std::unexpected("SMF: tensor '" + t.name +
                               "' byte size disagrees with its dims");
      // Overflow-safe range check: offset + size may wrap in u64.
      if (t.data_offset > bytes.size() ||
          t.byte_size > bytes.size() - t.data_offset)
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
    const uint8_t kind = r.Read<uint8_t>();
    // Range-check the op vocabulary here: ForwardBuilder switches over the
    // enum, and an out-of-range value would silently skip the op, surfacing
    // later as a misleading "not a constant tensor" error.
    if (r.ok && kind > static_cast<uint8_t>(SmfOpKind::kRelu))
      return std::unexpected("SMF: op " + std::to_string(i) +
                             " has unknown kind " + std::to_string(kind));
    op.kind = static_cast<SmfOpKind>(kind);
    op.name = r.ReadStr();
    const uint8_t n_in = r.Read<uint8_t>();
    for (uint8_t k = 0; k < n_in; ++k) op.inputs.push_back(r.ReadStr());
    op.output = r.ReadStr();
    model.ops.push_back(std::move(op));
  }

  if (!r.ok) return std::unexpected("SMF: truncated file '" + path + "'");

  // Each blob was individually range-checked against the file, but the
  // ranges must also be pairwise disjoint and lie past the metadata (the
  // writer's layout). Without this, a small file can declare its whole
  // extent as the data of arbitrarily many tensors, amplifying one crafted
  // or corrupt megabyte into an unbounded total allocation above.
  std::vector<std::pair<uint64_t, uint64_t>> ranges;  // (offset, size)
  for (const auto& t : model.tensors)
    if (t.is_const) ranges.emplace_back(t.data_offset, t.byte_size);
  std::sort(ranges.begin(), ranges.end());
  uint64_t prev_end = r.pos;  // first byte past the metadata section
  for (const auto& [off, size] : ranges) {
    if (off < prev_end)
      return std::unexpected(
          "SMF: constant tensor data ranges overlap each other or the "
          "metadata in '" + path + "'");
    prev_end = off + size;  // cannot wrap: range-checked against file size
  }
  return model;
}

std::expected<void, std::string> SaveSmf(const std::string& path,
                                         SmfModel& model) {
  // The record formats carry u16 string lengths and u8 rank/input counts;
  // Writer casts silently, so an oversized field would desynchronize every
  // record after it. Reject instead of writing a self-corrupting file.
  auto str_ok = [](const std::string& s) { return s.size() <= UINT16_MAX; };
  if (!str_ok(model.input_name) || !str_ok(model.output_name))
    return std::unexpected("SMF: I/O tensor name exceeds 65535 bytes");
  for (const auto& t : model.tensors) {
    if (!str_ok(t.name))
      return std::unexpected("SMF: tensor name exceeds 65535 bytes");
    if (t.dims.size() > UINT8_MAX)
      return std::unexpected("SMF: tensor '" + t.name + "' rank exceeds 255");
  }
  for (const auto& op : model.ops) {
    if (!str_ok(op.name) || !str_ok(op.output))
      return std::unexpected("SMF: op name exceeds 65535 bytes");
    if (op.inputs.size() > UINT8_MAX)
      return std::unexpected("SMF: op '" + op.name +
                             "' has more than 255 inputs");
    for (const auto& in : op.inputs)
      if (!str_ok(in))
        return std::unexpected("SMF: op '" + op.name +
                               "' input name exceeds 65535 bytes");
  }

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
