#include "compiler/frontend/ingressor/model_reader.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <thread>
#include <unordered_set>
#include <vector>

#include "compiler/diagnostics/tokenizing/error.h"
#include "source/identity/hash.h"
#include "source/parallel/parallel_for.h"

namespace seeml::update {

namespace tokenizing = seeml::diag::tokenizing;

namespace {

// Constant-tensor payloads below this total stay on the serial copy path:
// fanning a few KiB over the worker pool costs more than the copies.
constexpr uint64_t kParallelCopyThreshold = 4u << 20;

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

}  // namespace

std::expected<SmfModel, std::string> LoadSmf(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return tokenizing::FileError("cannot open", path);
  // Sized single read: stat, size the buffer once, one bulk transfer —
  // model files are the largest artifact the compiler ingests, and the
  // byte-at-a-time istreambuf_iterator form this replaces re-grew the
  // vector all the way up.
  f.seekg(0, std::ios::end);
  const std::streamoff end = f.tellg();
  if (end < 0) return tokenizing::FileError("cannot stat", path);
  f.seekg(0);
  std::vector<uint8_t> bytes(static_cast<size_t>(end));
  if (!bytes.empty() &&
      !f.read(reinterpret_cast<char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size())))
    return tokenizing::FileError("cannot read", path);

  Reader r{bytes.data(), bytes.size()};
  if (r.Read<uint32_t>() != kSmfMagic)
    return tokenizing::FileError("bad magic in", path);
  const uint32_t version = r.Read<uint32_t>();
  if (version < kSmfMinVersion || version > kSmfVersion)
    return tokenizing::FileError("unsupported version in", path);

  const uint32_t num_tensors = r.Read<uint32_t>();
  const uint32_t num_ops = r.Read<uint32_t>();

  SmfModel model;
  std::vector<size_t> const_tensors;  // indices of validated const tensors
  uint64_t const_bytes = 0;
  model.input_name = r.ReadStr();
  model.output_name = r.ReadStr();
  if (version >= 3) model.seq_len = r.Read<uint64_t>();
  // Counts are validated implicitly by the bounded Reader; reserving to the
  // declared sizes (capped against what the file could physically contain,
  // so a hostile header cannot demand gigabytes) avoids re-growth during the
  // parse. The caps divide by the minimum on-disk record size — capping at
  // bytes.size() *elements* would still let a small file demand
  // bytes.size() * sizeof(SmfTensor) of capacity up front.
  constexpr size_t kMinTensorRecordBytes = 28;  // name len + rank + flags +
                                                // 1 dim + offset + size
  constexpr size_t kMinOpRecordBytes = 6;  // kind + 2 empty names + input cnt
  model.tensors.reserve(
      std::min<size_t>(num_tensors, bytes.size() / kMinTensorRecordBytes));
  model.ops.reserve(std::min<size_t>(num_ops, bytes.size() / kMinOpRecordBytes));

  // Tensor names key every downstream binding (resolver, analyzer, sema);
  // a duplicate would silently resolve last-writer-wins, so it is a load
  // error, not a tolerated redundancy. Owning strings: the tensors vector
  // moves its names, which would dangle any view taken here.
  std::unordered_set<std::string> tensor_names;
  tensor_names.reserve(
      std::min<size_t>(num_tensors, bytes.size() / kMinTensorRecordBytes));

  for (uint32_t i = 0; i < num_tensors && r.ok; ++i) {
    SmfTensor t;
    t.name = r.ReadStr();
    if (r.ok && !tensor_names.insert(t.name).second)
      return tokenizing::TensorError(t.name, "is declared more than once");
    const uint8_t rank = r.Read<uint8_t>();
    const uint8_t flags = r.Read<uint8_t>();
    t.is_const = (flags & 1) != 0;
    for (uint8_t d = 0; d < rank; ++d) t.dims.push_back(r.Read<int64_t>());
    t.data_offset = r.Read<uint64_t>();
    t.byte_size = r.Read<uint64_t>();
    // A short read leaves zeroed fields; validating those zeros would blame
    // the tensor ("invalid dims") for what is really a cut-off file.
    if (!r.ok) return tokenizing::FileError("truncated file", path);

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
      return tokenizing::TensorError(t.name, "has invalid dims");

    if (t.is_const) {
      // The instruction stream sizes reads from dims while rodata packing and
      // emit-table patching size from byte_size — they must agree exactly.
      uint64_t expected_bytes = 0;
      if (!MulU64(volume, sizeof(float), &expected_bytes) ||
          t.byte_size != expected_bytes)
        return tokenizing::TensorError(t.name,
                                       "byte size disagrees with its dims");
      // Overflow-safe range check: offset + size may wrap in u64.
      if (t.data_offset > bytes.size() ||
          t.byte_size > bytes.size() - t.data_offset)
        return tokenizing::TensorError(t.name, "data range exceeds file size");
      // Payload copies are deferred: the scan pass touches metadata only, so
      // every blob is validated before the first byte moves and the copies
      // can be fanned out together afterwards.
      const_tensors.push_back(model.tensors.size());
      const_bytes += t.byte_size;
    }
    model.tensors.push_back(std::move(t));
  }

  for (uint32_t i = 0; i < num_ops && r.ok; ++i) {
    SmfOp op;
    const uint8_t kind = r.Read<uint8_t>();
    // Range-check before the cast: an unknown kind must be a load error, not
    // an out-of-range enum that a downstream switch silently skips. The
    // ceiling is per-version — a v3 kind inside a pre-v3 file is corruption,
    // not forward compatibility.
    const uint8_t kind_max = version >= 3 ? kSmfOpKindMax : kSmfOpKindMaxV2;
    if (kind > kind_max)
      return tokenizing::Error("unknown op kind " + std::to_string(kind) +
                               " in '" + path + "'");
    op.kind = static_cast<SmfOpKind>(kind);
    op.name = r.ReadStr();
    const uint8_t n_in = r.Read<uint8_t>();
    for (uint8_t k = 0; k < n_in; ++k) op.inputs.push_back(r.ReadStr());
    op.output = r.ReadStr();
    if (version >= 3) op.attr0 = r.Read<uint32_t>();
    model.ops.push_back(std::move(op));
  }

  if (!r.ok) return tokenizing::FileError("truncated file", path);

  // Materialize the validated constant payloads. Each copy writes only its
  // own tensor, so large models fan the copies out over ParallelFor
  // (deterministic: chunk geometry never depends on the worker count);
  // small ones stay serial.
  auto copy_payload = [&](size_t idx) {
    SmfTensor& t = model.tensors[idx];
    t.data.assign(bytes.begin() + static_cast<ptrdiff_t>(t.data_offset),
                  bytes.begin() +
                      static_cast<ptrdiff_t>(t.data_offset + t.byte_size));
  };
  if (const_bytes >= kParallelCopyThreshold && const_tensors.size() > 1) {
    ParallelFor(const_tensors.size(), 1,
                [&](size_t begin, size_t end, size_t /*chunk*/) {
                  for (size_t i = begin; i < end; ++i)
                    copy_payload(const_tensors[i]);
                });
  } else {
    for (size_t idx : const_tensors) copy_payload(idx);
  }

  model.content_hash = ContentHash64(bytes.data(), bytes.size());
  return model;
}

std::expected<std::vector<SmfModel>, std::string> LoadSmfMany(
    std::span<const std::string> paths) {
  std::vector<std::expected<SmfModel, std::string>> results(paths.size());
  if (paths.size() == 1) {
    results[0] = LoadSmf(paths[0]);
  } else if (!paths.empty()) {
    // Each loader writes only its own slot; LoadSmf is thread-compatible and
    // the worker pool serializes the in-flight data-parallel phases.
    std::vector<std::thread> loaders;
    loaders.reserve(paths.size());
    for (size_t i = 0; i < paths.size(); ++i)
      loaders.emplace_back([&results, &paths, i] {
        results[i] = LoadSmf(paths[i]);
      });
    for (std::thread& t : loaders) t.join();
  }

  std::vector<SmfModel> models;
  models.reserve(paths.size());
  for (auto& r : results) {
    if (!r) return std::unexpected(r.error());
    models.push_back(std::move(*r));
  }
  return models;
}

}  // namespace seeml::update
