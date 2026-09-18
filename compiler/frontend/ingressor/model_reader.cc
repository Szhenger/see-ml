#include "compiler/frontend/ingressor/model_reader.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "compiler/diagnostics/tokenizing/error.h"
#include "source/identity/hash.h"
#include "source/parallel/parallel_for.h"

namespace seeml::update {

namespace tokenizing = seeml::diag::tokenizing;

namespace {

// Constant-tensor payloads below this total stay on the serial copy path:
// fanning a few KiB over the worker pool costs more than the copies.
constexpr uint64_t kParallelCopyThreshold = 4u << 20;
// One parallel copy chunk: large enough that a chunk is bandwidth-, not
// dispatch-bound, small enough that the largest tensor spans many.
constexpr size_t kParallelCopyGrain = 8u << 20;

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

/// The model file's bytes for the duration of one load. On POSIX hosts a
/// private read-only mapping: the file is the largest artifact the compiler
/// ingests, and reading it into a buffer made load peak at file + payloads
/// with the buffer's pages still resident after it was freed (E2, #81). A
/// mapping is file-backed — clean, evictable, never part of the anonymous
/// footprint — and unmapping it is unconditional. Elsewhere (or if mmap
/// refuses) a default-initialized heap buffer: no zero-fill before the read
/// overwrites it, as a std::vector sized to the file would have done.
class FileBytes {
 public:
  FileBytes() = default;
  FileBytes(const FileBytes&) = delete;
  FileBytes& operator=(const FileBytes&) = delete;
  ~FileBytes() {
#if SEEML_PAYLOAD_MMAP
    if (mapped_) ::munmap(const_cast<uint8_t*>(data_), size_);
#endif
  }

  /// nullopt on success, else the verb of the failure ("cannot open", ...).
  std::optional<std::string_view> Open(const std::string& path) {
#if SEEML_PAYLOAD_MMAP
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return "cannot open";
    struct stat st {};
    if (::fstat(fd, &st) != 0 || st.st_size < 0) {
      ::close(fd);
      return "cannot stat";
    }
    size_ = static_cast<size_t>(st.st_size);
    if (size_ != 0) {
      void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
      if (p != MAP_FAILED) {
        ::close(fd);
        data_ = static_cast<const uint8_t*>(p);
        mapped_ = true;
        return std::nullopt;
      }
    }
    ::close(fd);
    if (size_ == 0) return std::nullopt;
#endif
    std::ifstream f(path, std::ios::binary);
    if (!f) return "cannot open";
    f.seekg(0, std::ios::end);
    const std::streamoff end = f.tellg();
    if (end < 0) return "cannot stat";
    f.seekg(0);
    size_ = static_cast<size_t>(end);
    heap_.reset(new uint8_t[size_]);  // default-init: no zero-fill
    if (size_ != 0 && !f.read(reinterpret_cast<char*>(heap_.get()),
                              static_cast<std::streamsize>(size_)))
      return "cannot read";
    data_ = heap_.get();
    return std::nullopt;
  }

  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
  bool mapped_ = false;
  std::unique_ptr<uint8_t[]> heap_;
};

}  // namespace

std::expected<SmfModel, std::string> LoadSmf(const std::string& path) {
  FileBytes bytes;
  if (auto verb = bytes.Open(path))
    return tokenizing::FileError(std::string(*verb), path);

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

    // Dims must be strictly positive — except a dynamic (-1) LEADING dim on
    // non-const tensors, which the compiler binds to the compiled batch
    // size — with a volume that cannot overflow the signed shape math
    // downstream (sir::Shape::volume / byteSize). A -1 anywhere else has no
    // binding rule: the driver reads dims.back() as the static input width,
    // and a dynamic width would flow into the SIR, bind zero-byte slots,
    // and seal a plan the runtime always rejects.
    uint64_t volume = 1;
    bool dims_ok = !t.dims.empty();
    for (size_t d = 0; d < t.dims.size(); ++d) {
      const int64_t dim = t.dims[d];
      if (dim == -1 && !t.is_const && d == 0) continue;
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
    const uint8_t kind_max = version >= 4   ? kSmfOpKindMax
                             : version == 3 ? kSmfOpKindMaxV3
                             : version == 2 ? kSmfOpKindMaxV2
                                            : kSmfOpKindMaxV1;
    if (kind > kind_max)
      return tokenizing::Error("unknown op kind " + std::to_string(kind) +
                               " in '" + path + "'");
    op.kind = static_cast<SmfOpKind>(kind);
    op.name = r.ReadStr();
    const uint8_t n_in = r.Read<uint8_t>();
    for (uint8_t k = 0; k < n_in; ++k) op.inputs.push_back(r.ReadStr());
    op.output = r.ReadStr();
    if (version >= 3) op.attr0 = r.Read<uint32_t>();
    if (version >= 5) op.attr1 = r.Read<uint32_t>();
    if (version >= 6) op.attr2 = r.Read<uint32_t>();
    // attr1 is defined only for kRope (the rotary base); on every other
    // kind it is reserved and must be zero, so a future meaning can never
    // be silently misread by a reader that predates it.
    if (op.attr1 != 0 && op.kind != SmfOpKind::kRope)
      return tokenizing::Error("op '" + op.name + "' carries a nonzero attr1, "
                               "which its kind does not define, in '" + path +
                               "'");
    // attr2 (v6) is the epsilon of a LayerNorm / RmsNorm, nothing else's;
    // when set it must be a finite positive float.
    if (op.attr2 != 0) {
      const bool norm = op.kind == SmfOpKind::kLayerNorm ||
                        op.kind == SmfOpKind::kRmsNorm;
      const float eps = std::bit_cast<float>(op.attr2);
      if (!norm)
        return tokenizing::Error("op '" + op.name + "' carries a nonzero "
                                 "attr2, which its kind does not define, in '" +
                                 path + "'");
      if (!std::isfinite(eps) || !(eps > 0.0f))
        return tokenizing::Error("op '" + op.name + "' has a normalization "
                                 "epsilon that is not a finite positive "
                                 "float, in '" + path + "'");
    }
    model.ops.push_back(std::move(op));
  }

  if (!r.ok) return tokenizing::FileError("truncated file", path);

  // Materialize the validated constant payloads. The copy is partitioned
  // over the payload BYTES, not the tensor list (E2, #81): a model is
  // typically dominated by one tensor — an embedding table — and a
  // per-tensor split copied it on a single thread while the rest idled.
  // Payloads are sized without a zero-fill (SmfBytes), so a chunk may land
  // anywhere inside a tensor; every byte is written exactly once, by
  // whichever chunk owns it, and the chunk geometry never depends on the
  // worker count.
  std::vector<uint64_t> starts;  // prefix sums over const_tensors
  starts.reserve(const_tensors.size() + 1);
  starts.push_back(0);
  for (size_t idx : const_tensors) {
    SmfTensor& t = model.tensors[idx];
    t.data.resize(static_cast<size_t>(t.byte_size));
    starts.push_back(starts.back() + t.byte_size);
  }
  auto copy_range = [&](uint64_t begin, uint64_t end) {
    // First tensor whose span reaches past `begin`.
    size_t i = static_cast<size_t>(
        std::upper_bound(starts.begin(), starts.end(), begin) -
        starts.begin() - 1);
    for (; i < const_tensors.size() && starts[i] < end; ++i) {
      SmfTensor& t = model.tensors[const_tensors[i]];
      const uint64_t lo = std::max(begin, starts[i]) - starts[i];
      const uint64_t hi = std::min(end, starts[i + 1]) - starts[i];
      std::memcpy(t.data.data() + lo, bytes.data() + t.data_offset + lo,
                  static_cast<size_t>(hi - lo));
    }
  };
  if (const_bytes >= kParallelCopyThreshold) {
    ParallelFor(static_cast<size_t>(const_bytes), kParallelCopyGrain,
                [&](size_t begin, size_t end, size_t /*chunk*/) {
                  copy_range(begin, end);
                });
  } else {
    copy_range(0, const_bytes);
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
