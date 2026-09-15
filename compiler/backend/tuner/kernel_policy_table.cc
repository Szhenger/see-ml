#include "compiler/backend/tuner/kernel_policy_table.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

#include "compiler/backend/architecture/host_arch.h"
#include "compiler/diagnostics/architecting/error.h"

namespace seeml::update {

namespace architecting = seeml::diag::architecting;

namespace {

std::unexpected<std::string> PolicyError(const std::string& message) {
  return architecting::Error(architecting::kKernelPolicy, message);
}

// --- A strict reader for the JSON the table is written in ------------------
// Objects, arrays, strings (with the standard escapes; \u for the BMP),
// numbers, true/false/null — RFC 8259, nothing more: no comments, no
// trailing commas, no NaN, one value per document, nesting bounded. The
// reader is only ever pointed at a file the build host wrote, but a
// reader that guesses is a reader that misreads, so every deviation is an
// error naming the byte offset.

struct JsonValue {
  enum class Kind { kNull, kBool, kNumber, kString, kArray, kObject };
  Kind kind = Kind::kNull;
  bool boolean = false;
  double number = 0.0;
  std::string text;
  std::vector<JsonValue> items;                            // kArray
  std::vector<std::pair<std::string, JsonValue>> members;  // kObject

  const JsonValue* Member(std::string_view key) const {
    for (const auto& [k, v] : members)
      if (k == key) return &v;
    return nullptr;
  }
};

class JsonReader {
 public:
  explicit JsonReader(std::string_view text) : text_(text) {}

  std::expected<JsonValue, std::string> ReadDocument() {
    auto v = ReadValue(0);
    if (!v) return v;
    SkipSpace();
    if (pos_ != text_.size()) return Error("trailing characters after value");
    return v;
  }

 private:
  static constexpr size_t kMaxDepth = 64;

  std::unexpected<std::string> Error(const std::string& what) const {
    return std::unexpected("at byte " + std::to_string(pos_) + ": " + what);
  }

  void SkipSpace() {
    while (pos_ < text_.size() &&
           (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\n' ||
            text_[pos_] == '\r'))
      ++pos_;
  }

  bool Consume(std::string_view literal) {
    if (text_.substr(pos_, literal.size()) != literal) return false;
    pos_ += literal.size();
    return true;
  }

  std::expected<JsonValue, std::string> ReadValue(size_t depth) {
    if (depth > kMaxDepth) return Error("nesting deeper than 64");
    SkipSpace();
    if (pos_ >= text_.size()) return Error("unexpected end of text");
    JsonValue v;
    const char c = text_[pos_];
    if (c == '{') return ReadObject(depth);
    if (c == '[') return ReadArray(depth);
    if (c == '"') {
      auto s = ReadString();
      if (!s) return std::unexpected(s.error());
      v.kind = JsonValue::Kind::kString;
      v.text = std::move(*s);
      return v;
    }
    if (Consume("true")) {
      v.kind = JsonValue::Kind::kBool;
      v.boolean = true;
      return v;
    }
    if (Consume("false")) {
      v.kind = JsonValue::Kind::kBool;
      return v;
    }
    if (Consume("null")) return v;
    if (c == '-' || (c >= '0' && c <= '9')) return ReadNumber();
    return Error(std::string("unexpected character '") + c + "'");
  }

  std::expected<JsonValue, std::string> ReadObject(size_t depth) {
    JsonValue v;
    v.kind = JsonValue::Kind::kObject;
    ++pos_;  // '{'
    SkipSpace();
    if (pos_ < text_.size() && text_[pos_] == '}') {
      ++pos_;
      return v;
    }
    for (;;) {
      SkipSpace();
      if (pos_ >= text_.size() || text_[pos_] != '"')
        return Error("expected a string key");
      auto key = ReadString();
      if (!key) return std::unexpected(key.error());
      SkipSpace();
      if (pos_ >= text_.size() || text_[pos_] != ':')
        return Error("expected ':' after key");
      ++pos_;
      auto val = ReadValue(depth + 1);
      if (!val) return val;
      for (const auto& [k, unused] : v.members)
        if (k == *key) return Error("duplicate key \"" + *key + "\"");
      v.members.emplace_back(std::move(*key), std::move(*val));
      SkipSpace();
      if (pos_ >= text_.size()) return Error("unterminated object");
      if (text_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (text_[pos_] == '}') {
        ++pos_;
        return v;
      }
      return Error("expected ',' or '}' in object");
    }
  }

  std::expected<JsonValue, std::string> ReadArray(size_t depth) {
    JsonValue v;
    v.kind = JsonValue::Kind::kArray;
    ++pos_;  // '['
    SkipSpace();
    if (pos_ < text_.size() && text_[pos_] == ']') {
      ++pos_;
      return v;
    }
    for (;;) {
      auto item = ReadValue(depth + 1);
      if (!item) return item;
      v.items.push_back(std::move(*item));
      SkipSpace();
      if (pos_ >= text_.size()) return Error("unterminated array");
      if (text_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (text_[pos_] == ']') {
        ++pos_;
        return v;
      }
      return Error("expected ',' or ']' in array");
    }
  }

  std::expected<std::string, std::string> ReadString() {
    ++pos_;  // opening quote
    std::string out;
    while (pos_ < text_.size()) {
      const char c = text_[pos_++];
      if (c == '"') return out;
      if (static_cast<unsigned char>(c) < 0x20)
        return Error("control character inside string");
      if (c != '\\') {
        out += c;
        continue;
      }
      if (pos_ >= text_.size()) return Error("unterminated escape");
      const char e = text_[pos_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          if (pos_ + 4 > text_.size()) return Error("short \\u escape");
          uint32_t cp = 0;
          for (int i = 0; i < 4; ++i) {
            const char h = text_[pos_++];
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= static_cast<uint32_t>(h - '0');
            else if (h >= 'a' && h <= 'f') cp |= static_cast<uint32_t>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') cp |= static_cast<uint32_t>(h - 'A' + 10);
            else return Error("bad hex digit in \\u escape");
          }
          if (cp >= 0xD800 && cp <= 0xDFFF)
            return Error("surrogate \\u escapes are not supported");
          if (cp < 0x80) {
            out += static_cast<char>(cp);
          } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
          } else {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
          }
          break;
        }
        default:
          return Error(std::string("unknown escape '\\") + e + "'");
      }
    }
    return Error("unterminated string");
  }

  std::expected<JsonValue, std::string> ReadNumber() {
    const size_t start = pos_;
    auto digits = [&] {
      const size_t s = pos_;
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9')
        ++pos_;
      return pos_ > s;
    };
    if (text_[pos_] == '-') ++pos_;
    if (pos_ < text_.size() && text_[pos_] == '0') {
      ++pos_;
    } else if (!digits()) {
      return Error("malformed number");
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      if (!digits()) return Error("malformed number fraction");
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-'))
        ++pos_;
      if (!digits()) return Error("malformed number exponent");
    }
    const std::string lexeme(text_.substr(start, pos_ - start));
    char* end = nullptr;
    const double d = std::strtod(lexeme.c_str(), &end);
    if (end != lexeme.c_str() + lexeme.size() || !std::isfinite(d))
      return Error("number out of range");
    JsonValue v;
    v.kind = JsonValue::Kind::kNumber;
    v.number = d;
    return v;
  }

  std::string_view text_;
  size_t pos_ = 0;
};

/// A JSON number that is exactly a u32.
std::expected<uint32_t, std::string> AsU32(const JsonValue* v,
                                           const std::string& what) {
  if (!v || v->kind != JsonValue::Kind::kNumber)
    return std::unexpected(what + " must be a number");
  if (v->number < 0.0 || v->number > 4294967295.0 ||
      v->number != std::floor(v->number))
    return std::unexpected(what + " must be a whole number in u32 range");
  return static_cast<uint32_t>(v->number);
}

std::expected<void, std::string> CheckEntry(const KernelPolicyEntry& e,
                                            const std::string& what) {
  if (e.gemm_tile_k == 0 || e.gemm_tile_n == 0)
    return std::unexpected(what + ": both tiles must be positive");
  if (e.gemm_tile_k % 4 != 0)
    return std::unexpected(what + ": gemm_tile_k " +
                           std::to_string(e.gemm_tile_k) +
                           " is not a multiple of the kernel's 4-wide unroll");
  return {};
}

}  // namespace

const KernelPolicyEntry* KernelPolicyTable::Find(
    std::string_view host_key) const {
  const auto it = hosts.find(std::string(host_key));
  return it == hosts.end() ? nullptr : &it->second;
}

std::expected<KernelPolicyTable, std::string> ParseKernelPolicyTable(
    std::string_view json) {
  auto doc = JsonReader(json).ReadDocument();
  if (!doc) return PolicyError("table is not valid JSON (" + doc.error() + ")");
  if (doc->kind != JsonValue::Kind::kObject)
    return PolicyError("table must be a JSON object");
  const JsonValue* schema = doc->Member("schema");
  if (!schema || schema->kind != JsonValue::Kind::kNumber || schema->number != 1.0)
    return PolicyError("table \"schema\" must be 1");
  const JsonValue* hosts = doc->Member("hosts");
  if (!hosts || hosts->kind != JsonValue::Kind::kObject)
    return PolicyError("table \"hosts\" must be an object keyed by host");

  KernelPolicyTable table;
  for (const auto& [key, host] : hosts->members) {
    if (host.kind != JsonValue::Kind::kObject)
      return PolicyError("host \"" + key + "\" must be an object");
    const JsonValue* cpu = host.Member("cpu");
    if (!cpu) continue;  // a host with a GPU policy only, some day
    if (cpu->kind != JsonValue::Kind::kObject)
      return PolicyError("host \"" + key + "\": \"cpu\" must be an object");
    KernelPolicyEntry e;
    auto k = AsU32(cpu->Member("gemm_tile_k"), "host \"" + key + "\": gemm_tile_k");
    if (!k) return PolicyError(k.error());
    auto n = AsU32(cpu->Member("gemm_tile_n"), "host \"" + key + "\": gemm_tile_n");
    if (!n) return PolicyError(n.error());
    e.gemm_tile_k = *k;
    e.gemm_tile_n = *n;
    if (auto ok = CheckEntry(e, "host \"" + key + "\""); !ok)
      return PolicyError(ok.error());
    table.hosts.emplace(key, e);
  }
  return table;
}

std::expected<KernelPolicyTable, std::string> LoadKernelPolicyTable(
    const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return PolicyError("cannot read kernel-policy table '" + path + "'");
  std::ostringstream buf;
  buf << in.rdbuf();
  auto table = ParseKernelPolicyTable(buf.str());
  if (!table)
    return std::unexpected(table.error() + " ['" + path + "']");
  return table;
}

std::expected<KernelPolicyEntry, std::string> ParseGemmTilesFlag(
    std::string_view text) {
  auto whole = [](std::string_view s, uint32_t* out) {
    if (s.empty() || s.size() > 10) return false;
    uint64_t v = 0;
    for (const char c : s) {
      if (c < '0' || c > '9') return false;
      v = v * 10 + static_cast<uint64_t>(c - '0');
    }
    if (v > 0xFFFFFFFFull) return false;
    *out = static_cast<uint32_t>(v);
    return true;
  };
  const size_t comma = text.find(',');
  KernelPolicyEntry e;
  if (comma == std::string_view::npos ||
      !whole(text.substr(0, comma), &e.gemm_tile_k) ||
      !whole(text.substr(comma + 1), &e.gemm_tile_n))
    return PolicyError("--gemm-tiles expects K,N as two whole numbers, got '" +
                       std::string(text) + "'");
  if (auto ok = CheckEntry(e, "--gemm-tiles"); !ok) return PolicyError(ok.error());
  return e;
}

std::expected<KernelPolicyChoice, std::string> ResolveKernelPolicy(
    const KernelPolicyRequest& request) {
  KernelPolicyChoice choice;
  choice.host_key = request.target_host ? *request.target_host
                                        : HostKey(DetectHostArch());
  if (request.target_host && !request.table_path)
    return PolicyError("--target-host names a table entry; pass the table "
                       "with --kernel-policy");
  if (request.explicit_tiles) {
    if (auto ok = CheckEntry(*request.explicit_tiles, "--gemm-tiles"); !ok)
      return PolicyError(ok.error());
    choice.tiles = *request.explicit_tiles;
    choice.source = "flag";
    return choice;
  }
  if (request.table_path) {
    auto table = LoadKernelPolicyTable(*request.table_path);
    if (!table) return std::unexpected(table.error());
    if (const KernelPolicyEntry* e = table->Find(choice.host_key)) {
      choice.tiles = *e;
      choice.source = "table";
      return choice;
    }
    choice.note = "kernel-policy table '" + *request.table_path +
                  "' has no entry for host \"" + choice.host_key +
                  "\"; the runtime's default GEMM tiles apply (run "
                  "tool/autotune.py on this host to add one)";
    architecting::DetectionFallback(architecting::kKernelPolicy, choice.note);
  }
  choice.source = "default";
  return choice;
}

}  // namespace seeml::update
