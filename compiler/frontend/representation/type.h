#ifndef SEEML_COMPILER_FRONTEND_REPRESENTATION_TYPE_H_
#define SEEML_COMPILER_FRONTEND_REPRESENTATION_TYPE_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

// =============================================================================
// The SIR type system: element types, shapes, and attribute values — the
// vocabulary every other core definition (value, operation, block) is written
// in. Part of the SIR core; see sir.h for the threading model.
// =============================================================================

namespace seeml::sir {

enum class DataType : uint8_t {
    F16, BF16, F32, F64,
    I8, I32, I64,
    Bool
};

constexpr size_t dtypeByteWidth(DataType dt) {
    switch (dt) {
        case DataType::Bool:
        case DataType::I8:   return 1;
        case DataType::F16:
        case DataType::BF16: return 2;
        case DataType::F32:
        case DataType::I32:  return 4;
        case DataType::F64:
        case DataType::I64:  return 8;
        default: return 0;
    }
}

constexpr std::string_view dtypeName(DataType dt) {
    switch (dt) {
        case DataType::F16:  return "f16";
        case DataType::BF16: return "bf16";
        case DataType::F32:  return "f32";
        case DataType::F64:  return "f64";
        case DataType::I8:   return "i8";
        case DataType::I32:  return "i32";
        case DataType::I64:  return "i64";
        case DataType::Bool: return "bool";
        default: return "unknown";
    }
}

struct Shape {
    static constexpr int64_t kDynamic = -1;

    std::vector<int64_t> dims;

    Shape() = default;
    explicit Shape(std::vector<int64_t> d) : dims(std::move(d)) {}
    Shape(std::initializer_list<int64_t> d) : dims(d) {}

    static Shape scalar() { return Shape{}; }

    int64_t rank() const { return static_cast<int64_t>(dims.size()); }
    bool isScalar() const { return dims.empty(); }
    bool isFullyStatic() const;
    /// Element count; kDynamic when any dimension is dynamic/negative or the
    /// product would overflow int64 ("unknown" saturates, never wraps).
    int64_t volume() const;
    /// Bytes at dtype `dt`; 0 when the volume is unknown or the byte count
    /// would overflow size_t.
    size_t byteSize(DataType dt) const;

    bool operator==(const Shape& o) const = default;
};

using AttributeValue = std::variant<
    int64_t, float, double, std::string,
    std::vector<int64_t>, std::vector<float>
>;

} // namespace seeml::sir

#endif // SEEML_COMPILER_FRONTEND_REPRESENTATION_TYPE_H_
