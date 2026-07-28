#include "compiler/frontend/representation/type.h"

#include <algorithm>

namespace seeml::sir {

bool Shape::isFullyStatic() const {
    return std::none_of(dims.begin(), dims.end(), [](int64_t d) { return d == kDynamic; });
}

int64_t Shape::volume() const {
    int64_t v = 1;
    for (auto d : dims) {
        // Any negative dim (kDynamic or invalid) and any product that would
        // overflow int64 saturate to "unknown" — signed overflow here would
        // be UB, and a wrapped element count is worse than no count.
        if (d < 0) return kDynamic;
        if (d != 0 && v > INT64_MAX / d) return kDynamic;
        v *= d;
    }
    return v;
}

size_t Shape::byteSize(DataType dt) const {
    const int64_t vol = volume();
    if (vol < 0) return 0;
    const size_t width = dtypeByteWidth(dt);
    if (width != 0 && static_cast<size_t>(vol) > SIZE_MAX / width) return 0;
    return static_cast<size_t>(vol) * width;
}

} // namespace seeml::sir
