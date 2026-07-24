#include "compiler/frontend/ingressor/model_format.h"

namespace seeml::update {

const SmfTensor* SmfModel::FindTensor(std::string_view name) const {
  for (const auto& t : tensors)
    if (t.name == name) return &t;
  return nullptr;
}

}  // namespace seeml::update
