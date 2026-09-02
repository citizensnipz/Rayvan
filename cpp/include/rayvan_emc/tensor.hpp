#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>

namespace rayvan::emc {

using Tensor = torch::Tensor;

inline void require_rank(const Tensor& tensor, std::int64_t rank, std::string_view name) {
    if (!tensor.defined() || tensor.dim() != rank) {
        throw std::invalid_argument(std::string(name) + " has an unexpected rank");
    }
}

inline void require_last_dim(const Tensor& tensor, std::int64_t size, std::string_view name) {
    require_rank(tensor, tensor.dim(), name);
    if (tensor.size(-1) != size) {
        throw std::invalid_argument(std::string(name) + " has an unexpected final dimension");
    }
}

inline void require_shape(
    const Tensor& tensor,
    std::span<const std::int64_t> expected,
    std::string_view name) {
    require_rank(tensor, static_cast<std::int64_t>(expected.size()), name);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (expected[index] >= 0 && tensor.size(static_cast<std::int64_t>(index)) != expected[index]) {
            throw std::invalid_argument(std::string(name) + " has an unexpected shape");
        }
    }
}

class InferenceScope final {
public:
    InferenceScope() : guard_() {}

private:
    torch::InferenceMode guard_;
};

}  // namespace rayvan::emc
