#pragma once

#include "rayvan_emc/tensor.hpp"

#include <cstdint>
#include <utility>

namespace rayvan::emc {

// Exact EMC gated-delta recurrence. Q/K are expected to be L2 normalized,
// alpha/beta are FP32 sigmoid outputs, and recurrent state is always FP32.
std::pair<Tensor, Tensor> delta_rule(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& alpha,
    const Tensor& beta,
    const Tensor& initial_state,
    std::int64_t chunk_size,
    std::int64_t max_scratch_bytes);

std::int64_t delta_backward_scratch_bytes(
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t chunk_size,
    std::int64_t head_dim);

std::int64_t delta_boundary_state_bytes(
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t sequence,
    std::int64_t chunk_size,
    std::int64_t head_dim);

bool delta_cuda_kernels_available() noexcept;

}  // namespace rayvan::emc
