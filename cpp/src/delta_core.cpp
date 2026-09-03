#include "rayvan_emc/n1/delta_core.hpp"

#include <torch/autograd.h>

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rayvan::emc {

#ifdef RAYVAN_EMC_CUDA_KERNELS
std::vector<Tensor> delta_cuda_forward(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, std::int64_t);
std::vector<Tensor> delta_cuda_backward(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const Tensor&, std::int64_t, std::int64_t);
#endif

namespace {

std::int64_t checked_bytes(std::initializer_list<std::int64_t> factors) {
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    std::int64_t bytes = 1;
    for (const auto factor : factors) {
        if (factor < 0) throw std::invalid_argument("invalid Delta allocation dimension");
        if (factor != 0 && bytes > maximum / factor) throw std::overflow_error("Delta allocation size overflow");
        bytes *= factor;
    }
    return bytes;
}

void validate_delta_inputs(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& alpha,
    const Tensor& beta,
    const Tensor& initial_state,
    std::int64_t chunk_size) {
    require_rank(query, 4, "Delta query");
    require_rank(key, 4, "Delta key");
    require_rank(value, 4, "Delta value");
    require_rank(alpha, 3, "Delta alpha");
    require_rank(beta, 3, "Delta beta");
    require_rank(initial_state, 4, "Delta initial state");
    if (query.sizes() != key.sizes() || query.sizes() != value.sizes()) {
        throw std::invalid_argument("Delta Q/K/V shapes must match");
    }
    const auto batch = query.size(0);
    const auto sequence = query.size(1);
    const auto heads = query.size(2);
    const auto dimension = query.size(3);
    if (batch <= 0 || sequence <= 0 || heads <= 0) {
        throw std::invalid_argument("Delta batch, sequence, and head dimensions must be positive");
    }
    if (alpha.sizes() != torch::IntArrayRef({batch, sequence, heads}) || beta.sizes() != alpha.sizes()) {
        throw std::invalid_argument("Delta alpha/beta shape mismatch");
    }
    if (initial_state.sizes() != torch::IntArrayRef({batch, heads, dimension, dimension})) {
        throw std::invalid_argument("Delta recurrent state shape mismatch");
    }
    if (query.device() != key.device() || query.device() != value.device() ||
        query.device() != alpha.device() || query.device() != beta.device() ||
        query.device() != initial_state.device()) {
        throw std::invalid_argument("Delta tensors must share a device");
    }
    if (query.scalar_type() != key.scalar_type() || query.scalar_type() != value.scalar_type()) {
        throw std::invalid_argument("Delta Q/K/V dtypes must match");
    }
    if (query.scalar_type() != torch::kFloat32 && query.scalar_type() != torch::kBFloat16) {
        throw std::invalid_argument("Delta Q/K/V must be FP32 or BF16");
    }
    if (alpha.scalar_type() != torch::kFloat32 || beta.scalar_type() != torch::kFloat32 ||
        initial_state.scalar_type() != torch::kFloat32) {
        throw std::invalid_argument("Delta alpha, beta, and recurrent state must be FP32");
    }
    if (dimension <= 0 || dimension > 64) throw std::invalid_argument("Delta head dimension must be in [1,64]");
    if (chunk_size != 16 && chunk_size != 32 && chunk_size != 64) {
        throw std::invalid_argument("Delta chunk size must be 16, 32, or 64");
    }
}

std::pair<Tensor, Tensor> reference_delta_rule(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& alpha,
    const Tensor& beta,
    const Tensor& initial_state) {
    auto state = initial_state;
    std::vector<Tensor> outputs;
    outputs.reserve(static_cast<std::size_t>(query.size(1)));
    for (std::int64_t token = 0; token < query.size(1); ++token) {
        const auto q = query.select(1, token).to(torch::kFloat32);
        const auto k = key.select(1, token).to(torch::kFloat32);
        const auto v = value.select(1, token).to(torch::kFloat32);
        const auto a = alpha.select(1, token).unsqueeze(-1);
        const auto b = beta.select(1, token).unsqueeze(-1);
        const auto retrieved = torch::matmul(state, k.unsqueeze(-1)).squeeze(-1);
        const auto residual = v - a * retrieved;
        state = a.unsqueeze(-1) * state + b.unsqueeze(-1) * residual.unsqueeze(-1) * k.unsqueeze(-2);
        outputs.push_back(torch::matmul(state, q.unsqueeze(-1)).squeeze(-1));
    }
    return {torch::stack(outputs, 1), state};
}

#ifdef RAYVAN_EMC_CUDA_KERNELS
class DeltaRuleAutograd final : public torch::autograd::Function<DeltaRuleAutograd> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* context,
        Tensor query,
        Tensor key,
        Tensor value,
        Tensor alpha,
        Tensor beta,
        Tensor initial_state,
        std::int64_t chunk_size,
        std::int64_t max_scratch_bytes) {
        query = query.contiguous();
        key = key.contiguous();
        value = value.contiguous();
        alpha = alpha.contiguous();
        beta = beta.contiguous();
        initial_state = initial_state.contiguous();
        const auto scratch = delta_backward_scratch_bytes(
            query.size(0), query.size(2), std::min(chunk_size, query.size(1)), query.size(3));
        const auto boundaries = delta_boundary_state_bytes(
            query.size(0), query.size(2), query.size(1), chunk_size, query.size(3));
        if (scratch > max_scratch_bytes) {
            throw std::runtime_error(
                "Delta backward scratch exceeds safety limit: " + std::to_string(scratch) +
                " bytes > " + std::to_string(max_scratch_bytes) + " bytes");
        }
        if (boundaries > max_scratch_bytes) {
            throw std::runtime_error(
                "Delta boundary states exceed safety limit: " + std::to_string(boundaries) +
                " bytes > " + std::to_string(max_scratch_bytes) + " bytes");
        }
        auto result = delta_cuda_forward(query, key, value, alpha, beta, initial_state, chunk_size);
        context->save_for_backward({query, key, value, alpha, beta, initial_state, result.at(2)});
        context->saved_data["chunk_size"] = chunk_size;
        context->saved_data["max_scratch_bytes"] = max_scratch_bytes;
        return {result.at(0), result.at(1)};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        torch::autograd::variable_list gradients) {
        const auto saved = context->get_saved_variables();
        auto grad_output = gradients.at(0).defined() ? gradients.at(0).contiguous() : torch::zeros_like(saved.at(0), torch::kFloat32);
        auto grad_final = gradients.at(1).defined() ? gradients.at(1).to(torch::kFloat32).contiguous() : torch::zeros_like(saved.at(5));
        const auto chunk_size = context->saved_data["chunk_size"].toInt();
        const auto max_scratch_bytes = context->saved_data["max_scratch_bytes"].toInt();
        auto result = delta_cuda_backward(
            saved.at(0), saved.at(1), saved.at(2), saved.at(3), saved.at(4), saved.at(5), saved.at(6),
            grad_output.to(torch::kFloat32), grad_final, chunk_size, max_scratch_bytes);
        return {result.at(0), result.at(1), result.at(2), result.at(3), result.at(4), result.at(5), Tensor(), Tensor()};
    }
};
#endif

}  // namespace

std::int64_t delta_backward_scratch_bytes(
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t chunk_size,
    std::int64_t head_dim) {
    if (batch < 0 || heads <= 0 || chunk_size <= 0 || head_dim <= 0) {
        throw std::invalid_argument("invalid Delta scratch dimensions");
    }
    if (chunk_size == std::numeric_limits<std::int64_t>::max()) {
        throw std::overflow_error("Delta scratch size overflow");
    }
    return checked_bytes({batch, heads, chunk_size + 1, head_dim, head_dim, 4});
}

std::int64_t delta_boundary_state_bytes(
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t sequence,
    std::int64_t chunk_size,
    std::int64_t head_dim) {
    if (batch < 0 || heads <= 0 || sequence <= 0 || chunk_size <= 0 || head_dim <= 0) {
        throw std::invalid_argument("invalid Delta boundary dimensions");
    }
    const auto chunks = 1 + (sequence - 1) / chunk_size;
    if (chunks == std::numeric_limits<std::int64_t>::max()) {
        throw std::overflow_error("Delta boundary size overflow");
    }
    return checked_bytes({batch, heads, chunks + 1, head_dim, head_dim, 4});
}

bool delta_cuda_kernels_available() noexcept {
#ifdef RAYVAN_EMC_CUDA_KERNELS
    return true;
#else
    return false;
#endif
}

std::pair<Tensor, Tensor> delta_rule(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& alpha,
    const Tensor& beta,
    const Tensor& initial_state,
    std::int64_t chunk_size,
    std::int64_t max_scratch_bytes) {
    validate_delta_inputs(query, key, value, alpha, beta, initial_state, chunk_size);
    const auto scratch = delta_backward_scratch_bytes(
        query.size(0), query.size(2), std::min(chunk_size, query.size(1)), query.size(3));
    const auto boundaries = delta_boundary_state_bytes(
        query.size(0), query.size(2), query.size(1), chunk_size, query.size(3));
    if (scratch > max_scratch_bytes) {
        throw std::runtime_error(
            "Delta backward scratch exceeds safety limit: " + std::to_string(scratch) +
            " bytes > " + std::to_string(max_scratch_bytes) + " bytes");
    }
    if (boundaries > max_scratch_bytes) {
        throw std::runtime_error(
            "Delta boundary states exceed safety limit: " + std::to_string(boundaries) +
            " bytes > " + std::to_string(max_scratch_bytes) + " bytes");
    }
    if (!query.is_cuda()) return reference_delta_rule(query, key, value, alpha, beta, initial_state);
#ifdef RAYVAN_EMC_CUDA_KERNELS
    auto result = DeltaRuleAutograd::apply(
        query, key, value, alpha, beta, initial_state, chunk_size, max_scratch_bytes);
    return {result.at(0), result.at(1)};
#else
    throw std::runtime_error("Delta CUDA tensors require a build with native CUDA kernels");
#endif
}

}  // namespace rayvan::emc
