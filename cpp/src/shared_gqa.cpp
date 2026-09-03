#include "rayvan_emc/n1/shared_gqa.hpp"

#include <ATen/ops/_fused_sdp_choice.h>
#include <ATen/ops/scaled_dot_product_attention.h>

#include <cmath>
#include <stdexcept>

namespace rayvan::emc {

RMSNormImpl::RMSNormImpl(std::int64_t width, double epsilon)
    : epsilon_(epsilon) {
    if (width <= 0 || epsilon <= 0.0) throw std::invalid_argument("invalid RMSNorm configuration");
    weight_ = register_parameter("weight", torch::ones({width}));
}

Tensor RMSNormImpl::forward(const Tensor& input) const {
    const auto source_type = input.scalar_type();
    const auto fp32 = input.to(torch::kFloat32);
    const auto inverse_rms = torch::rsqrt(fp32.square().mean(-1, true) + epsilon_);
    return (fp32 * inverse_rms).to(source_type) * weight_.to(source_type);
}

SharedCausalGQAImpl::SharedCausalGQAImpl(const ModelConfig& config)
    : width_(config.latent_dim), query_heads_(config.gqa_query_heads),
      kv_heads_(config.gqa_kv_heads), head_dim_(config.latent_dim / config.gqa_query_heads),
      rope_base_(config.rope_base),
      input_norm(register_module("input_norm", RMSNorm(config.latent_dim))),
      query_projection(register_module("query_projection", torch::nn::Linear(config.latent_dim, config.latent_dim))),
      key_projection(register_module("key_projection", torch::nn::Linear(config.latent_dim, config.gqa_kv_heads * head_dim_))),
      value_projection(register_module("value_projection", torch::nn::Linear(config.latent_dim, config.gqa_kv_heads * head_dim_))),
      output_projection(register_module("output_projection", torch::nn::Linear(config.latent_dim, config.latent_dim))) {
    if (head_dim_ % 2 != 0) throw std::invalid_argument("RoPE requires an even GQA head dimension");
}

Tensor SharedCausalGQAImpl::apply_rope(const Tensor& input) const {
    const auto sequence = input.size(2);
    const auto half = head_dim_ / 2;
    const auto options = input.options().dtype(torch::kFloat32);
    const auto dimensions = torch::arange(half, options);
    const auto inverse_frequency = torch::exp(
        dimensions * (-2.0 * std::log(rope_base_) / static_cast<double>(head_dim_)));
    const auto positions = torch::arange(sequence, options);
    const auto angles = positions.unsqueeze(-1) * inverse_frequency.unsqueeze(0);
    const auto cosine = angles.cos().reshape({1, 1, sequence, half});
    const auto sine = angles.sin().reshape({1, 1, sequence, half});
    const auto paired = input.to(torch::kFloat32).reshape({input.size(0), input.size(1), sequence, half, 2});
    const auto even = paired.select(-1, 0);
    const auto odd = paired.select(-1, 1);
    return torch::stack({even * cosine - odd * sine, even * sine + odd * cosine}, -1)
        .reshape_as(input)
        .to(input.scalar_type());
}

std::tuple<Tensor, Tensor, Tensor> SharedCausalGQAImpl::project_qkv(const Tensor& input) {
    require_rank(input, 3, "shared GQA input");
    if (input.size(-1) != width_) throw std::invalid_argument("shared GQA width mismatch");
    const auto normalized = input_norm->forward(input);
    const auto batch = input.size(0);
    const auto sequence = input.size(1);
    auto query = query_projection->forward(normalized)
                     .reshape({batch, sequence, query_heads_, head_dim_}).transpose(1, 2);
    auto key = key_projection->forward(normalized)
                   .reshape({batch, sequence, kv_heads_, head_dim_}).transpose(1, 2);
    auto value = value_projection->forward(normalized)
                     .reshape({batch, sequence, kv_heads_, head_dim_}).transpose(1, 2);
    return {apply_rope(query), apply_rope(key), value};
}

Tensor SharedCausalGQAImpl::forward(const Tensor& input) {
    auto [query, key, value] = project_qkv(input);
    // Some LibTorch/CUDA combinations select the unfused math kernel for the
    // experimental enable_gqa path. Explicit grouped expansion preserves GQA
    // semantics and lets the mature fused SDPA dispatch run where supported.
    if (query_heads_ != kv_heads_) {
        const auto group_size = query_heads_ / kv_heads_;
        key = key.repeat_interleave(group_size, 1);
        value = value.repeat_interleave(group_size, 1);
    }
    const auto attended = at::scaled_dot_product_attention(
        query, key, value, std::nullopt, 0.0, true, std::nullopt, false);
    const auto merged = attended.transpose(1, 2).contiguous().reshape_as(input);
    return input + output_projection->forward(merged).to(input.scalar_type());
}

std::int64_t SharedCausalGQAImpl::selected_backend(const Tensor& input) {
    auto [query, key, value] = project_qkv(input);
    if (query_heads_ != kv_heads_) {
        const auto group_size = query_heads_ / kv_heads_;
        key = key.repeat_interleave(group_size, 1);
        value = value.repeat_interleave(group_size, 1);
    }
    return at::_fused_sdp_choice(
        query, key, value, std::nullopt, 0.0, true, std::nullopt, false);
}

}  // namespace rayvan::emc
