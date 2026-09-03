#include "rayvan_emc/n1/delta_n1.hpp"

#include "rayvan_emc/n1/delta_core.hpp"

#include <torch/nn/functional/normalization.h>

#include <cmath>
#include <stdexcept>

namespace rayvan::emc {

DeltaN1Block::DeltaN1Block(const ModelConfig& config)
    : N1Block(config, N1Family::delta),
      width_(config.resolved_delta_internal_dim()),
      heads_(config.delta_heads),
      head_dim_(width_ / heads_),
      input_norm(register_module("input_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.latent_dim})))),
      query_projection(register_module("query_projection", torch::nn::Linear(config.latent_dim, width_))),
      key_projection(register_module("key_projection", torch::nn::Linear(config.latent_dim, width_))),
      value_projection(register_module("value_projection", torch::nn::Linear(config.latent_dim, width_))),
      alpha_projection(register_module("alpha_projection", torch::nn::Linear(config.latent_dim, heads_))),
      beta_projection(register_module("beta_projection", torch::nn::Linear(config.latent_dim, heads_))),
      output_gate(register_module("output_gate", torch::nn::Linear(config.latent_dim, width_))),
      output_adapter(register_module("output_adapter", torch::nn::Linear(width_, config.latent_dim))),
      post_norm(register_module("post_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.latent_dim})))),
      post_ffn(register_module(
          "post_ffn",
          torch::nn::Sequential(
              torch::nn::Linear(config.latent_dim, config.resolved_delta_ffn_dim()),
              torch::nn::GELU(),
              torch::nn::Linear(config.resolved_delta_ffn_dim(), config.latent_dim)))),
      initial_key(register_module("initial_key", torch::nn::Linear(config.latent_dim, width_))),
      initial_value(register_module("initial_value", torch::nn::Linear(config.latent_dim, width_))) {}

LeaseState DeltaN1Block::begin_lease(const Tensor& shared_state) {
    const auto summary = shared_state.mean(1);
    const auto key = initial_key->forward(summary).reshape({summary.size(0), heads_, head_dim_});
    const auto value = initial_value->forward(summary).reshape({summary.size(0), heads_, head_dim_});
    auto memory = torch::einsum("bhv,bhk->bhvk", {value.to(torch::kFloat32), key.to(torch::kFloat32)});
    memory = memory / std::sqrt(static_cast<double>(head_dim_));
    return {{{"memory", std::move(memory)}}};
}

BlockOutput DeltaN1Block::forward_chunk(
    const Tensor& chunk_latent,
    const Tensor& shared_state,
    const LeaseState& lease_state) {
    const auto conditioned = condition_chunk(chunk_latent, shared_state);
    const auto normalized = input_norm->forward(conditioned);
    const std::vector<std::int64_t> shape{conditioned.size(0), conditioned.size(1), heads_, head_dim_};
    const auto normalize_options = torch::nn::functional::NormalizeFuncOptions().dim(-1);
    const auto projected_query = query_projection->forward(normalized).reshape(shape);
    const auto recurrence_dtype = projected_query.scalar_type();
    const auto query = torch::nn::functional::normalize(projected_query, normalize_options).to(recurrence_dtype);
    const auto key = torch::nn::functional::normalize(
        key_projection->forward(normalized).reshape(shape), normalize_options).to(recurrence_dtype);
    const auto value = torch::tanh(value_projection->forward(normalized).reshape(shape)).to(recurrence_dtype);
    const auto alpha = torch::sigmoid(alpha_projection->forward(normalized)).to(torch::kFloat32);
    const auto beta = torch::sigmoid(beta_projection->forward(normalized)).to(torch::kFloat32);
    const auto initial = lease_state.tensors.at("memory").to(torch::kFloat32);
    auto [delta_output, final_state] = delta_rule(
        query, key, value, alpha, beta, initial, config_.chunk_size, config_.delta_max_scratch_bytes);
    auto gated = delta_output.reshape({conditioned.size(0), conditioned.size(1), width_}) *
        torch::sigmoid(output_gate->forward(normalized)).to(torch::kFloat32);
    auto proposal = output_adapter->forward(gated.to(conditioned.scalar_type()));
    proposal = proposal + post_ffn->forward(post_norm->forward(proposal));
    proposal = proposal.to(chunk_latent.scalar_type());
    return {
        proposal,
        state_proposal(proposal, shared_state),
        {{{"memory", std::move(final_state)}}}};
}

}  // namespace rayvan::emc
