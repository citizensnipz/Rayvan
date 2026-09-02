#pragma once

#include "rayvan_emc/config.hpp"
#include "rayvan_emc/tensor.hpp"

#include <optional>

namespace rayvan::emc {

struct IntegratorTrace {
    Tensor proposal_acceptance;    // [B,S,K]
    Tensor proposal_norms;         // [B,S,K]
    Tensor proposal_similarity;    // [B,S,K,K]
    Tensor proposal_contributions; // [B,S,K]
    Tensor integrated_update_norm; // [B,S]
    Tensor gate_magnitude;          // [B,S]
};

struct IntegratorOutput {
    Tensor latent; // [B,S,D]
    std::optional<IntegratorTrace> trace;
};

class IntegratorImpl final : public torch::nn::Module {
public:
    explicit IntegratorImpl(const ModelConfig& config);
    IntegratorOutput forward(
        const Tensor& latent,
        const Tensor& proposals,
        const Tensor& selected_weights,
        bool return_diagnostics = false);

private:
    std::int64_t latent_dim_;
    std::int64_t num_heads_;
    std::int64_t head_dim_;
    torch::nn::LayerNorm latent_norm{nullptr};
    torch::nn::LayerNorm proposal_norm{nullptr};
    torch::nn::Linear query_projection{nullptr};
    torch::nn::Linear key_projection{nullptr};
    torch::nn::Linear value_projection{nullptr};
    torch::nn::Linear attention_output{nullptr};
    Tensor routing_prior_scale;
    torch::nn::Linear update_projection{nullptr};
    torch::nn::Linear gate_projection{nullptr};
};
TORCH_MODULE(Integrator);

class N2IntegratorImpl final : public torch::nn::Module {
public:
    explicit N2IntegratorImpl(const ModelConfig& config);
    IntegratorOutput forward(
        const Tensor& latent,
        const Tensor& proposals,
        const Tensor& selected_weights,
        bool return_diagnostics = false);

private:
    Integrator proposal_integrator{nullptr};
};
TORCH_MODULE(N2Integrator);

}  // namespace rayvan::emc
