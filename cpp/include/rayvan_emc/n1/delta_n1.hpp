#pragma once

#include "rayvan_emc/n1/n1.hpp"

namespace rayvan::emc {

class DeltaN1Block final : public N1Block {
public:
    explicit DeltaN1Block(const ModelConfig& config);
    LeaseState begin_lease(const Tensor& shared_state) override;
    BlockOutput forward_chunk(
        const Tensor& chunk_latent,
        const Tensor& shared_state,
        const LeaseState& lease_state) override;

private:
    std::int64_t width_;
    std::int64_t heads_;
    std::int64_t head_dim_;
    torch::nn::LayerNorm input_norm{nullptr};
    torch::nn::Linear query_projection{nullptr};
    torch::nn::Linear key_projection{nullptr};
    torch::nn::Linear value_projection{nullptr};
    torch::nn::Linear alpha_projection{nullptr};
    torch::nn::Linear beta_projection{nullptr};
    torch::nn::Linear output_gate{nullptr};
    torch::nn::Linear output_adapter{nullptr};
    torch::nn::LayerNorm post_norm{nullptr};
    torch::nn::Sequential post_ffn;
    torch::nn::Linear initial_key{nullptr};
    torch::nn::Linear initial_value{nullptr};
};

}  // namespace rayvan::emc
