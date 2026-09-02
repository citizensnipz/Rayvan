#pragma once

#include "rayvan_emc/n1/n1.hpp"

namespace rayvan::emc {

class GPTN1Block final : public N1Block {
public:
    explicit GPTN1Block(const ModelConfig& config);
    LeaseState begin_lease(const Tensor& shared_state) override;
    BlockOutput forward_chunk(
        const Tensor& chunk_latent,
        const Tensor& shared_state,
        const LeaseState& lease_state) override;

private:
    torch::nn::LayerNorm attention_norm{nullptr};
    torch::nn::LayerNorm shared_norm{nullptr};
    torch::nn::MultiheadAttention attention{nullptr};
    torch::nn::LayerNorm feed_forward_norm{nullptr};
    torch::nn::Sequential feed_forward;
    Tensor causal_mask;
};

}  // namespace rayvan::emc
