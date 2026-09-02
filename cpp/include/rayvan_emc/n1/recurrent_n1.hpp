#pragma once

#include "rayvan_emc/n1/n1.hpp"

namespace rayvan::emc {

class RecurrentN1Block final : public N1Block {
public:
    explicit RecurrentN1Block(const ModelConfig& config);
    LeaseState begin_lease(const Tensor& shared_state) override;
    BlockOutput forward_chunk(
        const Tensor& chunk_latent,
        const Tensor& shared_state,
        const LeaseState& lease_state) override;

private:
    std::int64_t width_;
    torch::nn::LayerNorm input_norm{nullptr};
    torch::nn::Linear input_adapter{nullptr};
    torch::nn::GRU recurrent{nullptr};
    torch::nn::Linear output_adapter{nullptr};
    torch::nn::Linear state_initializer{nullptr};
};

}  // namespace rayvan::emc
