#pragma once

#include "rayvan_emc/n1/n1.hpp"

namespace rayvan::emc {

class SSMN1Block final : public N1Block {
public:
    explicit SSMN1Block(const ModelConfig& config);
    LeaseState begin_lease(const Tensor& shared_state) override;
    BlockOutput forward_chunk(
        const Tensor& chunk_latent,
        const Tensor& shared_state,
        const LeaseState& lease_state) override;

private:
    Tensor parallel_diagonal_scan(
        const Tensor& log_decay,
        const Tensor& candidate,
        const Tensor& initial_state) const;

    std::int64_t width_;
    std::int64_t kernel_size_;
    torch::nn::LayerNorm input_norm{nullptr};
    torch::nn::Linear input_adapter{nullptr};
    torch::nn::Conv1d causal_convolution{nullptr};
    torch::nn::Linear delta_projection{nullptr};
    torch::nn::Linear input_projection{nullptr};
    torch::nn::Linear gate_projection{nullptr};
    Tensor log_decay;
    torch::nn::Linear output_adapter{nullptr};
    torch::nn::Linear state_initializer{nullptr};
    Tensor scan_causal_mask;
};

}  // namespace rayvan::emc
