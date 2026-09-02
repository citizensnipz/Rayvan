#pragma once

#include "rayvan_emc/config.hpp"
#include "rayvan_emc/tensor.hpp"

#include <optional>

namespace rayvan::emc {

struct RoutingDecision {
    Tensor scores;                  // [B,N]
    Tensor pre_top_k_probabilities; // [B,N]
    Tensor selected_indices;        // [B,K]
    Tensor selected_weights;        // [B,K]
    Tensor selected_slots;          // [B,K]
};

class NexusImpl final : public torch::nn::Module {
public:
    explicit NexusImpl(const ModelConfig& config);
    RoutingDecision forward(
        const Tensor& shared_latent,
        std::int64_t top_k,
        const std::optional<Tensor>& availability_mask = std::nullopt);

private:
    ModelConfig config_;
    torch::nn::LayerNorm input_norm{nullptr};
    torch::nn::Linear score_projection{nullptr};
};
TORCH_MODULE(Nexus);

}  // namespace rayvan::emc
