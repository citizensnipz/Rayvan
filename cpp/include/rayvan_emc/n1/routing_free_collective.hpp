#pragma once

#include "rayvan_emc/config.hpp"
#include "rayvan_emc/n1/n1.hpp"
#include "rayvan_emc/n1/shared_gqa.hpp"
#include "rayvan_emc/tensor.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace rayvan::emc {

struct RoutingItem {
    Tensor contextual_tokens; // [B,C,D]
    Tensor latent_state;      // [B,M,D]
    std::int64_t chunk_index = 0;
};

struct ExpertActivation {
    Tensor internal_response; // [B,R]
    Tensor response_norm;     // [B]
    Tensor strength;          // [B]
    Tensor active;            // [B], bool
};

struct CollectiveExpertOutput {
    Tensor token_proposal;  // [A,C,D]
    Tensor latent_proposal; // [A,M,D]
    std::shared_ptr<N1PersistentState> full_state;
};

class CollectiveExpertImpl final : public torch::nn::Module {
public:
    CollectiveExpertImpl(
        const ModelConfig& config,
        N1Family family,
        std::int64_t expert_id,
        std::string name);

    ExpertActivation activation(const Tensor& routing_representation, double theta);
    std::shared_ptr<N1PersistentState> initialize_state(const Tensor& shared_latent);
    CollectiveExpertOutput forward_routing_item(
        const RoutingItem& item,
        const Tensor& active_request_indices,
        const std::shared_ptr<N1PersistentState>& state);

    [[nodiscard]] const Tensor& activation_bias() const noexcept { return activation_bias_; }
    [[nodiscard]] const std::shared_ptr<N1Node>& body_module() const noexcept { return body_; }
    [[nodiscard]] N1Family family() const noexcept { return family_; }
    [[nodiscard]] std::int64_t parameter_count() const;

private:
    ModelConfig config_;
    N1Family family_;
    std::shared_ptr<N1Node> body_;
    torch::nn::Linear activation_projection{nullptr};
    Tensor activation_bias_;
    torch::nn::Linear latent_proposal_projection{nullptr};
};

struct RoutingFreeCollectiveState {
    Tensor shared_latent; // [B,M,D]
    std::vector<std::shared_ptr<N1PersistentState>> expert_states;
};

struct RoutingFreeTrace {
    Tensor activation_response;              // [B,Q,N]
    Tensor activation_strength;              // [B,Q,N]
    Tensor activation_mask;                  // [B,Q,N]
    Tensor all_off_recovery;                  // [B,Q]
    Tensor raw_token_proposal_norm;           // [B,Q,N]
    Tensor raw_latent_proposal_norm;          // [B,Q,N]
    Tensor normalized_latent_proposal_norm;   // [B,Q,N]
    Tensor latent_attention_weights;          // [B,Q,M,N]
    Tensor latent_norm;                       // [B,Q]
    Tensor expert_biases;                     // [N]
    Tensor activation_density;                // scalar
    Tensor target_density;                    // scalar
    Tensor adaptive_lambda;                   // scalar used for this forward
    Tensor expert_balancing_loss;             // scalar, unweighted
    Tensor routing_item_balancing_loss;       // scalar, unweighted
    Tensor balancing_loss;                    // scalar, unweighted
    Tensor coactivation_matrix;               // [N,N]
    Tensor activation_correlation;            // [N,N]
    Tensor compute_share;                     // [N]
};

struct RoutingFreeCollectiveOutput {
    Tensor contextual_state; // [B,S,D], after shared GQA
    Tensor token_state;      // [B,S,D], integrated N1 output
    Tensor auxiliary_loss;   // scalar
    std::shared_ptr<RoutingFreeCollectiveState> state;
    std::optional<RoutingFreeTrace> trace;
};

class RoutingFreeCollectiveImpl final : public torch::nn::Module {
public:
    explicit RoutingFreeCollectiveImpl(const ModelConfig& config);
    RoutingFreeCollectiveOutput forward(
        const Tensor& embeddings,
        const std::shared_ptr<RoutingFreeCollectiveState>& state = nullptr,
        const std::optional<Tensor>& availability_mask = std::nullopt,
        const std::optional<Tensor>& force_active_mask = std::nullopt,
        const std::optional<Tensor>& zero_proposal_mask = std::nullopt,
        bool return_trace = false);

    [[nodiscard]] const std::vector<std::shared_ptr<CollectiveExpertImpl>>& experts() const noexcept {
        return expert_handles_;
    }
    [[nodiscard]] SharedCausalGQA& shared_attention() noexcept { return shared_attention_; }
    [[nodiscard]] const SharedCausalGQA& shared_attention() const noexcept { return shared_attention_; }
    [[nodiscard]] const Tensor& adaptive_lambda() const noexcept { return adaptive_lambda_; }
    // Profiling-only entry point used by the native CUDA benchmark.
    Tensor diagnostic_integrate_latent(
        const Tensor& latent,
        const Tensor& proposals,
        const Tensor& active_mask);

private:
    struct LatentIntegration {
        Tensor latent;
        Tensor attention;
        Tensor normalized_proposal_norm;
    };

    Tensor routing_representation(const RoutingItem& item);
    LatentIntegration integrate_latent(
        const Tensor& latent,
        const Tensor& proposals,
        const Tensor& active_mask);

    ModelConfig config_;
    SharedCausalGQA shared_attention_{nullptr};
    RMSNorm routing_context_norm{nullptr};
    RMSNorm routing_latent_norm{nullptr};
    torch::nn::Linear latent_initializer{nullptr};
    torch::nn::ModuleList experts_;
    std::vector<std::shared_ptr<CollectiveExpertImpl>> expert_handles_;
    RMSNorm latent_query_norm{nullptr};
    RMSNorm latent_proposal_norm{nullptr};
    torch::nn::Linear latent_query_projection{nullptr};
    torch::nn::Linear latent_key_projection{nullptr};
    torch::nn::Linear latent_value_projection{nullptr};
    torch::nn::Linear latent_output_projection{nullptr};
    torch::nn::Linear latent_gate_projection{nullptr};
    Tensor adaptive_lambda_;
};
TORCH_MODULE(RoutingFreeCollective);

}  // namespace rayvan::emc
