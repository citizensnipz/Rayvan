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

struct BasinMatch {
    Tensor basin_index;       // [B], int64
    Tensor distance;          // [B], normalized squared distance
    Tensor competence;        // [B]
    Tensor evidence;          // [B]
    Tensor uncertainty;       // [B]
    Tensor initialized;       // [B], bool
    Tensor resistance;        // [B]
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

    BasinMatch match_competence(const Tensor& need_embedding) const;
    void update_competence(
        const Tensor& need_embedding,
        const Tensor& matched_basin,
        const Tensor& utility,
        const Tensor& active,
        const Tensor& exploratory,
        const Tensor& novel);
    std::shared_ptr<N1PersistentState> initialize_state(const Tensor& shared_latent);
    CollectiveExpertOutput forward_routing_item(
        const RoutingItem& item,
        const Tensor& active_request_indices,
        const std::shared_ptr<N1PersistentState>& state);

    [[nodiscard]] const Tensor& basin_centers() const noexcept { return basin_centers_; }
    [[nodiscard]] const Tensor& basin_radii() const noexcept { return basin_radii_; }
    [[nodiscard]] const Tensor& basin_competence() const noexcept { return basin_competence_; }
    [[nodiscard]] const Tensor& basin_evidence() const noexcept { return basin_evidence_; }
    [[nodiscard]] const Tensor& basin_uncertainty() const noexcept { return basin_uncertainty_; }
    [[nodiscard]] const Tensor& basin_initialized() const noexcept { return basin_initialized_; }
    [[nodiscard]] const Tensor& marginal_utility() const noexcept { return marginal_utility_; }
    [[nodiscard]] const Tensor& utility_observations() const noexcept { return utility_observations_; }
    [[nodiscard]] const std::shared_ptr<N1Node>& body_module() const noexcept { return body_; }
    [[nodiscard]] N1Family family() const noexcept { return family_; }
    [[nodiscard]] std::int64_t parameter_count() const;

private:
    ModelConfig config_;
    N1Family family_;
    std::shared_ptr<N1Node> body_;
    torch::nn::Linear latent_proposal_projection{nullptr};
    Tensor basin_centers_;
    Tensor basin_radii_;
    Tensor basin_competence_;
    Tensor basin_evidence_;
    Tensor basin_utility_variance_;
    Tensor basin_uncertainty_;
    Tensor basin_initialized_;
    Tensor marginal_utility_;
    Tensor utility_observations_;
    Tensor compute_cost_;
};

struct RoutingFreeCollectiveState {
    Tensor shared_latent; // [B,M,D]
    std::vector<std::shared_ptr<N1PersistentState>> expert_states;
};

struct RoutingFreeTrace {
    Tensor need_embedding;                   // [B,Q,Z]
    Tensor matched_basin;                    // [B,Q,N]
    Tensor basin_distance;                   // [B,Q,N]
    Tensor resistance;                       // [B,Q,N]
    Tensor resonance_probability;            // [B,Q,N]
    Tensor activation_mask;                  // [B,Q,N]
    Tensor novelty_mask;                     // [B,Q]
    Tensor low_confidence_mask;               // [B,Q]
    Tensor exploration_mask;                 // [B,Q,N]
    Tensor raw_token_proposal_norm;           // [B,Q,N]
    Tensor raw_latent_proposal_norm;          // [B,Q,N]
    Tensor normalized_latent_proposal_norm;   // [B,Q,N]
    Tensor latent_attention_weights;          // [B,Q,M,N]
    Tensor latent_norm;                       // [B,Q]
    Tensor activation_density;                // scalar
    Tensor resonance_entropy;                 // scalar mean Bernoulli entropy
    Tensor coactivation_matrix;               // [N,N]
    Tensor activation_correlation;            // [N,N]
    Tensor compute_share;                     // [N]
    Tensor basin_centers;                     // [N,K,Z]
    Tensor basin_radii;                       // [N,K]
    Tensor basin_competence;                  // [N,K]
    Tensor basin_evidence;                    // [N,K]
    Tensor basin_uncertainty;                 // [N,K]
    Tensor basin_initialized;                 // [N,K]
    Tensor marginal_utility;                  // [N]
    Tensor utility_observations;              // [N]
    Tensor training_activation_density;       // scalar, cumulative
    Tensor training_novelty_rate;             // scalar, cumulative
    Tensor training_exploration_rate;         // scalar, cumulative
    Tensor training_resonance_entropy;        // scalar, cumulative
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
    // Consumes gradients retained from the ordinary language-model backward
    // pass and updates non-gradient competence memory.
    void update_competence_from_backward();
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

    struct PendingCompetenceObservation {
        Tensor need_embedding;
        Tensor matched_basin;
        Tensor resonance_probability;
        Tensor activation_mask;
        Tensor exploration_mask;
        Tensor novelty_mask;
        Tensor token_proposals;
        Tensor output_state;
    };

    Tensor computational_need(const RoutingItem& item);
    LatentIntegration integrate_latent(
        const Tensor& latent,
        const Tensor& proposals,
        const Tensor& active_mask);

    ModelConfig config_;
    SharedCausalGQA shared_attention_{nullptr};
    RMSNorm routing_latent_norm{nullptr};
    torch::nn::Linear need_projection_in{nullptr};
    torch::nn::Linear need_projection_out{nullptr};
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
    std::vector<PendingCompetenceObservation> pending_competence_;
    Tensor training_routing_items_;
    Tensor training_activation_count_;
    Tensor training_novelty_count_;
    Tensor training_exploration_count_;
    Tensor training_entropy_sum_;
};
TORCH_MODULE(RoutingFreeCollective);

}  // namespace rayvan::emc
