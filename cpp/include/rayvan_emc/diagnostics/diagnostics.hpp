#pragma once

#include "rayvan_emc/model.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rayvan::emc {

struct RoutingReport {
    std::vector<double> selection_frequency;
    std::vector<double> request_selection_fraction;
    std::vector<double> pre_top_k_probability;
    std::vector<double> selected_routing_weight;
    std::vector<std::vector<double>> slot_distribution;
    double normalized_entropy = 0.0;
    double effective_n1_count = 0.0;
    std::vector<bool> near_dead;
    double slot_monopoly = 0.0;
    bool partial_collapse = false;
    bool global_collapse = false;
};

struct IntegratorReport {
    std::vector<double> acceptance;
    std::vector<double> proposal_norm;
    std::vector<double> contribution;
    double mean_similarity = 0.0;
};

struct RoutingFreeReport {
    std::vector<double> activation_rate;
    std::vector<double> resonance_probability_mean;
    std::vector<double> resonance_probability_std;
    std::vector<double> resistance_mean;
    std::vector<double> compute_share;
    std::vector<double> token_proposal_norm;
    std::vector<double> raw_latent_proposal_norm;
    std::vector<double> normalized_latent_proposal_norm;
    std::vector<double> latent_attention;
    std::vector<std::vector<double>> coactivation;
    std::vector<std::vector<double>> activation_correlation;
    std::vector<double> parameter_norm;
    std::vector<double> gradient_norm;
    std::vector<double> update_norm;
    std::vector<double> basin_count;
    std::vector<std::vector<double>> basin_centers;
    std::vector<std::vector<double>> basin_radii;
    std::vector<std::vector<double>> basin_competence;
    std::vector<std::vector<double>> basin_evidence;
    std::vector<std::vector<double>> basin_uncertainty;
    std::vector<double> marginal_utility;
    std::vector<double> utility_observations;
    double activation_density = 0.0;
    double resonance_entropy = 0.0;
    double novelty_rate = 0.0;
    double low_confidence_rate = 0.0;
    double exploration_rate = 0.0;
    double training_activation_density = 0.0;
    double training_novelty_rate = 0.0;
    double training_exploration_rate = 0.0;
    double training_resonance_entropy = 0.0;
    double mean_active_experts = 0.0;
    double effective_expert_count = 0.0;
    double normalized_activation_entropy = 0.0;
    bool starvation = false;
    bool monopoly = false;
    bool all_on = false;
    bool all_off = false;
    bool proposal_scale_instability = false;
};

struct MemoryReport {
    std::uint64_t process_rss_bytes = 0;
    std::uint64_t process_peak_rss_bytes = 0;
    std::uint64_t parameter_bytes = 0;
    std::uint64_t gradient_bytes = 0;
    std::uint64_t optimizer_bytes = 0;
    std::uint64_t cuda_allocated_bytes = 0;
    std::uint64_t cuda_reserved_bytes = 0;
    std::uint64_t cuda_peak_allocated_bytes = 0;
    std::uint64_t cuda_peak_reserved_bytes = 0;
};

class DiagnosticAccumulator final {
public:
    explicit DiagnosticAccumulator(std::int64_t num_nodes);
    void update(const EMCOutput& output);
    [[nodiscard]] RoutingReport routing_report() const;
    [[nodiscard]] IntegratorReport integrator_report() const;
    [[nodiscard]] RoutingFreeReport routing_free_report() const;
    void reset();

private:
    std::int64_t num_nodes_;
    std::int64_t batches_ = 0;
    Tensor selection_count_;
    Tensor request_selection_count_;
    Tensor probability_sum_;
    Tensor selected_weight_sum_;
    Tensor selected_weight_count_;
    Tensor slot_count_;
    Tensor entropy_sum_;
    Tensor acceptance_sum_;
    Tensor proposal_norm_sum_;
    Tensor contribution_sum_;
    Tensor integrator_count_;
    Tensor similarity_sum_;
    Tensor similarity_count_;
    std::optional<RoutingFreeTrace> routing_free_trace_;
};

MemoryReport collect_memory_report(
    const EMCModel& model,
    const torch::optim::Optimizer* optimizer = nullptr,
    std::int64_t cuda_device = 0);
double global_parameter_norm(const EMCModel& model);
double global_gradient_norm(const EMCModel& model);

}  // namespace rayvan::emc
