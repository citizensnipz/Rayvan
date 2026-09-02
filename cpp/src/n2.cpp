#include "rayvan_emc/n2/dispatch.hpp"
#include "rayvan_emc/n2/integrator.hpp"
#include "rayvan_emc/n2/nexus.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace rayvan::emc {

DispatchPlan DispatchPlan::from_routing(const Tensor& selected_indices, std::int64_t num_experts) {
    require_rank(selected_indices, 2, "selected_indices");
    const auto batch = selected_indices.size(0);
    const auto selected = selected_indices.size(1);
    const auto assignments = batch * selected;
    auto expert_ids = selected_indices.reshape({-1});
    auto source_indices = torch::arange(batch, selected_indices.options())
                              .unsqueeze(1).expand({-1, selected}).reshape({-1});
    auto slot_indices = torch::arange(selected, selected_indices.options())
                            .unsqueeze(0).expand({batch, -1}).reshape({-1});
    // Composite keys give deterministic expert grouping and preserve source/slot order.
    auto permutation = (expert_ids * assignments + torch::arange(assignments, selected_indices.options())).argsort();
    auto assignment_indices = torch::arange(assignments, selected_indices.options());
    auto inverse = torch::empty_like(permutation).scatter(0, permutation, assignment_indices);
    auto counts = torch::bincount(expert_ids, {}, num_experts);
    auto offsets = torch::cat({counts.new_zeros({1}), counts.cumsum(0)});
    return {
        expert_ids,
        source_indices,
        slot_indices,
        permutation,
        inverse,
        expert_ids.index_select(0, permutation),
        source_indices.index_select(0, permutation),
        slot_indices.index_select(0, permutation),
        counts,
        offsets};
}

Tensor DispatchPlan::restore(
    const Tensor& grouped_proposals,
    std::int64_t batch,
    std::int64_t sequence,
    std::int64_t latent) const {
    const auto selected = slot_indices.numel() / batch;
    return grouped_proposals.index_select(0, inverse_permutation)
        .reshape({batch, selected, sequence, latent})
        .permute({0, 2, 1, 3});
}

NexusImpl::NexusImpl(const ModelConfig& config)
    : config_(config),
      input_norm(register_module("input_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.latent_dim})))),
      score_projection(register_module(
          "score_projection",
          torch::nn::Linear(config.latent_dim, static_cast<std::int64_t>(config.population.size())))) {}

RoutingDecision NexusImpl::forward(
    const Tensor& shared_latent,
    std::int64_t top_k,
    const std::optional<Tensor>& availability_mask) {
    require_rank(shared_latent, 3, "shared_latent");
    if (top_k <= 0 || top_k > static_cast<std::int64_t>(config_.population.size())) {
        throw std::invalid_argument("invalid Nexus top-K");
    }
    auto scores = score_projection->forward(input_norm->forward(shared_latent).mean(1));
    if (availability_mask) {
        require_rank(*availability_mask, 1, "availability_mask");
        if (availability_mask->numel() != scores.size(-1)) throw std::invalid_argument("availability mask size mismatch");
        const auto available = availability_mask->to(scores.device(), torch::kBool);
        if (available.sum().item<std::int64_t>() < top_k) throw std::invalid_argument("availability mask leaves fewer N1 nodes than top-K");
        scores = scores.masked_fill(~available, -std::numeric_limits<float>::infinity());
    }
    auto [selected_scores, selected_indices] = torch::topk(scores, top_k, -1);
    auto selected_weights = torch::softmax(selected_scores, -1);
    auto slots = torch::arange(top_k, selected_indices.options()).expand_as(selected_indices);
    return {scores, torch::softmax(scores, -1), selected_indices, selected_weights, slots};
}

IntegratorImpl::IntegratorImpl(const ModelConfig& config)
    : latent_dim_(config.latent_dim), num_heads_(config.integrator_heads),
      head_dim_(config.latent_dim / config.integrator_heads),
      latent_norm(register_module("latent_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.latent_dim})))),
      proposal_norm(register_module("proposal_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.latent_dim})))),
      query_projection(register_module("query_projection", torch::nn::Linear(config.latent_dim, config.latent_dim))),
      key_projection(register_module("key_projection", torch::nn::Linear(config.latent_dim, config.latent_dim))),
      value_projection(register_module("value_projection", torch::nn::Linear(config.latent_dim, config.latent_dim))),
      attention_output(register_module("attention_output", torch::nn::Linear(config.latent_dim, config.latent_dim))),
      update_projection(register_module("update_projection", torch::nn::Linear(config.latent_dim * 4, config.latent_dim))),
      gate_projection(register_module("gate_projection", torch::nn::Linear(config.latent_dim * 4, config.latent_dim))) {
    routing_prior_scale = register_parameter("routing_prior_scale", torch::tensor(0.5f));
}

IntegratorOutput IntegratorImpl::forward(
    const Tensor& latent,
    const Tensor& proposals,
    const Tensor& selected_weights,
    bool return_diagnostics) {
    require_rank(latent, 3, "latent");
    require_rank(proposals, 4, "proposals");
    require_rank(selected_weights, 2, "selected_weights");
    const auto batch = proposals.size(0);
    const auto sequence = proposals.size(1);
    const auto selected = proposals.size(2);
    if (proposals.size(3) != latent_dim_ || latent.sizes() != torch::IntArrayRef({batch, sequence, latent_dim_}) ||
        selected_weights.sizes() != torch::IntArrayRef({batch, selected})) {
        throw std::invalid_argument("Integrator shape mismatch");
    }

    const auto normalized_latent = latent_norm->forward(latent);
    const auto normalized_proposals = proposal_norm->forward(proposals);
    const auto query = query_projection->forward(normalized_latent).reshape({batch, sequence, num_heads_, head_dim_});
    const auto keys = key_projection->forward(normalized_proposals)
                          .reshape({batch, sequence, selected, num_heads_, head_dim_})
                          .permute({0, 1, 3, 2, 4});
    const auto values = value_projection->forward(normalized_proposals)
                            .reshape({batch, sequence, selected, num_heads_, head_dim_})
                            .permute({0, 1, 3, 2, 4});
    auto scores = torch::matmul(query.unsqueeze(-2), keys.transpose(-2, -1)).squeeze(-2);
    scores = scores / std::sqrt(static_cast<double>(head_dim_));
    const auto routing_weights = selected_weights.unsqueeze(1).expand({-1, sequence, -1});
    const auto routing_prior = routing_weights.clamp_min(1e-9).log().unsqueeze(2);
    scores = scores + routing_prior_scale * routing_prior;
    const auto head_acceptance = torch::softmax(scores, -1);
    const auto attended_heads = torch::matmul(head_acceptance.unsqueeze(-2), values).squeeze(-2);
    const auto attended = attention_output->forward(attended_heads.reshape({batch, sequence, latent_dim_}));

    const std::vector<std::int64_t> variance_dimension{2};
    auto [proposal_variance, proposal_mean] = torch::var_mean(proposals, variance_dimension, false, false);
    const auto integration_input = torch::cat(
        {normalized_latent, attended, proposal_mean, proposal_variance}, -1);
    const auto candidate_update = update_projection->forward(integration_input);
    const auto update_gate = torch::sigmoid(gate_projection->forward(integration_input));
    const auto next_latent = latent + update_gate * candidate_update;
    if (!return_diagnostics) return {next_latent, std::nullopt};

    const auto acceptance = head_acceptance.mean(2);
    const auto per_proposal_values = (head_acceptance.unsqueeze(-1) * values).permute({0, 1, 3, 2, 4});
    const auto contributions = per_proposal_values.flatten(3).norm(2, -1);
    const auto normalized = torch::nn::functional::normalize(
        proposals,
        torch::nn::functional::NormalizeFuncOptions().dim(-1));
    IntegratorTrace trace{
        acceptance.detach(),
        proposals.norm(2, -1).detach(),
        torch::matmul(normalized, normalized.transpose(-2, -1)).detach(),
        contributions.detach(),
        candidate_update.norm(2, -1).detach(),
        update_gate.abs().mean(-1).detach()};
    return {next_latent, std::move(trace)};
}

N2IntegratorImpl::N2IntegratorImpl(const ModelConfig& config)
    : proposal_integrator(register_module("proposal_integrator", Integrator(config))) {}

IntegratorOutput N2IntegratorImpl::forward(
    const Tensor& latent,
    const Tensor& proposals,
    const Tensor& selected_weights,
    bool return_diagnostics) {
    return proposal_integrator->forward(
        latent, proposals, selected_weights, return_diagnostics);
}

}  // namespace rayvan::emc
