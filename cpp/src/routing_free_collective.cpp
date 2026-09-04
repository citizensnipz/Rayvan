#include "rayvan_emc/n1/routing_free_collective.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace rayvan::emc {
namespace {

constexpr double kDistanceEpsilon = 1e-6;
constexpr double kUnavailableResistance = 1e6;

Tensor normalized_rows(const Tensor& value) {
    const auto fp32 = value.to(torch::kFloat32);
    return fp32 / fp32.norm(2, -1, true).clamp_min(1e-8);
}

}  // namespace

CollectiveExpertImpl::CollectiveExpertImpl(
    const ModelConfig& config,
    N1Family family,
    std::int64_t expert_id,
    std::string name)
    : config_(config), family_(family),
      body_(register_module("body", create_n1_node(config, family, expert_id, std::move(name)))),
      latent_proposal_projection(register_module(
          "latent_proposal_projection",
          torch::nn::Linear(config.latent_dim * 2, config.latent_dim))) {
    const auto options = torch::TensorOptions().dtype(torch::kFloat32);
    const auto basins = config.competence_basin_count;
    basin_centers_ = register_buffer(
        "basin_centers", torch::zeros({basins, config.competence_embedding_dim}, options));
    basin_radii_ = register_buffer(
        "basin_radii", torch::full({basins}, config.competence_radius_init, options));
    basin_competence_ = register_buffer("basin_competence", torch::zeros({basins}, options));
    basin_evidence_ = register_buffer("basin_evidence", torch::zeros({basins}, options));
    basin_utility_variance_ = register_buffer("basin_utility_variance", torch::zeros({basins}, options));
    basin_uncertainty_ = register_buffer(
        "basin_uncertainty", torch::full({basins}, config.competence_sigma_initial, options));
    basin_initialized_ = register_buffer(
        "basin_initialized", torch::zeros({basins}, torch::TensorOptions().dtype(torch::kBool)));
    marginal_utility_ = register_buffer("marginal_utility", torch::zeros({}, options));
    utility_observations_ = register_buffer("utility_observations", torch::zeros({}, options));
    compute_cost_ = register_buffer(
        "compute_cost", torch::tensor(static_cast<double>(parameter_count()) / 10'000'000.0, options));
}

BasinMatch CollectiveExpertImpl::match_competence(const Tensor& need_embedding) const {
    require_rank(need_embedding, 2, "computational-need embedding");
    if (need_embedding.size(1) != config_.competence_embedding_dim) {
        throw std::invalid_argument("computational-need embedding width mismatch");
    }
    const auto delta = need_embedding.to(torch::kFloat32).unsqueeze(1) - basin_centers_.unsqueeze(0);
    const auto distance = delta.square().sum(-1) /
        (basin_radii_.square().unsqueeze(0) + kDistanceEpsilon);
    const auto valid = basin_initialized_.unsqueeze(0).expand_as(distance);
    const auto selectable = torch::where(
        valid, distance, torch::full_like(distance, kUnavailableResistance));
    auto nearest = selectable.min(1);
    const auto nearest_distance = std::get<0>(nearest);
    const auto index = std::get<1>(nearest);
    const auto competence = basin_competence_.gather(0, index);
    const auto evidence = basin_evidence_.gather(0, index);
    const auto uncertainty = basin_uncertainty_.gather(0, index);
    const auto initialized = basin_initialized_.gather(0, index);
    const auto resistance = nearest_distance - config_.competence_lambda_q * competence +
        config_.competence_compute_cost_weight * compute_cost_;
    return {index, nearest_distance, competence, evidence, uncertainty, initialized, resistance};
}

void CollectiveExpertImpl::update_competence(
    const Tensor& need_embedding,
    const Tensor& matched_basin,
    const Tensor& utility_input,
    const Tensor& active,
    const Tensor& exploratory,
    const Tensor& novel) {
    torch::NoGradGuard no_grad;
    const auto z = need_embedding.detach().to(torch::kFloat32);
    const auto utility = utility_input.detach().to(torch::kFloat32).clamp(
        -config_.competence_utility_clip, config_.competence_utility_clip);
    const auto active_mask = active.to(torch::kBool);
    const auto creation_candidate = active_mask & exploratory.to(torch::kBool) &
        novel.to(torch::kBool) & (utility > 0.0);

    const auto active_count = active_mask.to(torch::kFloat32).sum();
    const auto active_utility = (utility * active_mask.to(torch::kFloat32)).sum() /
        active_count.clamp_min(1.0);
    const auto utility_decay = torch::pow(
        torch::full_like(active_count, 1.0 - config_.competence_alpha_q), active_count);
    marginal_utility_.copy_(torch::where(
        active_count > 0.0,
        utility_decay * marginal_utility_ + (1.0 - utility_decay) * active_utility,
        marginal_utility_));
    utility_observations_.add_(active_count);

    for (std::int64_t basin = 0; basin < config_.competence_basin_count; ++basin) {
        const auto initialized = basin_initialized_.select(0, basin);
        const auto mask = active_mask & (matched_basin == basin) & initialized & ~creation_candidate;
        const auto mask_float = mask.to(torch::kFloat32);
        const auto count = mask_float.sum();
        const auto has_observation = count > 0.0;
        const auto old_q = basin_competence_.select(0, basin);
        const auto old_n = basin_evidence_.select(0, basin);
        const auto old_mu = basin_centers_.select(0, basin);
        const auto old_radius = basin_radii_.select(0, basin);
        const auto old_variance = basin_utility_variance_.select(0, basin);
        const auto mean_utility = (utility * mask_float).sum() / count.clamp_min(1.0);
        const auto q_decay = torch::pow(
            torch::full_like(count, 1.0 - config_.competence_alpha_q), count);
        const auto new_q = q_decay * old_q + (1.0 - q_decay) * mean_utility;
        const auto new_n = old_n + count;

        const auto positive = utility.clamp_min(0.0) * mask_float;
        const auto center_delta = (positive.unsqueeze(1) * (z - old_mu.unsqueeze(0))).sum(0);
        const auto moved_mu = normalized_rows(
            (old_mu + config_.competence_eta_mu * center_delta).unsqueeze(0)).squeeze(0);
        const auto euclidean_distance = (z - old_mu.unsqueeze(0)).square().sum(-1).sqrt();
        const auto radius_delta = (positive * (euclidean_distance - old_radius)).sum();
        const auto new_radius = (old_radius + config_.competence_eta_r * radius_delta).clamp(
            config_.competence_radius_min, config_.competence_radius_max);

        const auto squared_residual = (utility - old_q).square();
        const auto mean_squared_residual = (squared_residual * mask_float).sum() /
            count.clamp_min(1.0);
        const auto variance_decay = torch::pow(
            torch::full_like(count, 1.0 - config_.competence_variance_alpha), count);
        const auto new_variance = variance_decay * old_variance +
            (1.0 - variance_decay) * mean_squared_residual;
        const auto prior_uncertainty =
            config_.competence_sigma_initial * config_.competence_sigma_initial *
            static_cast<double>(config_.competence_min_evidence) /
            (new_n + static_cast<double>(config_.competence_min_evidence));
        const auto new_uncertainty = (new_variance + prior_uncertainty).sqrt().clamp_min(
            config_.competence_sigma_floor);

        basin_competence_.select(0, basin).copy_(torch::where(has_observation, new_q, old_q));
        basin_evidence_.select(0, basin).copy_(torch::where(has_observation, new_n, old_n));
        basin_centers_.select(0, basin).copy_(torch::where(
            has_observation.expand_as(old_mu), moved_mu, old_mu));
        basin_radii_.select(0, basin).copy_(torch::where(has_observation, new_radius, old_radius));
        basin_utility_variance_.select(0, basin).copy_(torch::where(
            has_observation, new_variance, old_variance));
        basin_uncertainty_.select(0, basin).copy_(torch::where(
            has_observation, new_uncertainty, basin_uncertainty_.select(0, basin)));
    }

    // Commit at most one new basin per expert/cycle so one minibatch cannot
    // fill every slot with near-identical centers.
    const auto candidate_score = torch::where(
        creation_candidate, utility,
        torch::full_like(utility, -std::numeric_limits<float>::infinity()));
    const auto best_observation = candidate_score.argmax();
    const auto has_candidate = creation_candidate.any();
    const auto unused = ~basin_initialized_;
    const auto weak = (basin_evidence_ < config_.competence_replacement_max_evidence) |
        ((basin_competence_ <= config_.competence_replacement_max_q) &
         (basin_uncertainty_ >= config_.competence_confidence_sigma));
    const auto replaceable = unused | weak;
    const auto slot_order = torch::arange(
        config_.competence_basin_count, basin_evidence_.options());
    const auto replacement_score = torch::where(
        unused, -1e6 + slot_order,
        basin_evidence_ + 16.0 * basin_competence_.clamp_min(0.0) - basin_uncertainty_);
    const auto selectable_score = torch::where(
        replaceable, replacement_score,
        torch::full_like(replacement_score, std::numeric_limits<float>::infinity()));
    const auto replacement_basin = selectable_score.argmin();
    const auto can_create = has_candidate & replaceable.any();
    const auto replacement_mask =
        (slot_order.to(torch::kLong) == replacement_basin) & can_create;
    const auto new_center = z.index_select(0, best_observation.reshape({1})).squeeze(0);
    const auto new_utility = utility.gather(0, best_observation.reshape({1})).squeeze(0);
    basin_centers_.copy_(torch::where(
        replacement_mask.unsqueeze(1), new_center.unsqueeze(0), basin_centers_));
    basin_radii_.copy_(torch::where(
        replacement_mask, torch::full_like(basin_radii_, config_.competence_radius_init), basin_radii_));
    basin_competence_.copy_(torch::where(
        replacement_mask, new_utility.expand_as(basin_competence_), basin_competence_));
    basin_evidence_.copy_(torch::where(
        replacement_mask, torch::ones_like(basin_evidence_), basin_evidence_));
    basin_utility_variance_.copy_(torch::where(
        replacement_mask, torch::zeros_like(basin_utility_variance_), basin_utility_variance_));
    basin_uncertainty_.copy_(torch::where(
        replacement_mask,
        torch::full_like(basin_uncertainty_, config_.competence_sigma_initial),
        basin_uncertainty_));
    basin_initialized_.copy_(basin_initialized_ | replacement_mask);
}

std::shared_ptr<N1PersistentState> CollectiveExpertImpl::initialize_state(const Tensor& shared_latent) {
    return body_->initialize_routing_state(shared_latent);
}

CollectiveExpertOutput CollectiveExpertImpl::forward_routing_item(
    const RoutingItem& item,
    const Tensor& active_request_indices,
    const std::shared_ptr<N1PersistentState>& state) {
    auto body_output = body_->forward_routing_item(
        item.contextual_tokens, active_request_indices, state);
    const auto pooled = body_output.token_proposal.mean(1, true)
                            .expand({-1, config_.shared_state_slots, -1});
    const auto latent_proposal = latent_proposal_projection->forward(
        torch::cat({body_output.local_shared, pooled}, -1));
    return {std::move(body_output.token_proposal), latent_proposal, std::move(body_output.full_state)};
}

std::int64_t CollectiveExpertImpl::parameter_count() const {
    std::int64_t count = 0;
    for (const auto& parameter : parameters()) count += parameter.numel();
    return count;
}

RoutingFreeCollectiveImpl::RoutingFreeCollectiveImpl(const ModelConfig& config)
    : config_(config),
      shared_attention_(register_module("shared_attention", SharedCausalGQA(config))),
      routing_latent_norm(register_module("routing_latent_norm", RMSNorm(config.latent_dim))),
      need_projection_in(register_module(
          "need_projection_in", torch::nn::Linear(config.latent_dim, config.latent_dim))),
      need_projection_out(register_module(
          "need_projection_out", torch::nn::Linear(config.latent_dim, config.competence_embedding_dim))),
      latent_initializer(register_module(
          "latent_initializer",
          torch::nn::Linear(config.latent_dim, config.shared_state_slots * config.latent_dim))),
      experts_(register_module("experts", torch::nn::ModuleList())),
      latent_query_norm(register_module("latent_query_norm", RMSNorm(config.latent_dim))),
      latent_proposal_norm(register_module("latent_proposal_norm", RMSNorm(config.latent_dim))),
      latent_query_projection(register_module("latent_query_projection", torch::nn::Linear(config.latent_dim, config.latent_dim))),
      latent_key_projection(register_module("latent_key_projection", torch::nn::Linear(
          torch::nn::LinearOptions(config.latent_dim, config.latent_dim).bias(false)))),
      latent_value_projection(register_module("latent_value_projection", torch::nn::Linear(
          torch::nn::LinearOptions(config.latent_dim, config.latent_dim).bias(false)))),
      latent_output_projection(register_module("latent_output_projection", torch::nn::Linear(
          torch::nn::LinearOptions(config.latent_dim, config.latent_dim).bias(false)))),
      latent_gate_projection(register_module("latent_gate_projection", torch::nn::Linear(config.latent_dim * 2, config.latent_dim))) {
    const auto names = n1_node_names(config.population);
    expert_handles_.reserve(config.population.size());
    for (std::size_t index = 0; index < config.population.size(); ++index) {
        auto expert = std::make_shared<CollectiveExpertImpl>(
            config, config.population[index], static_cast<std::int64_t>(index), names[index]);
        experts_->push_back(expert);
        expert_handles_.push_back(std::move(expert));
    }
    const auto counter_options = torch::TensorOptions().dtype(torch::kFloat64);
    training_routing_items_ = register_buffer("training_routing_items", torch::zeros({}, counter_options));
    training_activation_count_ = register_buffer("training_activation_count", torch::zeros({}, counter_options));
    training_novelty_count_ = register_buffer("training_novelty_count", torch::zeros({}, counter_options));
    training_exploration_count_ = register_buffer("training_exploration_count", torch::zeros({}, counter_options));
    training_entropy_sum_ = register_buffer("training_entropy_sum", torch::zeros({}, counter_options));
}

Tensor RoutingFreeCollectiveImpl::computational_need(const RoutingItem& item) {
    const auto latent = routing_latent_norm->forward(item.latent_state.mean(1));
    return normalized_rows(need_projection_out->forward(torch::silu(need_projection_in->forward(latent))));
}

RoutingFreeCollectiveImpl::LatentIntegration RoutingFreeCollectiveImpl::integrate_latent(
    const Tensor& latent,
    const Tensor& proposals,
    const Tensor& active_mask) {
    require_rank(latent, 3, "collective latent");
    require_rank(proposals, 4, "collective latent proposals");
    const auto batch = latent.size(0);
    const auto slots = latent.size(1);
    const auto experts = proposals.size(1);
    const auto heads = config_.integrator_heads;
    const auto head_dim = config_.latent_dim / heads;
    const auto normalized = latent_proposal_norm->forward(proposals);
    const auto query = latent_query_projection->forward(latent_query_norm->forward(latent))
                           .reshape({batch, slots, heads, head_dim});
    const auto keys = latent_key_projection->forward(normalized)
                          .reshape({batch, experts, slots, heads, head_dim})
                          .permute({0, 2, 3, 1, 4});
    const auto values = latent_value_projection->forward(normalized)
                            .reshape({batch, experts, slots, heads, head_dim})
                            .permute({0, 2, 3, 1, 4});
    auto scores = torch::matmul(query.unsqueeze(-2), keys.transpose(-2, -1)).squeeze(-2);
    scores = scores / std::sqrt(static_cast<double>(head_dim));
    scores = scores.masked_fill(
        ~active_mask.reshape({batch, 1, 1, experts}),
        -std::numeric_limits<float>::infinity());
    const auto attention = torch::softmax(scores, -1);
    const auto attended = torch::matmul(attention.unsqueeze(-2), values).squeeze(-2)
                              .reshape({batch, slots, config_.latent_dim});
    const auto candidate = latent_output_projection->forward(attended);
    const auto gate = torch::sigmoid(latent_gate_projection->forward(
        torch::cat({latent_query_norm->forward(latent), candidate}, -1)));
    return {latent + gate * candidate, attention.mean(2),
            normalized.to(torch::kFloat32).norm(2, -1).mean(2)};
}

Tensor RoutingFreeCollectiveImpl::diagnostic_integrate_latent(
    const Tensor& latent,
    const Tensor& proposals,
    const Tensor& active_mask) {
    return integrate_latent(latent, proposals, active_mask).latent;
}

RoutingFreeCollectiveOutput RoutingFreeCollectiveImpl::forward(
    const Tensor& embeddings,
    const std::shared_ptr<RoutingFreeCollectiveState>& incoming_state,
    const std::optional<Tensor>& availability_mask,
    const std::optional<Tensor>& force_active_mask,
    const std::optional<Tensor>& zero_proposal_mask,
    bool return_trace) {
    require_rank(embeddings, 3, "routing-free embeddings");
    const auto batch = embeddings.size(0);
    const auto sequence = embeddings.size(1);
    const auto expert_count = static_cast<std::int64_t>(expert_handles_.size());
    const auto contextual = shared_attention_->forward(embeddings);

    auto state = std::make_shared<RoutingFreeCollectiveState>();
    if (incoming_state) {
        if (incoming_state->shared_latent.sizes() !=
                torch::IntArrayRef({batch, config_.shared_state_slots, config_.latent_dim}) ||
            static_cast<std::int64_t>(incoming_state->expert_states.size()) != expert_count) {
            throw std::invalid_argument("routing-free persistent state shape mismatch");
        }
        state->shared_latent = incoming_state->shared_latent;
        state->expert_states = incoming_state->expert_states;
    } else {
        state->shared_latent = latent_initializer->forward(contextual.mean(1))
                                   .reshape({batch, config_.shared_state_slots, config_.latent_dim});
        state->expert_states.reserve(expert_handles_.size());
        for (const auto& expert : expert_handles_) {
            state->expert_states.push_back(expert->initialize_state(state->shared_latent));
        }
    }

    Tensor available;
    if (availability_mask) {
        if (availability_mask->dim() != 1 || availability_mask->numel() != expert_count) {
            throw std::invalid_argument("availability mask must contain one value per expert");
        }
        available = availability_mask->to(embeddings.device(), torch::kBool).reshape({1, expert_count});
        if (!available.to(torch::kCPU).any().item<bool>()) {
            throw std::invalid_argument("availability mask must leave at least one expert enabled");
        }
    } else {
        available = torch::ones({1, expert_count}, embeddings.options().dtype(torch::kBool));
    }
    Tensor forced;
    if (force_active_mask) {
        forced = force_active_mask->to(embeddings.device(), torch::kBool);
        if (forced.dim() == 1) forced = forced.reshape({1, expert_count});
        if (forced.dim() != 2 || forced.size(1) != expert_count ||
            (forced.size(0) != 1 && forced.size(0) != batch)) {
            throw std::invalid_argument("force-active mask must have shape [N] or [B,N]");
        }
    } else {
        forced = torch::zeros({1, expert_count}, embeddings.options().dtype(torch::kBool));
    }
    Tensor zeroed;
    if (zero_proposal_mask) {
        if (zero_proposal_mask->dim() != 1 || zero_proposal_mask->numel() != expert_count) {
            throw std::invalid_argument("zero-proposal mask must contain one value per expert");
        }
        zeroed = zero_proposal_mask->to(embeddings.device(), torch::kBool);
    } else {
        zeroed = torch::zeros({expert_count}, embeddings.options().dtype(torch::kBool));
    }

    std::vector<Tensor> output_chunks, need_history, basin_history, distance_history;
    std::vector<Tensor> resistance_history, probability_history, mask_history, novelty_history;
    std::vector<Tensor> low_confidence_history, exploration_history, token_norm_history;
    std::vector<Tensor> raw_latent_norm_history, normalized_latent_norm_history;
    std::vector<Tensor> attention_history, latent_norm_history;
    const auto population_scale = 1.0 / std::sqrt(static_cast<double>(expert_count));

    std::int64_t chunk_index = 0;
    for (std::int64_t start = 0; start < sequence; start += config_.chunk_size, ++chunk_index) {
        const auto end = std::min(start + config_.chunk_size, sequence);
        const auto chunk = contextual.index({torch::indexing::Slice(), torch::indexing::Slice(start, end)});
        RoutingItem item{chunk, state->shared_latent, chunk_index};
        const auto need = computational_need(item);

        std::vector<Tensor> basin_indices, distances, resistances, evidences, uncertainties, initialized;
        for (const auto& expert : expert_handles_) {
            const auto match = expert->match_competence(need);
            basin_indices.push_back(match.basin_index);
            distances.push_back(match.distance);
            resistances.push_back(match.resistance);
            evidences.push_back(match.evidence);
            uncertainties.push_back(match.uncertainty);
            initialized.push_back(match.initialized);
        }
        const auto basin_index = torch::stack(basin_indices, 1);
        const auto distance = torch::stack(distances, 1);
        const auto resistance = torch::stack(resistances, 1);
        const auto evidence = torch::stack(evidences, 1);
        const auto uncertainty = torch::stack(uncertainties, 1);
        const auto initialized_mask = torch::stack(initialized, 1);
        const auto probability = torch::sigmoid(
            (config_.competence_rho - resistance) / config_.competence_tau);

        auto active = is_training()
            ? (torch::rand_like(probability) < probability)
            : (resistance < config_.competence_rho);
        active = active & available;
        active = active | (forced & available);

        const auto eligible_resistance = torch::where(
            available, resistance, torch::full_like(resistance, kUnavailableResistance));
        auto best = eligible_resistance.min(1);
        const auto best_resistance = std::get<0>(best);
        const auto best_expert = std::get<1>(best);
        const auto best_evidence = evidence.gather(1, best_expert.unsqueeze(1)).squeeze(1);
        const auto best_uncertainty = uncertainty.gather(1, best_expert.unsqueeze(1)).squeeze(1);
        const auto best_initialized = initialized_mask.gather(1, best_expert.unsqueeze(1)).squeeze(1);
        const auto novel = best_resistance > config_.competence_rho_novel;
        const auto low_confidence = ~best_initialized |
            (best_evidence < config_.competence_min_evidence) |
            (best_uncertainty > config_.competence_confidence_sigma);
        const auto explore_region = is_training() & (novel | low_confidence);

        auto exploration = torch::zeros_like(active);
        auto remaining = available.expand({batch, expert_count}) & ~active;
        for (std::int64_t sample = 0; sample < config_.competence_novel_exploration_samples; ++sample) {
            auto weight = uncertainty / (evidence + 1.0).sqrt();
            weight = weight * remaining.to(weight.scalar_type()) * explore_region.unsqueeze(1);
            const auto row_sum = weight.sum(1, true);
            const auto safe_weight = torch::where(row_sum > 0.0, weight, torch::ones_like(weight));
            const auto chosen = torch::multinomial(safe_weight, 1, false);
            const auto chosen_mask = torch::zeros_like(active).scatter(1, chosen, true) &
                remaining & explore_region.unsqueeze(1);
            exploration = exploration | chosen_mask;
            remaining = remaining & ~chosen_mask;
        }
        active = active | exploration;
        exploration = exploration | (novel.unsqueeze(1) & active & ~forced);
        const auto participation = torch::where(
            exploration | forced, torch::ones_like(probability), probability) *
            active.to(probability.scalar_type());

        std::vector<Tensor> token_proposals, latent_proposals;
        for (std::int64_t expert_index = 0; expert_index < expert_count; ++expert_index) {
            const auto indices = torch::nonzero(active.select(1, expert_index)).reshape({-1});
            auto result = expert_handles_[static_cast<std::size_t>(expert_index)]->forward_routing_item(
                item, indices, state->expert_states[static_cast<std::size_t>(expert_index)]);
            state->expert_states[static_cast<std::size_t>(expert_index)] = std::move(result.full_state);
            auto token_full = torch::zeros_like(chunk).index_copy(
                0, indices, result.token_proposal.to(chunk.scalar_type()));
            auto latent_full = torch::zeros_like(state->shared_latent).index_copy(
                0, indices, result.latent_proposal.to(state->shared_latent.scalar_type()));
            const auto keep = (~zeroed.select(0, expert_index)).to(token_full.scalar_type());
            token_proposals.push_back(token_full * keep);
            latent_proposals.push_back(latent_full * keep);
        }
        const auto token_stack = torch::stack(token_proposals, 1);
        const auto latent_stack = torch::stack(latent_proposals, 1);
        const auto token_update = (
            token_stack * participation.unsqueeze(-1).unsqueeze(-1).to(token_stack.scalar_type()))
                                      .sum(1) * population_scale;
        auto output_state = chunk + token_update;
        if (is_training() && torch::GradMode::is_enabled()) {
            output_state.retain_grad();
            pending_competence_.push_back(PendingCompetenceObservation{
                need.detach(), basin_index.detach(), participation.detach(), active.detach(),
                exploration.detach(), novel.detach(), token_stack.detach(), output_state});
        }
        output_chunks.push_back(output_state);

        const auto integration_active = active & ~zeroed.reshape({1, expert_count});
        const auto first_available = available & (available.to(torch::kLong).cumsum(1) == 1);
        const auto safe_integration_active = integration_active |
            (~integration_active.any(1, true) & first_available.expand({batch, expert_count}));
        const auto integrated = integrate_latent(state->shared_latent, latent_stack, safe_integration_active);
        state->shared_latent = integrated.latent;

        need_history.push_back(need);
        basin_history.push_back(basin_index);
        distance_history.push_back(distance);
        resistance_history.push_back(resistance);
        probability_history.push_back(probability);
        mask_history.push_back(active);
        novelty_history.push_back(novel);
        low_confidence_history.push_back(low_confidence);
        exploration_history.push_back(exploration);
        token_norm_history.push_back(token_stack.to(torch::kFloat32).norm(2, -1).mean(2));
        raw_latent_norm_history.push_back(latent_stack.to(torch::kFloat32).norm(2, -1).mean(2));
        normalized_latent_norm_history.push_back(integrated.normalized_proposal_norm);
        attention_history.push_back(integrated.attention);
        latent_norm_history.push_back(state->shared_latent.to(torch::kFloat32).norm(2, -1).mean(1));
    }

    const auto probabilities = torch::stack(probability_history, 1);
    const auto masks = torch::stack(mask_history, 1);
    const auto binary = masks.to(torch::kFloat32);
    const auto density = binary.mean();
    const auto bounded_probability = probabilities.clamp(1e-8, 1.0 - 1e-8);
    const auto resonance_entropy = -(
        bounded_probability * bounded_probability.log() +
        (1.0 - bounded_probability) * (1.0 - bounded_probability).log()).mean();
    if (is_training() && torch::GradMode::is_enabled()) {
        torch::NoGradGuard no_grad;
        const auto routing_items = static_cast<double>(batch * masks.size(1));
        training_routing_items_.add_(routing_items);
        training_activation_count_.add_(binary.sum().to(torch::kFloat64));
        training_novelty_count_.add_(
            torch::stack(novelty_history, 1).to(torch::kFloat64).sum());
        training_exploration_count_.add_(
            torch::stack(exploration_history, 1).to(torch::kFloat64).sum());
        training_entropy_sum_.add_(
            resonance_entropy.to(torch::kFloat64) * routing_items * expert_count);
    }

    std::optional<RoutingFreeTrace> trace;
    if (return_trace) {
        const auto flattened = binary.reshape({-1, expert_count});
        const auto item_count = std::max<std::int64_t>(flattened.size(0), 1);
        const auto coactivation = flattened.transpose(0, 1).matmul(flattened) /
                                  static_cast<double>(item_count);
        const auto centered = flattened - flattened.mean(0, true);
        const auto covariance = centered.transpose(0, 1).matmul(centered) /
                                static_cast<double>(item_count);
        const auto standard = covariance.diag().clamp_min(1e-12).sqrt();
        const auto correlation = covariance /
            (standard.unsqueeze(1) * standard.unsqueeze(0)).clamp_min(1e-12);
        const auto activation_totals = binary.sum({0, 1});
        std::vector<Tensor> centers, radii, competence, evidence, uncertainty, initialized;
        std::vector<Tensor> marginal_utility, utility_observations;
        for (const auto& expert : expert_handles_) {
            centers.push_back(expert->basin_centers());
            radii.push_back(expert->basin_radii());
            competence.push_back(expert->basin_competence());
            evidence.push_back(expert->basin_evidence());
            uncertainty.push_back(expert->basin_uncertainty());
            initialized.push_back(expert->basin_initialized());
            marginal_utility.push_back(expert->marginal_utility());
            utility_observations.push_back(expert->utility_observations());
        }
        trace = RoutingFreeTrace{
            torch::stack(need_history, 1).detach(), torch::stack(basin_history, 1).detach(),
            torch::stack(distance_history, 1).detach(), torch::stack(resistance_history, 1).detach(),
            probabilities.detach(), masks.detach(), torch::stack(novelty_history, 1).detach(),
            torch::stack(low_confidence_history, 1).detach(),
            torch::stack(exploration_history, 1).detach(),
            torch::stack(token_norm_history, 1).detach(),
            torch::stack(raw_latent_norm_history, 1).detach(),
            torch::stack(normalized_latent_norm_history, 1).detach(),
            torch::stack(attention_history, 1).detach(), torch::stack(latent_norm_history, 1).detach(),
            density.detach(), resonance_entropy.detach(), coactivation.detach(), correlation.detach(),
            (activation_totals / activation_totals.sum().clamp_min(1.0)).detach(),
            torch::stack(centers).detach(), torch::stack(radii).detach(),
            torch::stack(competence).detach(), torch::stack(evidence).detach(),
            torch::stack(uncertainty).detach(), torch::stack(initialized).detach(),
            torch::stack(marginal_utility).detach(), torch::stack(utility_observations).detach(),
            (training_activation_count_ /
                (training_routing_items_ * expert_count).clamp_min(1.0)).detach(),
            (training_novelty_count_ / training_routing_items_.clamp_min(1.0)).detach(),
            (training_exploration_count_ /
                (training_routing_items_ * expert_count).clamp_min(1.0)).detach(),
            (training_entropy_sum_ /
                (training_routing_items_ * expert_count).clamp_min(1.0)).detach()};
    }
    return {contextual, torch::cat(output_chunks, 1),
            torch::zeros({}, embeddings.options().dtype(torch::kFloat32)),
            std::move(state), std::move(trace)};
}

void RoutingFreeCollectiveImpl::update_competence_from_backward() {
    torch::NoGradGuard no_grad;
    for (auto& observation : pending_competence_) {
        const auto gradient = observation.output_state.grad();
        if (!gradient.defined()) continue;
        const auto weighted_proposal = observation.token_proposals.to(torch::kFloat32) *
            observation.resonance_probability.unsqueeze(-1).unsqueeze(-1);
        auto utility = -(
            gradient.detach().to(torch::kFloat32).unsqueeze(1) * weighted_proposal).sum({2, 3});
        utility = utility /
            static_cast<double>(observation.token_proposals.size(2) * observation.token_proposals.size(3));
        utility = (utility * config_.competence_utility_scale).clamp(
            -config_.competence_utility_clip, config_.competence_utility_clip);
        utility = utility * observation.activation_mask.to(utility.scalar_type());
        for (std::int64_t expert = 0;
             expert < static_cast<std::int64_t>(expert_handles_.size()); ++expert) {
            expert_handles_[static_cast<std::size_t>(expert)]->update_competence(
                observation.need_embedding, observation.matched_basin.select(1, expert),
                utility.select(1, expert), observation.activation_mask.select(1, expert),
                observation.exploration_mask.select(1, expert), observation.novelty_mask);
        }
    }
    pending_competence_.clear();
}

}  // namespace rayvan::emc
