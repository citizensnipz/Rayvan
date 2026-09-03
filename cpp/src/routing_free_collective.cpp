#include "rayvan_emc/n1/routing_free_collective.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace rayvan::emc {

CollectiveExpertImpl::CollectiveExpertImpl(
    const ModelConfig& config,
    N1Family family,
    std::int64_t expert_id,
    std::string name)
    : config_(config), family_(family),
      body_(register_module("body", create_n1_node(config, family, expert_id, std::move(name)))),
      activation_projection(register_module(
          "activation_projection",
          torch::nn::Linear(torch::nn::LinearOptions(config.latent_dim, config.activation_rank).bias(false)))),
      latent_proposal_projection(register_module(
          "latent_proposal_projection",
          torch::nn::Linear(config.latent_dim * 2, config.latent_dim))) {
    // Match the small transformer initialization used by the reference
    // implementation; Kaiming-uniform makes L2 gate norms dwarf theta.
    torch::nn::init::normal_(activation_projection->weight, 0.0, 0.02);
    // Official implementation stores -1e-6 and evaluates norm - bias. This
    // makes initial participation broad while preserving the paper equation.
    activation_bias_ = register_parameter("activation_bias", torch::full({1}, -1e-6));
}

ExpertActivation CollectiveExpertImpl::activation(
    const Tensor& routing_representation,
    double theta) {
    require_rank(routing_representation, 2, "routing representation");
    const auto internal = activation_projection->forward(routing_representation);
    const auto response = internal.to(torch::kFloat32).norm(2, -1);
    const auto strength = torch::relu(response - activation_bias_.to(torch::kFloat32));
    return {internal, response, strength, strength >= theta};
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
    return {
        std::move(body_output.token_proposal),
        latent_proposal,
        std::move(body_output.full_state)};
}

std::int64_t CollectiveExpertImpl::parameter_count() const {
    std::int64_t count = 0;
    for (const auto& parameter : parameters()) count += parameter.numel();
    return count;
}

RoutingFreeCollectiveImpl::RoutingFreeCollectiveImpl(const ModelConfig& config)
    : config_(config),
      shared_attention_(register_module("shared_attention", SharedCausalGQA(config))),
      routing_context_norm(register_module("routing_context_norm", RMSNorm(config.latent_dim))),
      routing_latent_norm(register_module("routing_latent_norm", RMSNorm(config.latent_dim))),
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
    adaptive_lambda_ = register_buffer(
        "adaptive_lambda",
        torch::tensor(config.routing_lambda_initial, torch::TensorOptions().dtype(torch::kFloat32)));
}

Tensor RoutingFreeCollectiveImpl::routing_representation(const RoutingItem& item) {
    const auto context = routing_context_norm->forward(item.contextual_tokens.mean(1));
    const auto latent = routing_latent_norm->forward(item.latent_state.mean(1));
    return (context + latent) * (1.0 / std::sqrt(2.0));
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
    return {
        latent + gate * candidate,
        attention.mean(2),
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

    std::vector<Tensor> output_chunks;
    std::vector<Tensor> response_history;
    std::vector<Tensor> strength_history;
    std::vector<Tensor> mask_history;
    std::vector<Tensor> recovery_history;
    std::vector<Tensor> token_norm_history;
    std::vector<Tensor> raw_latent_norm_history;
    std::vector<Tensor> normalized_latent_norm_history;
    std::vector<Tensor> attention_history;
    std::vector<Tensor> latent_norm_history;
    const auto population_scale = 1.0 / std::sqrt(static_cast<double>(expert_count));
    std::vector<Tensor> expert_bias_values;
    expert_bias_values.reserve(expert_handles_.size());
    for (const auto& expert : expert_handles_) expert_bias_values.push_back(expert->activation_bias());
    const auto activation_biases = torch::cat(expert_bias_values).to(torch::kFloat32);

    std::int64_t chunk_index = 0;
    for (std::int64_t start = 0; start < sequence; start += config_.chunk_size, ++chunk_index) {
        const auto end = std::min(start + config_.chunk_size, sequence);
        const auto chunk = contextual.index({torch::indexing::Slice(), torch::indexing::Slice(start, end)});
        RoutingItem item{chunk, state->shared_latent, chunk_index};
        const auto representation = routing_representation(item);

        std::vector<Tensor> responses;
        std::vector<Tensor> strengths;
        std::vector<Tensor> local_masks;
        responses.reserve(expert_handles_.size());
        strengths.reserve(expert_handles_.size());
        local_masks.reserve(expert_handles_.size());
        for (const auto& expert : expert_handles_) {
            const auto decision = expert->activation(representation, config_.routing_theta);
            responses.push_back(decision.response_norm);
            strengths.push_back(decision.strength);
            local_masks.push_back(decision.active);
        }
        const auto response = torch::stack(responses, 1);
        const auto strength = torch::stack(strengths, 1);
        auto active = torch::stack(local_masks, 1) & available;
        active = active | (forced & available);
        const auto recovery = ~active.any(1);
        // Symmetric all-expert recovery contains no ranking or comparison. It
        // guarantees a valid attention set and gives every local gate signal.
        active = active | (recovery.unsqueeze(1).expand_as(active) & available);
        const auto effective_strength = torch::where(
            recovery.unsqueeze(1),
            torch::nn::functional::softplus(response - activation_biases.reshape({1, expert_count})) + 1e-6,
            strength);

        std::vector<Tensor> token_proposals;
        std::vector<Tensor> latent_proposals;
        token_proposals.reserve(expert_handles_.size());
        latent_proposals.reserve(expert_handles_.size());
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
            token_stack * effective_strength.unsqueeze(-1).unsqueeze(-1).to(token_stack.scalar_type()))
                                      .sum(1) * population_scale;
        output_chunks.push_back(chunk + token_update);
        const auto integration_active = active & ~zeroed.reshape({1, expert_count});
        const auto safe_integration_active = torch::where(
            integration_active.any(1, true), integration_active, active);
        const auto integrated = integrate_latent(
            state->shared_latent, latent_stack, safe_integration_active);
        state->shared_latent = integrated.latent;

        response_history.push_back(response);
        strength_history.push_back(strength);
        mask_history.push_back(active);
        recovery_history.push_back(recovery);
        token_norm_history.push_back(token_stack.to(torch::kFloat32).norm(2, -1).mean(2));
        raw_latent_norm_history.push_back(latent_stack.to(torch::kFloat32).norm(2, -1).mean(2));
        normalized_latent_norm_history.push_back(integrated.normalized_proposal_norm);
        attention_history.push_back(integrated.attention);
        latent_norm_history.push_back(state->shared_latent.to(torch::kFloat32).norm(2, -1).mean(1));
    }

    const auto responses = torch::stack(response_history, 1);
    const auto strengths = torch::stack(strength_history, 1);
    const auto masks = torch::stack(mask_history, 1);
    const auto binary = masks.to(torch::kFloat32);
    // The ordinary path is exactly the paper's ReLU proxy. The exceptional
    // all-off recovery uses its smooth positive fallback so gates retain a
    // gradient instead of becoming permanently dead.
    const auto recovery = torch::stack(recovery_history, 1);
    const auto proxy = torch::where(
        recovery.unsqueeze(-1),
        torch::nn::functional::softplus(
            responses - activation_biases.reshape({1, 1, expert_count})) + 1e-6,
        strengths.to(torch::kFloat32));
    const auto expert_balancing = (binary.mean({0, 1}) * proxy.mean({0, 1})).mean();
    const auto item_balancing = (binary.mean(2) * proxy.mean(2)).mean();
    const auto balancing = config_.routing_mu * expert_balancing +
                           (1.0 - config_.routing_mu) * item_balancing;
    const auto density = binary.mean();
    const auto lambda_used = adaptive_lambda_.detach().clone().to(balancing.device());
    const auto auxiliary = is_training() ? lambda_used * balancing : balancing.new_zeros({});
    if (is_training()) {
        torch::NoGradGuard no_grad;
        const auto direction = torch::sign(density.detach() - config_.routing_target_density);
        const auto factor = torch::pow(
            torch::full_like(adaptive_lambda_, 1.0 + config_.routing_adaptation_rate), direction);
        adaptive_lambda_.copy_(
            (adaptive_lambda_ * factor).clamp_max(config_.routing_lambda_max));
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
        trace = RoutingFreeTrace{
            responses.detach(), strengths.detach(), masks.detach(),
            torch::stack(recovery_history, 1).detach(),
            torch::stack(token_norm_history, 1).detach(),
            torch::stack(raw_latent_norm_history, 1).detach(),
            torch::stack(normalized_latent_norm_history, 1).detach(),
            torch::stack(attention_history, 1).detach(),
            torch::stack(latent_norm_history, 1).detach(),
            activation_biases.detach(), density.detach(),
            torch::full_like(density, config_.routing_target_density),
            lambda_used.detach(), expert_balancing.detach(), item_balancing.detach(),
            balancing.detach(), coactivation.detach(), correlation.detach(),
            (activation_totals / activation_totals.sum().clamp_min(1.0)).detach()};
    }
    return {
        contextual,
        torch::cat(output_chunks, 1),
        auxiliary,
        std::move(state),
        std::move(trace)};
}

}  // namespace rayvan::emc
