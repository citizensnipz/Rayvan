#include "rayvan_emc/n1/gpt_n1.hpp"
#include "rayvan_emc/n1/delta_n1.hpp"
#include "rayvan_emc/n1/recurrent_n1.hpp"
#include "rayvan_emc/n1/ssm_n1.hpp"

#include <torch/nn/functional/normalization.h>

#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace rayvan::emc {

N1Block::N1Block(const ModelConfig& config, N1Family family)
    : config_(config),
      shared_condition(register_module("shared_condition", torch::nn::Linear(config.latent_dim, config.latent_dim))),
      state_output(register_module("state_output", torch::nn::Linear(config.latent_dim * 2, config.latent_dim))),
      family_(family) {}

Tensor N1Block::condition_chunk(const Tensor& chunk, const Tensor& shared_state) {
    const auto context = shared_condition->forward(shared_state.mean(1)).unsqueeze(1);
    return chunk + context;
}

Tensor N1Block::state_proposal(const Tensor& token_proposal, const Tensor& shared_state) {
    const auto pooled = token_proposal.mean(1, true).expand({-1, shared_state.size(1), -1});
    return state_output->forward(torch::cat({shared_state, pooled}, -1));
}

GPTN1Block::GPTN1Block(const ModelConfig& config)
    : N1Block(config, N1Family::gpt),
      attention_norm(register_module("attention_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.latent_dim})))),
      shared_norm(register_module("shared_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.latent_dim})))),
      attention(register_module(
          "attention",
          torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(config.latent_dim, config.attention_heads)))),
      feed_forward_norm(register_module("feed_forward_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.latent_dim})))),
      feed_forward(register_module(
          "feed_forward",
          torch::nn::Sequential(
              torch::nn::Linear(config.latent_dim, config.resolved_module_hidden_dim()),
              torch::nn::GELU(),
              torch::nn::Linear(config.resolved_module_hidden_dim(), config.latent_dim)))) {
    const auto length = config.chunk_size + config.shared_state_slots;
    causal_mask = register_buffer(
        "_causal_mask",
        torch::ones({length, length}, torch::TensorOptions().dtype(torch::kBool)).triu(1));
}

LeaseState GPTN1Block::begin_lease(const Tensor&) { return {}; }

BlockOutput GPTN1Block::forward_chunk(
    const Tensor& chunk_latent,
    const Tensor& shared_state,
    const LeaseState&) {
    const auto conditioned = condition_chunk(chunk_latent, shared_state);
    const auto memory = shared_norm->forward(shared_state);
    const auto combined = torch::cat({memory, conditioned}, 1);
    const auto normalized = attention_norm->forward(combined);
    const auto length = combined.size(1);
    const auto mask = causal_mask.index({torch::indexing::Slice(0, length), torch::indexing::Slice(0, length)});
    const auto qkv = torch::linear(
        normalized, attention->in_proj_weight, attention->in_proj_bias);
    const auto projections = qkv.chunk(3, -1);
    const auto batch = normalized.size(0);
    const auto heads = config_.attention_heads;
    const auto head_dim = config_.latent_dim / heads;
    const auto query = projections[0]
                           .reshape({batch, length, heads, head_dim})
                           .transpose(1, 2);
    const auto key = projections[1]
                         .reshape({batch, length, heads, head_dim})
                         .transpose(1, 2);
    const auto value = projections[2]
                           .reshape({batch, length, heads, head_dim})
                           .transpose(1, 2);
    auto scores = torch::matmul(
        query * (1.0 / std::sqrt(static_cast<double>(head_dim))),
        key.transpose(-2, -1));
    scores = scores.masked_fill(
        mask.reshape({1, 1, length, length}),
        -std::numeric_limits<float>::infinity());
    const auto attended_heads = torch::matmul(torch::softmax(scores, -1), value);
    const auto attended = attention->out_proj->forward(
        attended_heads.transpose(1, 2)
            .contiguous()
            .reshape({batch, length, config_.latent_dim}));
    const auto state = combined + attended;
    const auto processed = state + feed_forward->forward(feed_forward_norm->forward(state));
    const auto token_state = processed.index({torch::indexing::Slice(), torch::indexing::Slice(memory.size(1), torch::indexing::None)});
    const auto proposal = token_state - conditioned;
    return {proposal, state_proposal(proposal, shared_state), {}};
}

SSMN1Block::SSMN1Block(const ModelConfig& config)
    : N1Block(config, N1Family::ssm),
      width_(config.resolved_state_space_dim()),
      kernel_size_(config.state_space_kernel_size),
      input_norm(register_module("input_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.latent_dim})))),
      input_adapter(register_module("input_adapter", torch::nn::Linear(config.latent_dim, width_))),
      causal_convolution(register_module(
          "causal_convolution",
          torch::nn::Conv1d(torch::nn::Conv1dOptions(width_, width_, kernel_size_).groups(width_)))),
      delta_projection(register_module("delta_projection", torch::nn::Linear(width_, width_))),
      input_projection(register_module("input_projection", torch::nn::Linear(width_, width_))),
      gate_projection(register_module("gate_projection", torch::nn::Linear(width_, width_))),
      output_adapter(register_module("output_adapter", torch::nn::Linear(width_, config.latent_dim))),
      state_initializer(register_module("state_initializer", torch::nn::Linear(config.latent_dim, width_))) {
    log_decay = register_parameter("log_decay", torch::zeros({width_}));
    scan_causal_mask = register_buffer(
        "_scan_causal_mask",
        torch::ones({config.chunk_size, config.chunk_size}, torch::TensorOptions().dtype(torch::kBool)).tril());
}

LeaseState SSMN1Block::begin_lease(const Tensor& shared_state) {
    const auto state = state_initializer->forward(shared_state.mean(1));
    auto history = state.new_zeros({state.size(0), kernel_size_ - 1, width_});
    return {{{"state", state}, {"conv_history", std::move(history)}}};
}

Tensor SSMN1Block::parallel_diagonal_scan(
    const Tensor& input_log_decay,
    const Tensor& candidate,
    const Tensor& initial_state) const {
    const auto decay = torch::exp(input_log_decay);
    const auto write = (1.0 - decay) * candidate;
    const auto log_prefix = torch::cumsum(input_log_decay, 1);
    const auto prefix_by_dimension = log_prefix.transpose(1, 2);
    const auto log_coefficients = prefix_by_dimension.unsqueeze(-1) - prefix_by_dimension.unsqueeze(-2);
    const auto length = input_log_decay.size(1);
    const auto causal = scan_causal_mask.index({torch::indexing::Slice(0, length), torch::indexing::Slice(0, length)});
    const auto masked = log_coefficients.masked_fill(~causal.reshape({1, 1, length, length}), -std::numeric_limits<float>::infinity());
    const auto coefficients = torch::exp(masked);
    const auto accumulated = torch::einsum("bhtj,bjh->bth", {coefficients, write});
    return accumulated + torch::exp(log_prefix) * initial_state.unsqueeze(1);
}

BlockOutput SSMN1Block::forward_chunk(
    const Tensor& chunk_latent,
    const Tensor& shared_state,
    const LeaseState& lease_state) {
    const auto conditioned = condition_chunk(chunk_latent, shared_state);
    const auto internal = input_adapter->forward(input_norm->forward(conditioned));
    const auto history = lease_state.tensors.at("conv_history").to(internal.scalar_type());
    const auto convolution_input = torch::cat({history, internal}, 1);
    const auto convolved = causal_convolution->forward(convolution_input.transpose(1, 2)).transpose(1, 2);
    const auto delta = torch::nn::functional::softplus(delta_projection->forward(convolved)).to(torch::kFloat32);
    const auto sequence_log_decay = -torch::nn::functional::softplus(log_decay).reshape({1, 1, -1}) * delta;
    const auto candidate = torch::tanh(input_projection->forward(convolved)).to(torch::kFloat32);
    const auto initial_state = lease_state.tensors.at("state").to(torch::kFloat32);
    const auto states = parallel_diagonal_scan(sequence_log_decay, candidate, initial_state);
    const auto gate = torch::sigmoid(gate_projection->forward(convolved)).to(torch::kFloat32);
    const auto proposal = output_adapter->forward((gate * states).to(internal.scalar_type())).to(chunk_latent.scalar_type());
    const auto new_history = convolution_input.index({torch::indexing::Slice(), torch::indexing::Slice(-(kernel_size_ - 1), torch::indexing::None)});
    LeaseState next{{{"state", states.index({torch::indexing::Slice(), -1})}, {"conv_history", new_history}}};
    return {proposal, state_proposal(proposal, shared_state), std::move(next)};
}

RecurrentN1Block::RecurrentN1Block(const ModelConfig& config)
    : N1Block(config, N1Family::recurrent),
      width_(config.resolved_recurrent_dim()),
      input_norm(register_module("input_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.latent_dim})))),
      input_adapter(register_module("input_adapter", torch::nn::Linear(config.latent_dim, width_))),
      recurrent(register_module("recurrent", torch::nn::GRU(torch::nn::GRUOptions(width_, width_).batch_first(true)))),
      output_adapter(register_module("output_adapter", torch::nn::Linear(width_, config.latent_dim))),
      state_initializer(register_module("state_initializer", torch::nn::Linear(config.latent_dim, width_))) {}

LeaseState RecurrentN1Block::begin_lease(const Tensor& shared_state) {
    return {{{"hidden", state_initializer->forward(shared_state.mean(1))}}};
}

BlockOutput RecurrentN1Block::forward_chunk(
    const Tensor& chunk_latent,
    const Tensor& shared_state,
    const LeaseState& lease_state) {
    const auto conditioned = condition_chunk(chunk_latent, shared_state);
    const auto internal = input_adapter->forward(input_norm->forward(conditioned));
    const auto hidden = lease_state.tensors.at("hidden").to(internal.scalar_type()).unsqueeze(0);
    auto [recurrent_output, new_hidden] = recurrent->forward(internal, hidden);
    auto proposal = output_adapter->forward(recurrent_output).to(chunk_latent.scalar_type());
    LeaseState next{{{"hidden", new_hidden.squeeze(0)}}};
    return {proposal, state_proposal(proposal, shared_state), std::move(next)};
}

std::shared_ptr<N1Block> create_n1_block(const ModelConfig& config, N1Family family) {
    switch (family) {
    case N1Family::gpt: return std::make_shared<GPTN1Block>(config);
    case N1Family::ssm: return std::make_shared<SSMN1Block>(config);
    case N1Family::recurrent: return std::make_shared<RecurrentN1Block>(config);
    case N1Family::delta: return std::make_shared<DeltaN1Block>(config);
    }
    throw std::invalid_argument("unsupported N1 family");
}

std::vector<std::string> n1_node_names(const std::vector<N1Family>& families) {
    std::unordered_map<std::string, std::int64_t> totals;
    for (const auto family : families) ++totals[std::string(to_string(family))];
    std::unordered_map<std::string, std::int64_t> seen;
    std::vector<std::string> names;
    names.reserve(families.size());
    for (const auto family : families) {
        const std::string base(to_string(family));
        const auto ordinal = ++seen[base];
        names.push_back(totals[base] == 1 ? base : base + "-" + std::to_string(ordinal));
    }
    return names;
}

N1Node::N1Node(
    const ModelConfig& config,
    N1Family family,
    std::int64_t node_id,
    std::string node_name)
    : config_(config), family_(family), node_id_(node_id), node_name_(std::move(node_name)),
      blocks(register_module("blocks", torch::nn::ModuleList())),
      state_initializer(register_module(
          "state_initializer",
          torch::nn::Linear(config.latent_dim, config.shared_state_slots * config.latent_dim))) {
    for (std::int64_t index = 0; index < config.n1_depth; ++index) blocks->push_back(create_n1_block(config, family));
    for (const auto& parameter : parameters()) parameter_count_ += parameter.numel();
}

std::shared_ptr<N1Node> create_n1_node(
    const ModelConfig& config,
    N1Family family,
    std::int64_t node_id,
    std::string node_name) {
    return std::make_shared<N1Node>(config, family, node_id, std::move(node_name));
}

std::int64_t N1Node::approximate_flops(std::int64_t sequence_length) const {
    if (sequence_length <= 0) throw std::invalid_argument("sequence_length must be positive");
    auto flops = 2 * parameter_count_ * sequence_length;
    if (family_ == N1Family::gpt) {
        const auto chunks = (sequence_length + config_.chunk_size - 1) / config_.chunk_size;
        const auto chunk_length = std::min(sequence_length, config_.chunk_size);
        const auto attended = chunk_length + config_.shared_state_slots;
        flops += config_.n1_depth * chunks * 4 * attended * attended * config_.latent_dim;
    }
    return flops;
}

N1Output N1Node::forward(const N1Input& input) {
    require_rank(input.shared_latent, 3, "shared_latent");
    if (input.shared_latent.size(-1) != config_.latent_dim) throw std::invalid_argument("shared_latent final dimension mismatch");
    const auto batch = input.shared_latent.size(0);
    const auto sequence = input.shared_latent.size(1);
    if (batch == 0) return {input.shared_latent, nullptr, make_diagnostics(sequence, 0, std::nullopt, false)};

    auto request_indices = input.request_indices.value_or(
        torch::arange(batch, input.shared_latent.options().dtype(torch::kLong)));
    if (request_indices.dim() != 1 || request_indices.size(0) != batch) throw std::invalid_argument("request_indices shape mismatch");

    Tensor local_shared;
    std::vector<LeaseState> incoming;
    const bool reset = !input.local_state;
    if (input.local_state) {
        if (static_cast<std::int64_t>(input.local_state->block_states.size()) != config_.n1_depth) {
            throw std::invalid_argument("local state block count mismatch");
        }
        local_shared = input.local_state->shared_state;
        incoming = input.local_state->block_states;
    } else {
        local_shared = input.shared_state.value_or(
            state_initializer->forward(input.shared_latent.mean(1)).reshape({batch, config_.shared_state_slots, config_.latent_dim}));
        incoming.reserve(config_.n1_depth);
        for (const auto& module : *blocks) {
            auto* block = dynamic_cast<N1Block*>(module.get());
            if (!block) throw std::logic_error("invalid N1 block registration");
            incoming.push_back(block->begin_lease(local_shared));
        }
    }
    if (local_shared.sizes() != torch::IntArrayRef({batch, config_.shared_state_slots, config_.latent_dim})) {
        throw std::invalid_argument("shared_state shape mismatch");
    }

    ++execution_count_;
    const double residual_scale = 1.0 / std::sqrt(static_cast<double>(config_.n1_depth));
    auto current = input.shared_latent;
    const auto initial_shared = local_shared;
    std::vector<LeaseState> next_states;
    next_states.reserve(config_.n1_depth);
    for (std::int64_t block_index = 0; block_index < config_.n1_depth; ++block_index) {
        auto* block = dynamic_cast<N1Block*>(blocks[static_cast<std::size_t>(block_index)].get());
        auto lease = incoming[static_cast<std::size_t>(block_index)];
        std::vector<Tensor> chunks;
        for (std::int64_t start = 0; start < sequence; start += config_.chunk_size) {
            const auto end = std::min(start + config_.chunk_size, sequence);
            const auto chunk = current.index({torch::indexing::Slice(), torch::indexing::Slice(start, end)});
            auto output = block->forward_chunk(chunk, local_shared, lease);
            chunks.push_back(chunk + residual_scale * output.token_proposal);
            local_shared = local_shared + residual_scale * output.state_proposal;
            lease = std::move(output.new_lease_state);
        }
        current = torch::cat(chunks, 1);
        next_states.push_back(std::move(lease));
    }

    std::optional<Tensor> state_change;
    if (family_ != N1Family::gpt) state_change = (local_shared - initial_shared).to(torch::kFloat32).norm().detach();
    std::shared_ptr<N1PersistentState> persistent;
    if (family_ != N1Family::gpt) {
        persistent = std::make_shared<N1PersistentState>(N1PersistentState{std::move(next_states), local_shared});
    }
    return {current - input.shared_latent, std::move(persistent), make_diagnostics(sequence, batch, state_change, reset)};
}

std::shared_ptr<N1PersistentState> N1Node::initialize_routing_state(const Tensor& shared_state) {
    require_rank(shared_state, 3, "routing shared_state");
    if (shared_state.size(1) != config_.shared_state_slots || shared_state.size(2) != config_.latent_dim) {
        throw std::invalid_argument("routing shared_state shape mismatch");
    }
    std::vector<LeaseState> leases;
    leases.reserve(config_.n1_depth);
    for (const auto& module : *blocks) {
        auto* block = dynamic_cast<N1Block*>(module.get());
        if (!block) throw std::logic_error("invalid N1 block registration");
        leases.push_back(block->begin_lease(shared_state));
    }
    return std::make_shared<N1PersistentState>(N1PersistentState{std::move(leases), shared_state});
}

N1RoutingItemOutput N1Node::forward_routing_item(
    const Tensor& chunk,
    const Tensor& active_request_indices,
    const std::shared_ptr<N1PersistentState>& full_state) {
    require_rank(chunk, 3, "routing item chunk");
    require_rank(active_request_indices, 1, "active request indices");
    if (!full_state || static_cast<std::int64_t>(full_state->block_states.size()) != config_.n1_depth) {
        throw std::invalid_argument("routing item requires initialized full-batch expert state");
    }
    if (active_request_indices.scalar_type() != torch::kLong) {
        throw std::invalid_argument("active request indices must be int64");
    }
    const auto active_chunk = chunk.index_select(0, active_request_indices);
    if (active_request_indices.numel() == 0) {
        return {active_chunk, full_state->shared_state.index_select(0, active_request_indices), full_state};
    }

    ++execution_count_;
    const double residual_scale = 1.0 / std::sqrt(static_cast<double>(config_.n1_depth));
    auto current = active_chunk;
    auto local_shared = full_state->shared_state.index_select(0, active_request_indices);
    std::vector<LeaseState> next_full_leases;
    next_full_leases.reserve(config_.n1_depth);
    for (std::int64_t block_index = 0; block_index < config_.n1_depth; ++block_index) {
        auto* block = dynamic_cast<N1Block*>(blocks[static_cast<std::size_t>(block_index)].get());
        if (!block) throw std::logic_error("invalid N1 block registration");
        LeaseState packed;
        for (const auto& [name, value] : full_state->block_states[static_cast<std::size_t>(block_index)].tensors) {
            packed.tensors.emplace(name, value.index_select(0, active_request_indices));
        }
        auto output = block->forward_chunk(current, local_shared, packed);
        current = current + residual_scale * output.token_proposal;
        local_shared = local_shared + residual_scale * output.state_proposal;
        LeaseState merged;
        for (const auto& [name, old_value] : full_state->block_states[static_cast<std::size_t>(block_index)].tensors) {
            const auto& next_value = output.new_lease_state.tensors.at(name);
            merged.tensors.emplace(
                name,
                old_value.to(next_value.scalar_type()).index_copy(
                    0, active_request_indices, next_value));
        }
        next_full_leases.push_back(std::move(merged));
    }
    auto next_shared = full_state->shared_state.to(local_shared.scalar_type()).index_copy(
        0, active_request_indices, local_shared);
    auto next_state = std::make_shared<N1PersistentState>(
        N1PersistentState{std::move(next_full_leases), std::move(next_shared)});
    return {current - active_chunk, std::move(local_shared), std::move(next_state)};
}

N1Diagnostics N1Node::make_diagnostics(
    std::int64_t sequence_length,
    std::int64_t batch,
    const std::optional<Tensor>& state_change,
    bool state_reset) const {
    const auto chunks = (sequence_length + config_.chunk_size - 1) / config_.chunk_size;
    const bool stateful = family_ != N1Family::gpt;
    N1Diagnostics result;
    result.node_id = node_id_;
    result.node_name = node_name_;
    result.family = family_;
    result.blocks_executed = config_.n1_depth;
    result.chunks_per_block = chunks;
    result.block_invocations = config_.n1_depth * chunks;
    result.parameters = parameter_count_;
    result.approximate_flops = approximate_flops(sequence_length);
    result.output_latent_size = config_.latent_dim;
    result.stateful = stateful;
    result.state_resets = stateful && state_reset ? batch : 0;
    if (stateful) {
        result.continuation_probability = chunks == 0 ? 0.0 : static_cast<double>(chunks - 1) / chunks;
        result.average_lease_length = static_cast<double>(chunks);
        result.state_change_magnitude = state_change;
    }
    return result;
}

}  // namespace rayvan::emc
