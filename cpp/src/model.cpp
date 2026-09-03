#include "rayvan_emc/model.hpp"

#include <torch/script.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace rayvan::emc {
namespace {

std::string decode_bundle_name(std::string value) {
    constexpr std::string_view marker = "__DOT__";
    std::size_t position = 0;
    while ((position = value.find(marker, position)) != std::string::npos) {
        value.replace(position, marker.size(), ".");
        position += 1;
    }
    return value;
}

}  // namespace

OutputProjectionImpl::OutputProjectionImpl(
    std::int64_t input_features,
    std::int64_t output_features,
    Tensor tied_weight) {
    auto initialized_weight = torch::empty({output_features, input_features});
    torch::nn::init::kaiming_uniform_(
        initialized_weight, std::sqrt(5.0));
    bias_ = register_parameter("bias", torch::empty({output_features}));
    const auto bound = 1.0 / std::sqrt(static_cast<double>(input_features));
    torch::nn::init::uniform_(bias_, -bound, bound);
    if (tied_weight.defined()) {
        weight_ = std::move(tied_weight);
    } else {
        weight_ = register_parameter("weight", std::move(initialized_weight));
    }
}

Tensor OutputProjectionImpl::forward(const Tensor& input) const {
    return torch::linear(input, weight_, bias_);
}

EMCModelImpl::EMCModelImpl(ModelConfig config)
    : config_(std::move(config)), active_top_k_(config_.top_k),
      token_embedding(register_module("token_embedding", torch::nn::Embedding(config_.vocab_size, config_.latent_dim))),
      n1_nodes(register_module("n1_nodes", torch::nn::ModuleList())),
      output_norm(register_module("output_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({config_.latent_dim})))),
      output_projection(register_module(
          "output_projection",
          OutputProjection(
              config_.latent_dim,
              config_.vocab_size,
              config_.tie_embeddings ? token_embedding->weight : Tensor()))) {
    config_.validate();
    if (config_.n1_mode == N1Mode::legacy_nexus) {
        position_embedding = register_module(
            "position_embedding",
            torch::nn::Embedding(config_.max_sequence_length, config_.latent_dim));
        router = register_module("router", Nexus(config_));
        integrator = register_module("integrator", N2Integrator(config_));
        const auto names = n1_node_names(config_.population);
        node_handles_.reserve(config_.population.size());
        for (std::size_t index = 0; index < config_.population.size(); ++index) {
            auto node = create_n1_node(config_, config_.population[index], static_cast<std::int64_t>(index), names[index]);
            n1_nodes->push_back(node);
            node_handles_.push_back(std::move(node));
        }
    } else {
        collective = register_module("routing_free_collective", RoutingFreeCollective(config_));
    }
    if (config_.tie_embeddings) {
        torch::NoGradGuard no_grad;
        torch::nn::init::normal_(token_embedding->weight, 0.0, 0.02);
        for (auto& item : output_projection->named_parameters(false)) {
            if (item.key() == "bias") torch::nn::init::zeros_(item.value());
        }
    }
}

void EMCModelImpl::set_active_top_k(std::int64_t value) {
    if (config_.n1_mode != N1Mode::legacy_nexus) {
        throw std::logic_error("top-K is unavailable in routing-free collective mode");
    }
    if (value < 1 || value > static_cast<std::int64_t>(config_.population.size())) {
        throw std::invalid_argument("active top-K must be within the N1 population");
    }
    active_top_k_ = value;
}

RoutingDecision EMCModelImpl::force_routing(
    const RoutingDecision& routing,
    const Tensor& forced_input,
    std::int64_t batch) const {
    auto forced = forced_input.to(routing.scores.device(), torch::kLong);
    if (forced.dim() == 1) forced = forced.reshape({1, -1}).expand({batch, -1});
    if (forced.dim() != 2 || forced.size(0) != batch || forced.size(1) != active_top_k_) {
        throw std::invalid_argument("forced routing must have shape [batch, top_k]");
    }
    const auto population = static_cast<std::int64_t>(config_.population.size());
    if (((forced < 0) | (forced >= population)).any().item<bool>()) {
        throw std::invalid_argument("forced routing contains an out-of-range N1 ID");
    }
    if (active_top_k_ > 1) {
        const auto sorted = std::get<0>(forced.sort(-1));
        const auto left = sorted.index({torch::indexing::Slice(), torch::indexing::Slice(0, active_top_k_ - 1)});
        const auto right = sorted.index({torch::indexing::Slice(), torch::indexing::Slice(1, active_top_k_)});
        if ((left == right).any().item<bool>()) throw std::invalid_argument("forced routing cannot select one N1 twice");
    }
    const auto forced_scores = routing.scores.gather(-1, forced);
    return {
        routing.scores,
        routing.pre_top_k_probabilities,
        forced,
        torch::softmax(forced_scores, -1),
        torch::arange(active_top_k_, forced.options()).expand_as(forced)};
}

std::tuple<Tensor, std::shared_ptr<N2State>, std::vector<N1Diagnostics>, std::vector<std::int64_t>, DispatchPlan>
EMCModelImpl::execute_selected_nodes(
    const Tensor& latent,
    const Tensor& selected_indices,
    const std::shared_ptr<N2State>& state) {
    const auto batch = latent.size(0);
    const auto sequence = latent.size(1);
    const auto latent_dim = latent.size(2);
    auto dispatch = DispatchPlan::from_routing(selected_indices, static_cast<std::int64_t>(config_.population.size()));

    std::vector<N1Output> outputs;
    outputs.reserve(node_handles_.size());
    for (std::size_t node_id = 0; node_id < node_handles_.size(); ++node_id) {
        const auto mask = dispatch.sorted_expert_ids == static_cast<std::int64_t>(node_id);
        const auto request_rows = dispatch.sorted_source_indices.masked_select(mask);
        const auto node_batch = latent.index_select(0, request_rows);
        std::shared_ptr<N1PersistentState> local;
        if (state) {
            const auto iterator = state->local_states.find(static_cast<std::int64_t>(node_id));
            if (iterator != state->local_states.end()) local = iterator->second;
        }
        outputs.push_back(node_handles_[node_id]->forward({node_batch, std::nullopt, std::move(local), request_rows}));
    }

    auto next_state = std::make_shared<N2State>();
    std::vector<N1Diagnostics> diagnostics;
    std::vector<std::int64_t> executed;
    std::vector<Tensor> grouped;
    grouped.reserve(outputs.size());
    for (std::size_t node_id = 0; node_id < outputs.size(); ++node_id) {
        auto& output = outputs[node_id];
        grouped.push_back(output.proposal);
        if (output.proposal.size(0) == 0) continue;
        if (output.local_state) next_state->local_states.emplace(static_cast<std::int64_t>(node_id), output.local_state);
        diagnostics.push_back(output.diagnostics);
        executed.push_back(static_cast<std::int64_t>(node_id));
    }
    const auto proposals = dispatch.restore(torch::cat(grouped, 0), batch, sequence, latent_dim);
    return {proposals, std::move(next_state), std::move(diagnostics), std::move(executed), std::move(dispatch)};
}

EMCOutput EMCModelImpl::forward(const EMCInput& input) {
    require_rank(input.token_ids, 2, "token_ids");
    if (input.token_ids.scalar_type() != torch::kLong) throw std::invalid_argument("token_ids must be int64");
    const auto batch = input.token_ids.size(0);
    const auto sequence = input.token_ids.size(1);
    if (sequence > config_.max_sequence_length) throw std::invalid_argument("sequence exceeds configured maximum");
    auto embeddings = token_embedding->forward(input.token_ids);
    if (config_.n1_mode == N1Mode::routing_free_collective) {
        std::shared_ptr<RoutingFreeCollectiveState> persistent;
        if (input.state) persistent = input.state->collective;
        std::optional<Tensor> availability;
        std::optional<Tensor> forced;
        std::optional<Tensor> zeroed;
        if (input.intervention) {
            availability = input.intervention->availability_mask;
            forced = input.intervention->force_active_mask;
            zeroed = input.intervention->zero_proposal_mask;
            if (input.intervention->forced_nodes) {
                throw std::invalid_argument("forced_nodes is a Top-K intervention and is unavailable in routing-free mode");
            }
        }
        auto result = collective->forward(
            embeddings, persistent, availability, forced, zeroed, input.return_trace);
        auto next_state = std::make_shared<N2State>();
        next_state->collective = result.state;
        const auto logits = output_projection->forward(output_norm->forward(result.token_state));
        const auto float_options = embeddings.options().dtype(torch::kFloat32);
        const auto long_options = embeddings.options().dtype(torch::kLong);
        RoutingDecision no_router{
            torch::empty({batch, 0}, float_options),
            torch::empty({batch, 0}, float_options),
            torch::empty({batch, 0}, long_options),
            torch::empty({batch, 0}, float_options),
            torch::empty({batch, 0}, long_options)};
        return {
            logits,
            embeddings,
            result.contextual_state,
            std::move(no_router),
            torch::empty({batch, sequence, 0, config_.latent_dim}, embeddings.options()),
            result.token_state,
            std::nullopt,
            std::nullopt,
            std::move(next_state),
            result.auxiliary_loss,
            std::move(result.trace)};
    }
    const auto positions = torch::arange(sequence, input.token_ids.options());
    embeddings = embeddings + position_embedding->forward(positions);
    const auto shared_state = embeddings;

    std::optional<Tensor> availability;
    if (input.intervention) availability = input.intervention->availability_mask;
    auto routing = router->forward(shared_state, active_top_k_, availability);
    if (input.intervention && input.intervention->forced_nodes) {
        routing = force_routing(routing, *input.intervention->forced_nodes, batch);
    }

    auto [proposals, next_state, node_diagnostics, executed, dispatch] =
        execute_selected_nodes(shared_state, routing.selected_indices, input.state);
    if (input.intervention && input.intervention->zero_proposal_mask) {
        const auto& mask_input = *input.intervention->zero_proposal_mask;
        if (mask_input.dim() != 1 || mask_input.numel() != static_cast<std::int64_t>(config_.population.size())) {
            throw std::invalid_argument("zero proposal mask must contain one value per N1");
        }
        const auto zero = mask_input.to(proposals.device(), torch::kBool).index({routing.selected_indices});
        proposals = proposals.masked_fill(zero.unsqueeze(1).unsqueeze(-1), 0);
    }

    auto integrated = integrator->forward(shared_state, proposals, routing.selected_weights, input.return_trace);
    const auto logits = output_projection->forward(output_norm->forward(integrated.latent));
    std::optional<ExecutionTrace> execution_trace;
    if (input.return_trace) {
        ExecutionTrace trace;
        trace.selected_node_ids = routing.selected_indices.detach();
        trace.selected_node_weights = routing.selected_weights.detach();
        trace.pre_top_k_probabilities = routing.pre_top_k_probabilities.detach();
        trace.selected_slots = routing.selected_slots.detach();
        trace.dispatch_permutation = dispatch.permutation.detach();
        trace.dispatch_inverse_permutation = dispatch.inverse_permutation.detach();
        trace.dispatch_counts = dispatch.expert_counts.detach();
        trace.dispatch_offsets = dispatch.expert_offsets.detach();
        trace.executed_node_ids = std::move(executed);
        trace.actual_node_executions = routing.selected_indices.numel();
        trace.theoretical_all_node_executions = batch * static_cast<std::int64_t>(config_.population.size());
        trace.node_diagnostics = std::move(node_diagnostics);
        execution_trace = std::move(trace);
    }
    return {
        logits,
        embeddings,
        shared_state,
        std::move(routing),
        proposals,
        integrated.latent,
        std::move(integrated.trace),
        std::move(execution_trace),
        std::move(next_state),
        torch::zeros({}, embeddings.options().dtype(torch::kFloat32)),
        std::nullopt};
}

struct EMCModel::Storage {
    explicit Storage(ModelConfig config) : module(std::move(config)) {}
    EMCModelModule module;
};

EMCModel::EMCModel(ModelConfig config) : storage_(std::make_unique<Storage>(std::move(config))) {}
EMCModel::~EMCModel() = default;
EMCModel::EMCModel(EMCModel&&) noexcept = default;
EMCModel& EMCModel::operator=(EMCModel&&) noexcept = default;
EMCOutput EMCModel::forward(const EMCInput& input) { return storage_->module->forward(input); }
void EMCModel::train() { storage_->module->train(); }
void EMCModel::eval() { storage_->module->eval(); }
void EMCModel::to(const torch::Device& device) { storage_->module->to(device); }
void EMCModel::set_active_top_k(std::int64_t value) { storage_->module->set_active_top_k(value); }
std::int64_t EMCModel::active_top_k() const { return storage_->module->active_top_k(); }
const ModelConfig& EMCModel::config() const { return storage_->module->config(); }
std::vector<Tensor> EMCModel::parameters(bool recurse) const { return storage_->module->parameters(recurse); }
bool EMCModel::embeddings_tied() const { return storage_->module->embeddings_tied(); }
EMCModelModule EMCModel::module() const { return storage_->module; }

void EMCModel::save_weights(const std::filesystem::path& path) const {
    torch::serialize::OutputArchive archive;
    storage_->module->save(archive);
    archive.save_to(path.string());
}

void EMCModel::load_weights(const std::filesystem::path& path, const torch::Device& device) {
    storage_->module->to(torch::kCPU);
    torch::serialize::InputArchive archive;
    archive.load_from(path.string(), torch::kCPU);
    storage_->module->load(archive);
    storage_->module->to(device);
}

void EMCModel::import_python_weights(const std::filesystem::path& torchscript_bundle) {
    auto bundle = torch::jit::load(torchscript_bundle.string(), torch::kCPU);
    std::unordered_map<std::string, Tensor> source;
    for (const auto& item : bundle.named_buffers(true)) source.emplace(decode_bundle_name(item.name), item.value);
    torch::NoGradGuard no_grad;
    for (auto& item : storage_->module->named_parameters(true)) {
        const auto iterator = source.find(item.key());
        if (iterator == source.end()) throw std::runtime_error("Python bundle missing parameter: " + item.key());
        if (item.value().sizes() != iterator->second.sizes()) throw std::runtime_error("Python parameter shape mismatch: " + item.key());
        item.value().copy_(iterator->second.to(item.value().device(), item.value().scalar_type()));
    }
    for (auto& item : storage_->module->named_buffers(true)) {
        const auto iterator = source.find(item.key());
        if (iterator == source.end()) continue; // non-persistent cached masks are regenerated natively
        if (item.value().sizes() == iterator->second.sizes()) {
            item.value().copy_(iterator->second.to(item.value().device(), item.value().scalar_type()));
        }
    }
}

}  // namespace rayvan::emc
