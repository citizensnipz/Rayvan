#include "rayvan_emc/config.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace rayvan::emc {
namespace {

std::vector<std::string> split(std::string_view value, char delimiter) {
    std::vector<std::string> result;
    std::size_t start = 0;
    while (start <= value.size()) {
        const auto end = value.find(delimiter, start);
        result.emplace_back(value.substr(start, end == std::string_view::npos ? value.size() - start : end - start));
        if (end == std::string_view::npos) break;
        start = end + 1;
    }
    return result;
}

template <typename Integer>
Integer parse_integer(const std::unordered_map<std::string, std::string>& fields, const std::string& key) {
    const auto iterator = fields.find(key);
    if (iterator == fields.end()) throw std::runtime_error("missing config field: " + key);
    Integer value{};
    const auto* begin = iterator->second.data();
    const auto* end = begin + iterator->second.size();
    if (const auto result = std::from_chars(begin, end, value); result.ec != std::errc{} || result.ptr != end) {
        throw std::runtime_error("invalid integer config field: " + key);
    }
    return value;
}

double parse_double(const std::unordered_map<std::string, std::string>& fields, const std::string& key) {
    const auto iterator = fields.find(key);
    if (iterator == fields.end()) throw std::runtime_error("missing config field: " + key);
    try {
        std::size_t parsed = 0;
        const auto value = std::stod(iterator->second, &parsed);
        if (parsed != iterator->second.size()) throw std::runtime_error("trailing characters");
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error("invalid floating-point config field: " + key);
    }
}

}  // namespace

std::string_view to_string(N1Family family) noexcept {
    switch (family) {
    case N1Family::gpt: return "gpt";
    case N1Family::ssm: return "ssm";
    case N1Family::recurrent: return "recurrent";
    case N1Family::delta: return "delta";
    }
    return "unknown";
}

N1Family n1_family_from_string(std::string_view value) {
    if (value == "gpt") return N1Family::gpt;
    if (value == "ssm") return N1Family::ssm;
    if (value == "recurrent" || value == "gru") return N1Family::recurrent;
    if (value == "delta" || value == "deltanet") return N1Family::delta;
    throw std::invalid_argument("unknown N1 family: " + std::string(value));
}

std::string_view to_string(N1Mode mode) noexcept {
    return mode == N1Mode::routing_free_collective ? "routing_free_collective" : "legacy_nexus";
}

N1Mode n1_mode_from_string(std::string_view value) {
    if (value == "legacy_nexus" || value == "legacy") return N1Mode::legacy_nexus;
    if (value == "routing_free_collective" || value == "routing-free") return N1Mode::routing_free_collective;
    throw std::invalid_argument("unknown N1 mode: " + std::string(value));
}

std::string_view to_string(RoutingGranularity granularity) noexcept {
    switch (granularity) {
    case RoutingGranularity::token: return "token";
    case RoutingGranularity::chunk: return "chunk";
    case RoutingGranularity::sequence: return "sequence";
    case RoutingGranularity::latent_cycle: return "latent_cycle";
    }
    return "unknown";
}

RoutingGranularity routing_granularity_from_string(std::string_view value) {
    if (value == "token") return RoutingGranularity::token;
    if (value == "chunk") return RoutingGranularity::chunk;
    if (value == "sequence") return RoutingGranularity::sequence;
    if (value == "latent_cycle") return RoutingGranularity::latent_cycle;
    throw std::invalid_argument("unknown routing granularity: " + std::string(value));
}

std::string_view to_string(Precision precision) noexcept {
    return precision == Precision::bf16 ? "bf16" : "fp32";
}

Precision precision_from_string(std::string_view value) {
    if (value == "fp32") return Precision::fp32;
    if (value == "bf16") return Precision::bf16;
    throw std::invalid_argument("precision must be fp32 or bf16");
}

std::int64_t ModelConfig::resolved_module_hidden_dim() const noexcept {
    return module_hidden_dim > 0 ? module_hidden_dim : latent_dim * 4;
}
std::int64_t ModelConfig::resolved_state_space_dim() const noexcept {
    return state_space_dim > 0 ? state_space_dim : latent_dim * 4;
}
std::int64_t ModelConfig::resolved_recurrent_dim() const noexcept {
    return recurrent_dim > 0 ? recurrent_dim : latent_dim * 2;
}
std::int64_t ModelConfig::resolved_delta_internal_dim() const noexcept {
    return delta_internal_dim > 0 ? delta_internal_dim : latent_dim;
}
std::int64_t ModelConfig::resolved_delta_ffn_dim() const noexcept {
    return delta_ffn_dim > 0 ? delta_ffn_dim : latent_dim * 4;
}

void ModelConfig::validate() const {
    const std::vector<std::pair<std::string_view, std::int64_t>> positive{
        {"latent_dim", latent_dim}, {"vocab_size", vocab_size},
        {"max_sequence_length", max_sequence_length}, {"attention_heads", attention_heads},
        {"integrator_heads", integrator_heads}, {"state_space_kernel_size", state_space_kernel_size},
        {"chunk_size", chunk_size}, {"shared_state_slots", shared_state_slots},
        {"n1_depth", n1_depth}, {"top_k", top_k},
        {"gqa_query_heads", gqa_query_heads}, {"gqa_kv_heads", gqa_kv_heads},
        {"activation_rank", activation_rank},
        {"delta_heads", delta_heads}, {"delta_max_scratch_bytes", delta_max_scratch_bytes}};
    for (const auto& [name, value] : positive) {
        if (value <= 0) throw std::invalid_argument(std::string(name) + " must be positive");
    }
    if (population.empty()) throw std::invalid_argument("population cannot be empty");
    if (top_k > static_cast<std::int64_t>(population.size())) {
        throw std::invalid_argument("top_k cannot exceed the N1 population");
    }
    if (n1_depth < 2) throw std::invalid_argument("n1_depth must be at least two");
    if (latent_dim % attention_heads != 0 || latent_dim % integrator_heads != 0) {
        throw std::invalid_argument("latent_dim must be divisible by attention and Integrator head counts");
    }
    if (latent_dim % gqa_query_heads != 0 || gqa_query_heads % gqa_kv_heads != 0) {
        throw std::invalid_argument("GQA requires latent_dim divisible by query heads and query heads divisible by KV heads");
    }
    if (rope_base <= 1.0) throw std::invalid_argument("rope_base must exceed one");
    if (routing_target_density <= 0.0 || routing_target_density > 1.0) {
        throw std::invalid_argument("routing_target_density must be in (0,1]");
    }
    if (routing_mu < 0.0 || routing_mu > 1.0 || routing_lambda_initial <= 0.0 ||
        routing_adaptation_rate <= 0.0 || routing_lambda_max <= 0.0 || routing_theta < 0.0) {
        throw std::invalid_argument("invalid routing-free controller configuration");
    }
    if (n1_mode == N1Mode::routing_free_collective && routing_granularity != RoutingGranularity::chunk) {
        throw std::invalid_argument("heterogeneous routing-free N1 currently requires chunk granularity");
    }
    for (const auto value : {module_hidden_dim, state_space_dim, recurrent_dim, delta_internal_dim, delta_ffn_dim}) {
        if (value < 0) throw std::invalid_argument("optional widths cannot be negative");
    }
    if (resolved_delta_internal_dim() % delta_heads != 0) {
        throw std::invalid_argument("delta_internal_dim must be divisible by delta_heads");
    }
    if (resolved_delta_internal_dim() / delta_heads > 64) {
        throw std::invalid_argument("Delta head dimension exceeds the CUDA kernel limit of 64");
    }
    if (chunk_size != 16 && chunk_size != 32 && chunk_size != 64 &&
        std::find(population.begin(), population.end(), N1Family::delta) != population.end()) {
        throw std::invalid_argument("Delta chunk_size must be 16, 32, or 64");
    }
}

std::int64_t TrainingConfig::planned_steps() const {
    if (token_budget <= 0) return steps;
    const auto per_step = batch_size * sequence_length * gradient_accumulation_steps;
    return (token_budget + per_step - 1) / per_step;
}

void TrainingConfig::validate(const ModelConfig& model) const {
    model.validate();
    if (steps <= 0 && token_budget <= 0) throw std::invalid_argument("steps or token_budget must be positive");
    if (batch_size <= 0 || sequence_length <= 0 || gradient_accumulation_steps <= 0) {
        throw std::invalid_argument("batch, sequence and accumulation must be positive");
    }
    if (sequence_length > model.max_sequence_length) throw std::invalid_argument("training sequence exceeds model maximum");
    if (learning_rate <= 0.0 || weight_decay < 0.0 || gradient_clip_norm <= 0.0) {
        throw std::invalid_argument("invalid optimizer configuration");
    }
    if (!std::is_sorted(milestones.begin(), milestones.end()) ||
        std::adjacent_find(milestones.begin(), milestones.end()) != milestones.end()) {
        throw std::invalid_argument("milestones must be sorted and unique");
    }
}

void save_model_config(const ModelConfig& config, const std::filesystem::path& path) {
    config.validate();
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot write model config: " + path.string());
    stream << "format=rayvan-emc-config-v3\n"
           << "n1_mode=" << to_string(config.n1_mode) << '\n'
           << "latent_dim=" << config.latent_dim << '\n'
           << "vocab_size=" << config.vocab_size << '\n'
           << "max_sequence_length=" << config.max_sequence_length << '\n'
           << "attention_heads=" << config.attention_heads << '\n'
           << "integrator_heads=" << config.integrator_heads << '\n'
           << "module_hidden_dim=" << config.module_hidden_dim << '\n'
           << "state_space_dim=" << config.state_space_dim << '\n'
           << "state_space_kernel_size=" << config.state_space_kernel_size << '\n'
           << "recurrent_dim=" << config.recurrent_dim << '\n'
           << "delta_internal_dim=" << config.delta_internal_dim << '\n'
           << "delta_heads=" << config.delta_heads << '\n'
           << "delta_ffn_dim=" << config.delta_ffn_dim << '\n'
           << "delta_max_scratch_bytes=" << config.delta_max_scratch_bytes << '\n'
           << "chunk_size=" << config.chunk_size << '\n'
           << "shared_state_slots=" << config.shared_state_slots << '\n'
           << "n1_depth=" << config.n1_depth << '\n'
           << "top_k=" << config.top_k << '\n'
           << "gqa_query_heads=" << config.gqa_query_heads << '\n'
           << "gqa_kv_heads=" << config.gqa_kv_heads << '\n'
           << "rope_base=" << config.rope_base << '\n'
           << "routing_granularity=" << to_string(config.routing_granularity) << '\n'
           << "activation_rank=" << config.activation_rank << '\n'
           << "routing_target_density=" << config.routing_target_density << '\n'
           << "routing_mu=" << config.routing_mu << '\n'
           << "routing_lambda_initial=" << config.routing_lambda_initial << '\n'
           << "routing_adaptation_rate=" << config.routing_adaptation_rate << '\n'
           << "routing_lambda_max=" << config.routing_lambda_max << '\n'
           << "routing_theta=" << config.routing_theta << '\n'
           << "tie_embeddings=" << (config.tie_embeddings ? 1 : 0) << '\n'
           << "population=";
    for (std::size_t index = 0; index < config.population.size(); ++index) {
        if (index != 0) stream << ',';
        stream << to_string(config.population[index]);
    }
    stream << '\n';
}

ModelConfig load_model_config(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot read model config: " + path.string());
    std::unordered_map<std::string, std::string> fields;
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto separator = line.find('=');
        if (separator != std::string::npos) fields.emplace(line.substr(0, separator), line.substr(separator + 1));
    }
    const auto format = fields["format"];
    if (format != "rayvan-emc-config-v1" && format != "rayvan-emc-config-v2" &&
        format != "rayvan-emc-config-v3") {
        throw std::runtime_error("unsupported model config format");
    }
    ModelConfig config;
    if (format == "rayvan-emc-config-v3") config.n1_mode = n1_mode_from_string(fields.at("n1_mode"));
    config.latent_dim = parse_integer<std::int64_t>(fields, "latent_dim");
    config.vocab_size = parse_integer<std::int64_t>(fields, "vocab_size");
    config.max_sequence_length = parse_integer<std::int64_t>(fields, "max_sequence_length");
    config.attention_heads = parse_integer<std::int64_t>(fields, "attention_heads");
    config.integrator_heads = parse_integer<std::int64_t>(fields, "integrator_heads");
    config.module_hidden_dim = parse_integer<std::int64_t>(fields, "module_hidden_dim");
    config.state_space_dim = parse_integer<std::int64_t>(fields, "state_space_dim");
    config.state_space_kernel_size = parse_integer<std::int64_t>(fields, "state_space_kernel_size");
    config.recurrent_dim = parse_integer<std::int64_t>(fields, "recurrent_dim");
    if (format == "rayvan-emc-config-v2" || format == "rayvan-emc-config-v3") {
        config.delta_internal_dim = parse_integer<std::int64_t>(fields, "delta_internal_dim");
        config.delta_heads = parse_integer<std::int64_t>(fields, "delta_heads");
        config.delta_ffn_dim = parse_integer<std::int64_t>(fields, "delta_ffn_dim");
        config.delta_max_scratch_bytes = parse_integer<std::int64_t>(fields, "delta_max_scratch_bytes");
    }
    config.chunk_size = parse_integer<std::int64_t>(fields, "chunk_size");
    config.shared_state_slots = parse_integer<std::int64_t>(fields, "shared_state_slots");
    config.n1_depth = parse_integer<std::int64_t>(fields, "n1_depth");
    config.top_k = parse_integer<std::int64_t>(fields, "top_k");
    if (format == "rayvan-emc-config-v3") {
        config.gqa_query_heads = parse_integer<std::int64_t>(fields, "gqa_query_heads");
        config.gqa_kv_heads = parse_integer<std::int64_t>(fields, "gqa_kv_heads");
        config.rope_base = parse_double(fields, "rope_base");
        config.routing_granularity = routing_granularity_from_string(fields.at("routing_granularity"));
        config.activation_rank = parse_integer<std::int64_t>(fields, "activation_rank");
        config.routing_target_density = parse_double(fields, "routing_target_density");
        config.routing_mu = parse_double(fields, "routing_mu");
        config.routing_lambda_initial = parse_double(fields, "routing_lambda_initial");
        config.routing_adaptation_rate = parse_double(fields, "routing_adaptation_rate");
        config.routing_lambda_max = parse_double(fields, "routing_lambda_max");
        config.routing_theta = parse_double(fields, "routing_theta");
    }
    config.tie_embeddings = parse_integer<int>(fields, "tie_embeddings") != 0;
    config.population.clear();
    for (const auto& family : split(fields.at("population"), ',')) config.population.push_back(n1_family_from_string(family));
    config.validate();
    return config;
}

}  // namespace rayvan::emc
