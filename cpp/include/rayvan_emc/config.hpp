#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace rayvan::emc {

enum class N1Family { gpt, ssm, recurrent, delta };
enum class N1Mode { legacy_nexus, routing_free_collective };
enum class RoutingGranularity { token, chunk, sequence, latent_cycle };
enum class Precision { fp32, bf16 };

std::string_view to_string(N1Family family) noexcept;
N1Family n1_family_from_string(std::string_view value);
std::string_view to_string(N1Mode mode) noexcept;
N1Mode n1_mode_from_string(std::string_view value);
std::string_view to_string(RoutingGranularity granularity) noexcept;
RoutingGranularity routing_granularity_from_string(std::string_view value);
std::string_view to_string(Precision precision) noexcept;
Precision precision_from_string(std::string_view value);

struct ModelConfig {
    N1Mode n1_mode = N1Mode::legacy_nexus;
    std::int64_t latent_dim = 32;
    std::int64_t vocab_size = 256;
    std::int64_t max_sequence_length = 128;
    std::int64_t attention_heads = 4;
    std::int64_t integrator_heads = 4;
    std::int64_t module_hidden_dim = 0;
    std::int64_t state_space_dim = 0;
    std::int64_t state_space_kernel_size = 4;
    std::int64_t recurrent_dim = 0;
    std::int64_t delta_internal_dim = 0;
    std::int64_t delta_heads = 4;
    std::int64_t delta_ffn_dim = 0;
    std::int64_t delta_max_scratch_bytes = 64 * 1024 * 1024;
    std::int64_t chunk_size = 64;
    std::int64_t shared_state_slots = 4;
    std::int64_t n1_depth = 3;
    std::int64_t top_k = 2;
    std::int64_t gqa_query_heads = 8;
    std::int64_t gqa_kv_heads = 2;
    double rope_base = 10'000.0;
    RoutingGranularity routing_granularity = RoutingGranularity::chunk;
    std::int64_t activation_rank = 16;
    double routing_target_density = 0.50;
    double routing_mu = 0.50;
    double routing_lambda_initial = 1e-10;
    double routing_adaptation_rate = 0.02;
    double routing_lambda_max = 1.0;
    double routing_theta = 1.0;
    bool tie_embeddings = true;
    std::vector<N1Family> population{
        N1Family::gpt, N1Family::ssm, N1Family::recurrent, N1Family::delta};

    [[nodiscard]] std::int64_t resolved_module_hidden_dim() const noexcept;
    [[nodiscard]] std::int64_t resolved_state_space_dim() const noexcept;
    [[nodiscard]] std::int64_t resolved_recurrent_dim() const noexcept;
    [[nodiscard]] std::int64_t resolved_delta_internal_dim() const noexcept;
    [[nodiscard]] std::int64_t resolved_delta_ffn_dim() const noexcept;
    void validate() const;
};

struct TrainingConfig {
    std::int64_t steps = 100;
    std::int64_t token_budget = 0;
    std::int64_t batch_size = 16;
    std::int64_t sequence_length = 32;
    std::int64_t gradient_accumulation_steps = 1;
    double learning_rate = 3e-3;
    double weight_decay = 0.01;
    double gradient_clip_norm = 1.0;
    std::int64_t evaluation_interval = 25;
    std::int64_t evaluation_batches = 4;
    std::uint64_t seed = 42;
    Precision precision = Precision::fp32;
    std::vector<std::int64_t> milestones{100'000, 250'000, 500'000, 750'000, 1'000'000};

    [[nodiscard]] std::int64_t planned_steps() const;
    void validate(const ModelConfig& model) const;
};

// Stable line-oriented manifests keep checkpoint metadata inspectable without
// introducing a JSON dependency into the runtime.
void save_model_config(const ModelConfig& config, const std::filesystem::path& path);
ModelConfig load_model_config(const std::filesystem::path& path);

}  // namespace rayvan::emc
