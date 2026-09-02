#pragma once

#include "rayvan_emc/config.hpp"
#include "rayvan_emc/tensor.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rayvan::emc {

struct LeaseState {
    std::unordered_map<std::string, Tensor> tensors;
};

struct N1PersistentState {
    std::vector<LeaseState> block_states;
    Tensor shared_state;
};

struct N1Input {
    Tensor shared_latent;               // [B,S,D]
    std::optional<Tensor> shared_state; // [B,M,D]
    std::shared_ptr<N1PersistentState> local_state;
    std::optional<Tensor> request_indices; // [B]
};

struct N1Diagnostics {
    std::int64_t node_id = 0;
    std::string node_name;
    N1Family family = N1Family::gpt;
    std::int64_t blocks_executed = 0;
    std::int64_t chunks_per_block = 0;
    std::int64_t block_invocations = 0;
    std::int64_t parameters = 0;
    std::int64_t approximate_flops = 0;
    std::int64_t output_latent_size = 0;
    bool stateful = false;
    std::int64_t state_resets = 0;
    std::optional<double> continuation_probability;
    std::optional<double> average_lease_length;
    std::optional<Tensor> state_change_magnitude;
};

struct N1Output {
    Tensor proposal; // [B,S,D]
    std::shared_ptr<N1PersistentState> local_state;
    N1Diagnostics diagnostics;
};

struct BlockOutput {
    Tensor token_proposal; // [B,C,D]
    Tensor state_proposal; // [B,M,D]
    LeaseState new_lease_state;
};

class N1Block : public torch::nn::Module {
public:
    N1Block(const ModelConfig& config, N1Family family);
    ~N1Block() override = default;

    [[nodiscard]] N1Family family() const noexcept { return family_; }
    virtual LeaseState begin_lease(const Tensor& shared_state) = 0;
    virtual BlockOutput forward_chunk(
        const Tensor& chunk_latent,
        const Tensor& shared_state,
        const LeaseState& lease_state) = 0;

protected:
    Tensor condition_chunk(const Tensor& chunk, const Tensor& shared_state);
    Tensor state_proposal(const Tensor& token_proposal, const Tensor& shared_state);

    ModelConfig config_;
    torch::nn::Linear shared_condition{nullptr};
    torch::nn::Linear state_output{nullptr};

private:
    N1Family family_;
};

class N1Node final : public torch::nn::Module {
public:
    N1Node(const ModelConfig& config, N1Family family, std::int64_t node_id, std::string node_name);
    N1Output forward(const N1Input& input);

    [[nodiscard]] N1Family family() const noexcept { return family_; }
    [[nodiscard]] const std::string& node_name() const noexcept { return node_name_; }
    [[nodiscard]] std::int64_t execution_count() const noexcept { return execution_count_; }
    [[nodiscard]] std::int64_t parameter_count() const noexcept { return parameter_count_; }
    [[nodiscard]] std::int64_t approximate_flops(std::int64_t sequence_length) const;

private:
    N1Diagnostics make_diagnostics(
        std::int64_t sequence_length,
        std::int64_t batch,
        const std::optional<Tensor>& state_change,
        bool state_reset) const;

    ModelConfig config_;
    N1Family family_;
    std::int64_t node_id_;
    std::string node_name_;
    torch::nn::ModuleList blocks;
    torch::nn::Linear state_initializer{nullptr};
    std::int64_t parameter_count_ = 0;
    std::int64_t execution_count_ = 0;
};

std::shared_ptr<N1Block> create_n1_block(const ModelConfig& config, N1Family family);
std::shared_ptr<N1Node> create_n1_node(
    const ModelConfig& config,
    N1Family family,
    std::int64_t node_id,
    std::string node_name);
std::vector<std::string> n1_node_names(const std::vector<N1Family>& families);

}  // namespace rayvan::emc
