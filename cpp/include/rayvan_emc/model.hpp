#pragma once

#include "rayvan_emc/config.hpp"
#include "rayvan_emc/n1/n1.hpp"
#include "rayvan_emc/n1/routing_free_collective.hpp"
#include "rayvan_emc/n2/dispatch.hpp"
#include "rayvan_emc/n2/integrator.hpp"
#include "rayvan_emc/n2/nexus.hpp"
#include "rayvan_emc/tensor.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace rayvan::emc {

struct N2State {
    std::unordered_map<std::int64_t, std::shared_ptr<N1PersistentState>> local_states;
    std::shared_ptr<RoutingFreeCollectiveState> collective;
};

struct CausalIntervention {
    // False entries disable an N1 before top-K, causing natural next-best replacement.
    std::optional<Tensor> availability_mask; // [N], bool
    // Exact alternate IDs. Shape [B,K].
    std::optional<Tensor> forced_nodes;
    // Proposals from selected matching nodes are zeroed after execution.
    std::optional<Tensor> zero_proposal_mask; // [N], bool
    // Routing-free only: independently force the named experts on. [N] or [B,N].
    std::optional<Tensor> force_active_mask;
};

struct ExecutionTrace {
    Tensor selected_node_ids;       // [B,K]
    Tensor selected_node_weights;   // [B,K]
    Tensor pre_top_k_probabilities; // [B,N]
    Tensor selected_slots;          // [B,K]
    Tensor dispatch_permutation;    // [B*K]
    Tensor dispatch_inverse_permutation;
    Tensor dispatch_counts;         // [N]
    Tensor dispatch_offsets;        // [N+1]
    std::vector<std::int64_t> executed_node_ids;
    std::int64_t actual_node_executions = 0;
    std::int64_t theoretical_all_node_executions = 0;
    std::vector<N1Diagnostics> node_diagnostics;
};

struct EMCInput {
    Tensor token_ids; // [B,S], int64
    bool return_trace = false;
    std::optional<CausalIntervention> intervention;
    std::shared_ptr<N2State> state;
};

struct EMCOutput {
    Tensor logits;       // [B,S,V]
    Tensor embeddings;   // [B,S,D]
    Tensor shared_state; // [B,S,D], pre-N2 latent
    RoutingDecision routing;
    Tensor proposals;    // [B,S,K,D]
    Tensor integrated_state; // [B,S,D]
    std::optional<IntegratorTrace> integrator_trace;
    std::optional<ExecutionTrace> execution_trace;
    std::shared_ptr<N2State> state;
    Tensor routing_aux_loss; // scalar; zero for legacy and evaluation
    std::optional<RoutingFreeTrace> routing_free_trace;
};

class OutputProjectionImpl final : public torch::nn::Module {
public:
    OutputProjectionImpl(
        std::int64_t input_features,
        std::int64_t output_features,
        Tensor tied_weight = Tensor());
    Tensor forward(const Tensor& input) const;
    [[nodiscard]] const Tensor& weight_tensor() const noexcept { return weight_; }

private:
    Tensor weight_;
    Tensor bias_;
};
TORCH_MODULE(OutputProjection);

class EMCModelImpl final : public torch::nn::Module {
public:
    explicit EMCModelImpl(ModelConfig config);
    EMCOutput forward(const EMCInput& input);

    [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::int64_t active_top_k() const noexcept { return active_top_k_; }
    void set_active_top_k(std::int64_t value);
    [[nodiscard]] const std::vector<std::shared_ptr<N1Node>>& nodes() const noexcept { return node_handles_; }
    [[nodiscard]] RoutingFreeCollective& routing_free_collective() noexcept { return collective; }
    [[nodiscard]] const RoutingFreeCollective& routing_free_collective() const noexcept { return collective; }
    [[nodiscard]] bool embeddings_tied() const noexcept {
        return token_embedding->weight.unsafeGetTensorImpl() ==
            output_projection->weight_tensor().unsafeGetTensorImpl();
    }

private:
    RoutingDecision force_routing(const RoutingDecision& routing, const Tensor& forced, std::int64_t batch) const;
    std::tuple<Tensor, std::shared_ptr<N2State>, std::vector<N1Diagnostics>, std::vector<std::int64_t>, DispatchPlan>
    execute_selected_nodes(const Tensor& latent, const Tensor& selected_indices, const std::shared_ptr<N2State>& state);

    ModelConfig config_;
    std::int64_t active_top_k_;
    torch::nn::Embedding token_embedding{nullptr};
    torch::nn::Embedding position_embedding{nullptr};
    Nexus router{nullptr};
    torch::nn::ModuleList n1_nodes;
    std::vector<std::shared_ptr<N1Node>> node_handles_;
    N2Integrator integrator{nullptr};
    RoutingFreeCollective collective{nullptr};
    torch::nn::LayerNorm output_norm{nullptr};
    OutputProjection output_projection{nullptr};
};
using EMCModelModule = torch::nn::ModuleHolder<EMCModelImpl>;

// Owning API. PIMPL-style ownership keeps callers independent of module layout;
// tensor types remain ATen tensors inside the native library boundary.
class EMCModel final {
public:
    explicit EMCModel(ModelConfig config);
    ~EMCModel();
    EMCModel(EMCModel&&) noexcept;
    EMCModel& operator=(EMCModel&&) noexcept;
    EMCModel(const EMCModel&) = delete;
    EMCModel& operator=(const EMCModel&) = delete;

    EMCOutput forward(const EMCInput& input);
    void train();
    void eval();
    void to(const torch::Device& device);
    void save_weights(const std::filesystem::path& path) const;
    void load_weights(const std::filesystem::path& path, const torch::Device& device = torch::kCPU);
    void import_python_weights(const std::filesystem::path& torchscript_bundle);
    void set_active_top_k(std::int64_t value);
    [[nodiscard]] std::int64_t active_top_k() const;
    [[nodiscard]] const ModelConfig& config() const;
    [[nodiscard]] std::vector<Tensor> parameters(bool recurse = true) const;
    [[nodiscard]] bool embeddings_tied() const;
    [[nodiscard]] EMCModelModule module() const;

private:
    struct Storage;
    std::unique_ptr<Storage> storage_;
};

}  // namespace rayvan::emc
