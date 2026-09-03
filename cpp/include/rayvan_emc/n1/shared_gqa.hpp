#pragma once

#include "rayvan_emc/config.hpp"
#include "rayvan_emc/tensor.hpp"

#include <torch/torch.h>

#include <cstdint>

namespace rayvan::emc {

class RMSNormImpl final : public torch::nn::Module {
public:
    explicit RMSNormImpl(std::int64_t width, double epsilon = 1e-6);
    Tensor forward(const Tensor& input) const;

private:
    double epsilon_;
    Tensor weight_;
};
TORCH_MODULE(RMSNorm);

class SharedCausalGQAImpl final : public torch::nn::Module {
public:
    explicit SharedCausalGQAImpl(const ModelConfig& config);
    Tensor forward(const Tensor& input);

    // Diagnostic-only. This may synchronize and must not be called in the hot path.
    [[nodiscard]] std::int64_t selected_backend(const Tensor& input);
    [[nodiscard]] std::int64_t query_heads() const noexcept { return query_heads_; }
    [[nodiscard]] std::int64_t kv_heads() const noexcept { return kv_heads_; }
    [[nodiscard]] std::int64_t head_dim() const noexcept { return head_dim_; }

private:
    std::tuple<Tensor, Tensor, Tensor> project_qkv(const Tensor& input);
    Tensor apply_rope(const Tensor& input) const;

    std::int64_t width_;
    std::int64_t query_heads_;
    std::int64_t kv_heads_;
    std::int64_t head_dim_;
    double rope_base_;
    RMSNorm input_norm{nullptr};
    torch::nn::Linear query_projection{nullptr};
    torch::nn::Linear key_projection{nullptr};
    torch::nn::Linear value_projection{nullptr};
    torch::nn::Linear output_projection{nullptr};
};
TORCH_MODULE(SharedCausalGQA);

}  // namespace rayvan::emc
