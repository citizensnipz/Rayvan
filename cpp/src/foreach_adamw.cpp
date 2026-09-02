#include "rayvan_emc/training/foreach_adamw.hpp"

#include <ATen/ops/_foreach_add.h>
#include <ATen/ops/_foreach_addcdiv.h>
#include <ATen/ops/_foreach_addcmul.h>
#include <ATen/ops/_foreach_div.h>
#include <ATen/ops/_foreach_lerp.h>
#include <ATen/ops/_foreach_maximum.h>
#include <ATen/ops/_foreach_mul.h>
#include <ATen/ops/_foreach_sqrt.h>
#include <torch/torch.h>

#include <cmath>
#include <map>
#include <tuple>
#include <vector>

namespace rayvan::emc {
namespace {

using Tensor = torch::Tensor;

struct CompatibilityKey {
    c10::DeviceType device_type;
    c10::DeviceIndex device_index;
    at::ScalarType dtype;

    auto operator<=>(const CompatibilityKey&) const = default;
};

struct TensorBucket {
    std::vector<Tensor> parameters;
    std::vector<Tensor> gradients;
    std::vector<Tensor> exp_avgs;
    std::vector<Tensor> exp_avg_sqs;
    std::vector<Tensor> max_exp_avg_sqs;
    std::vector<at::Scalar> bias_correction2_sqrts;
    std::vector<at::Scalar> negative_step_sizes;
};

}  // namespace

torch::Tensor ForeachAdamW::step(LossClosure closure) {
    torch::NoGradGuard no_grad;
    Tensor loss;
    if (closure) {
        at::AutoGradMode enable_grad(true);
        loss = closure();
    }

    for (auto& group : param_groups_) {
        auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
        const auto [beta1, beta2] = options.betas();
        std::map<CompatibilityKey, TensorBucket> buckets;

        for (auto& parameter : group.params()) {
            if (!parameter.grad().defined()) continue;
            auto gradient = parameter.grad();
            TORCH_CHECK(!gradient.is_sparse(), "AdamW does not support sparse gradients");

            const auto identity = parameter.unsafeGetTensorImpl();
            auto iterator = state_.find(identity);
            if (iterator == state_.end()) {
                auto state = std::make_unique<torch::optim::AdamWParamState>();
                state->step(0);
                state->exp_avg(torch::zeros_like(parameter, torch::MemoryFormat::Preserve));
                state->exp_avg_sq(torch::zeros_like(parameter, torch::MemoryFormat::Preserve));
                if (options.amsgrad()) {
                    state->max_exp_avg_sq(torch::zeros_like(parameter, torch::MemoryFormat::Preserve));
                }
                iterator = state_.emplace(identity, std::move(state)).first;
            }

            auto& state = static_cast<torch::optim::AdamWParamState&>(*iterator->second);
            state.step(state.step() + 1);
            const double bias_correction1 = 1.0 - std::pow(beta1, state.step());
            const double bias_correction2 = 1.0 - std::pow(beta2, state.step());

            CompatibilityKey key{
                parameter.device().type(),
                parameter.device().has_index() ? parameter.device().index() : c10::DeviceIndex{-1},
                parameter.scalar_type()};
            auto& bucket = buckets[key];
            bucket.parameters.push_back(parameter);
            bucket.gradients.push_back(std::move(gradient));
            bucket.exp_avgs.push_back(state.exp_avg());
            bucket.exp_avg_sqs.push_back(state.exp_avg_sq());
            if (options.amsgrad()) bucket.max_exp_avg_sqs.push_back(state.max_exp_avg_sq());
            bucket.bias_correction2_sqrts.emplace_back(std::sqrt(bias_correction2));
            bucket.negative_step_sizes.emplace_back(-options.lr() / bias_correction1);
        }

        for (auto& [_, bucket] : buckets) {
            if (bucket.parameters.empty()) continue;
            if (options.weight_decay() != 0.0) {
                at::_foreach_mul_(bucket.parameters, 1.0 - options.lr() * options.weight_decay());
            }
            at::_foreach_lerp_(bucket.exp_avgs, bucket.gradients, 1.0 - beta1);
            at::_foreach_mul_(bucket.exp_avg_sqs, beta2);
            at::_foreach_addcmul_(
                bucket.exp_avg_sqs, bucket.gradients, bucket.gradients, 1.0 - beta2);

            std::vector<Tensor> denominator;
            if (options.amsgrad()) {
                at::_foreach_maximum_(bucket.max_exp_avg_sqs, bucket.exp_avg_sqs);
                denominator = at::_foreach_sqrt(bucket.max_exp_avg_sqs);
            } else {
                denominator = at::_foreach_sqrt(bucket.exp_avg_sqs);
            }
            at::_foreach_div_(denominator, bucket.bias_correction2_sqrts);
            at::_foreach_add_(denominator, options.eps());
            at::_foreach_addcdiv_(
                bucket.parameters,
                bucket.exp_avgs,
                denominator,
                bucket.negative_step_sizes);
        }
    }
    return loss;
}

}  // namespace rayvan::emc
