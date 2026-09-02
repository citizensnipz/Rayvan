#pragma once

#include "rayvan_emc/checkpoint/checkpoint.hpp"
#include "rayvan_emc/config.hpp"
#include "rayvan_emc/diagnostics/telemetry.hpp"
#include "rayvan_emc/model.hpp"
#include "rayvan_emc/training/dataset.hpp"
#include "rayvan_emc/training/foreach_adamw.hpp"

#include <filesystem>
#include <functional>
#include <limits>
#include <vector>

namespace rayvan::emc {

struct TrainingMetrics {
    std::int64_t step = 0;
    std::int64_t tokens_processed = 0;
    double training_loss = 0.0;
    double validation_loss = 0.0;
    double validation_perplexity = 0.0;
    double gradient_norm = 0.0;
    double update_norm = 0.0;
    double tokens_per_second = 0.0;
    double elapsed_seconds = 0.0;
    MemoryReport memory;
};

struct TrainingResult {
    std::vector<TrainingMetrics> history;
    std::int64_t steps_completed = 0;
    std::int64_t tokens_processed = 0;
    double final_training_loss = 0.0;
    double final_validation_loss = 0.0;
    double best_validation_loss = 0.0;
};

class Trainer final {
public:
    Trainer(EMCModel& model, TrainingConfig config, torch::Device device);
    TrainingResult train(
        const TokenStream& training,
        const TokenStream& validation,
        const std::filesystem::path& checkpoint_directory = {});
    CheckpointProgress resume(const std::filesystem::path& checkpoint);
    [[nodiscard]] std::pair<double, double> evaluate(
        const TokenStream& stream,
        std::int64_t batches);
    [[nodiscard]] ForeachAdamW& optimizer() noexcept { return optimizer_; }

private:
    Tensor next_token_loss(const Tensor& logits, const Tensor& targets) const;

    EMCModel& model_;
    TrainingConfig config_;
    torch::Device device_;
    ForeachAdamW optimizer_;
    torch::Generator train_generator_;
    torch::Generator evaluation_generator_;
    std::int64_t resumed_step_ = 0;
    std::int64_t resumed_tokens_ = 0;
    double resumed_best_validation_ =
        std::numeric_limits<double>::infinity();
};

}  // namespace rayvan::emc
