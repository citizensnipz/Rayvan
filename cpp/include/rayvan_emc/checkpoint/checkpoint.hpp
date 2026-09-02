#pragma once

#include "rayvan_emc/config.hpp"
#include "rayvan_emc/model.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace rayvan::emc {

inline constexpr std::int64_t checkpoint_format_version = 1;

struct CheckpointProgress {
    std::int64_t step = 0;
    std::int64_t tokens_processed = 0;
    double validation_loss = 0.0;
    double best_validation_loss = 0.0;
    std::uint64_t seed = 0;
    std::string precision = "fp32";
    std::optional<Tensor> cpu_rng_state;
    std::optional<Tensor> train_generator_state;
    std::optional<Tensor> evaluation_generator_state;
};

// A checkpoint is a directory: manifest.rayvan, model.pt and optional
// optimizer.pt. Evaluation loading never opens optimizer.pt.
void save_checkpoint(
    const std::filesystem::path& directory,
    const EMCModel& model,
    const torch::optim::Optimizer* optimizer,
    const CheckpointProgress& progress);
CheckpointProgress load_model_checkpoint(
    const std::filesystem::path& directory,
    EMCModel& model,
    const torch::Device& device = torch::kCPU);
CheckpointProgress load_training_checkpoint(
    const std::filesystem::path& directory,
    EMCModel& model,
    torch::optim::Optimizer& optimizer,
    const torch::Device& device);
std::filesystem::path milestone_checkpoint_path(
    const std::filesystem::path& root,
    std::int64_t tokens_processed);

}  // namespace rayvan::emc
