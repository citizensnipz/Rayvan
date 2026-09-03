#include "rayvan_emc/training/trainer.hpp"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/autocast_mode.h>
#include <torch/nn/utils/clip_grad.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace rayvan::emc {
namespace {

class AutocastScope final {
public:
    AutocastScope(const torch::Device& device, Precision precision)
        : device_type_(device.type()), previous_enabled_(at::autocast::is_autocast_enabled(device_type_)),
          previous_dtype_(at::autocast::get_autocast_dtype(device_type_)) {
        const bool enabled = device.is_cuda() && precision == Precision::bf16;
        if (enabled) at::autocast::set_autocast_dtype(device_type_, torch::kBFloat16);
        at::autocast::set_autocast_enabled(device_type_, enabled);
    }
    ~AutocastScope() {
        at::autocast::set_autocast_enabled(device_type_, previous_enabled_);
        at::autocast::set_autocast_dtype(device_type_, previous_dtype_);
    }

private:
    c10::DeviceType device_type_;
    bool previous_enabled_;
    at::ScalarType previous_dtype_;
};

double update_norm(const std::vector<Tensor>& before, const std::vector<Tensor>& after) {
    auto total = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat64));
    for (std::size_t index = 0; index < before.size(); ++index) {
        total += (after[index].detach().to(torch::kCPU, torch::kFloat64) - before[index]).pow(2).sum();
    }
    return std::sqrt(total.item<double>());
}

double parameter_norm(const std::vector<Tensor>& parameters) {
    auto total = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat64));
    for (const auto& parameter : parameters) total += parameter.detach().to(torch::kCPU, torch::kFloat64).pow(2).sum();
    return std::sqrt(total.item<double>());
}

double parameter_gradient_norm(const std::vector<Tensor>& parameters) {
    auto total = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat64));
    for (const auto& parameter : parameters) {
        if (parameter.grad().defined()) total += parameter.grad().detach().to(torch::kCPU, torch::kFloat64).pow(2).sum();
    }
    return std::sqrt(total.item<double>());
}

}  // namespace

Trainer::Trainer(EMCModel& model, TrainingConfig config, torch::Device device)
    : model_(model), config_(std::move(config)), device_(std::move(device)),
      optimizer_(
          model.parameters(),
          torch::optim::AdamWOptions(config_.learning_rate).weight_decay(config_.weight_decay)),
      train_generator_(torch::make_generator<at::CPUGeneratorImpl>(config_.seed)),
      evaluation_generator_(torch::make_generator<at::CPUGeneratorImpl>(config_.seed + 1)) {
    config_.validate(model.config());
    if (config_.precision == Precision::bf16 && !device_.is_cuda()) {
        throw std::invalid_argument("BF16 training requires a CUDA device in this port");
    }
    model_.to(device_);
}

CheckpointProgress Trainer::resume(
    const std::filesystem::path& checkpoint) {
    auto progress = load_training_checkpoint(
        checkpoint, model_, optimizer_, device_);
    if (progress.cpu_rng_state) {
        auto cpu_generator =
            at::detail::getDefaultCPUGenerator();
        cpu_generator.set_state(*progress.cpu_rng_state);
    }
    if (progress.train_generator_state) {
        train_generator_.set_state(*progress.train_generator_state);
    }
    if (progress.evaluation_generator_state) {
        evaluation_generator_.set_state(
            *progress.evaluation_generator_state);
    }
    resumed_step_ = progress.step;
    resumed_tokens_ = progress.tokens_processed;
    resumed_best_validation_ = progress.best_validation_loss;
    return progress;
}

Tensor Trainer::next_token_loss(const Tensor& logits, const Tensor& targets) const {
    return torch::nn::functional::cross_entropy(
        logits.reshape({-1, logits.size(-1)}),
        targets.reshape({-1}));
}

std::pair<double, double> Trainer::evaluate(const TokenStream& stream, std::int64_t batches) {
    if (batches <= 0) throw std::invalid_argument("evaluation batches must be positive");
    model_.eval();
    torch::NoGradGuard no_grad;
    double loss_sum = 0.0;
    for (std::int64_t batch = 0; batch < batches; ++batch) {
        const auto windows = stream.sample_batch(config_.batch_size, config_.sequence_length, evaluation_generator_, device_);
        const auto input = windows.index({torch::indexing::Slice(), torch::indexing::Slice(0, config_.sequence_length)});
        const auto targets = windows.index({torch::indexing::Slice(), torch::indexing::Slice(1, config_.sequence_length + 1)});
        AutocastScope autocast(device_, config_.precision);
        loss_sum += next_token_loss(model_.forward({input}).logits, targets).to(torch::kFloat32).item<double>();
    }
    const auto mean = loss_sum / batches;
    return {mean, std::exp(std::min(mean, 20.0))};
}

TrainingResult Trainer::train(
    const TokenStream& training,
    const TokenStream& validation,
    const std::filesystem::path& checkpoint_directory) {
    if (!checkpoint_directory.empty()) {
        std::filesystem::create_directories(checkpoint_directory);
    }
    const auto started = std::chrono::steady_clock::now();
    const auto planned_steps = config_.planned_steps();
    const auto tokens_per_step = config_.batch_size * config_.sequence_length * config_.gradient_accumulation_steps;
    std::int64_t tokens_processed = resumed_tokens_;
    std::size_t next_milestone = 0;
    while (
        next_milestone < config_.milestones.size() &&
        config_.milestones[next_milestone] <= tokens_processed) {
        ++next_milestone;
    }
    double best_validation = resumed_best_validation_;
    double last_training_loss = 0.0;
    double last_validation_loss = std::numeric_limits<double>::quiet_NaN();
    TrainingResult result;

    for (
        std::int64_t step = resumed_step_ + 1;
        step <= planned_steps;
        ++step) {
        model_.train();
        optimizer_.zero_grad();
        double accumulated_loss = 0.0;
        for (std::int64_t accumulation = 0; accumulation < config_.gradient_accumulation_steps; ++accumulation) {
            const auto windows = training.sample_batch(config_.batch_size, config_.sequence_length, train_generator_, device_);
            const auto input = windows.index({torch::indexing::Slice(), torch::indexing::Slice(0, config_.sequence_length)});
            const auto targets = windows.index({torch::indexing::Slice(), torch::indexing::Slice(1, config_.sequence_length + 1)});
            Tensor loss;
            {
                AutocastScope autocast(device_, config_.precision);
                const auto output = model_.forward({input});
                loss = next_token_loss(output.logits, targets) + output.routing_aux_loss;
            }
            accumulated_loss += loss.detach().to(torch::kFloat32).item<double>();
            (loss / static_cast<double>(config_.gradient_accumulation_steps)).backward();
        }
        const auto gradient_norm = torch::nn::utils::clip_grad_norm_(
            model_.parameters(), config_.gradient_clip_norm);
        const bool evaluate_now = step == 1 || step == planned_steps || step % config_.evaluation_interval == 0;
        std::vector<Tensor> before;
        std::vector<std::vector<Tensor>> expert_before;
        if (evaluate_now) {
            before.reserve(model_.parameters().size());
            for (const auto& parameter : model_.parameters()) before.push_back(parameter.detach().to(torch::kCPU).clone());
            if (model_.config().n1_mode == N1Mode::routing_free_collective) {
                for (const auto& expert : model_.module()->routing_free_collective()->experts()) {
                    std::vector<Tensor> snapshot;
                    for (const auto& parameter : expert->parameters()) snapshot.push_back(parameter.detach().to(torch::kCPU).clone());
                    expert_before.push_back(std::move(snapshot));
                }
            }
        }
        optimizer_.step();
        tokens_processed += tokens_per_step;
        last_training_loss = accumulated_loss / config_.gradient_accumulation_steps;

        const bool milestone_due = next_milestone < config_.milestones.size() && tokens_processed >= config_.milestones[next_milestone];
        if (evaluate_now || milestone_due) {
            auto [validation_loss, validation_perplexity] = evaluate(validation, config_.evaluation_batches);
            last_validation_loss = validation_loss;
            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            const auto current = model_.parameters();
            TrainingMetrics metrics;
            metrics.step = step;
            metrics.tokens_processed = tokens_processed;
            metrics.training_loss = last_training_loss;
            metrics.validation_loss = validation_loss;
            metrics.validation_perplexity = validation_perplexity;
            metrics.gradient_norm = gradient_norm;
            metrics.update_norm = before.empty() ? 0.0 : update_norm(before, current);
            metrics.tokens_per_second = tokens_processed / std::max(elapsed, 1e-9);
            metrics.elapsed_seconds = elapsed;
            metrics.memory = collect_memory_report(model_, &optimizer_, device_.is_cuda() ? device_.index() : 0);
            result.history.push_back(metrics);

            if (!checkpoint_directory.empty()) {
                DiagnosticAccumulator accumulator(
                    static_cast<std::int64_t>(
                        model_.config().population.size()));
                {
                    torch::NoGradGuard no_grad;
                    const auto windows = validation.sample_batch(
                        config_.batch_size,
                        config_.sequence_length,
                        evaluation_generator_,
                        device_);
                    const auto input = windows.index(
                        {torch::indexing::Slice(),
                         torch::indexing::Slice(
                             0, config_.sequence_length)});
                    AutocastScope autocast(device_, config_.precision);
                    accumulator.update(
                        model_.forward({input, true}));
                }
                TelemetryRecord telemetry;
                telemetry.step = step;
                telemetry.tokens_processed = tokens_processed;
                telemetry.loss = validation_loss;
                telemetry.perplexity = validation_perplexity;
                telemetry.tokens_per_second =
                    metrics.tokens_per_second;
                telemetry.wall_seconds = elapsed;
                telemetry.parameter_norm =
                    global_parameter_norm(model_);
                telemetry.gradient_norm = gradient_norm;
                telemetry.update_norm = metrics.update_norm;
                if (model_.config().n1_mode == N1Mode::routing_free_collective) {
                    telemetry.routing_free = accumulator.routing_free_report();
                    const auto& experts = model_.module()->routing_free_collective()->experts();
                    for (std::size_t index = 0; index < experts.size(); ++index) {
                        const auto parameters = experts[index]->parameters();
                        telemetry.routing_free->parameter_norm.push_back(parameter_norm(parameters));
                        telemetry.routing_free->gradient_norm.push_back(parameter_gradient_norm(parameters));
                        telemetry.routing_free->update_norm.push_back(
                            index < expert_before.size() ? update_norm(expert_before[index], parameters) : 0.0);
                    }
                } else {
                    telemetry.routing = accumulator.routing_report();
                    telemetry.integrator = accumulator.integrator_report();
                }
                telemetry.memory = metrics.memory;
                append_telemetry(
                    checkpoint_directory / "telemetry.tsv",
                    telemetry);
                CheckpointProgress progress{
                    step,
                    tokens_processed,
                    validation_loss,
                    std::min(best_validation, validation_loss),
                    config_.seed,
                    std::string(to_string(config_.precision))};
                progress.cpu_rng_state =
                    at::detail::getDefaultCPUGenerator().get_state();
                progress.train_generator_state =
                    train_generator_.get_state();
                progress.evaluation_generator_state =
                    evaluation_generator_.get_state();
                save_checkpoint(checkpoint_directory / "latest", model_, &optimizer_, progress);
                if (validation_loss < best_validation) save_checkpoint(checkpoint_directory / "best", model_, &optimizer_, progress);
                while (next_milestone < config_.milestones.size() && tokens_processed >= config_.milestones[next_milestone]) {
                    save_checkpoint(
                        milestone_checkpoint_path(checkpoint_directory, config_.milestones[next_milestone]),
                        model_, &optimizer_, progress);
                    ++next_milestone;
                }
            } else {
                while (next_milestone < config_.milestones.size() && tokens_processed >= config_.milestones[next_milestone]) ++next_milestone;
            }
            best_validation = std::min(best_validation, validation_loss);
        }
    }

    result.steps_completed = planned_steps;
    result.tokens_processed = tokens_processed;
    result.final_training_loss = last_training_loss;
    result.final_validation_loss = last_validation_loss;
    result.best_validation_loss = best_validation;
    return result;
}

}  // namespace rayvan::emc
