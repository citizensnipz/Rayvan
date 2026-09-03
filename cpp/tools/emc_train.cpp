#include "rayvan_emc/model.hpp"
#include "rayvan_emc/training/dataset.hpp"
#include "rayvan_emc/training/trainer.hpp"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/autocast_mode.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace emc = rayvan::emc;

namespace {

emc::ModelConfig research_config(emc::N1Mode mode) {
    emc::ModelConfig config;
    config.n1_mode = mode;
    config.latent_dim = 256;
    config.vocab_size = 50'257;
    config.max_sequence_length = 256;
    config.attention_heads = 8;
    config.integrator_heads = 8;
    config.module_hidden_dim = 6'144;
    config.state_space_dim = 960;
    config.state_space_kernel_size = 4;
    config.recurrent_dim = 704;
    config.delta_internal_dim = 512;
    config.delta_heads = 8;
    config.delta_ffn_dim = 5'120;
    config.delta_max_scratch_bytes = 64 * 1024 * 1024;
    config.chunk_size = 64;
    config.shared_state_slots = 4;
    config.n1_depth = 2;
    config.top_k = 2;
    config.tie_embeddings = true;
    config.population = {
        emc::N1Family::gpt, emc::N1Family::ssm,
        emc::N1Family::recurrent, emc::N1Family::delta};
    return config;
}

double token_accuracy(
    emc::EMCModel& model,
    const emc::TokenStream& stream,
    std::int64_t batches,
    torch::Generator& generator,
    const torch::Device& device) {
    model.eval();
    torch::NoGradGuard no_grad;
    at::autocast::set_autocast_dtype(at::kCUDA, torch::kBFloat16);
    at::autocast::set_autocast_enabled(at::kCUDA, true);
    std::int64_t correct = 0;
    std::int64_t total = 0;
    for (std::int64_t index = 0; index < batches; ++index) {
        const auto windows = stream.sample_batch(4, 256, generator, device);
        const auto input = windows.index({torch::indexing::Slice(), torch::indexing::Slice(0, 256)});
        const auto target = windows.index({torch::indexing::Slice(), torch::indexing::Slice(1, 257)});
        const auto prediction = model.forward({input}).logits.argmax(-1);
        correct += (prediction == target).sum().item<std::int64_t>();
        total += target.numel();
    }
    at::autocast::set_autocast_enabled(at::kCUDA, false);
    return static_cast<double>(correct) / static_cast<double>(total);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            throw std::invalid_argument(
                "usage: rayvan-emc-train <train.rvtok> <validation.rvtok> <checkpoint-directory> [--tokens N] [--mode legacy|routing-free] [--vocab-size N]");
        }
        std::int64_t token_budget = 1'000'000;
        std::int64_t vocab_size = 50'257;
        auto mode = emc::N1Mode::routing_free_collective;
        for (int index = 4; index < argc; ++index) {
            const std::string argument(argv[index]);
            if (argument == "--tokens" && index + 1 < argc) token_budget = std::stoll(argv[++index]);
            else if (argument == "--mode" && index + 1 < argc) mode = emc::n1_mode_from_string(argv[++index]);
            else if (argument == "--vocab-size" && index + 1 < argc) vocab_size = std::stoll(argv[++index]);
            else throw std::invalid_argument("unknown option: " + argument);
        }
        if (!torch::cuda::is_available()) throw std::runtime_error("CUDA is unavailable");
        torch::manual_seed(42);
        torch::set_num_threads(1);
        const auto training = emc::TokenStream::load(argv[1]);
        const auto validation = emc::TokenStream::load(argv[2]);
        auto config = research_config(mode);
        config.vocab_size = vocab_size;
        emc::EMCModel model(config);
        emc::TrainingConfig training_config;
        training_config.token_budget = token_budget;
        training_config.batch_size = 4;
        training_config.sequence_length = 256;
        training_config.gradient_accumulation_steps = 1;
        training_config.learning_rate = 3e-4;
        training_config.weight_decay = 0.01;
        training_config.gradient_clip_norm = 1.0;
        training_config.evaluation_interval = token_budget <= 100'000 ? 10 : 250;
        training_config.evaluation_batches = 8;
        training_config.seed = 42;
        training_config.precision = emc::Precision::bf16;
        training_config.milestones = {100'000, 250'000, 500'000, 750'000, 1'000'000};
        const torch::Device device(torch::kCUDA, 0);
        emc::Trainer trainer(model, training_config, device);
        const auto result = trainer.train(training, validation, argv[3]);
        auto evaluation_generator = torch::make_generator<at::CPUGeneratorImpl>(43);
        const auto accuracy = token_accuracy(model, validation, 16, evaluation_generator, device);
        std::cout << std::fixed << std::setprecision(6) << "{\n"
                  << "  \"mode\": \"" << emc::to_string(mode) << "\",\n"
                  << "  \"steps\": " << result.steps_completed << ",\n"
                  << "  \"tokens\": " << result.tokens_processed << ",\n"
                  << "  \"training_loss\": " << result.final_training_loss << ",\n"
                  << "  \"validation_loss\": " << result.final_validation_loss << ",\n"
                  << "  \"best_validation_loss\": " << result.best_validation_loss << ",\n"
                  << "  \"validation_perplexity\": " << std::exp(std::min(result.final_validation_loss, 20.0)) << ",\n"
                  << "  \"validation_token_accuracy\": " << accuracy << "\n"
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        at::autocast::set_autocast_enabled(at::kCUDA, false);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
