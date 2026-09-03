#include "rayvan_emc/diagnostics/diagnostics.hpp"
#include "rayvan_emc/model.hpp"
#include "rayvan_emc/n1/delta_core.hpp"
#include "rayvan_emc/n1/n1.hpp"
#include "rayvan_emc/n2/integrator.hpp"
#include "rayvan_emc/n2/nexus.hpp"
#include "rayvan_emc/training/foreach_adamw.hpp"

#include <ATen/autocast_mode.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cupti.h>
#include <cupti_driver_cbid.h>
#include <cuda_runtime.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace emc = rayvan::emc;

namespace {

std::atomic<std::uint64_t> launch_count{0};

void CUPTIAPI launch_callback(
    void*, CUpti_CallbackDomain domain, CUpti_CallbackId callback_id, const void* data) {
    if (domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
    const auto* callback = static_cast<const CUpti_CallbackData*>(data);
    if (callback->callbackSite != CUPTI_API_ENTER) return;
    if (callback_id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
        callback_id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz ||
        callback_id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx ||
        callback_id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz) {
        ++launch_count;
    }
}

void check_cupti(CUptiResult result, const char* operation) {
    if (result == CUPTI_SUCCESS) return;
    const char* message = nullptr;
    cuptiGetResultString(result, &message);
    throw std::runtime_error(std::string(operation) + " failed: " + (message ? message : "unknown"));
}

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(result));
    }
}

struct Measurement {
    double wall_ms = 0.0;
    double cuda_ms = 0.0;
    double launches = 0.0;
    std::uint64_t peak_temporary_bytes = 0;
};

template <typename Function>
Measurement measure(int iterations, Function&& function) {
    for (int index = 0; index < 3; ++index) function();
    torch::cuda::synchronize(0);
    constexpr auto aggregate = static_cast<std::size_t>(c10::CachingAllocator::StatType::AGGREGATE);
    const auto allocated_before = c10::cuda::CUDACachingAllocator::getDeviceStats(0)
                                      .allocated_bytes[aggregate].current;
    c10::cuda::CUDACachingAllocator::resetPeakStats(0);

    CUpti_SubscriberHandle subscriber{};
    check_cupti(cuptiSubscribe(&subscriber, launch_callback, nullptr), "cuptiSubscribe");
    check_cupti(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API), "cuptiEnableDomain");
    launch_count = 0;
    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    const auto wall_start = std::chrono::steady_clock::now();
    for (int index = 0; index < iterations; ++index) function();
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    const auto wall_stop = std::chrono::steady_clock::now();
    float elapsed = 0.0F;
    check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    check_cupti(cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_DRIVER_API), "cuptiDisableDomain");
    check_cupti(cuptiUnsubscribe(subscriber), "cuptiUnsubscribe");
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    const auto peak = stats.allocated_bytes[aggregate].peak;
    return {
        std::chrono::duration<double, std::milli>(wall_stop - wall_start).count() / iterations,
        static_cast<double>(elapsed) / iterations,
        static_cast<double>(launch_count.load()) / iterations,
        static_cast<std::uint64_t>(peak > allocated_before ? peak - allocated_before : 0)};
}

void clear_grad(const torch::Tensor& tensor) {
    tensor.mutable_grad() = torch::Tensor();
}

emc::ModelConfig research_config(std::int64_t chunk_size) {
    emc::ModelConfig config;
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
    config.chunk_size = chunk_size;
    config.shared_state_slots = 4;
    config.n1_depth = 3;
    config.top_k = 2;
    config.tie_embeddings = true;
    config.population = {
        emc::N1Family::gpt, emc::N1Family::ssm,
        emc::N1Family::recurrent, emc::N1Family::delta};
    config.validate();
    return config;
}

torch::Tensor language_loss(const torch::Tensor& logits, const torch::Tensor& targets) {
    return torch::nn::functional::cross_entropy(
        logits.reshape({-1, logits.size(-1)}), targets.reshape({-1}));
}

void print_measurement(const std::string& name, const Measurement& value) {
    std::cout << "  \"" << name << "\": {\"wall_ms\": " << value.wall_ms
              << ", \"cuda_ms\": " << value.cuda_ms
              << ", \"launches\": " << value.launches
              << ", \"peak_temporary_bytes\": " << value.peak_temporary_bytes << "},\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        int iterations = 10;
        bool bf16 = true;
        bool core_only = false;
        for (int index = 1; index < argc; ++index) {
            const std::string argument(argv[index]);
            if (argument == "--fp32") bf16 = false;
            else if (argument == "--core-only") core_only = true;
            else if (argument == "--iterations" && index + 1 < argc) iterations = std::stoi(argv[++index]);
            else throw std::invalid_argument("unknown option: " + argument);
        }
        if (iterations <= 0 || !torch::cuda::is_available()) {
            throw std::runtime_error("positive iterations and CUDA are required");
        }
        torch::manual_seed(20260902);
        torch::set_num_threads(1);
        const torch::Device device(torch::kCUDA, 0);
        const auto vector_dtype = bf16 ? torch::kBFloat16 : torch::kFloat32;
        at::autocast::set_autocast_dtype(at::kCUDA, torch::kBFloat16);
        at::autocast::set_autocast_enabled(at::kCUDA, bf16);

        constexpr std::int64_t batch = 4;
        constexpr std::int64_t sequence = 256;
        constexpr std::int64_t heads = 8;
        constexpr std::int64_t dimension = 64;
        const auto vector_options = torch::TensorOptions().device(device).dtype(vector_dtype);
        const auto float_options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
        auto query = torch::nn::functional::normalize(
            torch::randn({batch, sequence, heads, dimension}, vector_options),
            torch::nn::functional::NormalizeFuncOptions().dim(-1)).to(vector_dtype).requires_grad_(true);
        auto key = torch::nn::functional::normalize(
            torch::randn({batch, sequence, heads, dimension}, vector_options),
            torch::nn::functional::NormalizeFuncOptions().dim(-1)).to(vector_dtype).requires_grad_(true);
        auto value = torch::tanh(torch::randn({batch, sequence, heads, dimension}, vector_options))
                         .to(vector_dtype).requires_grad_(true);
        auto alpha = torch::sigmoid(torch::randn({batch, sequence, heads}, float_options)).requires_grad_(true);
        auto beta = torch::sigmoid(torch::randn({batch, sequence, heads}, float_options)).requires_grad_(true);
        auto initial = (torch::randn({batch, heads, dimension, dimension}, float_options) * 0.01).requires_grad_(true);
        const auto grad_output = torch::randn({batch, sequence, heads, dimension}, float_options);
        const auto grad_final = torch::randn({batch, heads, dimension, dimension}, float_options);

        std::cout << std::fixed << std::setprecision(4) << "{\n"
                  << "  \"device\": \"RTX 5070 / CUDA\",\n"
                  << "  \"precision\": \"" << (bf16 ? "bf16_vectors_fp32_state" : "fp32") << "\",\n";
        for (const auto chunk : {16LL, 32LL, 64LL}) {
            const auto forward = measure(iterations, [&] {
                torch::NoGradGuard no_grad;
                (void)emc::delta_rule(query, key, value, alpha, beta, initial, chunk, 64 * 1024 * 1024);
            });
            const auto backward = measure(iterations, [&] {
                clear_grad(query); clear_grad(key); clear_grad(value);
                clear_grad(alpha); clear_grad(beta); clear_grad(initial);
                const auto result = emc::delta_rule(query, key, value, alpha, beta, initial, chunk, 64 * 1024 * 1024);
                torch::autograd::backward(
                    {result.first, result.second}, {grad_output, grad_final});
            });
            print_measurement("core_chunk_" + std::to_string(chunk) + "_forward", forward);
            print_measurement("core_chunk_" + std::to_string(chunk) + "_forward_backward", backward);
            std::cout << "  \"core_chunk_" << chunk << "_scratch_bytes\": "
                      << emc::delta_backward_scratch_bytes(batch, heads, chunk, dimension) << ",\n";
        }

        if (!core_only) {
            const auto config = research_config(64);
            auto latent = torch::randn({batch, sequence, config.latent_dim}, float_options).to(vector_dtype).requires_grad_(true);
            for (std::int64_t family_id = 0; family_id < 3; ++family_id) {
                const auto family = config.population[static_cast<std::size_t>(family_id)];
                auto node = emc::create_n1_node(config, family, family_id, std::string(emc::to_string(family)));
                node->to(device);
                node->eval();
                const auto forward = measure(iterations, [&] {
                    torch::NoGradGuard no_grad;
                    (void)node->forward({latent}).proposal;
                });
                print_measurement("n1_" + std::string(emc::to_string(family)) + "_forward", forward);
            }
            for (const auto delta_chunk : {16LL, 32LL, 64LL}) {
                const auto delta_config = research_config(delta_chunk);
                auto node = emc::create_n1_node(delta_config, emc::N1Family::delta, 3, "delta");
                node->to(device);
                node->eval();
                const auto forward = measure(iterations, [&] {
                    torch::NoGradGuard no_grad;
                    (void)node->forward({latent}).proposal;
                });
                print_measurement("n1_delta_chunk_" + std::to_string(delta_chunk) + "_forward", forward);
                node->train();
                const auto forward_backward = measure(iterations, [&] {
                    node->zero_grad();
                    clear_grad(latent);
                    node->forward({latent}).proposal.square().mean().backward();
                });
                print_measurement(
                    "n1_delta_chunk_" + std::to_string(delta_chunk) + "_forward_backward",
                    forward_backward);
            }

            emc::Nexus nexus(config);
            nexus->to(device);
            const auto nexus_measurement = measure(iterations, [&] {
                torch::NoGradGuard no_grad;
                (void)nexus->forward(latent, 2);
            });
            print_measurement("nexus_forward", nexus_measurement);

            emc::N2Integrator integrator(config);
            integrator->to(device);
            const auto proposals = torch::randn({batch, sequence, 2, config.latent_dim}, vector_options);
            const auto weights = torch::softmax(torch::randn({batch, 2}, float_options), -1);
            const auto integrator_measurement = measure(iterations, [&] {
                torch::NoGradGuard no_grad;
                (void)integrator->forward(latent, proposals, weights, false).latent;
            });
            print_measurement("integrator_forward", integrator_measurement);

            emc::EMCModel model(config);
            model.to(device);
            auto tokens = torch::randint(config.vocab_size, {batch, sequence},
                torch::TensorOptions().device(device).dtype(torch::kLong));
            auto targets = torch::roll(tokens, {-1}, {1});
            model.eval();
            const auto natural = measure(iterations, [&] {
                torch::NoGradGuard no_grad;
                (void)model.forward({tokens}).logits;
            });
            print_measurement("mixed_natural_forward", natural);
            for (std::int64_t family_id = 0; family_id < 4; ++family_id) {
                const auto partner = family_id == 0 ? 1 : 0;
                auto forced = torch::empty({batch, 2}, torch::TensorOptions().device(device).dtype(torch::kLong));
                forced.select(1, 0).fill_(family_id);
                forced.select(1, 1).fill_(partner);
                emc::CausalIntervention intervention;
                intervention.forced_nodes = forced;
                const auto forced_measurement = measure(iterations, [&] {
                    torch::NoGradGuard no_grad;
                    (void)model.forward({tokens, false, intervention}).logits;
                });
                print_measurement("mixed_forced_" + std::string(emc::to_string(config.population[family_id])) + "_forward", forced_measurement);
            }

            model.train();
            emc::ForeachAdamW optimizer(model.parameters(), torch::optim::AdamWOptions(3e-4).weight_decay(0.01));
            const auto forward_backward = measure(iterations, [&] {
                optimizer.zero_grad();
                language_loss(model.forward({tokens}).logits, targets).backward();
            });
            print_measurement("mixed_natural_forward_backward", forward_backward);
            const auto optimizer_measurement = measure(iterations, [&] { optimizer.step(); });
            print_measurement("foreach_adamw", optimizer_measurement);
            const auto train = measure(iterations, [&] {
                optimizer.zero_grad();
                language_loss(model.forward({tokens}).logits, targets).backward();
                optimizer.step();
            });
            print_measurement("mixed_train_step", train);
            std::cout << "  \"mixed_tokens_per_second\": " << (batch * sequence * 1000.0 / train.wall_ms) << ",\n";
            const auto memory = emc::collect_memory_report(model, &optimizer, 0);
            std::cout << "  \"parameter_bytes\": " << memory.parameter_bytes << ",\n"
                      << "  \"optimizer_bytes\": " << memory.optimizer_bytes << ",\n"
                      << "  \"cuda_allocated_bytes\": " << memory.cuda_allocated_bytes << ",\n"
                      << "  \"cuda_reserved_bytes\": " << memory.cuda_reserved_bytes << ",\n"
                      << "  \"cuda_peak_allocated_bytes\": " << memory.cuda_peak_allocated_bytes << ",\n"
                      << "  \"cuda_peak_reserved_bytes\": " << memory.cuda_peak_reserved_bytes << "\n";
        } else {
            std::cout << "  \"core_only\": true\n";
        }
        std::cout << "}\n";
        at::autocast::set_autocast_enabled(at::kCUDA, false);
        return 0;
    } catch (const std::exception& error) {
        at::autocast::set_autocast_enabled(at::kCUDA, false);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
