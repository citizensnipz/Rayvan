#include "rayvan_emc/checkpoint/checkpoint.hpp"
#include "rayvan_emc/diagnostics/diagnostics.hpp"
#include "rayvan_emc/model.hpp"
#include "rayvan_emc/training/foreach_adamw.hpp"

#include <ATen/autocast_mode.h>
#include <cupti.h>
#include <cupti_driver_cbid.h>
#include <cuda_runtime.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
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
    throw std::runtime_error(std::string(operation) + ": " + (message ? message : "unknown"));
}

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(result));
}

template <typename Function>
double measure_ms(int iterations, Function&& function) {
    for (int index = 0; index < 2; ++index) function();
    torch::cuda::synchronize(0);
    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int index = 0; index < iterations; ++index) function();
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed = 0.0F;
    check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    return static_cast<double>(elapsed) / iterations;
}

template <typename Function>
double measure_launches(int iterations, Function&& function) {
    torch::cuda::synchronize(0);
    CUpti_SubscriberHandle subscriber{};
    check_cupti(cuptiSubscribe(&subscriber, launch_callback, nullptr), "cuptiSubscribe");
    check_cupti(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API), "cuptiEnableDomain");
    launch_count = 0;
    for (int index = 0; index < iterations; ++index) function();
    torch::cuda::synchronize(0);
    check_cupti(cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_DRIVER_API), "cuptiDisableDomain");
    check_cupti(cuptiUnsubscribe(subscriber), "cuptiUnsubscribe");
    return static_cast<double>(launch_count.load()) / iterations;
}

emc::ModelConfig research_config() {
    emc::ModelConfig config;
    config.n1_mode = emc::N1Mode::routing_free_collective;
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
    config.chunk_size = 64;
    config.shared_state_slots = 4;
    config.n1_depth = 2;
    return config;
}

torch::Tensor language_loss(const torch::Tensor& logits, const torch::Tensor& targets) {
    return torch::nn::functional::cross_entropy(
        logits.reshape({-1, logits.size(-1)}), targets.reshape({-1}));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (!torch::cuda::is_available()) throw std::runtime_error("CUDA is required");
        const int iterations = argc > 2 ? std::stoi(argv[2]) : 10;
        const std::filesystem::path checkpoint = argc > 1 ? argv[1] : std::filesystem::path();
        auto config = checkpoint.empty() ? research_config() : emc::load_model_config(checkpoint / "model.rvcfg");
        if (config.n1_mode != emc::N1Mode::routing_free_collective) {
            throw std::invalid_argument("routing-free checkpoint required");
        }
        torch::manual_seed(42);
        torch::set_num_threads(1);
        const torch::Device device(torch::kCUDA, 0);
        emc::EMCModel model(config);
        if (checkpoint.empty()) model.to(device);
        else (void)emc::load_model_checkpoint(checkpoint, model, device);
        model.eval();
        auto& collective = *model.module()->routing_free_collective();
        const auto float_options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
        const auto long_options = torch::TensorOptions().device(device).dtype(torch::kLong);
        const auto bool_options = torch::TensorOptions().device(device).dtype(torch::kBool);
        auto embeddings = torch::randn({4, 256, config.latent_dim}, float_options);
        auto latent = torch::randn({4, config.shared_state_slots, config.latent_dim}, float_options);
        auto proposals = torch::randn({4, 4, config.shared_state_slots, config.latent_dim}, float_options);
        auto active = torch::ones({4, 4}, bool_options);
        auto representation = torch::randn({4, config.competence_embedding_dim}, float_options);
        representation = representation / representation.norm(2, -1, true).clamp_min(1e-8);
        auto indices = torch::arange(4, long_options);
        auto tokens = torch::randint(config.vocab_size, {4, 256}, long_options);
        auto targets = torch::roll(tokens, {-1}, {1});

        at::autocast::set_autocast_dtype(at::kCUDA, torch::kBFloat16);
        at::autocast::set_autocast_enabled(at::kCUDA, true);
        torch::NoGradGuard no_grad;
        const auto shared_ms = measure_ms(iterations, [&] {
            (void)collective.shared_attention()->forward(embeddings);
        });
        const auto gate_ms = measure_ms(iterations, [&] {
            for (const auto& expert : collective.experts()) {
                (void)expert->match_competence(representation);
            }
        });
        std::vector<torch::Tensor> dispatch_masks;
        dispatch_masks.reserve(collective.experts().size());
        for (const auto& expert : collective.experts()) {
            dispatch_masks.push_back(expert->match_competence(representation).resistance < config.competence_rho);
        }
        const auto sparse_dispatch_ms = measure_ms(iterations, [&] {
            for (const auto& mask : dispatch_masks) (void)torch::nonzero(mask).reshape({-1});
        });
        std::vector<double> expert_ms;
        for (const auto& expert : collective.experts()) {
            auto state = expert->initialize_state(latent);
            emc::RoutingItem item{embeddings.index({torch::indexing::Slice(), torch::indexing::Slice(0, 64)}), latent, 0};
            expert_ms.push_back(measure_ms(iterations, [&] {
                (void)expert->forward_routing_item(item, indices, state);
            }));
        }
        const auto latent_ms = measure_ms(iterations, [&] {
            (void)collective.diagnostic_integrate_latent(latent, proposals, active);
        });
        const auto full_forward_ms = measure_ms(iterations, [&] {
            (void)model.forward({tokens}).logits;
        });
        const auto full_forward_launches = measure_launches(iterations, [&] {
            (void)model.forward({tokens}).logits;
        });
        const auto traced = model.forward({tokens, true});
        const auto backend = collective.shared_attention()->selected_backend(traced.embeddings);
        torch::AutoGradMode enable_grad(true);

        model.train();
        emc::ForeachAdamW optimizer(model.parameters(), torch::optim::AdamWOptions(3e-4).weight_decay(0.01));
        const auto forward_backward_ms = measure_ms(iterations, [&] {
            optimizer.zero_grad();
            const auto output = model.forward({tokens});
            (language_loss(output.logits, targets) + output.routing_aux_loss).backward();
        });
        const auto forward_backward_launches = measure_launches(iterations, [&] {
            optimizer.zero_grad();
            const auto output = model.forward({tokens});
            (language_loss(output.logits, targets) + output.routing_aux_loss).backward();
        });
        const auto optimizer_ms = measure_ms(iterations, [&] { optimizer.step(); });
        const auto optimizer_launches = measure_launches(iterations, [&] { optimizer.step(); });
        const auto memory = emc::collect_memory_report(model, &optimizer, 0);
        std::int64_t expert_parameter_count = 0;
        std::vector<std::int64_t> expert_parameter_counts;
        expert_parameter_counts.reserve(collective.experts().size());
        for (const auto& expert : collective.experts()) {
            const auto count = expert->parameter_count();
            expert_parameter_counts.push_back(count);
            expert_parameter_count += count;
        }
        const auto total_parameter_count = memory.parameter_bytes / 4;

        std::cout << std::fixed << std::setprecision(4) << "{\n"
                  << "  \"sdp_backend\": " << backend << ",\n"
                  << "  \"shared_gqa_ms\": " << shared_ms << ",\n"
                  << "  \"local_gates_ms\": " << gate_ms << ",\n";
        std::cout << "  \"sparse_dispatch_ms\": " << sparse_dispatch_ms << ",\n";
        for (std::size_t index = 0; index < expert_ms.size(); ++index) {
            std::cout << "  \"expert_" << emc::to_string(config.population[index])
                      << "_ms\": " << expert_ms[index] << ",\n";
        }
        std::cout << "  \"latent_integration_ms\": " << latent_ms << ",\n"
                  << "  \"sparse_full_forward_ms\": " << full_forward_ms << ",\n"
                  << "  \"sparse_full_forward_launches\": " << full_forward_launches << ",\n"
                  << "  \"forward_backward_ms\": " << forward_backward_ms << ",\n"
                  << "  \"forward_backward_launches\": " << forward_backward_launches << ",\n"
                  << "  \"optimizer_ms\": " << optimizer_ms << ",\n"
                  << "  \"optimizer_launches\": " << optimizer_launches << ",\n"
                  << "  \"tokens_per_second\": " << (4.0 * 256.0 * 1000.0 / (forward_backward_ms + optimizer_ms)) << ",\n"
                  << "  \"activation_density\": " << traced.routing_free_trace->activation_density.item<double>() << ",\n"
                  << "  \"parameter_count\": " << total_parameter_count << ",\n"
                  << "  \"shared_parameter_count\": " << total_parameter_count - expert_parameter_count << ",\n"
                  << "  \"expert_parameter_counts\": [";
        for (std::size_t index = 0; index < expert_parameter_counts.size(); ++index) {
            if (index != 0) std::cout << ", ";
            std::cout << expert_parameter_counts[index];
        }
        std::cout << "],\n"
                  << "  \"parameter_bytes\": " << memory.parameter_bytes << ",\n"
                  << "  \"optimizer_bytes\": " << memory.optimizer_bytes << ",\n"
                  << "  \"process_rss_bytes\": " << memory.process_rss_bytes << ",\n"
                  << "  \"process_peak_rss_bytes\": " << memory.process_peak_rss_bytes << ",\n"
                  << "  \"cuda_allocated_bytes\": " << memory.cuda_allocated_bytes << ",\n"
                  << "  \"cuda_reserved_bytes\": " << memory.cuda_reserved_bytes << ",\n"
                  << "  \"cuda_peak_allocated_bytes\": " << memory.cuda_peak_allocated_bytes << ",\n"
                  << "  \"cuda_peak_reserved_bytes\": " << memory.cuda_peak_reserved_bytes << "\n"
                  << "}\n";
        at::autocast::set_autocast_enabled(at::kCUDA, false);
        return 0;
    } catch (const std::exception& error) {
        at::autocast::set_autocast_enabled(at::kCUDA, false);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
