#include "rayvan_emc/model.hpp"
#include "rayvan_emc/training/foreach_adamw.hpp"

#include <cupti.h>
#include <cupti_driver_cbid.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

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
    throw std::runtime_error(std::string(operation) + " failed: " + (message ? message : "unknown CUPTI error"));
}

torch::Tensor bundle_tensor(const std::filesystem::path& path, const std::string& name) {
    auto module = torch::jit::load(path.string(), torch::kCPU);
    for (const auto& item : module.named_buffers(true)) {
        if (item.name == name) return item.value;
    }
    throw std::runtime_error("fixture tensor not found: " + name);
}

torch::Tensor loss_for(const torch::Tensor& logits, const torch::Tensor& targets) {
    return torch::nn::functional::cross_entropy(
        logits.reshape({-1, logits.size(-1)}), targets.reshape({-1}));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            throw std::invalid_argument(
                "usage: rayvan-emc-optimizer-profile <fixture-directory> [--stock]");
        }
        const std::filesystem::path fixture(argv[1]);
        const bool stock = argc > 2 && std::string(argv[2]) == "--stock";
        const torch::Device device(torch::kCUDA, 0);
        if (!torch::cuda::is_available()) throw std::runtime_error("CUDA is unavailable");

        torch::manual_seed(20260902);
        emc::EMCModel model(emc::load_model_config(fixture / "model.rvcfg"));
        model.import_python_weights(fixture / "weights.pt");
        model.to(device);
        model.train();
        auto tokens = bundle_tensor(fixture / "forward.pt", "tokens").to(device, torch::kLong);
        auto targets = bundle_tensor(fixture / "forward.pt", "targets").to(device, torch::kLong);

        std::unique_ptr<torch::optim::AdamW> optimizer;
        auto options = torch::optim::AdamWOptions(3e-4).weight_decay(0.01);
        if (stock) {
            optimizer = std::make_unique<torch::optim::AdamW>(model.parameters(), options);
        } else {
            optimizer = std::make_unique<emc::ForeachAdamW>(model.parameters(), options);
        }

        const auto forward_backward = [&] {
            optimizer->zero_grad();
            loss_for(model.forward({tokens}).logits, targets).backward();
        };
        for (int index = 0; index < 6; ++index) {
            forward_backward();
            optimizer->step();
        }
        forward_backward();
        torch::cuda::synchronize(0);

        constexpr auto aggregate = static_cast<std::size_t>(c10::CachingAllocator::StatType::AGGREGATE);
        const auto allocated_before = c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes[aggregate].current;
        c10::cuda::CUDACachingAllocator::resetPeakStats(0);

        CUpti_SubscriberHandle subscriber{};
        check_cupti(cuptiSubscribe(&subscriber, launch_callback, nullptr), "cuptiSubscribe");
        check_cupti(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API), "cuptiEnableDomain");
        launch_count = 0;
        cudaEvent_t start{};
        cudaEvent_t stop{};
        if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
            throw std::runtime_error("CUDA event creation failed");
        }
        cudaEventRecord(start);
        const auto wall_start = std::chrono::steady_clock::now();
        optimizer->step();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        const auto wall_stop = std::chrono::steady_clock::now();
        float cuda_ms = 0.0F;
        cudaEventElapsedTime(&cuda_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        check_cupti(cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_DRIVER_API), "cuptiDisableDomain");
        check_cupti(cuptiUnsubscribe(subscriber), "cuptiUnsubscribe");
        const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
        const auto temporary_bytes = stats.allocated_bytes[aggregate].peak > allocated_before
            ? stats.allocated_bytes[aggregate].peak - allocated_before
            : 0;
        const auto wall_ms = std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
        std::cout << "profiled_optimizer=" << (stock ? "stock" : "foreach")
                  << " parameters=" << model.parameters().size()
                  << " cuda_launches=" << launch_count.load()
                  << " cuda_execution_ms=" << cuda_ms
                  << " wall_ms=" << wall_ms
                  << " peak_temporary_cuda_bytes=" << temporary_bytes << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#include <atomic>
#include <chrono>
