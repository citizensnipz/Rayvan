#include "rayvan_emc/diagnostics/diagnostics.hpp"
#include "rayvan_emc/model.hpp"
#include "rayvan_emc/training/foreach_adamw.hpp"

#include <ATen/autocast_mode.h>
#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/torch.h>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace emc = rayvan::emc;

namespace {

struct ProcessSample {
    double cpu_seconds = 0.0;
    std::uint64_t rss = 0;
    std::uint64_t peak_rss = 0;
};

ProcessSample process_sample() {
    ProcessSample result;
#if defined(_WIN32)
    FILETIME create{}, exit{}, kernel{}, user{};
    if (GetProcessTimes(GetCurrentProcess(), &create, &exit, &kernel, &user)) {
        ULARGE_INTEGER kernel_value{}, user_value{};
        kernel_value.LowPart = kernel.dwLowDateTime;
        kernel_value.HighPart = kernel.dwHighDateTime;
        user_value.LowPart = user.dwLowDateTime;
        user_value.HighPart = user.dwHighDateTime;
        result.cpu_seconds = static_cast<double>(kernel_value.QuadPart + user_value.QuadPart) / 10'000'000.0;
    }
    PROCESS_MEMORY_COUNTERS_EX counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters))) {
        result.rss = counters.WorkingSetSize;
        result.peak_rss = counters.PeakWorkingSetSize;
    }
#endif
    return result;
}

class Autocast final {
public:
    Autocast(const torch::Device& device, bool bf16) : device_(device.type()) {
        previous_ = at::autocast::is_autocast_enabled(device_);
        dtype_ = at::autocast::get_autocast_dtype(device_);
        if (bf16) at::autocast::set_autocast_dtype(device_, torch::kBFloat16);
        at::autocast::set_autocast_enabled(device_, bf16);
    }
    ~Autocast() {
        at::autocast::set_autocast_enabled(device_, previous_);
        at::autocast::set_autocast_dtype(device_, dtype_);
    }
private:
    c10::DeviceType device_;
    bool previous_;
    at::ScalarType dtype_;
};

void synchronize(const torch::Device& device) {
    if (device.is_cuda()) torch::cuda::synchronize(device.index());
}

torch::Tensor bundle_tensor(const std::filesystem::path& path, const std::string& encoded_name) {
    auto module = torch::jit::load(path.string(), torch::kCPU);
    for (const auto& item : module.named_buffers(true)) {
        if (item.name == encoded_name) return item.value;
    }
    throw std::runtime_error("fixture tensor not found: " + encoded_name);
}

torch::Tensor loss_for(const torch::Tensor& logits, const torch::Tensor& targets) {
    return torch::nn::functional::cross_entropy(logits.reshape({-1, logits.size(-1)}), targets.reshape({-1}));
}

template <typename Function>
double time_iterations(const torch::Device& device, int iterations, Function&& function) {
    synchronize(device);
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) function();
    synchronize(device);
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::invalid_argument("usage: rayvan-emc-benchmark <fixture-directory> [--cpu] [--bf16] [--realistic] [--warmup N] [--iterations N]");
        const std::filesystem::path fixture(argv[1]);
        bool cpu = false;
        bool bf16 = false;
        bool realistic = false;
        int warmup = 10;
        int iterations = 100;
        for (int index = 2; index < argc; ++index) {
            const std::string argument(argv[index]);
            if (argument == "--cpu") cpu = true;
            else if (argument == "--bf16") bf16 = true;
            else if (argument == "--realistic") realistic = true;
            else if (argument == "--warmup" && index + 1 < argc) warmup = std::stoi(argv[++index]);
            else if (argument == "--iterations" && index + 1 < argc) iterations = std::stoi(argv[++index]);
            else throw std::invalid_argument("unknown benchmark option: " + argument);
        }
        if (warmup < 0 || iterations <= 0) throw std::invalid_argument("benchmark counts must be non-negative warmup and positive iterations");
        const torch::Device device = cpu ? torch::Device(torch::kCPU) : torch::Device(torch::kCUDA, 0);
        if (device.is_cuda() && !torch::cuda::is_available()) throw std::runtime_error("CUDA is unavailable");
        if (bf16 && !device.is_cuda()) throw std::runtime_error("BF16 benchmark requires CUDA");
        torch::set_num_threads(1);
        torch::manual_seed(20260902);
        emc::ModelConfig model_config;
        if (realistic) {
            model_config.latent_dim = 64;
            model_config.vocab_size = 8192;
            model_config.max_sequence_length = 128;
            model_config.attention_heads = 4;
            model_config.integrator_heads = 4;
            model_config.module_hidden_dim = 128;
            model_config.state_space_dim = 96;
            model_config.state_space_kernel_size = 3;
            model_config.recurrent_dim = 64;
            model_config.chunk_size = 16;
            model_config.shared_state_slots = 4;
            model_config.n1_depth = 2;
            model_config.top_k = 2;
            model_config.tie_embeddings = true;
            model_config.population = {emc::N1Family::gpt, emc::N1Family::ssm, emc::N1Family::recurrent};
        } else {
            model_config = emc::load_model_config(fixture / "model.rvcfg");
        }
        emc::EMCModel model(model_config);
        if (!realistic) model.import_python_weights(fixture / "weights.pt");
        model.to(device);
        auto tokens = realistic
            ? torch::randint(model_config.vocab_size, {4, 128}, torch::TensorOptions().device(device).dtype(torch::kLong))
            : bundle_tensor(fixture / "forward.pt", "tokens").to(device, torch::kLong);
        auto targets = realistic
            ? torch::roll(tokens, {-1}, {1})
            : bundle_tensor(fixture / "forward.pt", "targets").to(device, torch::kLong);

        model.eval();
        {
            torch::NoGradGuard no_grad;
            Autocast autocast(device, bf16);
            for (int index = 0; index < warmup; ++index) (void)model.forward({tokens}).logits;
        }
        const auto process_before = process_sample();
        double forward_seconds;
        {
            torch::NoGradGuard no_grad;
            Autocast autocast(device, bf16);
            forward_seconds = time_iterations(device, iterations, [&] { (void)model.forward({tokens}).logits; });
        }

        model.train();
        emc::ForeachAdamW optimizer(model.parameters(), torch::optim::AdamWOptions(3e-4).weight_decay(0.01));
        for (int index = 0; index < warmup; ++index) {
            optimizer.zero_grad();
            Autocast autocast(device, bf16);
            loss_for(model.forward({tokens}).logits, targets).backward();
        }
        const auto forward_backward_seconds = time_iterations(device, iterations, [&] {
            optimizer.zero_grad();
            Autocast autocast(device, bf16);
            loss_for(model.forward({tokens}).logits, targets).backward();
        });
        const auto optimizer_seconds = time_iterations(device, iterations, [&] { optimizer.step(); });
        const auto train_seconds = time_iterations(device, iterations, [&] {
            optimizer.zero_grad();
            Autocast autocast(device, bf16);
            loss_for(model.forward({tokens}).logits, targets).backward();
            optimizer.step();
        });
        const auto process_after = process_sample();
        const auto memory = emc::collect_memory_report(model, &optimizer, device.is_cuda() ? 0 : 0);
        const auto wall = forward_seconds + forward_backward_seconds + optimizer_seconds + train_seconds;
        const auto cpu_percent = wall > 0.0
            ? 100.0 * (process_after.cpu_seconds - process_before.cpu_seconds) /
                  wall / std::max(1u, std::thread::hardware_concurrency())
            : 0.0;
        const auto tokens_per_step = static_cast<double>(tokens.numel());

        std::cout << std::fixed << std::setprecision(6)
                  << "{\n"
                  << "  \"runtime\": \"cpp\",\n"
                  << "  \"device\": \"" << (device.is_cuda() ? "cuda" : "cpu") << "\",\n"
                  << "  \"precision\": \"" << (bf16 ? "bf16" : "fp32") << "\",\n"
                  << "  \"iterations\": " << iterations << ",\n"
                  << "  \"forward_ms\": " << forward_seconds * 1000.0 / iterations << ",\n"
                  << "  \"forward_backward_ms\": " << forward_backward_seconds * 1000.0 / iterations << ",\n"
                  << "  \"optimizer_step_ms\": " << optimizer_seconds * 1000.0 / iterations << ",\n"
                  << "  \"train_step_ms\": " << train_seconds * 1000.0 / iterations << ",\n"
                  << "  \"tokens_per_second\": " << tokens_per_step * iterations / train_seconds << ",\n"
                  << "  \"cpu_utilization_percent\": " << cpu_percent << ",\n"
                  << "  \"process_rss_bytes\": " << process_after.rss << ",\n"
                  << "  \"peak_rss_bytes\": " << process_after.peak_rss << ",\n"
                  << "  \"parameter_bytes\": " << memory.parameter_bytes << ",\n"
                  << "  \"optimizer_bytes\": " << memory.optimizer_bytes << ",\n"
                  << "  \"cuda_allocated_bytes\": " << memory.cuda_allocated_bytes << ",\n"
                  << "  \"cuda_reserved_bytes\": " << memory.cuda_reserved_bytes << ",\n"
                  << "  \"cuda_peak_allocated_bytes\": " << memory.cuda_peak_allocated_bytes << ",\n"
                  << "  \"cuda_peak_reserved_bytes\": " << memory.cuda_peak_reserved_bytes << "\n"
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
