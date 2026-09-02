#include "rayvan_emc/checkpoint/checkpoint.hpp"
#include "rayvan_emc/diagnostics/diagnostics.hpp"
#include "rayvan_emc/model.hpp"

#include <torch/cuda.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace emc = rayvan::emc;

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            throw std::invalid_argument(
                "usage: rayvan-emc-checkpoint-probe write <fixture> <checkpoint> | load <checkpoint> <cpu|cuda>");
        }
        const std::string mode(argv[1]);
        if (mode == "write") {
            const std::filesystem::path fixture(argv[2]);
            const std::filesystem::path checkpoint(argv[3]);
            emc::EMCModel model(emc::load_model_config(fixture / "model.rvcfg"));
            model.import_python_weights(fixture / "weights.pt");
            emc::save_checkpoint(checkpoint, model, nullptr, {0, 0, 0.0, 0.0, 20260902, "fp32"});
            std::cout << checkpoint.string() << '\n';
            return 0;
        }
        if (mode != "load") throw std::invalid_argument("unknown checkpoint probe mode");
        const std::filesystem::path checkpoint(argv[2]);
        const std::string requested_device(argv[3]);
        const torch::Device device = requested_device == "cuda"
            ? torch::Device(torch::kCUDA, 0)
            : torch::Device(torch::kCPU);
        if (device.is_cuda() && !torch::cuda::is_available()) throw std::runtime_error("CUDA unavailable");
        auto config = emc::load_model_config(checkpoint / "model.rvcfg");
        emc::EMCModel model(config);
        (void)emc::load_model_checkpoint(checkpoint, model, device);
        if (device.is_cuda()) torch::cuda::synchronize();
        const auto memory = emc::collect_memory_report(model, nullptr, 0);
        std::cout << "{\n"
                  << "  \"runtime\": \"cpp\",\n"
                  << "  \"checkpoint_load_rss_bytes\": " << memory.process_rss_bytes << ",\n"
                  << "  \"checkpoint_load_peak_rss_bytes\": " << memory.process_peak_rss_bytes << ",\n"
                  << "  \"cuda_allocated_bytes\": " << memory.cuda_allocated_bytes << ",\n"
                  << "  \"cuda_reserved_bytes\": " << memory.cuda_reserved_bytes << ",\n"
                  << "  \"cuda_peak_allocated_bytes\": " << memory.cuda_peak_allocated_bytes << "\n"
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
