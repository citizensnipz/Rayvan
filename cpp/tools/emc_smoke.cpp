#include "rayvan_emc/diagnostics/diagnostics.hpp"
#include "rayvan_emc/model.hpp"

#include <torch/torch.h>

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    try {
        torch::manual_seed(42);
        rayvan::emc::ModelConfig config;
        config.vocab_size = 257;
        config.max_sequence_length = 16;
        config.chunk_size = 8;
        config.n1_depth = 2;
        rayvan::emc::EMCModel model(config);
        const bool use_cuda = argc > 1 && std::string(argv[1]) == "--cuda";
        const torch::Device device = use_cuda ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
        if (use_cuda && !torch::cuda::is_available()) {
            std::cerr << "CUDA requested but this LibTorch build has no available CUDA device\n";
            return 2;
        }
        model.to(device);
        model.eval();
        rayvan::emc::InferenceScope inference;
        const auto tokens = torch::arange(16, torch::TensorOptions().dtype(torch::kLong).device(device)).reshape({1, 16}) % config.vocab_size;
        const auto output = model.forward({tokens, true});
        std::cout << "logits=" << output.logits.sizes()
                  << " top_k=" << output.routing.selected_indices.sizes()
                  << " proposals=" << output.proposals.sizes() << '\n';
        const auto memory = rayvan::emc::collect_memory_report(model, nullptr, 0);
        std::cout << "rss=" << memory.process_rss_bytes
                  << " parameter_bytes=" << memory.parameter_bytes
                  << " cuda_allocated=" << memory.cuda_allocated_bytes
                  << " cuda_reserved=" << memory.cuda_reserved_bytes << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
