#include "rayvan_emc/checkpoint/checkpoint.hpp"
#include "rayvan_emc/diagnostics/causal.hpp"
#include "rayvan_emc/model.hpp"
#include "rayvan_emc/training/dataset.hpp"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/autocast_mode.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace emc = rayvan::emc;

namespace {

struct Result {
    double loss = 0.0;
    double accuracy = 0.0;
    double latency_ms = 0.0;
    std::vector<std::int64_t> selections;
    std::vector<std::vector<std::int64_t>> temporal_selections;
    std::vector<double> token_proposal_norm;
    std::vector<double> raw_latent_norm;
    std::vector<double> normalized_latent_norm;
    std::vector<double> latent_attention;
    std::vector<double> latent_influence;
    std::vector<double> shared_latent_norm_by_chunk;
};

Result evaluate(
    emc::EMCModel& model,
    const std::vector<torch::Tensor>& windows,
    const std::optional<emc::CausalIntervention>& intervention) {
    model.eval();
    torch::NoGradGuard no_grad;
    double loss = 0.0;
    std::int64_t correct = 0;
    std::int64_t total = 0;
    std::vector<std::int64_t> selections(model.config().population.size(), 0);
    torch::Tensor temporal_sum;
    torch::Tensor token_norm_sum;
    torch::Tensor raw_latent_sum;
    torch::Tensor normalized_latent_sum;
    torch::Tensor attention_sum;
    torch::Tensor influence_sum;
    torch::Tensor shared_latent_sum;
    std::int64_t routing_items = 0;
    std::int64_t attention_items = 0;
    std::int64_t temporal_requests = 0;
    at::autocast::set_autocast_dtype(at::kCUDA, torch::kBFloat16);
    at::autocast::set_autocast_enabled(at::kCUDA, true);
    torch::cuda::synchronize(0);
    const auto started = std::chrono::steady_clock::now();
    for (const auto& batch : windows) {
        const auto input = batch.index({torch::indexing::Slice(), torch::indexing::Slice(0, 256)});
        const auto targets = batch.index({torch::indexing::Slice(), torch::indexing::Slice(1, 257)});
        const auto output = model.forward({input, true, intervention});
        loss += torch::nn::functional::cross_entropy(
                    output.logits.reshape({-1, output.logits.size(-1)}), targets.reshape({-1}))
                    .to(torch::kFloat32).item<double>();
        correct += (output.logits.argmax(-1) == targets).sum().item<std::int64_t>();
        total += targets.numel();
        if (output.routing_free_trace) {
            const auto& trace = *output.routing_free_trace;
            const auto counts = trace.activation_mask
                                    .to(torch::kCPU, torch::kLong).sum({0, 1}).contiguous();
            const auto* data = counts.const_data_ptr<std::int64_t>();
            for (std::int64_t index = 0; index < counts.numel(); ++index) {
                selections.at(static_cast<std::size_t>(index)) += data[index];
            }
            const auto temporal = trace.activation_mask.to(torch::kCPU, torch::kLong).sum(0);
            temporal_sum = temporal_sum.defined() ? temporal_sum + temporal : temporal;
            const auto token_norm = trace.raw_token_proposal_norm.to(torch::kCPU, torch::kFloat64).sum({0, 1});
            const auto raw_latent = trace.raw_latent_proposal_norm.to(torch::kCPU, torch::kFloat64).sum({0, 1});
            const auto normalized_latent = trace.normalized_latent_proposal_norm.to(torch::kCPU, torch::kFloat64).sum({0, 1});
            const auto attention = trace.latent_attention_weights.to(torch::kCPU, torch::kFloat64);
            const auto attention_by_item = attention.mean(2);
            const auto influence = (
                attention_by_item * trace.raw_latent_proposal_norm.to(torch::kCPU, torch::kFloat64)).sum({0, 1});
            token_norm_sum = token_norm_sum.defined() ? token_norm_sum + token_norm : token_norm;
            raw_latent_sum = raw_latent_sum.defined() ? raw_latent_sum + raw_latent : raw_latent;
            normalized_latent_sum = normalized_latent_sum.defined() ? normalized_latent_sum + normalized_latent : normalized_latent;
            const auto attention_reduced = attention.sum({0, 1, 2});
            attention_sum = attention_sum.defined() ? attention_sum + attention_reduced : attention_reduced;
            influence_sum = influence_sum.defined() ? influence_sum + influence : influence;
            const auto shared_latent = trace.latent_norm.to(torch::kCPU, torch::kFloat64).sum(0);
            shared_latent_sum = shared_latent_sum.defined() ? shared_latent_sum + shared_latent : shared_latent;
            routing_items += trace.activation_mask.size(0) * trace.activation_mask.size(1);
            attention_items += trace.latent_attention_weights.size(0) * trace.latent_attention_weights.size(1) *
                               trace.latent_attention_weights.size(2);
            temporal_requests += trace.activation_mask.size(0);
        } else {
            const auto ids = output.routing.selected_indices.to(torch::kCPU).reshape({-1});
            const auto* data = ids.const_data_ptr<std::int64_t>();
            for (std::int64_t index = 0; index < ids.numel(); ++index) ++selections.at(static_cast<std::size_t>(data[index]));
        }
    }
    torch::cuda::synchronize(0);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    at::autocast::set_autocast_enabled(at::kCUDA, false);
    Result result{
        loss / windows.size(),
        static_cast<double>(correct) / total,
        elapsed / windows.size(),
        std::move(selections)};
    if (temporal_sum.defined()) {
        const auto temporal = temporal_sum.contiguous();
        const auto* values = temporal.const_data_ptr<std::int64_t>();
        result.temporal_selections.resize(static_cast<std::size_t>(temporal.size(0)));
        for (std::int64_t chunk = 0; chunk < temporal.size(0); ++chunk) {
            auto& row = result.temporal_selections[static_cast<std::size_t>(chunk)];
            row.resize(static_cast<std::size_t>(temporal.size(1)));
            for (std::int64_t expert = 0; expert < temporal.size(1); ++expert) {
                row[static_cast<std::size_t>(expert)] = values[chunk * temporal.size(1) + expert];
            }
        }
        const auto to_vector = [](const torch::Tensor& tensor) {
            const auto contiguous = tensor.contiguous();
            const auto* values = contiguous.const_data_ptr<double>();
            return std::vector<double>(values, values + contiguous.numel());
        };
        result.token_proposal_norm = to_vector(token_norm_sum / routing_items);
        result.raw_latent_norm = to_vector(raw_latent_sum / routing_items);
        result.normalized_latent_norm = to_vector(normalized_latent_sum / routing_items);
        result.latent_attention = to_vector(attention_sum / attention_items);
        result.latent_influence = to_vector(influence_sum / routing_items);
        result.shared_latent_norm_by_chunk = to_vector(shared_latent_sum / temporal_requests);
    }
    return result;
}

void print_result(const std::string& name, const Result& result, bool comma) {
    std::cout << "  \"" << name << "\": {\"loss\": " << result.loss
              << ", \"perplexity\": " << std::exp(std::min(result.loss, 20.0))
              << ", \"accuracy\": " << result.accuracy
              << ", \"latency_ms\": " << result.latency_ms
              << ", \"selections\": [";
    for (std::size_t index = 0; index < result.selections.size(); ++index) {
        if (index) std::cout << ", ";
        std::cout << result.selections[index];
    }
    std::cout << "]}" << (comma ? "," : "") << "\n";
}

void print_vector(const std::vector<double>& values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) std::cout << ", ";
        std::cout << values[index];
    }
    std::cout << ']';
}

void print_natural_diagnostics(const Result& result) {
    std::cout << "  \"natural_diagnostics\": {\n"
              << "    \"temporal_selections\": [";
    for (std::size_t chunk = 0; chunk < result.temporal_selections.size(); ++chunk) {
        if (chunk) std::cout << ", ";
        std::cout << '[';
        for (std::size_t expert = 0; expert < result.temporal_selections[chunk].size(); ++expert) {
            if (expert) std::cout << ", ";
            std::cout << result.temporal_selections[chunk][expert];
        }
        std::cout << ']';
    }
    std::cout << "],\n    \"token_proposal_norm\": ";
    print_vector(result.token_proposal_norm);
    std::cout << ",\n    \"raw_latent_norm\": ";
    print_vector(result.raw_latent_norm);
    std::cout << ",\n    \"normalized_latent_norm\": ";
    print_vector(result.normalized_latent_norm);
    std::cout << ",\n    \"latent_attention\": ";
    print_vector(result.latent_attention);
    std::cout << ",\n    \"latent_influence\": ";
    print_vector(result.latent_influence);
    std::cout << ",\n    \"shared_latent_norm_by_chunk\": ";
    print_vector(result.shared_latent_norm_by_chunk);
    std::cout << "\n  }\n";
}

emc::CausalIntervention exact_pair(
    std::int64_t left,
    std::int64_t right,
    emc::N1Mode mode,
    const torch::Device& device,
    std::int64_t batch) {
    if (mode == emc::N1Mode::routing_free_collective) {
        auto mask = torch::zeros({4}, torch::TensorOptions().device(device).dtype(torch::kBool));
        mask.index_put_({left}, true);
        mask.index_put_({right}, true);
        emc::CausalIntervention intervention;
        intervention.availability_mask = mask;
        intervention.force_active_mask = mask;
        return intervention;
    }
    auto forced = torch::tensor({left, right}, torch::TensorOptions().dtype(torch::kLong))
                      .reshape({1, 2}).expand({batch, 2}).clone().to(device);
    return emc::force_alternate_n1(forced);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            throw std::invalid_argument("usage: rayvan-emc-causal <checkpoint> <validation.rvtok> [batches]");
        }
        const auto batches = argc > 3 ? std::stoll(argv[3]) : 16;
        if (batches <= 0 || !torch::cuda::is_available()) {
            throw std::runtime_error("positive batches and CUDA are required");
        }
        const std::filesystem::path checkpoint(argv[1]);
        const auto config = emc::load_model_config(checkpoint / "model.rvcfg");
        if (config.population.size() != 4 || config.population.back() != emc::N1Family::delta) {
            throw std::runtime_error("causal audit requires the four-family mixed checkpoint");
        }
        const torch::Device device(torch::kCUDA, 0);
        emc::EMCModel model(config);
        (void)emc::load_model_checkpoint(checkpoint, model, device);
        const auto validation = emc::TokenStream::load(argv[2]);
        auto generator = torch::make_generator<at::CPUGeneratorImpl>(43);
        std::vector<torch::Tensor> windows;
        windows.reserve(static_cast<std::size_t>(batches));
        for (std::int64_t index = 0; index < batches; ++index) {
            windows.push_back(validation.sample_batch(4, 256, generator, device));
        }
        (void)evaluate(model, {windows.front()}, std::nullopt);
        const auto natural = evaluate(model, windows, std::nullopt);
        const std::vector<std::string> names{"gpt", "ssm", "recurrent", "delta"};
        std::vector<std::pair<std::string, Result>> results;
        results.emplace_back("natural", natural);
        for (std::int64_t expert = 0; expert < 4; ++expert) {
            auto available = torch::ones({4}, torch::TensorOptions().device(device).dtype(torch::kBool));
            available.index_put_({expert}, false);
            results.emplace_back(
                "disable_" + names[static_cast<std::size_t>(expert)],
                evaluate(model, windows, emc::disable_n1(available)));
            auto zero = torch::zeros({4}, torch::TensorOptions().device(device).dtype(torch::kBool));
            zero.index_put_({expert}, true);
            results.emplace_back(
                "zero_" + names[static_cast<std::size_t>(expert)],
                evaluate(model, windows, emc::zero_n1_proposal(zero)));
            if (config.n1_mode == emc::N1Mode::routing_free_collective) {
                auto forced = torch::zeros({4}, torch::TensorOptions().device(device).dtype(torch::kBool));
                forced.index_put_({expert}, true);
                emc::CausalIntervention intervention;
                intervention.force_active_mask = forced;
                results.emplace_back(
                    "force_" + names[static_cast<std::size_t>(expert)],
                    evaluate(model, windows, intervention));
            }
        }
        for (std::int64_t left = 0; left < 4; ++left) {
            for (std::int64_t right = left + 1; right < 4; ++right) {
                results.emplace_back(
                    "pair_" + names[static_cast<std::size_t>(left)] + "_" + names[static_cast<std::size_t>(right)],
                    evaluate(model, windows, exact_pair(left, right, config.n1_mode, device, 4)));
            }
        }
        std::cout << std::fixed << std::setprecision(6) << "{\n";
        for (std::size_t index = 0; index < results.size(); ++index) {
            print_result(results[index].first, results[index].second, true);
        }
        print_natural_diagnostics(natural);
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& error) {
        at::autocast::set_autocast_enabled(at::kCUDA, false);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
