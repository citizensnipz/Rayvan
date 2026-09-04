#include "rayvan_emc/checkpoint/checkpoint.hpp"
#include "rayvan_emc/model.hpp"

#include <ATen/autocast_mode.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace emc = rayvan::emc;

namespace {

constexpr std::array<char, 8> evaluation_magic{'R', 'V', 'C', 'A', 'P', 'E', 'V', '1'};

template <typename T>
T read_value(std::istream& stream) {
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!stream) throw std::runtime_error("truncated capability evaluation file");
    return value;
}

std::string read_string(std::istream& stream) {
    const auto size = read_value<std::uint32_t>(stream);
    std::string value(size, '\0');
    stream.read(value.data(), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("truncated capability name");
    return value;
}

std::vector<std::int64_t> read_tokens(std::istream& stream, std::uint32_t size) {
    std::vector<std::int32_t> stored(size);
    stream.read(reinterpret_cast<char*>(stored.data()), static_cast<std::streamsize>(stored.size() * sizeof(std::int32_t)));
    if (!stream) throw std::runtime_error("truncated capability token sequence");
    return {stored.begin(), stored.end()};
}

struct Example {
    std::uint32_t capability = 0;
    std::vector<std::int64_t> prompt;
    std::vector<std::int64_t> target;
};

struct EvaluationSet {
    std::uint32_t vocab_size = 0;
    std::vector<std::string> capabilities;
    std::vector<Example> examples;
};

EvaluationSet load_evaluation(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot read capability evaluation file: " + path.string());
    std::array<char, 8> magic{};
    stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (magic != evaluation_magic) throw std::runtime_error("invalid capability evaluation magic");
    if (read_value<std::uint32_t>(stream) != 1) throw std::runtime_error("unsupported capability evaluation version");
    EvaluationSet result;
    result.vocab_size = read_value<std::uint32_t>(stream);
    const auto capability_count = read_value<std::uint32_t>(stream);
    result.capabilities.reserve(capability_count);
    for (std::uint32_t index = 0; index < capability_count; ++index) result.capabilities.push_back(read_string(stream));
    const auto example_count = read_value<std::uint64_t>(stream);
    result.examples.reserve(static_cast<std::size_t>(example_count));
    for (std::uint64_t index = 0; index < example_count; ++index) {
        Example example;
        example.capability = read_value<std::uint32_t>(stream);
        const auto prompt_size = read_value<std::uint32_t>(stream);
        const auto target_size = read_value<std::uint32_t>(stream);
        if (example.capability >= capability_count) throw std::runtime_error("invalid capability index");
        example.prompt = read_tokens(stream, prompt_size);
        example.target = read_tokens(stream, target_size);
        result.examples.push_back(std::move(example));
    }
    return result;
}

struct Metrics {
    explicit Metrics(std::size_t experts = 0)
        : activations(experts, 0.0), probability_sum(experts, 0.0), resistance_sum(experts, 0.0),
          token_norm_sum(experts, 0.0), latent_attention_sum(experts, 0.0),
          coactivation(experts, std::vector<double>(experts, 0.0)),
          ablation_loss_sum(experts, 0.0), ablation_correct(experts, 0),
          ablation_tokens(experts, 0), ablation_examples(experts, 0) {}

    std::int64_t examples = 0;
    std::int64_t skipped = 0;
    std::int64_t exact = 0;
    std::int64_t target_tokens = 0;
    std::int64_t correct = 0;
    double loss_sum = 0.0;
    double routing_items = 0.0;
    double latent_attention_items = 0.0;
    double novelty_items = 0.0;
    double low_confidence_items = 0.0;
    double exploration_activations = 0.0;
    double resonance_entropy_sum = 0.0;
    double training_activation_density = 0.0;
    double training_novelty_rate = 0.0;
    double training_exploration_rate = 0.0;
    double training_resonance_entropy = 0.0;
    double elapsed_seconds = 0.0;
    std::vector<double> activations;
    std::vector<double> probability_sum;
    std::vector<double> resistance_sum;
    std::vector<double> token_norm_sum;
    std::vector<double> latent_attention_sum;
    std::vector<std::vector<double>> coactivation;
    std::vector<double> ablation_loss_sum;
    std::vector<std::int64_t> ablation_correct;
    std::vector<std::int64_t> ablation_tokens;
    std::vector<std::int64_t> ablation_examples;
    double ablation_baseline_loss_sum = 0.0;
    std::int64_t ablation_baseline_correct = 0;
    std::int64_t ablation_baseline_tokens = 0;
};

struct ExampleResult {
    bool skipped = false;
    std::int64_t tokens = 0;
    std::int64_t correct = 0;
    double loss_sum = 0.0;
};

void add_tensor(std::vector<double>& destination, const torch::Tensor& tensor) {
    const auto values = tensor.to(torch::kCPU, torch::kFloat64).contiguous().reshape({-1});
    const auto* data = values.const_data_ptr<double>();
    for (std::int64_t index = 0; index < values.numel(); ++index) destination.at(static_cast<std::size_t>(index)) += data[index];
}

ExampleResult evaluate_example(
    emc::EMCModel& model,
    const Example& example,
    Metrics* metrics,
    const torch::Device& device,
    const std::optional<emc::CausalIntervention>& intervention = std::nullopt) {
    if (example.prompt.empty() || example.target.empty() ||
        example.prompt.size() + example.target.size() - 1 > static_cast<std::size_t>(model.config().max_sequence_length)) {
        return {.skipped = true};
    }
    std::vector<std::int64_t> combined = example.prompt;
    combined.insert(combined.end(), example.target.begin(), example.target.end());
    const auto sequence = static_cast<std::int64_t>(combined.size() - 1);
    auto input = torch::from_blob(combined.data(), {1, sequence}, torch::TensorOptions().dtype(torch::kLong)).clone().to(device);
    auto answer_targets = torch::from_blob(
        const_cast<std::int64_t*>(example.target.data()),
        {static_cast<std::int64_t>(example.target.size())},
        torch::TensorOptions().dtype(torch::kLong)).clone().to(device);
    const auto output = model.forward({input, metrics != nullptr, intervention});
    const auto answer_start = static_cast<std::int64_t>(example.prompt.size() - 1);
    const auto answer_logits = output.logits.index({0, torch::indexing::Slice(answer_start, answer_start + answer_targets.numel())}).to(torch::kFloat32);
    const auto loss = torch::nn::functional::cross_entropy(
        answer_logits,
        answer_targets,
        torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kSum));
    const auto correct = (answer_logits.argmax(-1) == answer_targets).sum().item<std::int64_t>();

    if (metrics) {
        if (!output.routing_free_trace) throw std::runtime_error("routing-free trace is unavailable");
        const auto& trace = *output.routing_free_trace;
        const auto binary = trace.activation_mask.to(torch::kFloat64);
        const auto items = static_cast<double>(binary.size(0) * binary.size(1));
        metrics->routing_items += items;
        metrics->latent_attention_items += items * trace.latent_attention_weights.size(2);
        metrics->novelty_items += trace.novelty_mask.to(torch::kFloat64).sum().item<double>();
        metrics->low_confidence_items += trace.low_confidence_mask.to(torch::kFloat64).sum().item<double>();
        metrics->exploration_activations += trace.exploration_mask.to(torch::kFloat64).sum().item<double>();
        metrics->resonance_entropy_sum += trace.resonance_entropy.item<double>() * items;
        metrics->training_activation_density = trace.training_activation_density.item<double>();
        metrics->training_novelty_rate = trace.training_novelty_rate.item<double>();
        metrics->training_exploration_rate = trace.training_exploration_rate.item<double>();
        metrics->training_resonance_entropy = trace.training_resonance_entropy.item<double>();
        add_tensor(metrics->activations, binary.sum({0, 1}));
        add_tensor(metrics->probability_sum, trace.resonance_probability.to(torch::kFloat64).sum({0, 1}));
        add_tensor(metrics->resistance_sum, trace.resistance.to(torch::kFloat64).sum({0, 1}));
        add_tensor(metrics->token_norm_sum, trace.raw_token_proposal_norm.to(torch::kFloat64).sum({0, 1}));
        add_tensor(metrics->latent_attention_sum, trace.latent_attention_weights.to(torch::kFloat64).sum({0, 1, 2}));
        const auto coactivation = binary.reshape({-1, binary.size(-1)}).transpose(0, 1).matmul(binary.reshape({-1, binary.size(-1)}));
        const auto coactivation_cpu = coactivation.to(torch::kCPU, torch::kFloat64).contiguous();
        const auto* values = coactivation_cpu.const_data_ptr<double>();
        for (std::int64_t row = 0; row < coactivation_cpu.size(0); ++row) {
            for (std::int64_t column = 0; column < coactivation_cpu.size(1); ++column) {
                metrics->coactivation[static_cast<std::size_t>(row)][static_cast<std::size_t>(column)] +=
                    values[row * coactivation_cpu.size(1) + column];
            }
        }
    }
    return {
        false,
        answer_targets.numel(),
        correct,
        loss.item<double>()};
}

double safe_divide(double numerator, double denominator) {
    return denominator > 0.0 ? numerator / denominator : 0.0;
}

double entropy(const std::vector<double>& counts) {
    double total = 0.0;
    for (const auto value : counts) total += value;
    double result = 0.0;
    for (const auto value : counts) {
        const auto probability = safe_divide(value, total);
        if (probability > 0.0) result -= probability * std::log(probability);
    }
    return result;
}

void print_vector(const std::vector<double>& values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

void print_tensor_vector(const torch::Tensor& tensor) {
    const auto cpu = tensor.detach().to(torch::kCPU, torch::kFloat64).contiguous().reshape({-1});
    std::cout << '[';
    const auto* values = cpu.const_data_ptr<double>();
    for (std::int64_t index = 0; index < cpu.numel(); ++index) {
        if (index) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

void print_tensor_matrix(const torch::Tensor& tensor) {
    const auto cpu = tensor.detach().to(torch::kCPU, torch::kFloat64).contiguous();
    std::cout << '[';
    for (std::int64_t row = 0; row < cpu.size(0); ++row) {
        if (row) std::cout << ',';
        print_tensor_vector(cpu.select(0, row));
    }
    std::cout << ']';
}

std::vector<double> divided(const std::vector<double>& values, double denominator) {
    auto result = values;
    for (auto& value : result) value = safe_divide(value, denominator);
    return result;
}

void print_metrics(const Metrics& metrics, const std::vector<std::string>& experts) {
    const auto rates = divided(metrics.activations, metrics.routing_items);
    double active_total = 0.0;
    for (const auto value : metrics.activations) active_total += value;
    const auto shares = divided(metrics.activations, active_total);
    const auto assignment_entropy = entropy(metrics.activations);
    double election_entropy = 0.0;
    for (const auto rate : rates) {
        if (rate > 0.0) election_entropy -= rate * std::log(rate);
        if (rate < 1.0) election_entropy -= (1.0 - rate) * std::log(1.0 - rate);
    }
    election_entropy /= std::max<std::size_t>(rates.size(), 1);
    const auto dominant = static_cast<std::size_t>(std::distance(shares.begin(), std::max_element(shares.begin(), shares.end())));
    std::cout << "{\"examples\":" << metrics.examples
              << ",\"skipped\":" << metrics.skipped
              << ",\"target_tokens\":" << metrics.target_tokens
              << ",\"loss\":" << safe_divide(metrics.loss_sum, metrics.target_tokens)
              << ",\"perplexity\":" << std::exp(std::min(safe_divide(metrics.loss_sum, metrics.target_tokens), 20.0))
              << ",\"token_accuracy\":" << safe_divide(metrics.correct, metrics.target_tokens)
              << ",\"exact_accuracy\":" << safe_divide(metrics.exact, metrics.examples)
              << ",\"throughput_tokens_per_second\":" << safe_divide(metrics.target_tokens, metrics.elapsed_seconds)
              << ",\"routing_items\":" << metrics.routing_items
              << ",\"activation_rate\":";
    print_vector(rates);
    std::cout << ",\"compute_share\":";
    print_vector(shares);
    std::cout << ",\"mean_active_experts\":" << safe_divide(active_total, metrics.routing_items)
              << ",\"density\":" << safe_divide(active_total, metrics.routing_items * experts.size())
              << ",\"normalized_assignment_entropy\":" << safe_divide(assignment_entropy, std::log(static_cast<double>(experts.size())))
              << ",\"effective_experts\":" << std::exp(assignment_entropy)
              << ",\"normalized_binary_election_entropy\":" << safe_divide(election_entropy, std::log(2.0))
              << ",\"resonance_entropy\":" << safe_divide(metrics.resonance_entropy_sum, metrics.routing_items)
              << ",\"novelty_rate\":" << safe_divide(metrics.novelty_items, metrics.routing_items)
              << ",\"low_confidence_rate\":" << safe_divide(metrics.low_confidence_items, metrics.routing_items)
              << ",\"exploration_rate\":" << safe_divide(metrics.exploration_activations, metrics.routing_items * experts.size())
              << ",\"training_activation_density\":" << metrics.training_activation_density
              << ",\"training_novelty_rate\":" << metrics.training_novelty_rate
              << ",\"training_exploration_rate\":" << metrics.training_exploration_rate
              << ",\"training_resonance_entropy\":" << metrics.training_resonance_entropy
              << ",\"dominant_expert\":\"" << experts[dominant] << "\""
              << ",\"dominant_share\":" << shares[dominant]
              << ",\"starvation\":" << (std::any_of(rates.begin(), rates.end(), [](double value) { return value < 0.01; }) ? "true" : "false")
              << ",\"monopoly\":" << (shares[dominant] > 0.80 ? "true" : "false")
              << ",\"resonance_probability_mean\":";
    print_vector(divided(metrics.probability_sum, metrics.routing_items));
    std::cout << ",\"resistance_mean\":";
    print_vector(divided(metrics.resistance_sum, metrics.routing_items));
    std::cout << ",\"token_proposal_norm\":";
    print_vector(divided(metrics.token_norm_sum, metrics.routing_items));
    std::cout << ",\"latent_attention\":";
    print_vector(divided(metrics.latent_attention_sum, metrics.latent_attention_items));
    std::cout << ",\"disable_expert_loss_delta\":[";
    const auto natural_loss = safe_divide(metrics.ablation_baseline_loss_sum, metrics.ablation_baseline_tokens);
    for (std::size_t index = 0; index < experts.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << safe_divide(metrics.ablation_loss_sum[index], metrics.ablation_tokens[index]) - natural_loss;
    }
    std::cout << "],\"disable_expert_token_accuracy_delta\":[";
    const auto natural_accuracy = safe_divide(metrics.ablation_baseline_correct, metrics.ablation_baseline_tokens);
    for (std::size_t index = 0; index < experts.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << safe_divide(metrics.ablation_correct[index], metrics.ablation_tokens[index]) - natural_accuracy;
    }
    std::cout << "]}";
}

std::pair<double, double> mutual_information(const std::vector<Metrics>& lanes) {
    if (lanes.empty()) return {0.0, 0.0};
    const auto experts = lanes.front().activations.size();
    std::vector<double> capability(lanes.size(), 0.0);
    std::vector<double> expert(experts, 0.0);
    double total = 0.0;
    for (std::size_t lane = 0; lane < lanes.size(); ++lane) {
        for (std::size_t node = 0; node < experts; ++node) {
            const auto count = lanes[lane].activations[node];
            capability[lane] += count;
            expert[node] += count;
            total += count;
        }
    }
    double information = 0.0;
    for (std::size_t lane = 0; lane < lanes.size(); ++lane) {
        for (std::size_t node = 0; node < experts; ++node) {
            const auto joint = safe_divide(lanes[lane].activations[node], total);
            const auto lane_probability = safe_divide(capability[lane], total);
            const auto expert_probability = safe_divide(expert[node], total);
            if (joint > 0.0) information += joint * std::log(joint / (lane_probability * expert_probability));
        }
    }
    const auto denominator = std::sqrt(entropy(capability) * entropy(expert));
    return {information, safe_divide(information, denominator)};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) throw std::invalid_argument("usage: rayvan-emc-capability-eval <checkpoint> <evaluation.rvcap> [ablation-examples-per-capability] [output.json]");
        const auto ablation_limit = argc > 3 ? std::stoll(argv[3]) : 8;
        std::ofstream output_file;
        if (argc > 4) {
            output_file.open(argv[4], std::ios::binary | std::ios::trunc);
            if (!output_file) throw std::runtime_error("cannot open capability output file");
            std::cout.rdbuf(output_file.rdbuf());
        }
        if (ablation_limit < 0 || !torch::cuda::is_available()) throw std::runtime_error("nonnegative ablation count and CUDA are required");
        const std::filesystem::path checkpoint(argv[1]);
        const auto config = emc::load_model_config(checkpoint / "model.rvcfg");
        if (config.n1_mode != emc::N1Mode::routing_free_collective) throw std::runtime_error("routing-free checkpoint required");
        const auto evaluation = load_evaluation(argv[2]);
        if (evaluation.vocab_size != config.vocab_size) throw std::runtime_error("evaluation/checkpoint vocabulary mismatch");
        const torch::Device device(torch::kCUDA, 0);
        emc::EMCModel model(config);
        const auto progress = emc::load_model_checkpoint(checkpoint, model, device);
        model.eval();
        torch::NoGradGuard no_grad;
        at::autocast::set_autocast_dtype(at::kCUDA, torch::kBFloat16);
        at::autocast::set_autocast_enabled(at::kCUDA, true);
        const std::vector<std::string> experts{"gpt", "ssm", "recurrent", "delta"};
        if (config.population.size() != experts.size()) throw std::runtime_error("four-family capability evaluator expected");
        std::vector<Metrics> lanes;
        lanes.reserve(evaluation.capabilities.size());
        for (std::size_t index = 0; index < evaluation.capabilities.size(); ++index) lanes.emplace_back(experts.size());

        (void)evaluate_example(model, evaluation.examples.front(), nullptr, device);
        torch::cuda::synchronize(0);
        const auto started = std::chrono::steady_clock::now();
        for (const auto& example : evaluation.examples) {
            auto& lane = lanes.at(example.capability);
            const auto item_started = std::chrono::steady_clock::now();
            const auto result = evaluate_example(model, example, &lane, device);
            torch::cuda::synchronize(0);
            lane.elapsed_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - item_started).count();
            if (result.skipped) {
                ++lane.skipped;
                continue;
            }
            ++lane.examples;
            lane.target_tokens += result.tokens;
            lane.correct += result.correct;
            lane.exact += result.correct == result.tokens ? 1 : 0;
            lane.loss_sum += result.loss_sum;
        }

        std::vector<std::int64_t> ablation_seen(evaluation.capabilities.size(), 0);
        for (const auto& example : evaluation.examples) {
            if (ablation_seen.at(example.capability) >= ablation_limit) continue;
            if (example.prompt.empty() || example.target.empty() || example.prompt.size() + example.target.size() - 1 > static_cast<std::size_t>(config.max_sequence_length)) continue;
            ++ablation_seen.at(example.capability);
            auto& lane = lanes.at(example.capability);
            const auto baseline = evaluate_example(model, example, nullptr, device);
            lane.ablation_baseline_loss_sum += baseline.loss_sum;
            lane.ablation_baseline_correct += baseline.correct;
            lane.ablation_baseline_tokens += baseline.tokens;
            for (std::size_t expert = 0; expert < experts.size(); ++expert) {
                auto available = torch::ones({static_cast<std::int64_t>(experts.size())}, torch::TensorOptions().device(device).dtype(torch::kBool));
                available.index_put_({static_cast<std::int64_t>(expert)}, false);
                emc::CausalIntervention intervention;
                intervention.availability_mask = available;
                const auto result = evaluate_example(model, example, nullptr, device, intervention);
                lane.ablation_loss_sum[expert] += result.loss_sum;
                lane.ablation_correct[expert] += result.correct;
                lane.ablation_tokens[expert] += result.tokens;
                lane.ablation_examples[expert] += 1;
            }
        }
        torch::cuda::synchronize(0);
        const auto total_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        at::autocast::set_autocast_enabled(at::kCUDA, false);

        Metrics overall(experts.size());
        for (const auto& lane : lanes) {
            overall.examples += lane.examples;
            overall.skipped += lane.skipped;
            overall.exact += lane.exact;
            overall.target_tokens += lane.target_tokens;
            overall.correct += lane.correct;
            overall.loss_sum += lane.loss_sum;
            overall.routing_items += lane.routing_items;
            overall.latent_attention_items += lane.latent_attention_items;
            overall.novelty_items += lane.novelty_items;
            overall.low_confidence_items += lane.low_confidence_items;
            overall.exploration_activations += lane.exploration_activations;
            overall.resonance_entropy_sum += lane.resonance_entropy_sum;
            overall.elapsed_seconds += lane.elapsed_seconds;
            overall.training_activation_density = lane.training_activation_density;
            overall.training_novelty_rate = lane.training_novelty_rate;
            overall.training_exploration_rate = lane.training_exploration_rate;
            overall.training_resonance_entropy = lane.training_resonance_entropy;
            for (std::size_t expert = 0; expert < experts.size(); ++expert) {
                overall.activations[expert] += lane.activations[expert];
                overall.probability_sum[expert] += lane.probability_sum[expert];
                overall.resistance_sum[expert] += lane.resistance_sum[expert];
                overall.token_norm_sum[expert] += lane.token_norm_sum[expert];
                overall.latent_attention_sum[expert] += lane.latent_attention_sum[expert];
            overall.ablation_loss_sum[expert] += lane.ablation_loss_sum[expert];
                overall.ablation_correct[expert] += lane.ablation_correct[expert];
                overall.ablation_tokens[expert] += lane.ablation_tokens[expert];
                overall.ablation_examples[expert] += lane.ablation_examples[expert];
            }
            overall.ablation_baseline_loss_sum += lane.ablation_baseline_loss_sum;
            overall.ablation_baseline_correct += lane.ablation_baseline_correct;
            overall.ablation_baseline_tokens += lane.ablation_baseline_tokens;
        }
        const auto [information, normalized_information] = mutual_information(lanes);
        const auto& collective = *model.module()->routing_free_collective();
        std::vector<double> parameter_norms;
        parameter_norms.reserve(experts.size());
        for (const auto& expert : collective.experts()) {
            double squared = 0.0;
            for (const auto& parameter : expert->parameters()) squared += parameter.detach().to(torch::kCPU, torch::kFloat64).pow(2).sum().item<double>();
            parameter_norms.push_back(std::sqrt(squared));
        }

        std::cout << std::fixed << std::setprecision(8)
                  << "{\n  \"checkpoint_tokens\":" << progress.tokens_processed
                  << ",\n  \"checkpoint_step\":" << progress.step
                  << ",\n  \"evaluation_wall_seconds\":" << total_elapsed
                  << ",\n  \"expert_parameter_norm\":";
        print_vector(parameter_norms);
        std::cout << ",\n  \"competence_memory\":{\n";
        for (std::size_t index = 0; index < experts.size(); ++index) {
            const auto& expert = collective.experts()[index];
            std::cout << "    \"" << experts[index] << "\":{\"basin_count\":"
                      << expert->basin_initialized().sum().item<std::int64_t>()
                      << ",\"centers\":";
            print_tensor_matrix(expert->basin_centers());
            std::cout << ",\"radii\":";
            print_tensor_vector(expert->basin_radii());
            std::cout << ",\"competence\":";
            print_tensor_vector(expert->basin_competence());
            std::cout << ",\"evidence\":";
            print_tensor_vector(expert->basin_evidence());
            std::cout << ",\"uncertainty\":";
            print_tensor_vector(expert->basin_uncertainty());
            std::cout << ",\"marginal_utility\":" << expert->marginal_utility().item<double>()
                      << ",\"utility_observations\":" << expert->utility_observations().item<double>()
                      << "}" << (index + 1 == experts.size() ? "\n" : ",\n");
        }
        std::cout << "  }"
                  << ",\n  \"capability_expert_mutual_information_nats\":" << information
                  << ",\n  \"capability_expert_normalized_mutual_information\":" << normalized_information
                  << ",\n  \"overall\":";
        print_metrics(overall, experts);
        std::cout << ",\n  \"capabilities\":{\n";
        for (std::size_t index = 0; index < lanes.size(); ++index) {
            std::cout << "    \"" << evaluation.capabilities[index] << "\":";
            print_metrics(lanes[index], experts);
            std::cout << (index + 1 == lanes.size() ? "\n" : ",\n");
        }
        std::cout << "  }\n}\n";
        return 0;
    } catch (const std::exception& error) {
        at::autocast::set_autocast_enabled(at::kCUDA, false);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
