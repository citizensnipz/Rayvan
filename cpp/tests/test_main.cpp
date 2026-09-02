#include "rayvan_emc/checkpoint/checkpoint.hpp"
#include "rayvan_emc/diagnostics/causal.hpp"
#include "rayvan_emc/diagnostics/diagnostics.hpp"
#include "rayvan_emc/model.hpp"
#include "rayvan_emc/n1/n1.hpp"
#include "rayvan_emc/n2/dispatch.hpp"
#include "rayvan_emc/n2/integrator.hpp"
#include "rayvan_emc/n2/nexus.hpp"
#include "rayvan_emc/training/dataset.hpp"
#include "rayvan_emc/training/trainer.hpp"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/autocast_mode.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace emc = rayvan::emc;

namespace {

#undef CHECK
#define CHECK(condition) do { if (!(condition)) throw std::runtime_error(std::string("check failed: ") + #condition); } while (false)

std::string decode(std::string value) {
    const std::string marker = "__DOT__";
    std::size_t position = 0;
    while ((position = value.find(marker, position)) != std::string::npos) {
        value.replace(position, marker.size(), ".");
        ++position;
    }
    return value;
}

using Bundle = std::unordered_map<std::string, torch::Tensor>;

Bundle load_bundle(const std::filesystem::path& path) {
    auto module = torch::jit::load(path.string(), torch::kCPU);
    Bundle result;
    for (const auto& item : module.named_buffers(true)) result.emplace(decode(item.name), item.value);
    return result;
}

emc::ModelConfig small_config() {
    emc::ModelConfig config;
    config.latent_dim = 16;
    config.vocab_size = 67;
    config.max_sequence_length = 8;
    config.attention_heads = 4;
    config.integrator_heads = 4;
    config.module_hidden_dim = 32;
    config.state_space_dim = 24;
    config.state_space_kernel_size = 3;
    config.recurrent_dim = 20;
    config.chunk_size = 4;
    config.shared_state_slots = 2;
    config.n1_depth = 2;
    config.top_k = 2;
    config.tie_embeddings = true;
    config.population = {emc::N1Family::gpt, emc::N1Family::ssm, emc::N1Family::recurrent};
    return config;
}

torch::Tensor loss_for(const torch::Tensor& logits, const torch::Tensor& targets) {
    return torch::nn::functional::cross_entropy(logits.reshape({-1, logits.size(-1)}), targets.reshape({-1}));
}

void close(
    const torch::Tensor& actual,
    const torch::Tensor& expected,
    double atol = 2e-5,
    double rtol = 2e-4,
    std::string_view label = "tensor") {
    if (!torch::allclose(actual.detach().to(torch::kCPU, expected.scalar_type()), expected, rtol, atol, true)) {
        const auto difference = (actual.detach().to(torch::kCPU, torch::kFloat64) - expected.to(torch::kFloat64)).abs().max().item<double>();
        throw std::runtime_error(std::string(label) + " mismatch; max absolute difference=" + std::to_string(difference));
    }
}

std::filesystem::path fixture_root() {
    return std::filesystem::path(RAYVAN_EMC_SOURCE_DIR) / "tests" / "fixtures" / "reference";
}

std::filesystem::path temporary_root() {
    return std::filesystem::temp_directory_path() /
        ("rayvan-emc-native-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
}

void model_construction() {
    emc::EMCModel model(small_config());
    CHECK(model.parameters().size() > 20);
    CHECK(model.module()->nodes().size() == 3);
}

void n1_forward(emc::N1Family family) {
    auto config = small_config();
    auto node = emc::create_n1_node(config, family, 0, std::string(emc::to_string(family)));
    const auto latent = torch::randn({2, 8, config.latent_dim}, torch::TensorOptions().requires_grad(true));
    const auto output = node->forward({latent});
    CHECK(output.proposal.sizes() == latent.sizes());
    CHECK(torch::isfinite(output.proposal).all().item<bool>());
    output.proposal.square().mean().backward();
    CHECK(latent.grad().defined());
}

void independent_parameters() {
    auto config = small_config();
    config.population = {emc::N1Family::gpt, emc::N1Family::gpt, emc::N1Family::gpt};
    emc::EMCModel model(config);
    const auto& nodes = model.module()->nodes();
    std::vector<std::unordered_set<const void*>> identities;
    for (const auto& node : nodes) {
        std::unordered_set<const void*> set;
        for (const auto& parameter : node->parameters()) set.insert(parameter.unsafeGetTensorImpl());
        identities.push_back(std::move(set));
    }
    for (std::size_t left = 0; left < identities.size(); ++left) {
        for (std::size_t right = left + 1; right < identities.size(); ++right) {
            for (const auto* identity : identities[left]) CHECK(!identities[right].contains(identity));
        }
    }
}

void nexus_routing() {
    const auto config = small_config();
    emc::Nexus nexus(config);
    const auto decision = nexus->forward(torch::randn({4, 8, config.latent_dim}), 2);
    CHECK(decision.scores.sizes() == torch::IntArrayRef({4, 3}));
    CHECK(decision.selected_indices.sizes() == torch::IntArrayRef({4, 2}));
    close(decision.pre_top_k_probabilities.sum(-1), torch::ones({4}));
    close(decision.selected_weights.sum(-1), torch::ones({4}));
}

void exact_top_k() {
    emc::EMCModel model(small_config());
    const auto output = model.forward({torch::randint(67, {5, 8})});
    CHECK(output.routing.selected_indices.size(1) == 2);
    const auto sorted = std::get<0>(output.routing.selected_indices.sort(-1));
    CHECK(!(sorted.index({torch::indexing::Slice(), 0}) == sorted.index({torch::indexing::Slice(), 1})).any().item<bool>());
}

void sparse_execution() {
    emc::EMCModel model(small_config());
    const auto forced = torch::tensor({{0, 1}}, torch::kLong);
    emc::CausalIntervention intervention;
    intervention.forced_nodes = forced;
    model.forward({torch::randint(67, {1, 8}), true, intervention});
    const auto& nodes = model.module()->nodes();
    CHECK(nodes[0]->execution_count() == 1);
    CHECK(nodes[1]->execution_count() == 1);
    CHECK(nodes[2]->execution_count() == 0);
}

void dispatch_ordering() {
    const auto selected = torch::tensor({{2, 0}, {1, 2}, {0, 1}}, torch::kLong);
    const auto plan = emc::DispatchPlan::from_routing(selected, 3);
    const auto source = torch::arange(6, torch::kFloat32).reshape({6, 1, 1});
    const auto grouped = source.index_select(0, plan.permutation);
    const auto restored = plan.restore(grouped, 3, 1, 1).reshape({3, 2});
    close(restored, torch::arange(6, torch::kFloat32).reshape({3, 2}));
}

void integrator_gradients() {
    const auto config = small_config();
    emc::N2Integrator integrator(config);
    auto latent = torch::randn({2, 8, 16}, torch::TensorOptions().requires_grad(true));
    auto proposals = torch::randn({2, 8, 2, 16}, torch::TensorOptions().requires_grad(true));
    auto weights = torch::softmax(torch::randn({2, 2}), -1);
    const auto output = integrator->forward(latent, proposals, weights, true);
    CHECK(output.latent.sizes() == latent.sizes());
    CHECK(output.trace.has_value());
    output.latent.square().mean().backward();
    CHECK(latent.grad().defined() && proposals.grad().defined());
    CHECK(integrator->parameters().front().grad().defined());
}

void weight_tying() {
    emc::EMCModel model(small_config());
    CHECK(model.embeddings_tied());
    auto config = small_config();
    config.tie_embeddings = false;
    emc::EMCModel untied(config);
    CHECK(!untied.embeddings_tied());
}

void fp32_forward() {
    emc::EMCModel model(small_config());
    const auto output = model.forward({torch::randint(67, {2, 8})});
    CHECK(output.logits.scalar_type() == torch::kFloat32);
    CHECK(torch::isfinite(output.logits).all().item<bool>());
}

void bf16_forward() {
    if (!torch::cuda::is_available()) {
        std::cout << "SKIP bf16 forward: CUDA unavailable\n";
        return;
    }
    emc::EMCModel model(small_config());
    model.to(torch::Device(torch::kCUDA, 0));
    at::autocast::set_autocast_dtype(at::kCUDA, torch::kBFloat16);
    at::autocast::set_autocast_enabled(at::kCUDA, true);
    const auto output = model.forward({torch::randint(67, {2, 8}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong))});
    at::autocast::set_autocast_enabled(at::kCUDA, false);
    CHECK(torch::isfinite(output.logits).all().item<bool>());
}

void checkpoint_round_trip() {
    const auto root = temporary_root();
    std::filesystem::create_directories(root);
    emc::EMCModel source(small_config());
    source.eval();
    const auto tokens = torch::randint(67, {2, 8});
    const auto expected = source.forward({tokens}).logits.detach();
    torch::optim::AdamW optimizer(source.parameters(), torch::optim::AdamWOptions(1e-3));
    auto generator = torch::make_generator<at::CPUGeneratorImpl>(991);
    emc::CheckpointProgress saved{3, 96, 1.2, 1.1, 42, "fp32"};
    saved.cpu_rng_state =
        at::detail::getDefaultCPUGenerator().get_state();
    saved.train_generator_state = generator.get_state();
    saved.evaluation_generator_state = generator.get_state();
    emc::save_checkpoint(
        root / "checkpoint", source, &optimizer, saved);
    emc::EMCModel loaded(small_config());
    torch::optim::AdamW loaded_optimizer(loaded.parameters(), torch::optim::AdamWOptions(1e-3));
    const auto progress = emc::load_training_checkpoint(root / "checkpoint", loaded, loaded_optimizer, torch::kCPU);
    loaded.eval();
    CHECK(progress.step == 3 && progress.tokens_processed == 96);
    CHECK(progress.cpu_rng_state.has_value());
    CHECK(progress.train_generator_state.has_value());
    CHECK(torch::equal(
        *progress.train_generator_state,
        *saved.train_generator_state));
    close(loaded.forward({tokens}).logits, expected, 0.0, 0.0);
    std::filesystem::remove_all(root);
}

void milestone_save() {
    const auto root = temporary_root();
    emc::EMCModel model(small_config());
    const auto destination = emc::milestone_checkpoint_path(root, 100000);
    emc::save_checkpoint(destination, model, nullptr, {1, 100000, 2.0, 2.0, 42, "fp32"});
    CHECK(std::filesystem::exists(destination / "manifest.rayvan"));
    CHECK(!std::filesystem::exists(destination / "optimizer.pt"));
    std::filesystem::remove_all(root);
}

void diagnostics() {
    emc::EMCModel model(small_config());
    emc::DiagnosticAccumulator diagnostics(3);
    diagnostics.update(model.forward({torch::randint(67, {4, 8}), true}));
    const auto routing = diagnostics.routing_report();
    const auto integrator = diagnostics.integrator_report();
    CHECK(routing.selection_frequency.size() == 3);
    CHECK(std::abs(std::accumulate(routing.selection_frequency.begin(), routing.selection_frequency.end(), 0.0) - 1.0) < 1e-9);
    CHECK(integrator.acceptance.size() == 3);
    CHECK(emc::collect_memory_report(model).parameter_bytes > 0);
}

void causal_interventions() {
    emc::EMCModel model(small_config());
    const auto tokens = torch::randint(67, {2, 8});
    auto unavailable = torch::tensor({false, true, true}, torch::kBool);
    const auto disabled = model.forward({tokens, true, emc::disable_n1(unavailable)});
    CHECK(!(disabled.routing.selected_indices == 0).any().item<bool>());
    const auto forced_ids = torch::tensor({{0, 2}, {1, 2}}, torch::kLong);
    const auto forced = model.forward({tokens, true, emc::force_alternate_n1(forced_ids)});
    CHECK(torch::equal(forced.routing.selected_indices, forced_ids));
    auto zero_intervention = emc::force_alternate_n1(
        torch::tensor({{0, 1}, {1, 2}}, torch::kLong));
    zero_intervention.zero_proposal_mask =
        torch::tensor({false, true, false}, torch::kBool);
    const auto zeroed = model.forward({tokens, true, zero_intervention});
    const auto selected_zero = (zeroed.routing.selected_indices == 1).unsqueeze(1).unsqueeze(-1).expand_as(zeroed.proposals);
    CHECK(zeroed.proposals.masked_select(selected_zero).abs().max().item<float>() == 0.0f);
}

void deterministic_token_stream() {
    const auto root = temporary_root();
    std::filesystem::create_directories(root);
    std::vector<std::int32_t> tokens(100);
    std::iota(tokens.begin(), tokens.end(), 0);
    emc::TokenStream::save(root / "train.rvtok", tokens, "fixture/train");
    const auto loaded = emc::TokenStream::load(root / "train.rvtok");
    CHECK(loaded.metadata().token_count == tokens.size());
    CHECK(loaded.metadata().fingerprint == emc::token_fingerprint(tokens, "fixture/train"));
    auto first = torch::make_generator<at::CPUGeneratorImpl>(42);
    auto second = torch::make_generator<at::CPUGeneratorImpl>(42);
    CHECK(torch::equal(loaded.sample_batch(3, 8, first, torch::kCPU), loaded.sample_batch(3, 8, second, torch::kCPU)));
    std::filesystem::remove_all(root);
}

void forward_parity() {
    const auto root = fixture_root();
    const auto expected = load_bundle(root / "forward.pt");
    emc::EMCModel model(emc::load_model_config(root / "model.rvcfg"));
    model.import_python_weights(root / "weights.pt");
    const auto weights = load_bundle(root / "weights.pt");
    const auto named_parameters = model.module()->named_parameters(true);
    close(
        *named_parameters.find("n1_nodes.0.blocks.0.attention.in_proj_weight"),
        weights.at("n1_nodes.0.blocks.0.attention.in_proj_weight"),
        0.0,
        0.0,
        "imported GPT attention weights");
    model.eval();
    const auto output = model.forward({expected.at("tokens"), true});
    close(output.embeddings, expected.at("embeddings"), 2e-5, 2e-4, "embeddings");
    close(output.shared_state, expected.at("shared_state"), 2e-5, 2e-4, "shared_state");
    close(output.routing.scores, expected.at("router_scores"), 2e-5, 2e-4, "router_scores");
    close(output.routing.pre_top_k_probabilities, expected.at("router_probabilities"), 2e-5, 2e-4, "router_probabilities");
    if (!torch::equal(output.routing.selected_indices.cpu(), expected.at("top_k_indices"))) {
        throw std::runtime_error("top-K mismatch without an accepted silent fallback; inspect router_scores for a genuine tie");
    }
    close(output.routing.selected_weights, expected.at("selected_weights"), 2e-5, 2e-4, "selected_weights");
    for (std::size_t index = 0; index < model.module()->nodes().size(); ++index) {
        const auto proposal = model.module()->nodes()[index]->forward({output.embeddings}).proposal;
        close(
            proposal,
            expected.at("n1_proposal_" + std::to_string(index)),
            4e-5,
            4e-4,
            "individual N1 proposal " + std::to_string(index));
    }
    close(output.proposals, expected.at("proposals"), 3e-5, 3e-4, "proposals");
    close(output.integrated_state, expected.at("integrated_state"), 3e-5, 3e-4, "integrated_state");
    close(output.logits, expected.at("logits"), 4e-5, 4e-4, "logits");
    close(loss_for(output.logits, expected.at("targets")).reshape({1}), expected.at("loss"), 4e-5, 4e-4, "loss");
    CHECK(torch::equal(output.execution_trace->dispatch_permutation.cpu(), expected.at("dispatch_permutation")));
}

void gradient_parity() {
    const auto root = fixture_root();
    const auto forward = load_bundle(root / "forward.pt");
    const auto expected = load_bundle(root / "gradients.pt");
    emc::EMCModel model(emc::load_model_config(root / "model.rvcfg"));
    model.import_python_weights(root / "weights.pt");
    model.eval();
    emc::CausalIntervention intervention;
    intervention.forced_nodes = expected.at("forced_nodes");
    const auto output = model.forward({forward.at("tokens"), true, intervention});
    const auto loss = loss_for(output.logits, forward.at("targets"));
    loss.backward();
    close(loss.reshape({1}), expected.at("loss"), 4e-5, 4e-4, "gradient loss");
    const auto named = model.module()->named_parameters(true);
    for (const auto& [name, gradient] : expected) {
        if (!name.starts_with("gradient.")) continue;
        const auto parameter_name = name.substr(std::string("gradient.").size());
        const auto* parameter = named.find(parameter_name);
        CHECK(parameter != nullptr && parameter->grad().defined());
        close(parameter->grad(), gradient, 8e-5, 8e-4, parameter_name);
    }
}

void training_parity() {
    const auto root = fixture_root();
    const auto forward = load_bundle(root / "forward.pt");
    const auto expected = load_bundle(root / "training.pt");
    emc::EMCModel model(emc::load_model_config(root / "model.rvcfg"));
    model.import_python_weights(root / "weights.pt");
    model.train();
    torch::optim::AdamW optimizer(model.parameters(), torch::optim::AdamWOptions(3e-4).weight_decay(0.01));
    std::vector<torch::Tensor> losses;
    std::vector<torch::Tensor> routes;
    std::vector<torch::Tensor> norms;
    for (int step = 0; step < 3; ++step) {
        optimizer.zero_grad();
        const auto output = model.forward({forward.at("tokens"), true});
        const auto loss = loss_for(output.logits, forward.at("targets"));
        loss.backward();
        double squared = 0.0;
        for (const auto& parameter : model.parameters()) {
            if (parameter.grad().defined()) squared += parameter.grad().to(torch::kFloat32).square().sum().item<double>();
        }
        optimizer.step();
        losses.push_back(loss.detach());
        routes.push_back(output.routing.selected_indices.detach());
        norms.push_back(torch::tensor(std::sqrt(squared)));
    }
    close(torch::stack(losses), expected.at("losses"), 2e-4, 2e-3, "training losses");
    CHECK(torch::equal(torch::stack(routes).cpu(), expected.at("routes")));
    close(torch::stack(norms), expected.at("gradient_norms"), 3e-4, 3e-3, "training gradient norms");
    const auto named = model.module()->named_parameters(true);
    for (const auto& [name, value] : expected) {
        if (!name.starts_with("final.")) continue;
        const auto* parameter = named.find(name.substr(6));
        CHECK(parameter != nullptr);
        close(*parameter, value, 3e-4, 3e-3, name);
    }
}

void trainer_resume_and_telemetry() {
    const auto root = temporary_root();
    std::filesystem::create_directories(root);
    std::vector<std::int32_t> values(256);
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = static_cast<std::int32_t>(index % 67);
    }
    emc::TokenStream::save(
        root / "train.rvtok", values, "trainer/train");
    emc::TokenStream::save(
        root / "validation.rvtok", values, "trainer/validation");
    const auto training =
        emc::TokenStream::load(root / "train.rvtok");
    const auto validation =
        emc::TokenStream::load(root / "validation.rvtok");

    emc::TrainingConfig config;
    config.steps = 2;
    config.batch_size = 2;
    config.sequence_length = 8;
    config.evaluation_interval = 1;
    config.evaluation_batches = 1;
    config.milestones = {16, 32, 48};
    emc::EMCModel model(small_config());
    emc::Trainer trainer(model, config, torch::kCPU);
    const auto first = trainer.train(
        training, validation, root / "checkpoints");
    CHECK(first.steps_completed == 2);
    CHECK(first.tokens_processed == 32);
    CHECK(std::filesystem::exists(
        root / "checkpoints" / "telemetry.tsv"));
    CHECK(std::filesystem::exists(
        root / "checkpoints" / "latest" / "rng.pt"));

    config.steps = 3;
    emc::EMCModel resumed_model(small_config());
    emc::Trainer resumed(resumed_model, config, torch::kCPU);
    const auto progress =
        resumed.resume(root / "checkpoints" / "latest");
    CHECK(progress.step == 2);
    const auto second = resumed.train(
        training, validation, root / "resumed");
    CHECK(second.steps_completed == 3);
    CHECK(second.tokens_processed == 48);
    std::filesystem::remove_all(root);
}

void no_delta_path() {
    bool rejected = false;
    try { (void)emc::n1_family_from_string("delta"); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
}

}  // namespace

int main() {
    torch::set_num_threads(1);
    torch::manual_seed(42);
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"model construction", model_construction},
        {"GPT N1 forward", [] { n1_forward(emc::N1Family::gpt); }},
        {"SSM N1 forward", [] { n1_forward(emc::N1Family::ssm); }},
        {"recurrent N1 forward", [] { n1_forward(emc::N1Family::recurrent); }},
        {"independent same-family parameters", independent_parameters},
        {"Nexus routing", nexus_routing},
        {"exact top-K count", exact_top_k},
        {"unselected N1 sparse execution", sparse_execution},
        {"dispatch inverse ordering", dispatch_ordering},
        {"Integrator shape and gradient", integrator_gradients},
        {"weight tying", weight_tying},
        {"FP32 forward", fp32_forward},
        {"BF16 forward", bf16_forward},
        {"checkpoint round trip", checkpoint_round_trip},
        {"milestone checkpoint save", milestone_save},
        {"diagnostics", diagnostics},
        {"causal disable replace zero force", causal_interventions},
        {"deterministic token stream", deterministic_token_stream},
        {"Python C++ forward parity", forward_parity},
        {"Python C++ gradient parity", gradient_parity},
        {"tiny multi-step training parity", training_parity},
        {"native trainer resume and telemetry", trainer_resume_and_telemetry},
        {"Delta excluded", no_delta_path},
    };
    std::size_t failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    }
    std::cout << (tests.size() - failures) << '/' << tests.size() << " tests passed\n";
    return failures == 0 ? 0 : 1;
}
