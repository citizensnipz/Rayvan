#include "rayvan_emc/diagnostics/diagnostics.hpp"
#include "rayvan_emc/diagnostics/telemetry.hpp"

#include <torch/optim/adamw.h>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <unistd.h>
#include <fstream>
#endif

#if defined(RAYVAN_TORCH_CUDA)
#include <c10/cuda/CUDACachingAllocator.h>
#define RAYVAN_HAS_CUDA_ALLOCATOR 1
#endif

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace rayvan::emc {
namespace {

std::vector<double> doubles(const Tensor& tensor) {
    const auto contiguous = tensor.to(torch::kCPU, torch::kFloat64).contiguous().reshape({-1});
    const auto* data = contiguous.data_ptr<double>();
    return {data, data + contiguous.numel()};
}

std::uint64_t tensor_bytes(const Tensor& tensor) {
    return tensor.defined() ? static_cast<std::uint64_t>(tensor.numel() * tensor.element_size()) : 0;
}

std::pair<std::uint64_t, std::uint64_t> process_memory() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters))) {
        return {
            static_cast<std::uint64_t>(counters.WorkingSetSize),
            static_cast<std::uint64_t>(counters.PeakWorkingSetSize)};
    }
#elif defined(__linux__)
    std::ifstream stream("/proc/self/statm");
    std::uint64_t total = 0;
    std::uint64_t resident = 0;
    if (stream >> total >> resident) {
        const auto bytes = resident * static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
        return {bytes, bytes};
    }
#endif
    return {0, 0};
}

void write_vector(std::ostream& stream, const std::vector<double>& values) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) stream << ',';
        stream << values[index];
    }
}

}  // namespace

DiagnosticAccumulator::DiagnosticAccumulator(std::int64_t num_nodes) : num_nodes_(num_nodes) {
    if (num_nodes <= 0) throw std::invalid_argument("num_nodes must be positive");
}

void DiagnosticAccumulator::update(const EMCOutput& output) {
    const auto& routing = output.routing;
    const auto device = routing.scores.device();
    const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);
    const auto batch = routing.scores.size(0);
    const auto selected = routing.selected_indices.size(1);
    if (batches_ == 0) {
        selection_count_ = torch::zeros({num_nodes_}, options);
        request_selection_count_ = torch::zeros({num_nodes_}, options);
        probability_sum_ = torch::zeros({num_nodes_}, options);
        selected_weight_sum_ = torch::zeros({num_nodes_}, options);
        selected_weight_count_ = torch::zeros({num_nodes_}, options);
        slot_count_ = torch::zeros({num_nodes_, selected}, options);
        entropy_sum_ = torch::zeros({}, options);
        acceptance_sum_ = torch::zeros({num_nodes_}, options);
        proposal_norm_sum_ = torch::zeros({num_nodes_}, options);
        contribution_sum_ = torch::zeros({num_nodes_}, options);
        integrator_count_ = torch::zeros({num_nodes_}, options);
        similarity_sum_ = torch::zeros({}, options);
        similarity_count_ = torch::zeros({}, options);
    }
    if (slot_count_.size(1) != selected) throw std::invalid_argument("top-K changed while diagnostics were accumulating");

    const auto ids = routing.selected_indices.reshape({-1});
    const auto weights = routing.selected_weights.reshape({-1}).to(torch::kFloat64);
    selection_count_.scatter_add_(0, ids, torch::ones_like(weights));
    selected_weight_sum_.scatter_add_(0, ids, weights);
    selected_weight_count_.scatter_add_(0, ids, torch::ones_like(weights));
    const auto request_hits = torch::zeros({batch, num_nodes_}, options)
                                  .scatter(1, routing.selected_indices, 1.0)
                                  .clamp_max(1.0)
                                  .sum(0);
    request_selection_count_ += request_hits;
    probability_sum_ += routing.pre_top_k_probabilities.to(torch::kFloat64).sum(0);
    for (std::int64_t slot = 0; slot < selected; ++slot) {
        auto slot_ids = routing.selected_indices.index({torch::indexing::Slice(), slot});
        auto column = slot_count_.index({torch::indexing::Slice(), slot});
        column.scatter_add_(0, slot_ids, torch::ones({batch}, options));
    }
    const auto probabilities = routing.pre_top_k_probabilities.clamp_min(1e-12).to(torch::kFloat64);
    entropy_sum_ += -(probabilities * probabilities.log()).sum();

    if (output.integrator_trace) {
        const auto& trace = *output.integrator_trace;
        const auto token_count = trace.proposal_acceptance.size(0) * trace.proposal_acceptance.size(1);
        const auto expanded_ids = routing.selected_indices.unsqueeze(1)
                                      .expand({-1, trace.proposal_acceptance.size(1), -1})
                                      .reshape({-1});
        acceptance_sum_.scatter_add_(0, expanded_ids, trace.proposal_acceptance.reshape({-1}).to(torch::kFloat64));
        proposal_norm_sum_.scatter_add_(0, expanded_ids, trace.proposal_norms.reshape({-1}).to(torch::kFloat64));
        contribution_sum_.scatter_add_(0, expanded_ids, trace.proposal_contributions.reshape({-1}).to(torch::kFloat64));
        integrator_count_.scatter_add_(0, expanded_ids, torch::ones({token_count * selected}, options));
        const auto similarity = trace.proposal_similarity.to(torch::kFloat64);
        const auto eye = torch::eye(selected, options).reshape({1, 1, selected, selected});
        similarity_sum_ += (similarity * (1.0 - eye)).sum();
        similarity_count_ += batch * trace.proposal_similarity.size(1) * selected * std::max<std::int64_t>(selected - 1, 0);
    }
    ++batches_;
}

RoutingReport DiagnosticAccumulator::routing_report() const {
    if (batches_ == 0) throw std::logic_error("no routing diagnostics accumulated");
    const auto total_assignments = selection_count_.sum().clamp_min(1.0);
    const auto total_requests = request_selection_count_.sum() / slot_count_.size(1);
    RoutingReport report;
    report.selection_frequency = doubles(selection_count_ / total_assignments);
    report.request_selection_fraction = doubles(request_selection_count_ / total_requests.clamp_min(1.0));
    report.pre_top_k_probability = doubles(probability_sum_ / total_requests.clamp_min(1.0));
    report.selected_routing_weight = doubles(selected_weight_sum_ / selected_weight_count_.clamp_min(1.0));
    const auto slots = slot_count_.to(torch::kCPU, torch::kFloat64);
    report.slot_distribution.resize(static_cast<std::size_t>(num_nodes_));
    for (std::int64_t node = 0; node < num_nodes_; ++node) {
        report.slot_distribution[static_cast<std::size_t>(node)] = doubles(slots.index({node}) / slots.index({node}).sum().clamp_min(1.0));
    }
    const auto requests = total_requests.item<double>();
    const auto mean_entropy = entropy_sum_.item<double>() / std::max(requests, 1.0);
    report.normalized_entropy = num_nodes_ > 1 ? mean_entropy / std::log(static_cast<double>(num_nodes_)) : 1.0;
    report.effective_n1_count = std::exp(mean_entropy);
    report.near_dead.reserve(report.request_selection_fraction.size());
    for (const auto fraction : report.request_selection_fraction) report.near_dead.push_back(fraction < 0.01);
    const auto slot_totals = slot_count_.sum(0).clamp_min(1.0);
    report.slot_monopoly = (slot_count_ / slot_totals.unsqueeze(0)).max().item<double>();
    const auto maximum = *std::max_element(report.request_selection_fraction.begin(), report.request_selection_fraction.end());
    report.partial_collapse = report.slot_monopoly >= 0.80 || std::any_of(report.near_dead.begin(), report.near_dead.end(), [](bool value) { return value; });
    report.global_collapse = maximum >= 0.95 || report.effective_n1_count < 1.5;
    return report;
}

IntegratorReport DiagnosticAccumulator::integrator_report() const {
    if (batches_ == 0) throw std::logic_error("no Integrator diagnostics accumulated");
    IntegratorReport report;
    report.acceptance = doubles(acceptance_sum_ / integrator_count_.clamp_min(1.0));
    report.proposal_norm = doubles(proposal_norm_sum_ / integrator_count_.clamp_min(1.0));
    report.contribution = doubles(contribution_sum_ / integrator_count_.clamp_min(1.0));
    report.mean_similarity = (similarity_sum_ / similarity_count_.clamp_min(1.0)).item<double>();
    return report;
}

void DiagnosticAccumulator::reset() {
    batches_ = 0;
    selection_count_ = Tensor();
    request_selection_count_ = Tensor();
    probability_sum_ = Tensor();
    selected_weight_sum_ = Tensor();
    selected_weight_count_ = Tensor();
    slot_count_ = Tensor();
    entropy_sum_ = Tensor();
    acceptance_sum_ = Tensor();
    proposal_norm_sum_ = Tensor();
    contribution_sum_ = Tensor();
    integrator_count_ = Tensor();
    similarity_sum_ = Tensor();
    similarity_count_ = Tensor();
}

MemoryReport collect_memory_report(
    const EMCModel& model,
    const torch::optim::Optimizer* optimizer,
    std::int64_t cuda_device) {
    MemoryReport report;
    const auto [rss, peak_rss] = process_memory();
    report.process_rss_bytes = rss;
    report.process_peak_rss_bytes = peak_rss;
    std::unordered_set<const void*> seen;
    for (const auto& parameter : model.parameters()) {
        const auto* identity = parameter.unsafeGetTensorImpl();
        if (seen.insert(identity).second) report.parameter_bytes += tensor_bytes(parameter);
        if (parameter.grad().defined()) report.gradient_bytes += tensor_bytes(parameter.grad());
    }
    if (optimizer) {
        for (const auto& [_, state] : optimizer->state()) {
            const auto* adam = dynamic_cast<const torch::optim::AdamWParamState*>(state.get());
            if (!adam) continue;
            report.optimizer_bytes += tensor_bytes(adam->exp_avg()) + tensor_bytes(adam->exp_avg_sq()) + tensor_bytes(adam->max_exp_avg_sq());
        }
    }
#if defined(RAYVAN_HAS_CUDA_ALLOCATOR)
    if (torch::cuda::is_available()) {
        const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(static_cast<c10::DeviceIndex>(cuda_device));
        constexpr auto aggregate = static_cast<std::size_t>(c10::CachingAllocator::StatType::AGGREGATE);
        report.cuda_allocated_bytes = stats.allocated_bytes[aggregate].current;
        report.cuda_reserved_bytes = stats.reserved_bytes[aggregate].current;
        report.cuda_peak_allocated_bytes = stats.allocated_bytes[aggregate].peak;
        report.cuda_peak_reserved_bytes = stats.reserved_bytes[aggregate].peak;
    }
#else
    (void)cuda_device;
#endif
    return report;
}

double global_parameter_norm(const EMCModel& model) {
    auto total = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat64));
    for (const auto& parameter : model.parameters()) total += parameter.detach().to(torch::kCPU, torch::kFloat64).pow(2).sum();
    return std::sqrt(total.item<double>());
}

double global_gradient_norm(const EMCModel& model) {
    auto total = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat64));
    for (const auto& parameter : model.parameters()) {
        if (parameter.grad().defined()) total += parameter.grad().detach().to(torch::kCPU, torch::kFloat64).pow(2).sum();
    }
    return std::sqrt(total.item<double>());
}

void append_telemetry(const std::filesystem::path& path, const TelemetryRecord& record) {
    const bool header = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
    std::ofstream stream(path, std::ios::binary | std::ios::app);
    if (!stream) throw std::runtime_error("cannot append telemetry: " + path.string());
    if (header) {
        stream << "step\ttokens\tloss\tppl\ttokens_per_second\twall_seconds\tparameter_norm\tgradient_norm\tupdate_norm"
                  "\trouting_frequency\trouting_request_fraction\trouting_probability\trouting_weight\trouting_entropy"
                  "\teffective_n1\tslot_monopoly\tintegrator_acceptance\tproposal_norm\tcontribution\tproposal_similarity"
                  "\tprocess_rss\tparameter_bytes\toptimizer_bytes\tcuda_allocated\tcuda_reserved\tcuda_peak\n";
    }
    stream << std::setprecision(10)
           << record.step << '\t' << record.tokens_processed << '\t' << record.loss << '\t' << record.perplexity << '\t'
           << record.tokens_per_second << '\t' << record.wall_seconds << '\t' << record.parameter_norm << '\t'
           << record.gradient_norm << '\t' << record.update_norm << '\t';
    write_vector(stream, record.routing.selection_frequency); stream << '\t';
    write_vector(stream, record.routing.request_selection_fraction); stream << '\t';
    write_vector(stream, record.routing.pre_top_k_probability); stream << '\t';
    write_vector(stream, record.routing.selected_routing_weight); stream << '\t'
           << record.routing.normalized_entropy << '\t' << record.routing.effective_n1_count << '\t'
           << record.routing.slot_monopoly << '\t';
    write_vector(stream, record.integrator.acceptance); stream << '\t';
    write_vector(stream, record.integrator.proposal_norm); stream << '\t';
    write_vector(stream, record.integrator.contribution); stream << '\t'
           << record.integrator.mean_similarity << '\t' << record.memory.process_rss_bytes << '\t'
           << record.memory.parameter_bytes << '\t' << record.memory.optimizer_bytes << '\t'
           << record.memory.cuda_allocated_bytes << '\t' << record.memory.cuda_reserved_bytes << '\t'
           << record.memory.cuda_peak_allocated_bytes << '\n';
}

}  // namespace rayvan::emc
