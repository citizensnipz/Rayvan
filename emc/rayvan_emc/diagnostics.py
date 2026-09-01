from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .model import EMCCycleTrace, EMCModel


@dataclass(frozen=True)
class ParameterCounts:
    total: int
    approximate_active_per_cycle: int
    approximate_parameter_uses_per_forward: int



@dataclass(frozen=True)
class EMCParameterBreakdown:
    token_embeddings: int
    position_embeddings: int
    shared_core: int
    router: int
    modules_individual: tuple[int, ...]
    modules_combined: int
    integrator: int
    final_normalization: int
    output_projection_unique: int
    output_weight_tied: bool
    total_parameters: int
    trainable_parameters: int

def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _count_unique_parameters(modules: tuple[nn.Module, ...]) -> int:
    seen: set[int] = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                total += parameter.numel()
    return total



def parameter_breakdown(model: nn.Module) -> EMCParameterBreakdown:
    seen: set[int] = set()
    token_embeddings = _consume_unique_parameters(model.token_embedding, seen)
    position_embeddings = _consume_unique_parameters(model.position_embedding, seen)
    shared_core_module = getattr(model, "shared_core", None)
    shared_core = (
        _consume_unique_parameters(shared_core_module, seen)
        if shared_core_module is not None
        else 0
    )
    router = _consume_unique_parameters(model.router, seen)
    modules_individual = tuple(
        _consume_unique_parameters(module, seen) for module in model.emc_modules
    )
    integrator = _consume_unique_parameters(model.integrator, seen)
    final_normalization = _consume_unique_parameters(model.output_norm, seen)
    output_projection = _consume_unique_parameters(model.output_projection, seen)
    total = sum(parameter.numel() for parameter in model.parameters())
    return EMCParameterBreakdown(
        token_embeddings=token_embeddings,
        position_embeddings=position_embeddings,
        shared_core=shared_core,
        router=router,
        modules_individual=modules_individual,
        modules_combined=sum(modules_individual),
        integrator=integrator,
        final_normalization=final_normalization,
        output_projection_unique=output_projection,
        output_weight_tied=(
            model.output_projection.weight is model.token_embedding.weight
        ),
        total_parameters=total,
        trainable_parameters=sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )


def format_parameter_breakdown(breakdown: EMCParameterBreakdown) -> str:
    module_lines = "\n".join(
        f"  module m{index}:           {_format_parameters(count):>9}"
        for index, count in enumerate(breakdown.modules_individual)
    )
    tied_note = " (weight tied to embeddings)" if breakdown.output_weight_tied else ""
    return "\n".join(
        (
            "EMC parameter breakdown:",
            f"  token embeddings:      {_format_parameters(breakdown.token_embeddings):>9}",
            f"  position embeddings:   {_format_parameters(breakdown.position_embeddings):>9}",
            f"  shared core:           {_format_parameters(breakdown.shared_core):>9}",
            f"  router / Nexus:        {_format_parameters(breakdown.router):>9}",
            f"  modules combined:      {_format_parameters(breakdown.modules_combined):>9}",
            module_lines,
            f"  Integrator:            {_format_parameters(breakdown.integrator):>9}",
            f"  final normalization:   {_format_parameters(breakdown.final_normalization):>9}",
            f"  output projection:     {_format_parameters(breakdown.output_projection_unique):>9}{tied_note}",
            f"  total parameters:      {_format_parameters(breakdown.total_parameters):>9}",
            f"  trainable parameters:  {_format_parameters(breakdown.trainable_parameters):>9}",
        )
    )


def _consume_unique_parameters(module: nn.Module, seen: set[int]) -> int:
    total = 0
    for parameter in module.parameters():
        identity = id(parameter)
        if identity not in seen:
            seen.add(identity)
            total += parameter.numel()
    return total


def _format_parameters(count: int) -> str:
    return f"{count / 1_000_000:.3f}M"

def parameter_counts(model: nn.Module) -> ParameterCounts:
    total = count_parameters(model)
    if not hasattr(model, "emc_modules"):
        return ParameterCounts(total, total, total)

    module_counts = [count_parameters(module) for module in model.emc_modules]
    average_module = sum(module_counts) // len(module_counts)
    router_and_integrator = _count_unique_parameters(
        (model.router, model.integrator)
    )
    single_modules: tuple[nn.Module, ...] = (
        model.token_embedding,
        model.position_embedding,
        model.output_norm,
        model.output_projection,
    )
    shared_core_module = getattr(model, "shared_core", None)
    if shared_core_module is not None:
        single_modules = (*single_modules, shared_core_module)
    single_pass = _count_unique_parameters(single_modules)
    active_top_k = (
        model.config.resolved_active_top_k
        if getattr(model.config, "architecture_stage", "token") == "n1_chunked"
        else model.config.modules_per_cycle
    )
    selected_modules = active_top_k * average_module
    active_per_cycle = single_pass + router_and_integrator + selected_modules
    if getattr(model.config, "architecture_stage", "token") == "n1_chunked":
        active_uses = active_per_cycle
    else:
        active_uses = single_pass + model.config.num_cycles * (
            router_and_integrator + selected_modules
        )
    return ParameterCounts(total, active_per_cycle, active_uses)


@dataclass(frozen=True)
class RoutingReport:
    selection_counts: tuple[tuple[int, ...], ...]
    routing_distribution_per_cycle: tuple[tuple[float, ...], ...]
    traffic_fraction: tuple[float, ...]
    average_routing_probability: tuple[float, ...]
    mean_router_entropy: tuple[float, ...]
    family_traffic_fraction: tuple[tuple[str, float], ...]
    all_modules_used: bool
    dominant_modules: tuple[int, ...]
    dominant_traffic_fraction: float
    top_1_traffic_share: float
    top_2_traffic_share: float
    minimum_module_share: float
    normalized_routing_entropy: float
    effective_active_modules: float
    severe_collapse: bool
    routing_collapsed: bool
    routing_differs_across_inputs: bool
    routing_differs_across_cycles: bool
    maximum_router_gradient_norm: float
    module_gradient_norms: tuple[float, ...]
    module_parameter_counts: tuple[int, ...]
    module_update_norms: tuple[float, ...]
    module_updates_diverged: bool
    average_integrator_acceptance: tuple[float, ...]
    average_proposal_norm: tuple[float, ...]
    average_proposal_contribution: tuple[float, ...]
    family_integrator_acceptance: tuple[tuple[str, float], ...]
    mean_proposal_similarity: float
    mean_integrated_update_norm: float
    mean_gate_magnitude: float
    expert_names: tuple[str, ...] = ()


class EMCDiagnostics:
    def __init__(self, model: EMCModel) -> None:
        config = model.config
        shape = (config.num_cycles, config.num_modules)
        self.module_families = model.module_families
        self.expert_names = tuple(
            getattr(
                model,
                "expert_names",
                tuple(f"m{index}" for index in range(config.num_modules)),
            )
        )
        self.selection_counts = torch.zeros(*shape, dtype=torch.long)
        self.routing_probability_sums = torch.zeros(*shape, dtype=torch.float64)
        self.acceptance_sums = torch.zeros(*shape, dtype=torch.float64)
        self.proposal_norm_sums = torch.zeros(*shape, dtype=torch.float64)
        self.contribution_sums = torch.zeros(*shape, dtype=torch.float64)
        self.entropy_sums = torch.zeros(config.num_cycles, dtype=torch.float64)
        self.observations = torch.zeros(config.num_cycles, dtype=torch.long)
        self.similarity_sum = 0.0
        self.similarity_count = 0
        self.integrated_update_norm_sum = 0.0
        self.gate_magnitude_sum = 0.0
        self.integrator_observations = 0
        self.routes = [set() for _ in range(config.num_cycles)]
        self.maximum_router_gradient_norm = 0.0
        self.maximum_module_gradient_norms = [0.0] * config.num_modules
        self.initial_module_fingerprints = tuple(
            _module_fingerprint(module) for module in model.emc_modules
        )

    def observe_trace(self, trace: tuple[EMCCycleTrace, ...]) -> None:
        for cycle_trace in trace:
            cycle_index = cycle_trace.cycle - 1
            if cycle_trace.selected_indices is not None:
                decisions = cycle_trace.selected_indices.reshape(
                    -1, cycle_trace.selected_indices.size(-1)
                )
            else:
                decisions = torch.tensor(
                    [cycle_trace.selected_modules], dtype=torch.long
                )
            counts = torch.bincount(
                decisions.reshape(-1), minlength=self.selection_counts.size(1)
            )
            self.selection_counts[cycle_index] += counts
            probabilities = torch.softmax(
                cycle_trace.router_scores.to(torch.float64), dim=-1
            )
            self.routing_probability_sums[cycle_index] += probabilities.reshape(
                -1, probabilities.size(-1)
            ).sum(dim=0)
            entropies = -(
                probabilities * probabilities.clamp_min(1e-12).log()
            ).sum(dim=-1)
            self.entropy_sums[cycle_index] += entropies.sum()
            self.observations[cycle_index] += entropies.numel()
            self.routes[cycle_index].update(
                tuple(decision.tolist()) for decision in decisions
            )

            integrator_trace = cycle_trace.integrator_trace
            if integrator_trace is None or cycle_trace.selected_indices is None:
                continue
            selected = cycle_trace.selected_indices.reshape(-1)
            for source, destination in (
                (integrator_trace.proposal_acceptance, self.acceptance_sums),
                (integrator_trace.proposal_norms, self.proposal_norm_sums),
                (integrator_trace.proposal_contributions, self.contribution_sums),
            ):
                destination[cycle_index].scatter_add_(
                    0, selected, source.to(torch.float64).reshape(-1)
                )
            similarity = integrator_trace.proposal_similarity
            selected_count = similarity.size(-1)
            if selected_count > 1:
                off_diagonal = ~torch.eye(
                    selected_count, dtype=torch.bool
                ).reshape(1, 1, selected_count, selected_count)
                values = similarity[off_diagonal.expand_as(similarity)]
                self.similarity_sum += values.sum().item()
                self.similarity_count += values.numel()
            self.integrated_update_norm_sum += (
                integrator_trace.integrated_update_norm.sum().item()
            )
            self.gate_magnitude_sum += integrator_trace.gate_magnitude.sum().item()
            self.integrator_observations += (
                integrator_trace.gate_magnitude.numel()
            )

    def observe_router_gradients(self, model: EMCModel) -> None:
        squared_norm = 0.0
        for parameter in model.router.parameters():
            if parameter.grad is not None:
                squared_norm += parameter.grad.detach().float().square().sum().item()
        self.maximum_router_gradient_norm = max(
            self.maximum_router_gradient_norm, squared_norm**0.5
        )

    def observe_module_gradients(self, model: EMCModel) -> None:
        for index, module in enumerate(model.emc_modules):
            squared_norm = sum(
                parameter.grad.detach().float().square().sum().item()
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.maximum_module_gradient_norms[index] = max(
                self.maximum_module_gradient_norms[index], squared_norm**0.5
            )

    def report(self, model: EMCModel) -> RoutingReport:
        per_cycle_distributions: list[tuple[float, ...]] = []
        mean_entropies: list[float] = []
        for cycle_index in range(self.selection_counts.size(0)):
            counts = self.selection_counts[cycle_index]
            total = counts.sum().item()
            distribution = (
                counts.to(torch.float64) / total
                if total
                else torch.zeros_like(counts, dtype=torch.float64)
            )
            per_cycle_distributions.append(tuple(distribution.tolist()))
            observations = self.observations[cycle_index].item()
            mean_entropies.append(
                self.entropy_sums[cycle_index].item() / observations
                if observations
                else 0.0
            )

        overall_counts = self.selection_counts.sum(dim=0)
        overall_total = overall_counts.sum().item()
        traffic = (
            overall_counts.to(torch.float64) / overall_total
            if overall_total
            else torch.zeros_like(overall_counts, dtype=torch.float64)
        )
        total_observations = self.observations.sum().item()
        average_probability = (
            self.routing_probability_sums.sum(dim=0) / total_observations
            if total_observations
            else torch.zeros_like(traffic)
        )
        selection_denominator = overall_counts.clamp_min(1).to(torch.float64)
        average_acceptance = self.acceptance_sums.sum(dim=0) / selection_denominator
        average_norm = self.proposal_norm_sums.sum(dim=0) / selection_denominator
        average_contribution = (
            self.contribution_sums.sum(dim=0) / selection_denominator
        )
        dominant_count = min(2, traffic.numel())
        dominant_modules = tuple(
            torch.topk(traffic, k=dominant_count).indices.tolist()
        )
        top_1_share = traffic.max().item() if traffic.numel() else 0.0
        top_2_share = sum(traffic[index].item() for index in dominant_modules)
        minimum_share = traffic.min().item() if traffic.numel() else 0.0
        traffic_entropy = -(
            traffic * traffic.clamp_min(torch.finfo(traffic.dtype).tiny).log()
        ).sum().item()
        normalized_traffic_entropy = (
            traffic_entropy / math.log(traffic.numel())
            if traffic.numel() > 1
            else 1.0
        )
        effective_active_modules = math.exp(traffic_entropy) if overall_total else 0.0
        used_modules = int(torch.count_nonzero(overall_counts).item())
        severe_collapse = (
            used_modules <= model.config.modules_per_cycle
            or top_1_share >= 0.8
            or (traffic.numel() > 2 and top_2_share >= 0.9)
        )
        distributions = [torch.tensor(values) for values in per_cycle_distributions]
        differs_across_cycles = (
            any(
                not torch.allclose(distributions[0], distribution)
                for distribution in distributions[1:]
            )
            if distributions
            else False
        )
        final_fingerprints = tuple(
            _module_fingerprint(module) for module in model.emc_modules
        )
        update_vectors = tuple(
            final - initial
            for initial, final in zip(
                self.initial_module_fingerprints, final_fingerprints, strict=True
            )
        )
        update_norms = tuple(update.norm().item() for update in update_vectors)
        same_fingerprint_shape = (
            len({update.numel() for update in update_vectors}) <= 1
        )
        if same_fingerprint_shape and update_vectors:
            updates_diverged = any(
                not torch.allclose(update_vectors[0], update)
                for update in update_vectors[1:]
            )
        else:
            updates_diverged = (
                sum(norm > 1e-12 for norm in update_norms) > 1
            )

        family_traffic = _aggregate_by_family(
            self.module_families, traffic
        )
        family_acceptance = _aggregate_by_family(
            self.module_families,
            self.acceptance_sums.sum(dim=0),
            normalize=True,
        )
        return RoutingReport(
            selection_counts=tuple(
                tuple(row.tolist()) for row in self.selection_counts
            ),
            routing_distribution_per_cycle=tuple(per_cycle_distributions),
            traffic_fraction=tuple(traffic.tolist()),
            average_routing_probability=tuple(average_probability.tolist()),
            mean_router_entropy=tuple(mean_entropies),
            family_traffic_fraction=family_traffic,
            all_modules_used=used_modules == model.config.num_modules,
            dominant_modules=dominant_modules,
            dominant_traffic_fraction=top_2_share,
            top_1_traffic_share=top_1_share,
            top_2_traffic_share=top_2_share,
            minimum_module_share=minimum_share,
            normalized_routing_entropy=normalized_traffic_entropy,
            effective_active_modules=effective_active_modules,
            severe_collapse=severe_collapse,
            routing_collapsed=severe_collapse,
            routing_differs_across_inputs=any(
                len(cycle_routes) > 1 for cycle_routes in self.routes
            ),
            routing_differs_across_cycles=differs_across_cycles,
            maximum_router_gradient_norm=self.maximum_router_gradient_norm,
            module_gradient_norms=tuple(self.maximum_module_gradient_norms),
            module_parameter_counts=tuple(
                count_parameters(module) for module in model.emc_modules
            ),
            module_update_norms=update_norms,
            module_updates_diverged=updates_diverged,
            average_integrator_acceptance=tuple(average_acceptance.tolist()),
            average_proposal_norm=tuple(average_norm.tolist()),
            average_proposal_contribution=tuple(
                average_contribution.tolist()
            ),
            family_integrator_acceptance=family_acceptance,
            mean_proposal_similarity=(
                self.similarity_sum / self.similarity_count
                if self.similarity_count
                else 0.0
            ),
            mean_integrated_update_norm=(
                self.integrated_update_norm_sum / self.integrator_observations
                if self.integrator_observations
                else 0.0
            ),
            mean_gate_magnitude=(
                self.gate_magnitude_sum / self.integrator_observations
                if self.integrator_observations
                else 0.0
            ),
            expert_names=self.expert_names,
        )


def _aggregate_by_family(
    families: tuple[str, ...],
    values: Tensor,
    *,
    normalize: bool = False,
) -> tuple[tuple[str, float], ...]:
    totals: dict[str, float] = {}
    for family, value in zip(families, values.tolist(), strict=True):
        totals[family] = totals.get(family, 0.0) + float(value)
    if normalize:
        denominator = sum(totals.values())
        if denominator:
            totals = {
                family: value / denominator
                for family, value in totals.items()
            }
    return tuple(sorted(totals.items()))


def _module_fingerprint(module: nn.Module, samples_per_parameter: int = 16) -> Tensor:
    samples: list[Tensor] = []
    for parameter in module.parameters():
        flattened = parameter.detach().reshape(-1)
        sample_count = min(samples_per_parameter, flattened.numel())
        indices = torch.linspace(
            0,
            flattened.numel() - 1,
            steps=sample_count,
            device=flattened.device,
        ).long()
        samples.append(flattened[indices].float().cpu())
    return torch.cat(samples)
