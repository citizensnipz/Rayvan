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


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def parameter_counts(model: nn.Module) -> ParameterCounts:
    total = count_parameters(model)
    if not isinstance(model, EMCModel):
        return ParameterCounts(total, total, total)

    module_counts = [count_parameters(module) for module in model.emc_modules]
    average_module = sum(module_counts) // len(module_counts)
    router_and_integrator = count_parameters(model.router) + count_parameters(
        model.integrator
    )
    single_pass = (
        count_parameters(model.token_embedding)
        + count_parameters(model.position_embedding)
        + count_parameters(model.output_norm)
        + count_parameters(model.output_projection)
    )
    selected_modules = model.config.modules_per_cycle * average_module
    active_per_cycle = single_pass + router_and_integrator + selected_modules
    active_uses = single_pass + model.config.num_cycles * (
        router_and_integrator + selected_modules
    )
    return ParameterCounts(total, active_per_cycle, active_uses)


@dataclass(frozen=True)
class RoutingReport:
    selection_counts: tuple[tuple[int, ...], ...]
    routing_distribution_per_cycle: tuple[tuple[float, ...], ...]
    traffic_fraction: tuple[float, ...]
    mean_router_entropy: tuple[float, ...]
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
    module_update_norms: tuple[float, ...]
    module_updates_diverged: bool


class EMCDiagnostics:
    def __init__(self, model: EMCModel) -> None:
        config = model.config
        self.selection_counts = torch.zeros(
            config.num_cycles, config.num_modules, dtype=torch.long
        )
        self.entropy_sums = torch.zeros(config.num_cycles, dtype=torch.float64)
        self.observations = torch.zeros(config.num_cycles, dtype=torch.long)
        self.routes = [set() for _ in range(config.num_cycles)]
        self.maximum_router_gradient_norm = 0.0
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
            entropies = -(
                probabilities * probabilities.clamp_min(1e-12).log()
            ).sum(dim=-1)
            self.entropy_sums[cycle_index] += entropies.sum()
            self.observations[cycle_index] += entropies.numel()
            self.routes[cycle_index].update(
                tuple(decision.tolist()) for decision in decisions
            )

    def observe_router_gradients(self, model: EMCModel) -> None:
        squared_norm = 0.0
        for parameter in model.router.parameters():
            if parameter.grad is not None:
                squared_norm += parameter.grad.detach().float().square().sum().item()
        self.maximum_router_gradient_norm = max(
            self.maximum_router_gradient_norm, squared_norm**0.5
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
        differs_across_cycles = any(
            not torch.allclose(distributions[0], distribution)
            for distribution in distributions[1:]
        ) if distributions else False
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
        updates_diverged = any(
            not torch.allclose(update_vectors[0], update)
            for update in update_vectors[1:]
        ) if update_vectors else False

        return RoutingReport(
            selection_counts=tuple(
                tuple(row.tolist()) for row in self.selection_counts
            ),
            routing_distribution_per_cycle=tuple(per_cycle_distributions),
            traffic_fraction=tuple(traffic.tolist()),
            mean_router_entropy=tuple(mean_entropies),
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
            module_update_norms=update_norms,
            module_updates_diverged=updates_diverged,
        )


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
