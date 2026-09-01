from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn

from .chunked import ChunkedEMCModel, ChunkedExecutionTrace
from .model import EMCModel, EMCOutput
from .n2 import N2ExecutionTrace


TELEMETRY_SCHEMA_VERSION = 1


@dataclass
class _Window:
    num_modules: int
    families: tuple[str, ...]
    selection_counts: list[int] = field(init=False)
    request_counts: list[int] = field(init=False)
    request_pool_counts: list[int] = field(init=False)
    routing_unit_counts: list[int] = field(init=False)
    slot_counts: list[Counter[int]] = field(init=False)
    probability_sums: list[float] = field(init=False)
    probability_observations: list[int] = field(init=False)
    weight_sums: list[float] = field(init=False)
    weight_observations: list[int] = field(init=False)
    acceptance_sums: list[float] = field(init=False)
    acceptance_observations: list[int] = field(init=False)
    contribution_sums: list[float] = field(init=False)
    contribution_observations: list[int] = field(init=False)
    state_contribution_sums: list[float] = field(init=False)
    state_contribution_observations: list[int] = field(init=False)
    active_steps: list[int] = field(init=False)
    last_active_step: list[int] = field(init=False)
    total_selections: int = 0
    requests: int = 0
    request_pool_observations: int = 0
    routing_units: int = 0

    def __post_init__(self) -> None:
        self.selection_counts = [0] * self.num_modules
        self.request_counts = [0] * self.num_modules
        self.request_pool_counts = [0] * self.num_modules
        self.routing_unit_counts = [0] * self.num_modules
        self.slot_counts = [Counter() for _ in range(self.num_modules)]
        self.probability_sums = [0.0] * self.num_modules
        self.probability_observations = [0] * self.num_modules
        self.weight_sums = [0.0] * self.num_modules
        self.weight_observations = [0] * self.num_modules
        self.acceptance_sums = [0.0] * self.num_modules
        self.acceptance_observations = [0] * self.num_modules
        self.contribution_sums = [0.0] * self.num_modules
        self.contribution_observations = [0] * self.num_modules
        self.state_contribution_sums = [0.0] * self.num_modules
        self.state_contribution_observations = [0] * self.num_modules
        self.active_steps = [0] * self.num_modules
        self.last_active_step = [-1] * self.num_modules

    def observe_request(self, selected_by_request: Tensor, request_pool: Tensor | None) -> None:
        rows = selected_by_request.reshape(selected_by_request.size(0), -1)
        self.requests += rows.size(0)
        for module in range(self.num_modules):
            self.request_counts[module] += int((rows == module).any(dim=-1).sum().item())
        if request_pool is None:
            return
        pool_rows = request_pool.reshape(request_pool.size(0), -1)
        self.request_pool_observations += pool_rows.size(0)
        for module in range(self.num_modules):
            self.request_pool_counts[module] += int(
                (pool_rows == module).any(dim=-1).sum().item()
            )

    def observe_routing(
        self,
        selected: Tensor,
        scores: Tensor,
        weights: Tensor,
        *,
        step: int,
        acceptance: Tensor | None = None,
        contribution: Tensor | None = None,
        state_contribution: Tensor | None = None,
    ) -> None:
        selected_rows = selected.long().reshape(-1, selected.size(-1))
        weight_rows = weights.float().reshape(-1, weights.size(-1))
        score_rows = scores.float().reshape(-1, scores.size(-1))
        probabilities = torch.softmax(score_rows, dim=-1)
        self.routing_units += selected_rows.size(0)
        self.total_selections += selected_rows.numel()
        acceptance_rows = _proposal_rows(acceptance, selected_rows.shape)
        contribution_rows = _proposal_rows(contribution, selected_rows.shape)
        state_rows = _proposal_rows(state_contribution, selected_rows.shape)
        for module in range(self.num_modules):
            present = selected_rows == module
            count = int(present.sum().item())
            self.selection_counts[module] += count
            self.routing_unit_counts[module] += int(present.any(dim=-1).sum().item())
            self.probability_sums[module] += float(probabilities[:, module].sum().item())
            self.probability_observations[module] += probabilities.size(0)
            if count:
                self.weight_sums[module] += float(weight_rows[present].sum().item())
                self.weight_observations[module] += count
                for slot in range(selected_rows.size(-1)):
                    self.slot_counts[module][slot] += int(
                        (selected_rows[:, slot] == module).sum().item()
                    )
                if self.last_active_step[module] != step:
                    self.active_steps[module] += 1
                    self.last_active_step[module] = step
                _accumulate_selected(
                    present,
                    acceptance_rows,
                    self.acceptance_sums,
                    self.acceptance_observations,
                    module,
                )
                _accumulate_selected(
                    present,
                    contribution_rows,
                    self.contribution_sums,
                    self.contribution_observations,
                    module,
                )
                _accumulate_selected(
                    present,
                    state_rows,
                    self.state_contribution_sums,
                    self.state_contribution_observations,
                    module,
                )


class _N2TensorWindow:
    """GPU-side N2 telemetry reduced to fixed-size per-expert tensors."""

    def __init__(
        self, num_modules: int, families: tuple[str, ...], device: torch.device
    ) -> None:
        self.num_modules = num_modules
        self.families = families
        self.device = device
        self.selection_counts = torch.zeros(
            num_modules, dtype=torch.long, device=device
        )
        self.request_counts = torch.zeros(
            num_modules, dtype=torch.long, device=device
        )
        self.routing_unit_counts = torch.zeros(
            num_modules, dtype=torch.long, device=device
        )
        self.slot_counts = torch.zeros(
            num_modules, num_modules, dtype=torch.long, device=device
        )
        self.probability_sums = torch.zeros(
            num_modules, dtype=torch.float32, device=device
        )
        self.probability_observations = torch.zeros(
            num_modules, dtype=torch.long, device=device
        )
        self.weight_sums = torch.zeros(
            num_modules, dtype=torch.float32, device=device
        )
        self.weight_observations = torch.zeros(
            num_modules, dtype=torch.long, device=device
        )
        self.acceptance_sums = torch.zeros(
            num_modules, dtype=torch.float32, device=device
        )
        self.acceptance_observations = torch.zeros(
            num_modules, dtype=torch.long, device=device
        )
        self.contribution_sums = torch.zeros(
            num_modules, dtype=torch.float32, device=device
        )
        self.contribution_observations = torch.zeros(
            num_modules, dtype=torch.long, device=device
        )
        self.active_steps = torch.zeros(
            num_modules, dtype=torch.long, device=device
        )
        self.last_active_step = torch.full(
            (num_modules,), -1, dtype=torch.long, device=device
        )
        self.total_selections = 0
        self.requests = 0
        self.routing_units = 0

    def observe(
        self,
        selected: Tensor,
        scores: Tensor,
        weights: Tensor,
        *,
        step: int,
        acceptance: Tensor | None,
        contribution: Tensor | None,
    ) -> None:
        selected_rows = selected.to(device=self.device, dtype=torch.long).reshape(
            -1, selected.size(-1)
        )
        score_rows = scores.to(device=self.device, dtype=torch.float32).reshape(
            -1, scores.size(-1)
        )
        weight_rows = weights.to(device=self.device, dtype=torch.float32).reshape(
            -1, weights.size(-1)
        )
        one_hot = torch.nn.functional.one_hot(
            selected_rows, num_classes=self.num_modules
        )
        counts = one_hot.sum(dim=(0, 1))
        request_presence = one_hot.amax(dim=1).sum(dim=0)
        self.selection_counts.add_(counts)
        self.request_counts.add_(request_presence)
        self.routing_unit_counts.add_(request_presence)
        self.slot_counts[:, : selected_rows.size(1)].add_(
            one_hot.sum(dim=0).transpose(0, 1)
        )
        self.probability_sums.add_(torch.softmax(score_rows, dim=-1).sum(dim=0))
        self.probability_observations.add_(score_rows.size(0))
        self.weight_sums.scatter_add_(
            0, selected_rows.reshape(-1), weight_rows.reshape(-1)
        )
        self.weight_observations.add_(counts)
        acceptance_rows = _proposal_rows(acceptance, selected_rows.shape)
        if acceptance_rows is not None:
            self.acceptance_sums.scatter_add_(
                0,
                selected_rows.reshape(-1),
                acceptance_rows.to(self.device).reshape(-1),
            )
            self.acceptance_observations.add_(counts)
        contribution_rows = _proposal_rows(contribution, selected_rows.shape)
        if contribution_rows is not None:
            self.contribution_sums.scatter_add_(
                0,
                selected_rows.reshape(-1),
                contribution_rows.to(self.device).reshape(-1),
            )
            self.contribution_observations.add_(counts)
        active = counts > 0
        self.active_steps.add_(active & (self.last_active_step != step))
        self.last_active_step.masked_fill_(active, step)
        self.total_selections += selected_rows.numel()
        self.requests += selected_rows.size(0)
        self.routing_units += selected_rows.size(0)

    def materialize(self) -> _Window:
        window = _Window(self.num_modules, self.families)
        window.selection_counts = self.selection_counts.cpu().tolist()
        window.request_counts = self.request_counts.cpu().tolist()
        window.routing_unit_counts = self.routing_unit_counts.cpu().tolist()
        slot_rows = self.slot_counts.cpu().tolist()
        window.slot_counts = [
            Counter(
                {
                    slot: int(count)
                    for slot, count in enumerate(row)
                    if count
                }
            )
            for row in slot_rows
        ]
        window.probability_sums = self.probability_sums.cpu().tolist()
        window.probability_observations = (
            self.probability_observations.cpu().tolist()
        )
        window.weight_sums = self.weight_sums.cpu().tolist()
        window.weight_observations = self.weight_observations.cpu().tolist()
        window.acceptance_sums = self.acceptance_sums.cpu().tolist()
        window.acceptance_observations = (
            self.acceptance_observations.cpu().tolist()
        )
        window.contribution_sums = self.contribution_sums.cpu().tolist()
        window.contribution_observations = (
            self.contribution_observations.cpu().tolist()
        )
        window.active_steps = self.active_steps.cpu().tolist()
        window.last_active_step = self.last_active_step.cpu().tolist()
        window.total_selections = self.total_selections
        window.requests = self.requests
        window.routing_units = self.routing_units
        return window


class ModuleTelemetry:
    """Streaming, activation-free telemetry for EMC training milestones."""

    def __init__(self, model: EMCModel | ChunkedEMCModel) -> None:
        self.num_modules = model.config.num_modules
        self.families = tuple(model.module_families)
        self.chunked = isinstance(model, ChunkedEMCModel)
        self.n2 = getattr(model.config, "architecture_stage", None) == "n2"
        self.expert_names = tuple(
            getattr(
                model,
                "expert_names",
                tuple(f"m{index}" for index in range(self.num_modules)),
            )
        )
        self.device = next(model.parameters()).device
        if self.n2:
            self.cumulative = _N2TensorWindow(
                self.num_modules, self.families, self.device
            )
            self.interval = _N2TensorWindow(
                self.num_modules, self.families, self.device
            )
        else:
            self.cumulative = _Window(self.num_modules, self.families)
            self.interval = _Window(self.num_modules, self.families)
        self.last_snapshot_step = 0

    def observe(self, output: EMCOutput, step: int) -> None:
        if isinstance(output.chunk_trace, ChunkedExecutionTrace):
            self._observe_chunked(output.chunk_trace, step)
        elif isinstance(output.chunk_trace, N2ExecutionTrace):
            self._observe_n2(output, step)
        else:
            self._observe_token(output, step)

    def _observe_token(self, output: EMCOutput, step: int) -> None:
        if not output.trace:
            return
        per_request = torch.cat(
            [trace.selected_indices for trace in output.trace if trace.selected_indices is not None],
            dim=1,
        )
        for window in (self.interval, self.cumulative):
            window.observe_request(per_request, None)
        for trace in output.trace:
            if trace.selected_indices is None:
                continue
            integrator = trace.integrator_trace
            for window in (self.interval, self.cumulative):
                window.observe_routing(
                    trace.selected_indices,
                    trace.router_scores,
                    trace.router_weights,
                    step=step,
                    acceptance=(integrator.proposal_acceptance if integrator else None),
                    contribution=(integrator.proposal_contributions if integrator else None),
                )

    def _observe_chunked(self, trace: ChunkedExecutionTrace, step: int) -> None:
        if not trace.chunks:
            return
        per_request = torch.cat(
            [chunk.active_modules for chunk in trace.chunks], dim=1
        )
        for window in (self.interval, self.cumulative):
            window.observe_request(per_request, trace.request_pool.module_indices)
        for chunk in trace.chunks:
            for window in (self.interval, self.cumulative):
                window.observe_routing(
                    chunk.active_modules,
                    chunk.routing_scores,
                    chunk.routing_weights,
                    step=step,
                    acceptance=chunk.token_integrator_trace.proposal_acceptance,
                    contribution=chunk.token_integrator_trace.proposal_contributions,
                    state_contribution=(
                        chunk.state_integrator_trace.proposal_contributions
                    ),
                )

    def _observe_n2(self, output: EMCOutput, step: int) -> None:
        if not output.trace:
            return
        trace = output.trace[0]
        if trace.selected_indices is None:
            return
        selected = trace.selected_indices[:, 0]
        scores = trace.router_scores[:, 0]
        weights = trace.router_weights[:, 0]
        integrator = trace.integrator_trace
        acceptance = (
            integrator.proposal_acceptance.mean(dim=1)
            if integrator is not None
            else None
        )
        contribution = (
            integrator.proposal_contributions.mean(dim=1)
            if integrator is not None
            else None
        )
        for window in (self.interval, self.cumulative):
            if not isinstance(window, _N2TensorWindow):
                raise RuntimeError("N2 telemetry requires a tensor window")
            window.observe(
                selected,
                scores,
                weights,
                step=step,
                acceptance=acceptance,
                contribution=contribution,
            )


    def snapshot(
        self,
        model: nn.Module,
        *,
        milestone_tokens: int,
        observed_tokens: int,
        step: int,
        active_top_k: int | None,
        module_signals: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        interval_steps = max(0, step - self.last_snapshot_step)
        signals = {
            int(row["module"]): row
            for row in (module_signals or {}).get("modules", [])
        }
        interval_window = (
            self.interval.materialize()
            if isinstance(self.interval, _N2TensorWindow)
            else self.interval
        )
        cumulative_window = (
            self.cumulative.materialize()
            if isinstance(self.cumulative, _N2TensorWindow)
            else self.cumulative
        )
        interval_modules = self._module_rows(
            interval_window, model, signals, interval_steps
        )
        cumulative_modules = self._module_rows(
            cumulative_window, model, signals, step
        )
        record = {
            "milestone_tokens": milestone_tokens,
            "observed_tokens": observed_tokens,
            "step": step,
            "active_top_k": active_top_k,
            "routing_unit": (
                "chunk" if self.chunked else ("request" if self.n2 else "token-cycle")
            ),
            "interval_start_step": self.last_snapshot_step + 1,
            "interval_steps": interval_steps,
            "modules": interval_modules,
            "families": _family_rows(interval_modules),
            "cumulative_modules": cumulative_modules,
            "cumulative_families": _family_rows(cumulative_modules),
            "update_sampling": (
                "gradient and parameter norms plus the single optimizer update "
                "that crossed the milestone; no gradients or activations retained"
            ),
            "causal": None,
        }
        self.interval = (
            _N2TensorWindow(self.num_modules, self.families, self.device)
            if self.n2
            else _Window(self.num_modules, self.families)
        )
        self.last_snapshot_step = step
        return record

    def _module_rows(
        self,
        window: _Window,
        model: nn.Module,
        signals: Mapping[int, Mapping[str, Any]],
        step_count: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for module in range(self.num_modules):
            signal = signals.get(module, {})
            selection_count = window.selection_counts[module]
            rows.append(
                {
                    "module": module,
                    "family": self.families[module],
                    "expert_name": self.expert_names[module],
                    "selection_frequency": _fraction(
                        selection_count, window.total_selections
                    ),
                    "request_selection_fraction": _fraction(
                        window.request_counts[module], window.requests
                    ),
                    "request_pool_fraction": (
                        _fraction(
                            window.request_pool_counts[module],
                            window.request_pool_observations,
                        )
                        if window.request_pool_observations
                        else None
                    ),
                    "routing_unit_selection_fraction": _fraction(
                        window.routing_unit_counts[module], window.routing_units
                    ),
                    "chunk_selection_fraction": (
                        _fraction(
                            window.routing_unit_counts[module], window.routing_units
                        )
                        if self.chunked
                        else None
                    ),
                    "selection_slot_distribution": {
                        str(slot): _fraction(count, selection_count)
                        for slot, count in sorted(window.slot_counts[module].items())
                    },
                    "mean_router_probability_before_top_k": _mean_from_sum(
                        window.probability_sums[module],
                        window.probability_observations[module],
                    ),
                    "mean_normalized_selected_weight": _mean_from_sum(
                        window.weight_sums[module], window.weight_observations[module]
                    ),
                    "mean_integrator_acceptance": _mean_from_sum(
                        window.acceptance_sums[module],
                        window.acceptance_observations[module],
                    ),
                    "mean_integrator_token_contribution": _mean_from_sum(
                        window.contribution_sums[module],
                        window.contribution_observations[module],
                    ),
                    "mean_integrator_state_contribution": _mean_from_sum(
                        window.state_contribution_sums[module],
                        window.state_contribution_observations[module],
                    ),
                    "gradient_norm": signal.get("gradient_norm"),
                    "parameter_norm": signal.get("parameter_norm", _parameter_norm(model.emc_modules[module])),
                    "update_norm": signal.get("update_norm"),
                    "recent_update_norm": signal.get("update_norm"),
                    "active_step_fraction_since_previous_milestone": _fraction(
                        window.active_steps[module], step_count
                    ),
                    "selection_count": selection_count,
                    "routing_units_observed": window.routing_units,
                }
            )
        return rows


def write_developmental_record(
    output_directory: str | Path,
    record: Mapping[str, Any],
) -> Path:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "telemetry.json"
    payload: dict[str, Any]
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"schema_version": TELEMETRY_SCHEMA_VERSION, "milestones": []}
    milestones = [
        row
        for row in payload.get("milestones", [])
        if int(row["milestone_tokens"]) != int(record["milestone_tokens"])
    ]
    milestones.append(dict(record))
    milestones.sort(key=lambda row: int(row["milestone_tokens"]))
    payload["milestones"] = milestones
    _write_outputs(destination, payload)
    return path


def attach_causal_report(
    output_directory: str | Path,
    milestone_tokens: int,
    report: Mapping[str, Any],
) -> None:
    destination = Path(output_directory)
    path = destination / "telemetry.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    causal = report.get("causal_ablations", {})
    routing = report.get("nexus_analysis", {})
    compact = {
        "mean_effects_by_capability_and_family": causal.get(
            "performance_drop_when_family_removed", {}
        ),
        "module_effects_by_capability": causal.get("matrices", {}).get(
            "disable_module", {}
        ),
        "router_selection_frequency_by_capability": routing.get(
            "routing_frequency_by_capability", {}
        ),
        "router_module_frequency_by_capability": routing.get(
            "module_frequency_by_capability", {}
        ),
        "router_probability_by_capability": routing.get(
            "router_probability_by_capability", {}
        ),
        "router_collapse_status": report.get("router_collapse", {}).get("status"),
        "specialization_status": causal.get("specialization_status"),
        "active_interventions_evaluated": causal.get(
            "active_interventions_evaluated", 0
        ),
    }
    found = False
    for record in payload["milestones"]:
        if int(record["milestone_tokens"]) == milestone_tokens:
            record["causal"] = compact
            found = True
            break
    if not found:
        raise ValueError(f"unknown telemetry milestone {milestone_tokens}")
    _write_outputs(destination, payload)


def compare_developmental_runs(
    runs: Mapping[str, str | Path], output_directory: str | Path
) -> dict[str, Any]:
    payloads = {
        name: json.loads((Path(path) / "telemetry.json").read_text(encoding="utf-8"))
        for name, path in runs.items()
    }
    final: dict[str, Any] = {}
    for name, payload in payloads.items():
        milestones = payload.get("milestones", [])
        if not milestones:
            raise ValueError(f"run {name!r} has no telemetry milestones")
        last = milestones[-1]
        delta = next(
            (
                row
                for row in last.get("modules", [])
                if row.get("family") == "delta"
            ),
            None,
        )
        delta_trajectory = [
            {
                "milestone_tokens": milestone["milestone_tokens"],
                **next(
                    (
                        row
                        for row in milestone.get("modules", [])
                        if row.get("family") == "delta"
                    ),
                    {},
                ),
            }
            for milestone in milestones
        ]
        final[name] = {
            "final_milestone_tokens": last["milestone_tokens"],
            "validation_loss": (last.get("metrics") or {}).get(
                "validation_loss"
            ),
            "validation_perplexity": (last.get("metrics") or {}).get(
                "validation_perplexity"
            ),
            "validation_token_accuracy": (last.get("metrics") or {}).get(
                "validation_token_accuracy"
            ),
            "delta": delta,
            "delta_trajectory": delta_trajectory,
            "expert_death_timing": _expert_death_timing(milestones),
            "router_collapse_status": (last.get("causal") or {}).get(
                "router_collapse_status"
            ),
            "causal": last.get("causal"),
        }
    comparison = {
        "schema_version": 1,
        "interpretation_rule": (
            "Utilization balance is not a success criterion. Compare validation "
            "quality and causal usefulness before judging anti-starvation."
        ),
        "runs": payloads,
        "final": final,
    }
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8"
    )
    (destination / "comparison.md").write_text(
        _comparison_markdown(comparison), encoding="utf-8"
    )
    return comparison


def _proposal_rows(values: Tensor | None, selected_shape: torch.Size) -> Tensor | None:
    if values is None:
        return None
    tensor = values.float()
    if tensor.size(-1) != selected_shape[-1]:
        return None
    flattened = tensor.reshape(-1, tensor.size(-1))
    if flattened.size(0) == selected_shape[0]:
        return flattened
    while tensor.ndim > 2:
        tensor = tensor.mean(dim=-2)
    rows = tensor.reshape(-1, tensor.size(-1))
    if rows.size(0) != selected_shape[0]:
        return None
    return rows


def _accumulate_selected(
    selected: Tensor,
    values: Tensor | None,
    sums: list[float],
    observations: list[int],
    module: int,
) -> None:
    if values is None:
        return
    sums[module] += float(values[selected].sum().item())
    observations[module] += int(selected.sum().item())


def _parameter_norm(module: nn.Module) -> float:
    return math.sqrt(
        sum(parameter.detach().float().square().sum().item() for parameter in module.parameters())
    )


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean_from_sum(total: float, count: int) -> float | None:
    return total / count if count else None


def _family_rows(modules: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in modules:
        grouped.setdefault(str(row["family"]), []).append(row)
    output: list[dict[str, Any]] = []
    for family, rows in grouped.items():
        selection_weight = sum(int(row["selection_count"]) for row in rows)
        output.append(
            {
                "family": family,
                "modules": [int(row["module"]) for row in rows],
                "selection_frequency": sum(float(row["selection_frequency"]) for row in rows),
                "request_selection_fraction": min(
                    1.0, sum(float(row["request_selection_fraction"]) for row in rows)
                ),
                "routing_unit_selection_fraction": min(
                    1.0,
                    sum(
                        float(row["routing_unit_selection_fraction"])
                        for row in rows
                    ),
                ),
                "chunk_selection_fraction": _optional_sum(
                    row.get("chunk_selection_fraction") for row in rows
                ),
                "selection_slot_distribution": _family_slot_distribution(
                    rows, selection_weight
                ),
                "mean_router_probability_before_top_k": _optional_sum(
                    row.get("mean_router_probability_before_top_k") for row in rows
                ),
                "mean_normalized_selected_weight": _weighted_mean(
                    rows, "mean_normalized_selected_weight", "selection_count"
                ),
                "mean_integrator_acceptance": _weighted_mean(
                    rows, "mean_integrator_acceptance", "selection_count"
                ),
                "mean_integrator_token_contribution": _weighted_mean(
                    rows, "mean_integrator_token_contribution", "selection_count"
                ),
                "mean_integrator_state_contribution": _weighted_mean(
                    rows, "mean_integrator_state_contribution", "selection_count"
                ),
                "gradient_norm": _root_sum_square(row.get("gradient_norm") for row in rows),
                "parameter_norm": _root_sum_square(row.get("parameter_norm") for row in rows),
                "update_norm": _root_sum_square(row.get("update_norm") for row in rows),
                "recent_update_norm": _root_sum_square(
                    row.get("recent_update_norm") for row in rows
                ),
                "active_step_fraction_since_previous_milestone": min(
                    1.0,
                    sum(
                        float(row["active_step_fraction_since_previous_milestone"])
                        for row in rows
                    ),
                ),
                "selection_count": selection_weight,
            }
        )
    return output




def _family_slot_distribution(
    rows: Iterable[Mapping[str, Any]], selection_count: int
) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for row in rows:
        module_count = int(row["selection_count"])
        for slot, fraction in row.get("selection_slot_distribution", {}).items():
            counts[str(slot)] += float(fraction) * module_count
    return {
        slot: count / selection_count if selection_count else 0.0
        for slot, count in sorted(counts.items())
    }
def _optional_sum(values: Iterable[Any]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return sum(valid) if valid else None


def _weighted_mean(
    rows: Iterable[Mapping[str, Any]], value_key: str, weight_key: str
) -> float | None:
    valid = [
        (float(row[value_key]), int(row[weight_key]))
        for row in rows
        if row.get(value_key) is not None and int(row[weight_key]) > 0
    ]
    weight = sum(item[1] for item in valid)
    return sum(value * count for value, count in valid) / weight if weight else None


def _root_sum_square(values: Iterable[Any]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return math.sqrt(sum(value * value for value in valid)) if valid else None


def _write_outputs(destination: Path, payload: Mapping[str, Any]) -> None:
    (destination / "telemetry.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(destination / "telemetry.csv", payload)
    _write_causal_csv(destination / "causal.csv", payload)
    (destination / "developmental-report.md").write_text(
        _developmental_markdown(payload), encoding="utf-8"
    )
    plot_directory = destination / "plots"
    plot_directory.mkdir(exist_ok=True)
    for metric in (
        "selection_frequency",
        "mean_router_probability_before_top_k",
        "gradient_norm",
        "update_norm",
    ):
        _write_svg_plot(plot_directory / f"{metric}.svg", payload, metric)
    _write_causal_svg_plot(
        plot_directory / "causal_loss_impact.svg", payload
    )


def _write_csv(path: Path, payload: Mapping[str, Any]) -> None:
    fields = [
        "milestone_tokens",
        "observed_tokens",
        "step",
        "scope",
        "module",
        "family",
        "selection_frequency",
        "request_selection_fraction",
        "chunk_selection_fraction",
        "mean_router_probability_before_top_k",
        "mean_normalized_selected_weight",
        "mean_integrator_acceptance",
        "mean_integrator_token_contribution",
        "gradient_norm",
        "parameter_norm",
        "update_norm",
        "active_step_fraction_since_previous_milestone",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for milestone in payload.get("milestones", []):
            for scope, rows in (
                ("module", milestone.get("modules", [])),
                ("family", milestone.get("families", [])),
            ):
                for row in rows:
                    writer.writerow(
                        {
                            "milestone_tokens": milestone["milestone_tokens"],
                            "observed_tokens": milestone["observed_tokens"],
                            "step": milestone["step"],
                            "scope": scope,
                            **{field: row.get(field) for field in fields[4:]},
                        }
                    )


def _write_causal_csv(path: Path, payload: Mapping[str, Any]) -> None:
    fields = [
        "milestone_tokens",
        "capability",
        "family",
        "mean_loss_increase",
        "mean_token_accuracy_degradation",
        "fraction_measurably_worsened",
        "fraction_measurably_improved",
        "router_selection_frequency",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for milestone in payload.get("milestones", []):
            causal = milestone.get("causal") or {}
            matrix = causal.get("mean_effects_by_capability_and_family", {})
            routing = causal.get("router_selection_frequency_by_capability", {})
            for capability, families in matrix.items():
                for family, metrics in families.items():
                    writer.writerow(
                        {
                            "milestone_tokens": milestone["milestone_tokens"],
                            "capability": capability,
                            "family": family,
                            "mean_loss_increase": metrics.get(
                                "mean_loss_increase"
                            ),
                            "mean_token_accuracy_degradation": metrics.get(
                                "mean_token_accuracy_degradation"
                            ),
                            "fraction_measurably_worsened": metrics.get(
                                "fraction_measurably_worsened"
                            ),
                            "fraction_measurably_improved": metrics.get(
                                "fraction_measurably_improved"
                            ),
                            "router_selection_frequency": routing.get(
                                capability, {}
                            ).get(family),
                        }
                    )


def _developmental_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# EMC Developmental Telemetry",
        "",
        "Interval metrics describe training since the previous milestone; cumulative metrics remain in telemetry.json.",
        "",
        "| tokens | family | pre-top-K p | selection | gradient norm | update norm | active steps |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for milestone in payload.get("milestones", []):
        for row in milestone.get("families", []):
            lines.append(
                "| {tokens:,} | {family} | {prob} | {selection:.4f} | {gradient} | {update} | {active:.4f} |".format(
                    tokens=int(milestone["milestone_tokens"]),
                    family=row["family"],
                    prob=_format_number(row.get("mean_router_probability_before_top_k")),
                    selection=float(row["selection_frequency"]),
                    gradient=_format_number(row.get("gradient_norm")),
                    update=_format_number(row.get("update_norm")),
                    active=float(row["active_step_fraction_since_previous_milestone"]),
                )
            )
        causal = milestone.get("causal")
        if causal:
            lines.extend(
                [
                    "",
                    f"## Causal usefulness at {int(milestone['milestone_tokens']):,} tokens",
                    "",
                    f"Router collapse: `{causal.get('router_collapse_status')}`. Specialization: `{causal.get('specialization_status')}`.",
                    "",
                    "Full capability × family effects: `milestone-diagnostics/{tokens}/report.json`.".format(
                        tokens=milestone["milestone_tokens"]
                    ),
                ]
            )
            lines.extend(
                [
                    "",
                    "| capability | family | router selection | loss increase | token-accuracy degradation | hurt | help |",
                    "|---|---|---:|---:|---:|---:|---:|",
                ]
            )
            effects = causal.get("mean_effects_by_capability_and_family", {})
            routing = causal.get("router_selection_frequency_by_capability", {})
            for capability, families in effects.items():
                for family, metrics in families.items():
                    lines.append(
                        f"| {capability} | {family} | "
                        f"{_format_number(routing.get(capability, {}).get(family))} | "
                        f"{_format_number(metrics.get('mean_loss_increase'))} | "
                        f"{_format_number(metrics.get('mean_token_accuracy_degradation'))} | "
                        f"{_format_number(metrics.get('fraction_measurably_worsened'))} | "
                        f"{_format_number(metrics.get('fraction_measurably_improved'))} |"
                    )
    return "\n".join(lines) + "\n"


def _write_svg_plot(path: Path, payload: Mapping[str, Any], metric: str) -> None:
    milestones = payload.get("milestones", [])
    families = sorted(
        {row["family"] for item in milestones for row in item.get("families", [])}
    )
    values = [
        float(row[metric])
        for item in milestones
        for row in item.get("families", [])
        if row.get(metric) is not None
    ]
    width, height, pad = 720, 360, 48
    maximum = max(values, default=1.0)
    minimum = min(values, default=0.0)
    if math.isclose(maximum, minimum):
        maximum = minimum + 1.0
    token_values = [int(item["milestone_tokens"]) for item in milestones]
    min_token = min(token_values, default=0)
    max_token = max(token_values, default=1)
    if min_token == max_token:
        max_token = min_token + 1
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{pad}" y="24" font-family="sans-serif" font-size="16">{metric}</text>',
        f'<path d="M {pad} {height-pad} H {width-pad} M {pad} {height-pad} V {pad}" stroke="#555" fill="none"/>',
    ]
    for family_index, family in enumerate(families):
        points: list[str] = []
        for item in milestones:
            row = next(
                (candidate for candidate in item.get("families", []) if candidate["family"] == family),
                None,
            )
            if row is None or row.get(metric) is None:
                continue
            x = pad + (int(item["milestone_tokens"]) - min_token) / (max_token - min_token) * (width - 2 * pad)
            y = height - pad - (float(row[metric]) - minimum) / (maximum - minimum) * (height - 2 * pad)
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            color = colors[family_index % len(colors)]
            lines.append(
                f'<polyline points="{" ".join(points)}" stroke="{color}" fill="none" stroke-width="2"/>'
            )
            lines.append(
                f'<text x="{width-pad-100}" y="{pad+18*family_index}" fill="{color}" font-family="sans-serif" font-size="12">{family}</text>'
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")



def _write_causal_svg_plot(path: Path, payload: Mapping[str, Any]) -> None:
    causal_payload = {
        "schema_version": payload.get("schema_version"),
        "milestones": [],
    }
    for milestone in payload.get("milestones", []):
        matrix = (milestone.get("causal") or {}).get(
            "mean_effects_by_capability_and_family", {}
        )
        family_values: dict[str, list[float]] = {}
        for targets in matrix.values():
            for family, metrics in targets.items():
                value = metrics.get("mean_loss_increase")
                if value is not None:
                    family_values.setdefault(family, []).append(float(value))
        causal_payload["milestones"].append(
            {
                "milestone_tokens": milestone["milestone_tokens"],
                "families": [
                    {
                        "family": family,
                        "causal_loss_impact": sum(values) / len(values),
                    }
                    for family, values in sorted(family_values.items())
                ],
            }
        )
    _write_svg_plot(path, causal_payload, "causal_loss_impact")



def _expert_death_timing(
    milestones: Iterable[Mapping[str, Any]],
) -> dict[str, int | str | None]:
    trajectory = list(milestones)
    families = sorted(
        {
            row["family"]
            for milestone in trajectory
            for row in milestone.get("families", [])
        }
    )
    result: dict[str, int | str | None] = {}
    for family in families:
        values = [
            (
                int(milestone["milestone_tokens"]),
                float(
                    next(
                        row
                        for row in milestone.get("families", [])
                        if row["family"] == family
                    )["selection_frequency"]
                ),
            )
            for milestone in trajectory
            if any(
                row["family"] == family
                for row in milestone.get("families", [])
            )
        ]
        if not any(value > 0 for _, value in values):
            result[family] = "never_active"
            continue
        death = next(
            (
                tokens
                for index, (tokens, value) in enumerate(values)
                if value == 0
                and any(previous > 0 for _, previous in values[:index])
                and all(later == 0 for _, later in values[index:])
            ),
            None,
        )
        result[family] = death
    return result
def _comparison_markdown(comparison: Mapping[str, Any]) -> str:
    lines = [
        "# Anti-Starvation Comparison",
        "",
        str(comparison["interpretation_rule"]),
        "",
        "| run | final tokens | validation loss | perplexity | Delta selection | Delta pre-top-K p | Delta update norm | collapse |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in comparison["final"].items():
        delta = row.get("delta") or {}
        lines.append(
            f"| {name} | {int(row['final_milestone_tokens']):,} | "
            f"{_format_number(row.get('validation_loss'))} | "
            f"{_format_number(row.get('validation_perplexity'))} | "
            f"{_format_number(delta.get('selection_frequency'))} | "
            f"{_format_number(delta.get('mean_router_probability_before_top_k'))} | "
            f"{_format_number(delta.get('update_norm'))} | "
            f"{row.get('router_collapse_status')} |"
        )
    return "\n".join(lines) + "\n"


def _format_number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"
