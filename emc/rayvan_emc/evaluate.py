from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import platform
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .capability_tasks import (
    CAPABILITIES,
    CAPABILITY_GENERATOR_VERSION,
    DEFAULT_HELD_OUT_COMBINATIONS,
    DEFAULT_MIXTURE_WEIGHTS,
    CapabilitySuiteConfig,
    CapabilityTaskSuite,
    DiagnosticExample,
)
from .checkpoint import LoadedModelCheckpoint, load_model_checkpoint
from .chunked import ChunkedEMCModel, ChunkedExecutionTrace
from .diagnostics import count_parameters, parameter_counts
from .generation import generate_text
from .model import EMCModel, EMCOutput
from .tokenization import TextTokenizer

REPORT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class DiagnosticEvaluationConfig:
    seed: int = 42
    examples_per_capability: int = 100
    ablation_examples_per_capability: int = 8
    held_out_only: bool = False
    run_module_ablations: bool = True
    run_family_ablations: bool = True
    run_zero_proposal: bool = True
    run_forced_alternatives: bool = True
    max_notable_examples: int = 12
    device: str = "cpu"
    precision: str = "fp32"
    smoke: bool = False

    def __post_init__(self) -> None:
        if self.examples_per_capability <= 0:
            raise ValueError("examples_per_capability must be positive")
        if self.ablation_examples_per_capability < 0:
            raise ValueError("ablation_examples_per_capability cannot be negative")
        if self.max_notable_examples < 0:
            raise ValueError("max_notable_examples cannot be negative")
        if self.precision not in {"fp32", "fp16", "bf16", "auto"}:
            raise ValueError("unsupported precision")


@dataclass(frozen=True)
class InterventionResult:
    example_index: int
    capability: str
    intervention_type: str
    target: str
    baseline_exact: float
    intervened_exact: float
    baseline_token_accuracy: float
    intervened_token_accuracy: float


@dataclass
class _TraceSummary:
    module_counts: Counter[int]
    family_counts: Counter[str]
    request_pool_counts: Counter[int]
    score_sums: dict[int, float]
    score_observations: dict[int, int]
    routing_entropies: list[float]
    unique_modules: set[int]
    integrator: dict[str, dict[int, list[float]]]
    proposal_similarities: list[float]
    gate_magnitudes: list[float]
    balancing_biases: list[list[float]]
    lease_ages: list[float]
    switch_rates: list[float]
    continuation_rates: list[float]
    lease_state_norms: list[float]
    lease_state_changes: list[float]
    state_reset_count: int
    requested_pairs: int
    executed_pairs: int
    discarded_pairs: int
    actual_unique_executed_per_chunk: list[int]
    population_fraction_touched: float


def evaluate_checkpoint(
    checkpoint: str | Path,
    output_directory: str | Path,
    config: DiagnosticEvaluationConfig | None = None,
) -> dict[str, Any]:
    resolved = config or DiagnosticEvaluationConfig()
    loaded = load_model_checkpoint(checkpoint, device=resolved.device)
    report = evaluate_suite(
        loaded.model,
        loaded.tokenizer,
        resolved,
        checkpoint=str(checkpoint),
        checkpoint_training_config=loaded.training_config,
    )
    write_report(report, output_directory)
    return report


def evaluate_suite(
    model: nn.Module,
    tokenizer: TextTokenizer,
    config: DiagnosticEvaluationConfig | None = None,
    *,
    checkpoint: str | None = None,
    checkpoint_training_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = config or DiagnosticEvaluationConfig()
    suite_config = CapabilitySuiteConfig(seed=resolved.seed)
    suite = CapabilityTaskSuite(suite_config)
    examples = suite.balanced_evaluation(
        resolved.examples_per_capability,
        held_out_only=resolved.held_out_only,
    )
    device = torch.device(resolved.device)
    was_training = model.training
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    baseline_records: list[dict[str, Any]] = []
    baseline_outputs: dict[int, EMCOutput] = {}
    evaluated_tokens = 0
    for index, example in enumerate(examples):
        record, output, token_count = _evaluate_example(
            model, tokenizer, example, resolved, return_trace=True
        )
        record["example_index"] = index
        baseline_records.append(record)
        evaluated_tokens += token_count
        if output is not None:
            baseline_outputs[index] = output

    interventions = _run_ablations(
        model,
        tokenizer,
        examples,
        baseline_records,
        baseline_outputs,
        resolved,
    )
    cycle_results = _evaluate_cycles(model, tokenizer, examples, resolved)
    elapsed = time.perf_counter() - started
    peak_vram = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    capability_results = _aggregate_capability_results(baseline_records)
    overall_metrics = _overall_results(baseline_records, capability_results)
    routing = _aggregate_routing(baseline_records, model)
    integrator = _aggregate_integrator(baseline_records, model)
    causal = _aggregate_interventions(interventions, model)
    surface_analysis = _surface_vs_operation(baseline_records)
    collapse = _collapse_analysis(baseline_records, model)
    lease = _lease_analysis(baseline_records)
    sparsity = _sparsity_analysis(baseline_records, model)
    module_diagnostics = _module_diagnostics(model, baseline_records)
    stratified = _stratified_results(baseline_records)
    difficulty_curves = _difficulty_curves(baseline_records, model)
    notable = _notable_examples(
        examples,
        baseline_records,
        interventions,
        resolved.max_notable_examples,
    )
    generated_samples = _language_samples(model, tokenizer, examples, resolved)
    counts = parameter_counts(model)
    active_compute = {
        "approximate_active_parameters_per_cycle": (
            counts.approximate_active_per_cycle
        ),
        "approximate_parameter_uses_per_forward": (
            counts.approximate_parameter_uses_per_forward
        ),
        "active_flops": None,
        "note": "FLOP estimates are unavailable for heterogeneous chunk backends",
    }
    learning_status = _learning_status(capability_results)
    full_scale = 100 / resolved.examples_per_capability
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "manifest": {
            "code_version": git_commit(),
            "generator_version": CAPABILITY_GENERATOR_VERSION,
            "seed": resolved.seed,
            "model_config": asdict(model.config),
            "training_config": dict(checkpoint_training_config or {}),
            "suite_config": {
                "examples_per_capability": resolved.examples_per_capability,
                "ablation_examples_per_capability": resolved.ablation_examples_per_capability,
                "held_out_only": resolved.held_out_only,
                "held_out_combinations": list(DEFAULT_HELD_OUT_COMBINATIONS),
                "task_mixture": dict(DEFAULT_MIXTURE_WEIGHTS),
            },
            "checkpoint": checkpoint,
            "device": str(device),
            "precision": _resolved_precision(resolved.precision, device),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "runtime": {
            "wall_time_seconds": elapsed,
            "evaluated_tokens": evaluated_tokens,
            "tokens_per_second": evaluated_tokens / elapsed if elapsed else 0.0,
            "parameter_count": count_parameters(model),
            "active_compute": active_compute,
            "peak_vram_bytes": peak_vram,
            "estimated_100_examples_per_capability_seconds": elapsed * full_scale,
            "smoke_estimate": resolved.smoke,
        },
        "executive_summary": {
            "overall_loss": overall_metrics["overall_loss"],
            "language_perplexity": overall_metrics["language_perplexity"],
            "overall_learning_status": learning_status,
            "strongest_capability": _extreme_capability(capability_results, maximum=True),
            "weakest_capability": _extreme_capability(capability_results, maximum=False),
            "router_collapse_status": collapse["status"],
            "specialization_evidence": causal["specialization_statement"],
            "surface_routing_evidence": surface_analysis["statement"],
            "cycle_usefulness": _cycle_statement(cycle_results),
            "causal_family_importance": causal["most_important_family_by_capability"],
        },
        "overall_metrics": overall_metrics,
        "capability_results": capability_results,
        "stratified_results": stratified,
        "difficulty_curves": difficulty_curves,
        "nexus_analysis": routing,
        "integrator_analysis": integrator,
        "causal_ablations": causal,
        "surface_vs_computation": surface_analysis,
        "router_collapse": collapse,
        "lease_temporal_analysis": lease,
        "sparse_execution": sparsity,
        "cycle_analysis": cycle_results,
        "module_diagnostics": module_diagnostics,
        "generated_language_samples": generated_samples,
        "notable_examples": notable,
    }
    model.train(was_training)
    return report


def compare_checkpoints(
    checkpoints: Iterable[str | Path],
    output_directory: str | Path,
    config: DiagnosticEvaluationConfig | None = None,
) -> dict[str, Any]:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    reports: list[tuple[str, dict[str, Any]]] = []
    for index, checkpoint in enumerate(checkpoints):
        name = f"checkpoint-{index + 1}-{Path(checkpoint).stem}"
        report = evaluate_checkpoint(checkpoint, destination / name, config)
        reports.append((name, report))
    capabilities: dict[str, dict[str, float | None]] = {}
    for capability in CAPABILITIES:
        capabilities[capability] = {
            name: report["capability_results"].get(capability, {}).get("exact_accuracy")
            for name, report in reports
        }
    comparison = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "checkpoints": [
            {"name": name, "checkpoint": report["manifest"]["checkpoint"]}
            for name, report in reports
        ],
        "capability_exact_accuracy": capabilities,
        "router_collapse_status": {
            name: report["router_collapse"]["status"] for name, report in reports
        },
        "wall_time_seconds": {
            name: report["runtime"]["wall_time_seconds"] for name, report in reports
        },
    }
    (destination / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    _write_comparison_csv(comparison, destination / "comparison.csv")
    (destination / "comparison.md").write_text(
        _comparison_markdown(comparison), encoding="utf-8"
    )
    return comparison


def write_report(report: Mapping[str, Any], output_directory: str | Path) -> None:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    _write_metrics_csv(report, destination / "metrics.csv")
    (destination / "report.md").write_text(_report_markdown(report), encoding="utf-8")


def _evaluate_example(
    model: nn.Module,
    tokenizer: TextTokenizer,
    example: DiagnosticExample,
    config: DiagnosticEvaluationConfig,
    *,
    return_trace: bool,
    availability_mask: Tensor | None = None,
    zero_mask: Tensor | None = None,
    forced_modules: Tensor | None = None,
    cycle_limit: int | None = None,
) -> tuple[dict[str, Any], EMCOutput | None, int]:
    prompt_ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(example.target)
    all_ids = prompt_ids + target_ids
    maximum = int(model.config.max_sequence_length)
    if not prompt_ids or not target_ids or len(all_ids) - 1 > maximum:
        return (
            {
                "metadata": example.metadata_dict(),
                "skipped": True,
                "skip_reason": "empty encoding or sequence exceeds model context",
                "exact_match": 0.0,
                "token_accuracy": 0.0,
                "loss": None,
                "prediction": "",
                "target": example.target,
                "trace": None,
            },
            None,
            0,
        )
    device = next(model.parameters()).device
    inputs = torch.tensor(all_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    targets = torch.tensor(all_ids[1:], dtype=torch.long, device=device)
    kwargs: dict[str, Any] = {
        "return_trace": return_trace,
        "availability_mask": availability_mask,
        "diagnostic_zero_proposal_mask": zero_mask,
        "diagnostic_forced_modules": forced_modules,
    }
    if cycle_limit is not None and isinstance(model, EMCModel):
        kwargs["evaluation_cycle_limit"] = cycle_limit
    with torch.inference_mode(), _autocast(config.precision, device):
        raw_output = model(inputs, **kwargs)
    output = raw_output if isinstance(raw_output, EMCOutput) else None
    logits = output.logits if output is not None else raw_output
    answer_start = len(prompt_ids) - 1
    answer_logits = logits[0, answer_start : answer_start + len(target_ids)].float()
    answer_targets = targets[answer_start : answer_start + len(target_ids)]
    predicted_ids = answer_logits.argmax(dim=-1)
    exact = float(torch.equal(predicted_ids, answer_targets))
    token_accuracy = float((predicted_ids == answer_targets).float().mean().item())
    loss = float(F.cross_entropy(answer_logits, answer_targets).item())
    prediction = tokenizer.decode(predicted_ids.detach().cpu().tolist())
    trace = _summarize_trace(output, model) if output is not None else None
    record = {
        "metadata": example.metadata_dict(),
        "skipped": False,
        "exact_match": exact,
        "token_accuracy": token_accuracy,
        "loss": loss,
        "prediction": prediction,
        "target": example.target,
        "trace": _trace_to_dict(trace) if trace is not None else None,
    }
    return record, output, len(all_ids) - 1


def _run_ablations(
    model: nn.Module,
    tokenizer: TextTokenizer,
    examples: tuple[DiagnosticExample, ...],
    records: list[dict[str, Any]],
    outputs: Mapping[int, EMCOutput],
    config: DiagnosticEvaluationConfig,
) -> list[InterventionResult]:
    if config.ablation_examples_per_capability == 0:
        return []
    families = tuple(model.module_families)
    unique_families = tuple(dict.fromkeys(families))
    selected_per_capability: Counter[str] = Counter()
    results: list[InterventionResult] = []
    for index, (example, baseline) in enumerate(zip(examples, records, strict=True)):
        capability = example.diagnostic_metadata.capability
        if baseline["skipped"] or selected_per_capability[capability] >= config.ablation_examples_per_capability:
            continue
        selected_per_capability[capability] += 1
        if index not in outputs:
            continue
        baseline_exact = float(baseline["exact_match"])
        baseline_token = float(baseline["token_accuracy"])
        interventions: list[tuple[str, str, dict[str, Tensor | None]]] = []
        if config.run_module_ablations:
            for module_index in range(model.config.num_modules):
                availability = torch.ones(model.config.num_modules, dtype=torch.bool)
                availability[module_index] = False
                interventions.append(("disable_module", str(module_index), {"availability_mask": availability}))
        if config.run_family_ablations:
            for family in unique_families:
                availability = torch.tensor([name != family for name in families], dtype=torch.bool)
                interventions.append(("disable_family", family, {"availability_mask": availability}))
        if config.run_zero_proposal:
            for family in unique_families:
                zero = torch.tensor([name == family for name in families], dtype=torch.bool)
                interventions.append(("zero_family_proposal", family, {"zero_mask": zero}))
        if config.run_forced_alternatives:
            trace = baseline.get("trace") or {}
            counts = trace.get("module_counts", {})
            normal_order = sorted(range(model.config.num_modules), key=lambda item: -int(counts.get(str(item), 0)))
            top_family = families[normal_order[0]] if normal_order else families[0]
            top_k = model.config.resolved_active_top_k if isinstance(model, ChunkedEMCModel) else model.config.modules_per_cycle
            for family in unique_families:
                if family == top_family:
                    continue
                candidate = next(i for i, name in enumerate(families) if name == family)
                forced = [candidate]
                forced.extend(i for i in normal_order if i != candidate and i not in forced)
                forced.extend(i for i in range(model.config.num_modules) if i not in forced)
                interventions.append(("force_family_alternative", family, {"forced_modules": torch.tensor(forced[:top_k])}))
        for intervention_type, target, kwargs in interventions:
            try:
                intervened, _, _ = _evaluate_example(
                    model,
                    tokenizer,
                    example,
                    config,
                    return_trace=False,
                    availability_mask=kwargs.get("availability_mask"),
                    zero_mask=kwargs.get("zero_mask"),
                    forced_modules=kwargs.get("forced_modules"),
                )
            except ValueError:
                continue
            results.append(
                InterventionResult(
                    example_index=index,
                    capability=capability,
                    intervention_type=intervention_type,
                    target=target,
                    baseline_exact=baseline_exact,
                    intervened_exact=float(intervened["exact_match"]),
                    baseline_token_accuracy=baseline_token,
                    intervened_token_accuracy=float(intervened["token_accuracy"]),
                )
            )
    return results


def _evaluate_cycles(
    model: nn.Module,
    tokenizer: TextTokenizer,
    examples: tuple[DiagnosticExample, ...],
    config: DiagnosticEvaluationConfig,
) -> dict[str, Any]:
    if not isinstance(model, EMCModel) or model.config.num_cycles <= 1:
        return {"supported": False, "reason": "checkpoint does not expose multiple EMC cycles"}
    sample_limit = min(8, config.examples_per_capability)
    rows: dict[str, dict[str, dict[str, float]]] = {}
    for capability in CAPABILITIES:
        selected = [example for example in examples if example.diagnostic_metadata.capability == capability][:sample_limit]
        rows[capability] = {}
        for cycle in range(1, model.config.num_cycles + 1):
            metrics = [
                _evaluate_example(model, tokenizer, example, config, return_trace=False, cycle_limit=cycle)[0]
                for example in selected
            ]
            valid = [row for row in metrics if not row["skipped"]]
            rows[capability][str(cycle)] = {
                "exact_accuracy": _mean(row["exact_match"] for row in valid),
                "token_accuracy": _mean(row["token_accuracy"] for row in valid),
                "loss": _mean(row["loss"] for row in valid),
            }
    return {"supported": True, "capability_by_cycle": rows}


def _summarize_trace(output: EMCOutput, model: nn.Module) -> _TraceSummary:
    summary = _empty_trace()
    if output.trace:
        for cycle in output.trace:
            if cycle.selected_indices is None:
                continue
            selected = cycle.selected_indices.long()
            _observe_selected(summary, selected, model.module_families)
            probabilities = torch.softmax(cycle.router_scores.float(), dim=-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            summary.routing_entropies.extend(entropy.reshape(-1).tolist())
            _observe_scores(summary, cycle.router_scores)
            if cycle.integrator_trace is not None:
                _observe_integrator(summary, selected, cycle.integrator_trace, "token")
        summary.requested_pairs = sum(summary.module_counts.values())
        summary.executed_pairs = sum(len(cycle.selected_modules) for cycle in output.trace)
        summary.discarded_pairs = 0
        summary.actual_unique_executed_per_chunk = [len(cycle.selected_modules) for cycle in output.trace]
        summary.population_fraction_touched = len(summary.unique_modules) / model.config.num_modules
    if isinstance(output.chunk_trace, ChunkedExecutionTrace):
        chunked = output.chunk_trace
        summary.request_pool_counts.update(chunked.request_pool.module_indices.reshape(-1).tolist())
        _observe_scores(summary, chunked.request_pool.scores)
        for chunk in chunked.chunks:
            selected = chunk.active_modules.long()
            _observe_selected(summary, selected, model.module_families)
            _observe_scores(summary, chunk.routing_scores)
            summary.routing_entropies.extend(chunk.routing_entropy.reshape(-1).tolist())
            summary.lease_ages.extend(chunk.lease_ages[chunk.lease_ages > 0].float().tolist())
            summary.switch_rates.append(float(chunk.switch_rate))
            summary.continuation_rates.append(float(chunk.retained_rate))
            summary.balancing_biases.append(chunk.balance_bias.float().tolist())
            summary.lease_state_norms.append(float(chunk.lease_state_norm))
            summary.lease_state_changes.append(float(chunk.lease_state_change))
            summary.state_reset_count += int(chunk.state_reset_count)
            _observe_integrator(summary, selected, chunk.token_integrator_trace, "token")
            _observe_integrator(summary, selected, chunk.state_integrator_trace, "state")
            summary.requested_pairs += int(chunk.computed_chunk_module_pairs)
            summary.executed_pairs += int(chunk.retained_chunk_module_pairs)
            summary.discarded_pairs += int(chunk.computed_chunk_module_pairs - chunk.retained_chunk_module_pairs)
            summary.actual_unique_executed_per_chunk.append(len(chunk.executed_modules))
        summary.population_fraction_touched = float(chunked.population_fraction_touched)
    return summary


def _empty_trace() -> _TraceSummary:
    return _TraceSummary(
        module_counts=Counter(), family_counts=Counter(), request_pool_counts=Counter(),
        score_sums=defaultdict(float), score_observations=defaultdict(int),
        routing_entropies=[], unique_modules=set(),
        integrator={name: defaultdict(list) for name in ("acceptance", "token_contribution", "state_contribution", "proposal_norm")},
        proposal_similarities=[], gate_magnitudes=[], balancing_biases=[], lease_ages=[],
        switch_rates=[], continuation_rates=[], lease_state_norms=[], lease_state_changes=[],
        state_reset_count=0, requested_pairs=0, executed_pairs=0, discarded_pairs=0,
        actual_unique_executed_per_chunk=[], population_fraction_touched=0.0,
    )


def _observe_selected(summary: _TraceSummary, selected: Tensor, families: tuple[str, ...]) -> None:
    values = selected.reshape(-1).tolist()
    summary.module_counts.update(values)
    summary.family_counts.update(families[index] for index in values)
    summary.unique_modules.update(values)


def _observe_scores(summary: _TraceSummary, scores: Tensor) -> None:
    flattened = scores.float().reshape(-1, scores.size(-1))
    for index in range(flattened.size(-1)):
        values = flattened[:, index]
        finite = values[torch.isfinite(values)]
        if finite.numel():
            summary.score_sums[index] += float(finite.sum().item())
            summary.score_observations[index] += int(finite.numel())


def _observe_integrator(summary: _TraceSummary, selected: Tensor, trace: Any, kind: str) -> None:
    for metric_name, values in (
        ("acceptance", trace.proposal_acceptance),
        ("proposal_norm", trace.proposal_norms),
        (f"{kind}_contribution", trace.proposal_contributions),
    ):
        if metric_name not in summary.integrator:
            continue
        expanded = selected
        if expanded.ndim == 2 and values.ndim == 3:
            expanded = expanded.unsqueeze(1).expand(-1, values.size(1), -1)
        for module_index, value in zip(expanded.reshape(-1).tolist(), values.float().reshape(-1).tolist(), strict=True):
            summary.integrator[metric_name][module_index].append(value)
    similarity = trace.proposal_similarity.float()
    if similarity.size(-1) > 1:
        mask = ~torch.eye(similarity.size(-1), dtype=torch.bool)
        summary.proposal_similarities.extend(similarity[..., mask].reshape(-1).tolist())
    summary.gate_magnitudes.extend(trace.gate_magnitude.float().reshape(-1).tolist())


def _trace_to_dict(summary: _TraceSummary) -> dict[str, Any]:
    return {
        "module_counts": {str(key): value for key, value in summary.module_counts.items()},
        "family_counts": dict(summary.family_counts),
        "request_pool_counts": {str(key): value for key, value in summary.request_pool_counts.items()},
        "mean_router_scores": {str(key): summary.score_sums[key] / summary.score_observations[key] for key in summary.score_observations},
        "mean_routing_entropy": _mean(summary.routing_entropies),
        "unique_modules": sorted(summary.unique_modules),
        "integrator": {
            metric: {str(index): _mean(values) for index, values in modules.items()}
            for metric, modules in summary.integrator.items()
        },
        "mean_proposal_similarity": _mean(summary.proposal_similarities),
        "mean_gate_magnitude": _mean(summary.gate_magnitudes),
        "mean_balance_bias": _column_means(summary.balancing_biases),
        "lease_ages": summary.lease_ages,
        "switch_rates": summary.switch_rates,
        "continuation_rates": summary.continuation_rates,
        "lease_state_norms": summary.lease_state_norms,
        "lease_state_changes": summary.lease_state_changes,
        "state_reset_count": summary.state_reset_count,
        "requested_pairs": summary.requested_pairs,
        "executed_pairs": summary.executed_pairs,
        "discarded_pairs": summary.discarded_pairs,
        "actual_unique_executed_per_chunk": summary.actual_unique_executed_per_chunk,
        "population_fraction_touched": summary.population_fraction_touched,
    }


def _aggregate_capability_results(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped = _group_records(records, lambda row: row["metadata"]["capability"])
    result: dict[str, dict[str, Any]] = {}
    for capability, rows in grouped.items():
        valid = [row for row in rows if not row["skipped"]]
        mean_loss = _mean(row["loss"] for row in valid)
        result[capability] = {
            "examples": len(valid), "skipped": len(rows) - len(valid),
            "exact_accuracy": _mean(row["exact_match"] for row in valid),
            "token_accuracy": _mean(row["token_accuracy"] for row in valid),
            "cross_entropy": mean_loss,
            "perplexity": math.exp(min(mean_loss, 20.0)) if mean_loss is not None else None,
        }
    return result


def _overall_results(
    records: list[dict[str, Any]],
    capability_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    valid = [row for row in records if not row["skipped"]]
    language = capability_results.get("language", {})
    return {
        "examples": len(valid),
        "skipped": len(records) - len(valid),
        "overall_loss": _mean(row["loss"] for row in valid),
        "overall_exact_accuracy": _mean(row["exact_match"] for row in valid),
        "overall_token_accuracy": _mean(row["token_accuracy"] for row in valid),
        "language_perplexity": language.get("perplexity"),
    }


def _difficulty_curves(
    records: list[dict[str, Any]], model: nn.Module
) -> dict[str, dict[str, dict[str, Any]]]:
    curves: dict[str, dict[str, dict[str, Any]]] = {}
    families = tuple(dict.fromkeys(model.module_families))
    for capability in CAPABILITIES:
        capability_rows = [
            row
            for row in records
            if row["metadata"]["capability"] == capability
        ]
        curves[capability] = {}
        for difficulty, rows in _group_records(
            capability_rows, lambda row: row["metadata"]["difficulty"]
        ).items():
            valid = [row for row in rows if not row["skipped"]]
            family_counts: Counter[str] = Counter()
            for row in valid:
                family_counts.update((row.get("trace") or {}).get("family_counts", {}))
            total_routes = sum(family_counts.values())
            curves[capability][difficulty] = {
                "examples": len(valid),
                "exact_accuracy": _mean(row["exact_match"] for row in valid),
                "token_accuracy": _mean(row["token_accuracy"] for row in valid),
                "cross_entropy": _mean(row["loss"] for row in valid),
                "routing_entropy": _mean(
                    (row.get("trace") or {}).get("mean_routing_entropy")
                    for row in valid
                ),
                "family_routing_frequency": {
                    family: (
                        family_counts[family] / total_routes if total_routes else 0.0
                    )
                    for family in families
                },
            }
    return curves


def _stratified_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = {
        "surface_format": lambda row: row["metadata"]["surface_format"],
        "difficulty": lambda row: str(row["metadata"]["difficulty"]),
        "sequence_length": lambda row: _length_bucket(int(row["metadata"]["sequence_length"])),
        "distractor_count": lambda row: _distractor_bucket(int(row["metadata"]["distractor_count"])),
    }
    output: dict[str, Any] = {}
    for name, key_function in dimensions.items():
        output[name] = {}
        for key, rows in _group_records(records, key_function).items():
            valid = [row for row in rows if not row["skipped"]]
            output[name][key] = {
                "examples": len(valid),
                "exact_accuracy": _mean(row["exact_match"] for row in valid),
                "token_accuracy": _mean(row["token_accuracy"] for row in valid),
            }
    return output


def _aggregate_routing(records: list[dict[str, Any]], model: nn.Module) -> dict[str, Any]:
    families = tuple(dict.fromkeys(model.module_families))
    by_capability = _routing_matrix(records, "capability", families)
    by_surface = _routing_matrix(records, "surface_format", families)
    return {
        "routing_frequency_by_capability": by_capability,
        "routing_frequency_by_surface": by_surface,
        "module_frequency_by_capability": _module_routing_matrix(records, "capability", model.config.num_modules),
        "module_frequency_by_surface": _module_routing_matrix(records, "surface_format", model.config.num_modules),
        "router_scores_by_capability": _router_score_matrix(records, "capability", model.config.num_modules),
        "request_pool_by_capability": _request_pool_matrix(records, "capability", model.config.num_modules),
        "routing_entropy_by_capability": _trace_scalar_by(records, "capability", "mean_routing_entropy"),
        "average_unique_modules_per_request": _mean(len((row.get("trace") or {}).get("unique_modules", [])) for row in records if row.get("trace")),
        "population_fraction_touched_by_capability": _trace_scalar_by(records, "capability", "population_fraction_touched"),
        "balancing_bias_by_capability": _vector_by(records, "capability", "mean_balance_bias"),
    }


def _aggregate_integrator(records: list[dict[str, Any]], model: nn.Module) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in ("acceptance", "token_contribution", "state_contribution", "proposal_norm"):
        output[f"{metric}_by_capability"] = _integrator_matrix(records, "capability", metric, model)
        output[f"{metric}_by_surface"] = _integrator_matrix(records, "surface_format", metric, model)
    output["proposal_similarity_by_capability"] = _trace_scalar_by(records, "capability", "mean_proposal_similarity")
    output["gate_magnitude_by_capability"] = _trace_scalar_by(records, "capability", "mean_gate_magnitude")
    output["nexus_vs_integrator_disagreement"] = _nexus_integrator_disagreement(records, model)
    return output


def _aggregate_interventions(results: list[InterventionResult], model: nn.Module) -> dict[str, Any]:
    matrices: dict[str, dict[str, dict[str, Any]]] = {}
    for intervention_type in sorted({result.intervention_type for result in results}):
        matrices[intervention_type] = {}
        relevant = [result for result in results if result.intervention_type == intervention_type]
        for capability in CAPABILITIES:
            matrices[intervention_type][capability] = {}
            for target in sorted({result.target for result in relevant}):
                rows = [result for result in relevant if result.capability == capability and result.target == target]
                if not rows:
                    continue
                baseline = _mean(row.baseline_exact for row in rows)
                intervened = _mean(row.intervened_exact for row in rows)
                baseline_token = _mean(row.baseline_token_accuracy for row in rows)
                intervened_token = _mean(row.intervened_token_accuracy for row in rows)
                matrices[intervention_type][capability][target] = {
                    "examples": len(rows), "baseline_accuracy": baseline,
                    "intervened_accuracy": intervened,
                    "accuracy_delta_points": 100 * (intervened - baseline),
                    "token_accuracy_delta_points": 100 * (intervened_token - baseline_token),
                }
    family_matrix = matrices.get("disable_family", {})
    important: dict[str, str | None] = {}
    drops: list[float] = []
    for capability in CAPABILITIES:
        values = family_matrix.get(capability, {})
        if values:
            target, metric = min(values.items(), key=lambda item: item[1]["accuracy_delta_points"])
            important[capability] = target
            drops.append(float(metric["accuracy_delta_points"]))
        else:
            important[capability] = None
    statement = "No measurable causal family specialization in sampled ablations."
    if drops and min(drops) < 0:
        statement = "At least one capability loses accuracy when its most important family is removed; inspect the family matrix for specificity."
    return {
        "matrices": matrices,
        "performance_drop_when_family_removed": family_matrix,
        "most_important_family_by_capability": important,
        "specialization_statement": statement,
        "module_families": list(model.module_families),
    }


def _surface_vs_operation(records: list[dict[str, Any]]) -> dict[str, Any]:
    observations: list[tuple[str, str, str]] = []
    for row in records:
        trace = row.get("trace")
        if not trace:
            continue
        for family, count in trace["family_counts"].items():
            observations.extend((row["metadata"]["operation"], row["metadata"]["surface_format"], family) for _ in range(count))
    operation_mi = _mutual_information((operation, family) for operation, _, family in observations)
    surface_mi = _mutual_information((surface, family) for _, surface, family in observations)
    if operation_mi > surface_mi * 1.1:
        statement = "Module selection correlates more strongly with operation identity than surface format."
    elif surface_mi > operation_mi * 1.1:
        statement = "Module selection correlates more strongly with surface format than operation identity."
    else:
        statement = "Operation and surface format have similar association with module selection."
    return {
        "operation_selection_mutual_information_nats": operation_mi,
        "surface_selection_mutual_information_nats": surface_mi,
        "operation_to_surface_ratio": operation_mi / surface_mi if surface_mi else None,
        "statement": statement,
        "operation_family_matrix": _categorical_matrix(observations, 0, 2),
        "surface_family_matrix": _categorical_matrix(observations, 1, 2),
    }


def _collapse_analysis(records: list[dict[str, Any]], model: nn.Module) -> dict[str, Any]:
    overall = _collapse_metrics(records, model.config.num_modules)
    by_capability = {
        capability: _collapse_metrics([row for row in records if row["metadata"]["capability"] == capability], model.config.num_modules)
        for capability in CAPABILITIES
    }
    dominant = [metrics["top_module"] for metrics in by_capability.values() if metrics["observations"]]
    same_dominant_fraction = max(Counter(dominant).values()) / len(dominant) if dominant else 0.0
    globally_concentrated = overall["top_1_concentration"] >= 0.8 or overall["normalized_entropy"] < 0.35
    status = "global_collapse" if globally_concentrated and same_dominant_fraction >= 0.8 else "no_global_collapse"
    return {
        "status": status,
        "overall": overall,
        "by_capability": by_capability,
        "same_dominant_module_fraction": same_dominant_fraction,
        "rule": "Capability concentration alone is specialization; global concentration plus the same dominant module across capabilities is collapse.",
    }


def _lease_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        traces = [row["trace"] for row in rows if row.get("trace")]
        ages = [value for trace in traces for value in trace["lease_ages"]]
        switches = [value for trace in traces for value in trace["switch_rates"]]
        continuations = [value for trace in traces for value in trace["continuation_rates"]]
        return {
            "average_lease_length": _mean(ages),
            "median_lease_length": statistics.median(ages) if ages else None,
            "switches_per_chunk": _mean(switches),
            "switches_per_request": _mean(sum(trace["switch_rates"]) for trace in traces),
            "same_module_continuation_probability": _mean(continuations),
            "lease_state_norm": _mean(value for trace in traces for value in trace["lease_state_norms"]),
            "state_change_per_chunk": _mean(value for trace in traces for value in trace["lease_state_changes"]),
            "state_reset_count": sum(trace["state_reset_count"] for trace in traces),
        }
    return {
        "overall": metrics(records),
        "by_capability": {capability: metrics([row for row in records if row["metadata"]["capability"] == capability]) for capability in CAPABILITIES},
    }


def _sparsity_analysis(records: list[dict[str, Any]], model: nn.Module) -> dict[str, Any]:
    parameter_estimate = parameter_counts(model)
    module_parameter_counts = [
        count_parameters(module) for module in model.emc_modules
    ]
    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        traces = [row["trace"] for row in rows if row.get("trace")]
        requested = sum(trace["requested_pairs"] for trace in traces)
        executed = sum(trace["executed_pairs"] for trace in traces)
        discarded = sum(trace["discarded_pairs"] for trace in traces)
        routed_counts: Counter[int] = Counter()
        for trace in traces:
            routed_counts.update(
                {
                    int(index): count
                    for index, count in trace["module_counts"].items()
                }
            )
        routed_total = sum(routed_counts.values())
        route_weighted_module_parameters = (
            sum(
                routed_counts[index] * module_parameter_counts[index]
                for index in range(model.config.num_modules)
            )
            / routed_total
            if routed_total
            else None
        )
        return {
            "logical_top_k": (
                model.config.resolved_active_top_k
                if isinstance(model, ChunkedEMCModel)
                else model.config.modules_per_cycle
            ),
            "actual_modules_executed_per_chunk": _mean(
                value
                for trace in traces
                for value in trace["actual_unique_executed_per_chunk"]
            ),
            "average_module_population_touched_per_request": _mean(
                trace["population_fraction_touched"] for trace in traces
            ),
            "chunk_module_computations_requested": requested,
            "chunk_module_computations_executed": executed,
            "discarded_module_computations": discarded,
            "requested_executed_discrepancy_fraction": (
                abs(requested - executed) / requested if requested else 0.0
            ),
            "large_discrepancy": bool(
                requested and abs(requested - executed) / requested > 0.05
            ),
            "approximate_active_parameter_uses_per_forward": (
                parameter_estimate.approximate_parameter_uses_per_forward
            ),
            "route_weighted_module_parameters_per_selected_slot": (
                route_weighted_module_parameters
            ),
            "active_flops": None,
        }
    return {
        "overall": metrics(records),
        "by_capability": {
            capability: metrics(
                [
                    row
                    for row in records
                    if row["metadata"]["capability"] == capability
                ]
            )
            for capability in CAPABILITIES
        },
    }


def _module_diagnostics(model: nn.Module, records: list[dict[str, Any]]) -> dict[str, Any]:
    proposal_norms = _integrator_matrix(records, "capability", "proposal_norm", model)
    modules = []
    for index, module in enumerate(model.emc_modules):
        gradient_squared = sum(parameter.grad.detach().float().square().sum().item() for parameter in module.parameters() if parameter.grad is not None)
        parameter_squared = sum(parameter.detach().float().square().sum().item() for parameter in module.parameters())
        modules.append({
            "module": index, "family": model.module_families[index],
            "gradient_norm": gradient_squared ** 0.5,
            "parameter_norm": parameter_squared ** 0.5,
            "update_norm": None,
        })
    return {
        "modules": modules,
        "proposal_norms_by_capability": proposal_norms,
        "stateful_diagnostics": _lease_analysis(records),
        "sampling_policy": "Collected only on explicit evaluation forwards; no per-step synchronization.",
    }


def _notable_examples(
    examples: tuple[DiagnosticExample, ...], records: list[dict[str, Any]],
    interventions: list[InterventionResult], limit: int,
) -> list[dict[str, Any]]:
    candidates: list[tuple[int, int, str]] = []
    causal_by_example: dict[int, list[InterventionResult]] = defaultdict(list)
    for intervention in interventions:
        causal_by_example[intervention.example_index].append(intervention)
    for index, row in enumerate(records):
        trace = row.get("trace") or {}
        total = sum(trace.get("module_counts", {}).values())
        top = max(trace.get("module_counts", {}).values(), default=0)
        confident_failure = not row["exact_match"] and total and top / total >= 0.75
        disagreement = False
        acceptance = {
            key: value
            for key, value in trace.get("integrator", {})
            .get("acceptance", {})
            .items()
            if value is not None
        }
        if trace.get("module_counts") and acceptance:
            selected_top = max(trace["module_counts"], key=trace["module_counts"].get)
            accepted_top = max(acceptance, key=acceptance.get)
            disagreement = selected_top != accepted_top
        causal_effects = causal_by_example.get(index, [])
        strongest_effect = (
            min(
                causal_effects,
                key=lambda effect: (
                    effect.intervened_token_accuracy
                    - effect.baseline_token_accuracy
                ),
            )
            if causal_effects
            else None
        )
        causal_drop = (
            strongest_effect.intervened_token_accuracy
            - strongest_effect.baseline_token_accuracy
            if strongest_effect is not None
            else 0.0
        )
        if causal_drop < 0:
            candidates.append((0, index, "causally important module intervention"))
        elif confident_failure:
            candidates.append((1, index, "confident routing failure"))
        elif disagreement:
            candidates.append((2, index, "Nexus and Integrator disagree"))
        elif not row["exact_match"]:
            candidates.append((3, index, "model failure"))
        else:
            candidates.append((4, index, "representative success"))
    output = []
    for _, index, reason in sorted(candidates)[:limit]:
        example = examples[index]
        row = records[index]
        output.append({
            "reason": reason,
            "prompt": example.prompt,
            "target": example.target,
            "prediction": row["prediction"],
            "metadata": row["metadata"],
            "routing": row.get("trace", {}).get("family_counts") if row.get("trace") else None,
            "integrator_acceptance": row.get("trace", {}).get("integrator", {}).get("acceptance") if row.get("trace") else None,
            "causal_effects": [
                {
                    "intervention": effect.intervention_type,
                    "target": effect.target,
                    "exact_delta_points": 100
                    * (effect.intervened_exact - effect.baseline_exact),
                    "token_accuracy_delta_points": 100
                    * (
                        effect.intervened_token_accuracy
                        - effect.baseline_token_accuracy
                    ),
                }
                for effect in causal_by_example.get(index, [])
            ],
        })
    return output


def _language_samples(model: nn.Module, tokenizer: TextTokenizer, examples: tuple[DiagnosticExample, ...], config: DiagnosticEvaluationConfig) -> list[str]:
    language = [example for example in examples if example.diagnostic_metadata.capability == "language"][:2]
    samples = []
    for example in language:
        try:
            samples.append(generate_text(model, tokenizer, example.prompt, max_new_tokens=24, greedy=True))
        except (ValueError, RuntimeError):
            break
    return samples


def _routing_matrix(records: list[dict[str, Any]], dimension: str, families: tuple[str, ...]) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {}
    for key, rows in _group_records(records, lambda row: row["metadata"][dimension]).items():
        counts: Counter[str] = Counter()
        for row in rows:
            if row.get("trace"):
                counts.update(row["trace"]["family_counts"])
        total = sum(counts.values())
        matrix[key] = {family: counts[family] / total if total else 0.0 for family in families}
    return matrix


def _module_routing_matrix(records: list[dict[str, Any]], dimension: str, num_modules: int) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {}
    for key, rows in _group_records(records, lambda row: row["metadata"][dimension]).items():
        counts: Counter[str] = Counter()
        for row in rows:
            if row.get("trace"):
                counts.update(row["trace"]["module_counts"])
        total = sum(counts.values())
        matrix[key] = {str(index): counts[str(index)] / total if total else 0.0 for index in range(num_modules)}
    return matrix


def _router_score_matrix(records: list[dict[str, Any]], dimension: str, num_modules: int) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    for key, rows in _group_records(records, lambda row: row["metadata"][dimension]).items():
        output[key] = {str(index): _mean((row.get("trace") or {}).get("mean_router_scores", {}).get(str(index)) for row in rows) for index in range(num_modules)}
    return output


def _request_pool_matrix(records: list[dict[str, Any]], dimension: str, num_modules: int) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for key, rows in _group_records(records, lambda row: row["metadata"][dimension]).items():
        counts: Counter[str] = Counter()
        for row in rows:
            counts.update((row.get("trace") or {}).get("request_pool_counts", {}))
        total = sum(counts.values())
        output[key] = {str(index): counts[str(index)] / total if total else 0.0 for index in range(num_modules)}
    return output


def _integrator_matrix(records: list[dict[str, Any]], dimension: str, metric: str, model: nn.Module) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    for key, rows in _group_records(records, lambda row: row["metadata"][dimension]).items():
        by_family: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            values = (row.get("trace") or {}).get("integrator", {}).get(metric, {})
            for module, value in values.items():
                by_family[model.module_families[int(module)]].append(value)
        output[key] = {family: _mean(values) for family, values in by_family.items()}
    return output


def _nexus_integrator_disagreement(records: list[dict[str, Any]], model: nn.Module) -> dict[str, Any]:
    rows = []
    for capability, grouped in _group_records(records, lambda row: row["metadata"]["capability"]).items():
        counts: Counter[int] = Counter()
        acceptance: dict[int, list[float]] = defaultdict(list)
        for row in grouped:
            trace = row.get("trace") or {}
            counts.update({int(key): value for key, value in trace.get("module_counts", {}).items()})
            for key, value in trace.get("integrator", {}).get("acceptance", {}).items():
                acceptance[int(key)].append(value)
        total = sum(counts.values())
        for index in range(model.config.num_modules):
            frequency = counts[index] / total if total else 0.0
            mean_acceptance = _mean(acceptance[index])
            rows.append({"capability": capability, "module": index, "family": model.module_families[index], "selection_frequency": frequency, "acceptance": mean_acceptance, "high_selection_low_acceptance": bool(mean_acceptance is not None and frequency >= 0.25 and mean_acceptance < 0.2)})
    return {"rows": rows, "flagged": [row for row in rows if row["high_selection_low_acceptance"]]}


def _trace_scalar_by(records: list[dict[str, Any]], dimension: str, field: str) -> dict[str, float | None]:
    return {key: _mean((row.get("trace") or {}).get(field) for row in rows) for key, rows in _group_records(records, lambda row: row["metadata"][dimension]).items()}


def _vector_by(records: list[dict[str, Any]], dimension: str, field: str) -> dict[str, list[float]]:
    return {key: _column_means([(row.get("trace") or {}).get(field, []) for row in rows]) for key, rows in _group_records(records, lambda row: row["metadata"][dimension]).items()}


def _collapse_metrics(records: list[dict[str, Any]], num_modules: int) -> dict[str, Any]:
    counts = Counter()
    unique = []
    request_pool = Counter()
    for row in records:
        trace = row.get("trace") or {}
        counts.update({int(key): value for key, value in trace.get("module_counts", {}).items()})
        request_pool.update(trace.get("request_pool_counts", {}))
        if trace:
            unique.append(len(trace.get("unique_modules", [])))
    total = sum(counts.values())
    distribution = [counts[index] / total if total else 0.0 for index in range(num_modules)]
    entropy = -sum(value * math.log(value) for value in distribution if value > 0)
    sorted_values = sorted(distribution, reverse=True)
    pool_total = sum(request_pool.values())
    pool_top = max(request_pool.values(), default=0) / pool_total if pool_total else 0.0
    return {
        "observations": total,
        "distribution": distribution,
        "normalized_entropy": entropy / math.log(num_modules) if num_modules > 1 else 1.0,
        "effective_module_count": math.exp(entropy) if total else 0.0,
        "top_module": max(range(num_modules), key=lambda index: distribution[index]) if total else None,
        "top_1_concentration": sorted_values[0] if sorted_values else 0.0,
        "top_2_concentration": sum(sorted_values[:2]),
        "minimum_utilization": min(distribution) if distribution else 0.0,
        "request_pool_concentration": pool_top,
        "chunk_routing_concentration": sorted_values[0] if sorted_values else 0.0,
        "average_unique_modules_per_request": _mean(unique),
    }


def _mutual_information(pairs: Iterable[tuple[str, str]]) -> float:
    joint = Counter(pairs)
    total = sum(joint.values())
    if not total:
        return 0.0
    left = Counter()
    right = Counter()
    for (a, b), count in joint.items():
        left[a] += count
        right[b] += count
    return sum((count / total) * math.log((count * total) / (left[a] * right[b])) for (a, b), count in joint.items())


def _categorical_matrix(observations: list[tuple[str, str, str]], row_index: int, column_index: int) -> dict[str, dict[str, int]]:
    rows = sorted({observation[row_index] for observation in observations})
    columns = sorted({observation[column_index] for observation in observations})
    counts = Counter((observation[row_index], observation[column_index]) for observation in observations)
    return {row: {column: counts[(row, column)] for column in columns} for row in rows}


def _group_records(records: Iterable[dict[str, Any]], key_function: Any) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(key_function(record))].append(record)
    return dict(grouped)


def _mean(values: Iterable[float | int | None]) -> float | None:
    valid = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return sum(valid) / len(valid) if valid else None


def _column_means(rows: Iterable[list[float]]) -> list[float | None]:
    valid = [row for row in rows if row]
    if not valid:
        return []
    width = min(len(row) for row in valid)
    return [_mean(row[index] for row in valid) for index in range(width)]


def _length_bucket(length: int) -> str:
    if length <= 64:
        return "<=64"
    if length <= 128:
        return "65-128"
    if length <= 256:
        return "129-256"
    if length <= 512:
        return "257-512"
    return ">512"


def _distractor_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 4:
        return "low"
    if count <= 12:
        return "medium"
    return "high"


def _learning_status(results: Mapping[str, Mapping[str, Any]]) -> str:
    valid = [metrics["token_accuracy"] for metrics in results.values() if metrics.get("examples")]
    if not valid:
        return "insufficient_evaluable_examples"
    return "learning_signal_present" if max(valid) > 0.25 else "no_clear_learning_signal"


def _extreme_capability(results: Mapping[str, Mapping[str, Any]], *, maximum: bool) -> str | None:
    valid = [(name, metrics["token_accuracy"]) for name, metrics in results.items() if metrics.get("examples")]
    if not valid:
        return None
    function = max if maximum else min
    return function(valid, key=lambda item: item[1])[0]


def _cycle_statement(cycle_results: Mapping[str, Any]) -> str:
    if not cycle_results.get("supported"):
        return str(cycle_results.get("reason"))
    improved = 0
    for values in cycle_results["capability_by_cycle"].values():
        first = values[min(values, key=int)]["token_accuracy"]
        last = values[max(values, key=int)]["token_accuracy"]
        if first is not None and last is not None and last > first:
            improved += 1
    return f"Additional cycles improve sampled token accuracy for {improved} capabilities."


def _resolved_precision(precision: str, device: torch.device) -> str:
    if precision == "auto":
        return "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else ("fp16" if device.type == "cuda" else "fp32")
    return precision


def _autocast(precision: str, device: torch.device):
    resolved = _resolved_precision(precision, device)
    if device.type != "cuda" or resolved == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if resolved == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def git_commit() -> str | None:
    root = Path(__file__).resolve().parents[2]
    git_directory = root / ".git"
    try:
        if git_directory.is_file():
            target = git_directory.read_text(encoding="utf-8").strip()
            git_directory = (root / target.removeprefix("gitdir:").strip()).resolve()
        head = (git_directory / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or None
        reference = head.removeprefix("ref:").strip()
        loose_reference = git_directory / reference
        if loose_reference.is_file():
            return loose_reference.read_text(encoding="utf-8").strip() or None
        packed = git_directory / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith(("#", "^")):
                    commit, name = line.split(" ", 1)
                    if name == reference:
                        return commit
    except (OSError, ValueError):
        return None
    return None


def _write_metrics_csv(report: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "capability", "metric", "value"))
        writer.writeheader()
        for metric, value in report["overall_metrics"].items():
            writer.writerow(
                {
                    "section": "overall",
                    "capability": "overall",
                    "metric": metric,
                    "value": value,
                }
            )
        for capability, metrics in report["capability_results"].items():
            for metric, value in metrics.items():
                writer.writerow({"section": "capability", "capability": capability, "metric": metric, "value": value})
        for capability, difficulties in report["difficulty_curves"].items():
            for difficulty, metrics in difficulties.items():
                for metric in (
                    "exact_accuracy",
                    "token_accuracy",
                    "cross_entropy",
                    "routing_entropy",
                ):
                    writer.writerow(
                        {
                            "section": f"difficulty-{difficulty}",
                            "capability": capability,
                            "metric": metric,
                            "value": metrics[metric],
                        }
                    )
        for capability, targets in report["causal_ablations"]["performance_drop_when_family_removed"].items():
            for family, metrics in targets.items():
                writer.writerow({"section": "disable_family", "capability": capability, "metric": family, "value": metrics["accuracy_delta_points"]})


def _write_comparison_csv(comparison: Mapping[str, Any], path: Path) -> None:
    names = [row["name"] for row in comparison["checkpoints"]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("capability", *names))
        writer.writeheader()
        for capability, values in comparison["capability_exact_accuracy"].items():
            writer.writerow({"capability": capability, **values})


def _report_markdown(report: Mapping[str, Any]) -> str:
    summary = report["executive_summary"]
    lines = [
        "# Executive Summary", "",
        f"- Learning: `{summary['overall_learning_status']}`",
        f"- Overall loss: `{_format_value(summary['overall_loss'])}`",
        f"- Language perplexity: `{_format_value(summary['language_perplexity'])}`",
        f"- Strongest / weakest: `{summary['strongest_capability']}` / `{summary['weakest_capability']}`",
        f"- Router collapse: `{summary['router_collapse_status']}`",
        f"- Specialization: {summary['specialization_evidence']}",
        f"- Surface vs computation: {summary['surface_routing_evidence']}",
        f"- Cycles: {summary['cycle_usefulness']}", "",
        "# Overall Metrics", "",
        _markdown_table({"overall": report["overall_metrics"]}), "",
        "# Capability Results", "",
        _markdown_table(report["capability_results"], ("exact_accuracy", "token_accuracy", "cross_entropy", "perplexity")), "",
        "# Generalization and Difficulty Curves", "",
        _markdown_table(_difficulty_table_rows(report["difficulty_curves"])), "",
        "# Nexus Analysis", "", "## Routing Frequency by Capability", "",
        _markdown_table(report["nexus_analysis"]["routing_frequency_by_capability"]), "",
        "## Routing Frequency by Surface", "",
        _markdown_table(report["nexus_analysis"]["routing_frequency_by_surface"]), "",
        "# Integrator Analysis", "",
        "## Acceptance by Capability", "",
        _markdown_table(report["integrator_analysis"]["acceptance_by_capability"]), "",
        "## Nexus Selection versus Integrator Acceptance", "",
        f"Flagged rows: {len(report['integrator_analysis']['nexus_vs_integrator_disagreement']['flagged'])}", "",
        "# Causal Ablations", "", "## Performance Drop When Family Removed", "",
        _nested_metric_table(report["causal_ablations"]["performance_drop_when_family_removed"], "accuracy_delta_points"), "",
        "# Surface-vs-Computation Analysis", "", report["surface_vs_computation"]["statement"], "",
        f"Operation MI: `{report['surface_vs_computation']['operation_selection_mutual_information_nats']:.6f}`; surface MI: `{report['surface_vs_computation']['surface_selection_mutual_information_nats']:.6f}`.", "",
        "# Temporal/Lease Analysis", "", _markdown_table({"overall": report["lease_temporal_analysis"]["overall"]}), "",
        "# Sparse Execution", "", _markdown_table({"overall": report["sparse_execution"]["overall"]}), "",
        "# Cycle Analysis", "", f"Supported: `{report['cycle_analysis'].get('supported')}`", "",
        "# Module Diagnostics", "", _list_table(report["module_diagnostics"]["modules"]), "",
        "# Notable Examples", "",
    ]
    for example in report["notable_examples"]:
        lines.extend(
            [
                f"## {example['reason']}",
                "",
                f"- Capability: `{example['metadata']['capability']}`",
                f"- Surface: `{example['metadata']['surface_format']}`",
                f"- Target: `{example['target']}`",
                f"- Prediction: `{example['prediction']}`",
                f"- Causal effects: `{json.dumps(example['causal_effects'])}`",
                "",
                "```text",
                example["prompt"].rstrip(),
                "```",
                "",
            ]
        )
    return "\n".join(lines)




def _difficulty_table_rows(
    curves: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for capability, difficulties in curves.items():
        for difficulty, values in difficulties.items():
            row = {
                "exact_accuracy": values["exact_accuracy"],
                "token_accuracy": values["token_accuracy"],
                "cross_entropy": values["cross_entropy"],
                "routing_entropy": values["routing_entropy"],
            }
            row.update(
                {
                    f"route_{family}": frequency
                    for family, frequency in values[
                        "family_routing_frequency"
                    ].items()
                }
            )
            rows[f"{capability}/difficulty-{difficulty}"] = row
    return rows
def _comparison_markdown(comparison: Mapping[str, Any]) -> str:
    return "\n".join(["# Checkpoint Comparison", "", _markdown_table(comparison["capability_exact_accuracy"]), "", "## Router Collapse Status", "", _markdown_table({"status": comparison["router_collapse_status"]})])


def _markdown_table(rows: Mapping[str, Mapping[str, Any]], columns: Iterable[str] | None = None) -> str:
    if not rows:
        return "No data."
    resolved_columns = list(columns or sorted({column for values in rows.values() for column in values}))
    lines = ["| row | " + " | ".join(resolved_columns) + " |", "|---|" + "---|" * len(resolved_columns)]
    for name, values in rows.items():
        lines.append("| " + str(name) + " | " + " | ".join(_format_value(values.get(column)) for column in resolved_columns) + " |")
    return "\n".join(lines)


def _nested_metric_table(rows: Mapping[str, Mapping[str, Mapping[str, Any]]], metric: str) -> str:
    flattened = {row: {column: values.get(metric) for column, values in columns.items()} for row, columns in rows.items()}
    return _markdown_table(flattened)


def _list_table(rows: list[Mapping[str, Any]]) -> str:
    if not rows:
        return "No data."
    return _markdown_table({str(index): row for index, row in enumerate(rows)})


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "—"
    return str(value).replace("|", "\\|")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Rayvan computational capability diagnostics")
    parser.add_argument("--suite", choices=("capability",), default="capability")
    parser.add_argument("--checkpoint", action="append", required=True, help="repeat for comparison mode")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--examples-per-capability", type=int, default=100)
    parser.add_argument("--ablation-examples-per-capability", type=int, default=8)
    parser.add_argument("--diagnostic-smoke", action="store_true")
    parser.add_argument("--held-out-only", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-module-ablations", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = 20 if args.diagnostic_smoke else args.examples_per_capability
    ablation_examples = min(2, examples) if args.diagnostic_smoke else args.ablation_examples_per_capability
    config = DiagnosticEvaluationConfig(
        seed=args.seed,
        examples_per_capability=examples,
        ablation_examples_per_capability=0 if args.skip_ablations else ablation_examples,
        run_module_ablations=not args.skip_ablations and not args.skip_module_ablations,
        run_family_ablations=not args.skip_ablations,
        run_zero_proposal=not args.skip_ablations,
        run_forced_alternatives=not args.skip_ablations,
        held_out_only=args.held_out_only,
        device=args.device,
        precision=args.precision,
        smoke=args.diagnostic_smoke,
    )
    if len(args.checkpoint) == 1:
        evaluate_checkpoint(args.checkpoint[0], args.output_dir, config)
    else:
        compare_checkpoints(args.checkpoint, args.output_dir, config)
    print(f"wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
