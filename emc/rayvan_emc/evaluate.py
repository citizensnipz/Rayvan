from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import platform
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
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
from .checkpoint import load_model_checkpoint
from .chunked import ChunkedEMCModel, ChunkedExecutionTrace
from .diagnostics import count_parameters, parameter_counts
from .generation import generate_text
from .model import EMCCycleTrace, EMCModel, EMCOutput
from .n2 import N2ExecutionTrace
from .tokenization import TextTokenizer

REPORT_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class RouterDiagnosticThresholds:
    near_dead_selection_frequency: float = 0.01
    near_universal_request_fraction: float = 0.95
    global_fixed_set_request_fraction: float = 0.95
    low_utilization_entropy: float = 0.35
    mild_imbalance_gini: float = 0.20
    partial_collapse_gini: float = 0.45

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")


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
    deep_diagnostics: bool = False
    router_thresholds: RouterDiagnosticThresholds = field(
        default_factory=RouterDiagnosticThresholds
    )

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
    baseline_loss: float
    intervened_loss: float
    status: str
    validation: str


@dataclass
class _TraceSummary:
    module_counts: Counter[int]
    family_counts: Counter[str]
    module_routing_unit_counts: Counter[int]
    family_routing_unit_counts: Counter[str]
    module_slot_counts: dict[int, Counter[int]]
    routing_units: int
    request_pool_counts: Counter[int]
    score_sums: dict[int, float]
    score_observations: dict[int, int]
    probability_sums: dict[int, float]
    probability_observations: dict[int, int]
    selected_weight_sums: dict[int, float]
    selected_weight_observations: dict[int, int]
    pre_top_k_entropies: list[float]
    post_top_k_entropies: list[float]
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
    executed_module_sets: list[tuple[int, ...]]
    population_fraction_touched: float


def evaluate_checkpoint(
    checkpoint: str | Path,
    output_directory: str | Path,
    config: DiagnosticEvaluationConfig | None = None,
) -> dict[str, Any]:
    resolved = config or DiagnosticEvaluationConfig()
    loaded = load_model_checkpoint(checkpoint, device=resolved.device)
    try:
        report = evaluate_suite(
            loaded.model,
            loaded.tokenizer,
            resolved,
            checkpoint=str(checkpoint),
            checkpoint_training_config=loaded.training_config,
            checkpoint_training_diagnostics=getattr(
                loaded, "training_diagnostics", None
            ),
        )
        write_report(report, output_directory)
        return report
    finally:
        if next(loaded.model.parameters()).device.type == "cuda":
            loaded.model.to("cpu")
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_suite(
    model: nn.Module,
    tokenizer: TextTokenizer,
    config: DiagnosticEvaluationConfig | None = None,
    *,
    checkpoint: str | None = None,
    checkpoint_training_config: Mapping[str, Any] | None = None,
    checkpoint_training_diagnostics: Mapping[str, Any] | None = None,
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
    evaluated_tokens = 0
    for index, example in enumerate(examples):
        record, _, token_count = _evaluate_example(
            model, tokenizer, example, resolved, return_trace=True
        )
        record["example_index"] = index
        baseline_records.append(record)
        evaluated_tokens += token_count

    interventions = _run_ablations(
        model,
        tokenizer,
        examples,
        baseline_records,
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
    collapse = _collapse_analysis(
        baseline_records, model, resolved.router_thresholds
    )
    lease = _lease_analysis(baseline_records)
    sparsity = _sparsity_analysis(baseline_records, model)
    module_diagnostics = _module_diagnostics(
        model, baseline_records, checkpoint_training_diagnostics
    )
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
    integrity_warnings = _diagnostic_integrity_warnings(
        overall_metrics,
        interventions,
        causal,
        collapse,
        surface_analysis,
        module_diagnostics,
        cycle_results,
    )
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
                "deep_diagnostics": resolved.deep_diagnostics,
                "router_diagnostic_thresholds": asdict(resolved.router_thresholds),
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
            "routing": collapse["summary"],
            "router_collapse_status": collapse["status"],
            "specialization_status": causal["specialization_status"],
            "specialization_evidence": causal["specialization_statement"],
            "surface_routing_evidence": surface_analysis["statement"],
            "causal_diagnostics": causal["summary"],
            "cycles": _cycle_statement(cycle_results),
            "cycle_usefulness": _cycle_statement(cycle_results),
            "causal_family_importance": causal["most_important_family_by_capability"],
            "integrity_warning_count": len(integrity_warnings),
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
        "diagnostic_integrity_warnings": integrity_warnings,
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
    serializable_comparison = _replace_non_finite(comparison)
    (destination / "comparison.json").write_text(
        json.dumps(
            serializable_comparison,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    _write_comparison_csv(
        serializable_comparison, destination / "comparison.csv"
    )
    (destination / "comparison.md").write_text(
        _comparison_markdown(serializable_comparison), encoding="utf-8"
    )
    return comparison


def write_report(report: Mapping[str, Any], output_directory: str | Path) -> None:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    serializable_report = _replace_non_finite(report)
    (destination / "report.json").write_text(
        json.dumps(
            serializable_report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    _write_metrics_csv(serializable_report, destination / "metrics.csv")
    (destination / "report.md").write_text(
        _report_markdown(serializable_report), encoding="utf-8"
    )


def _replace_non_finite(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: _replace_non_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_replace_non_finite(item) for item in value]
    return value


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
    config: DiagnosticEvaluationConfig,
) -> list[InterventionResult]:
    if config.ablation_examples_per_capability == 0:
        return []
    families = tuple(model.module_families)
    unique_families = tuple(dict.fromkeys(families))
    sample_limit = config.ablation_examples_per_capability
    if config.deep_diagnostics:
        sample_limit = max(sample_limit, 32)
    selected_per_capability: Counter[str] = Counter()
    results: list[InterventionResult] = []
    for index, (example, baseline) in enumerate(zip(examples, records, strict=True)):
        capability = example.diagnostic_metadata.capability
        if baseline["skipped"] or selected_per_capability[capability] >= sample_limit:
            continue
        selected_per_capability[capability] += 1
        baseline_exact = float(baseline["exact_match"])
        baseline_token = float(baseline["token_accuracy"])
        baseline_loss = float(baseline["loss"])
        interventions: list[tuple[str, str, dict[str, Tensor | None]]] = []
        if config.run_module_ablations:
            for module_index in range(model.config.num_modules):
                availability = torch.ones(model.config.num_modules, dtype=torch.bool)
                availability[module_index] = False
                interventions.append(
                    (
                        "disable_module",
                        str(module_index),
                        {"availability_mask": availability},
                    )
                )
        if config.run_zero_proposal:
            for module_index in range(model.config.num_modules):
                zero = torch.zeros(model.config.num_modules, dtype=torch.bool)
                zero[module_index] = True
                interventions.append(
                    (
                        "zero_module_proposal",
                        str(module_index),
                        {"zero_mask": zero},
                    )
                )
        if config.run_family_ablations:
            for family in unique_families:
                availability = torch.tensor(
                    [name != family for name in families], dtype=torch.bool
                )
                interventions.append(
                    (
                        "disable_family",
                        family,
                        {"availability_mask": availability},
                    )
                )
        if config.run_zero_proposal:
            for family in unique_families:
                zero = torch.tensor(
                    [name == family for name in families], dtype=torch.bool
                )
                interventions.append(
                    ("zero_family_proposal", family, {"zero_mask": zero})
                )
        if config.run_forced_alternatives:
            trace = baseline.get("trace") or {}
            counts = trace.get("module_counts", {})
            normal_order = sorted(
                range(model.config.num_modules),
                key=lambda item: -int(counts.get(str(item), 0)),
            )
            top_family = families[normal_order[0]] if normal_order else families[0]
            top_k = int(
                getattr(model, "active_top_k", model.config.modules_per_cycle)
            )
            for candidate in range(model.config.num_modules):
                forced = [candidate]
                forced.extend(
                    index
                    for index in normal_order
                    if index != candidate and index not in forced
                )
                interventions.append(
                    (
                        "force_node_alternative",
                        str(candidate),
                        {"forced_modules": torch.tensor(forced[:top_k])},
                    )
                )
            for family in unique_families:
                if family == top_family:
                    continue
                candidate = next(
                    i for i, name in enumerate(families) if name == family
                )
                forced = [candidate]
                forced.extend(
                    i for i in normal_order if i != candidate and i not in forced
                )
                forced.extend(
                    i
                    for i in range(model.config.num_modules)
                    if i not in forced
                )
                interventions.append(
                    (
                        "force_family_alternative",
                        family,
                        {"forced_modules": torch.tensor(forced[:top_k])},
                    )
                )
        for intervention_type, target, kwargs in interventions:
            try:
                intervened, _, _ = _evaluate_example(
                    model,
                    tokenizer,
                    example,
                    config,
                    return_trace=True,
                    availability_mask=kwargs.get("availability_mask"),
                    zero_mask=kwargs.get("zero_mask"),
                    forced_modules=kwargs.get("forced_modules"),
                )
                status, validation = _validate_intervention(
                    intervention_type,
                    target,
                    families,
                    baseline.get("trace"),
                    intervened.get("trace"),
                )
            except ValueError as error:
                intervened = baseline
                status = "unsupported_intervention"
                validation = str(error)
            results.append(
                InterventionResult(
                    example_index=index,
                    capability=capability,
                    intervention_type=intervention_type,
                    target=target,
                    baseline_exact=baseline_exact,
                    intervened_exact=float(intervened["exact_match"]),
                    baseline_token_accuracy=baseline_token,
                    intervened_token_accuracy=float(
                        intervened["token_accuracy"]
                    ),
                    baseline_loss=baseline_loss,
                    intervened_loss=float(intervened["loss"]),
                    status=status,
                    validation=validation,
                )
            )
    return results


def _validate_intervention(
    intervention_type: str,
    target: str,
    families: tuple[str, ...],
    baseline_trace: Mapping[str, Any] | None,
    intervened_trace: Mapping[str, Any] | None,
) -> tuple[str, str]:
    if not baseline_trace or not intervened_trace:
        return (
            "unsupported_intervention",
            "forward trace unavailable; active path could not be validated",
        )
    node_interventions = {
        "disable_module",
        "zero_module_proposal",
        "force_node_alternative",
    }
    target_modules = (
        {int(target)}
        if intervention_type in node_interventions
        else {index for index, family in enumerate(families) if family == target}
    )
    baseline_active = set(baseline_trace.get("unique_modules", ()))
    intervened_active = set(intervened_trace.get("unique_modules", ()))
    if intervention_type not in {
        "force_family_alternative",
        "force_node_alternative",
    } and not (baseline_active & target_modules):
        return (
            "target_not_selected",
            "target was absent from the baseline active forward path",
        )
    if intervention_type in {"disable_module", "disable_family"}:
        if intervened_active & target_modules:
            return (
                "intervention_no_effect",
                "disabled target remained present in the active routing trace",
            )
        return (
            "active_intervention",
            "target was active at baseline and absent from intervened routing",
        )
    if intervention_type in {"zero_family_proposal", "zero_module_proposal"}:
        norms = intervened_trace.get("integrator", {}).get("proposal_norm", {})
        observed = [
            norms.get(str(index))
            for index in target_modules & intervened_active
            if str(index) in norms
        ]
        if not observed:
            return (
                "unsupported_intervention",
                "proposal-norm telemetry unavailable for the selected target",
            )
        if any(value is not None and abs(float(value)) > 1e-8 for value in observed):
            return (
                "intervention_no_effect",
                "target proposal remained nonzero at the Integrator input",
            )
        return (
            "active_intervention",
            "selected target proposal was zero at the Integrator input",
        )
    if intervention_type in {
        "force_family_alternative",
        "force_node_alternative",
    }:
        if not (intervened_active & target_modules):
            return (
                "intervention_no_effect",
                "forced target did not appear in intervened routing",
            )
        routing_changed = bool(
            baseline_trace.get("module_counts")
            != intervened_trace.get("module_counts")
            or baseline_trace.get("module_slot_counts")
            != intervened_trace.get("module_slot_counts")
        )
        if not routing_changed:
            return (
                "intervention_no_effect",
                "forced routing matched baseline counts and slot occupancy",
            )
        return (
            "active_intervention",
            "forced target appeared and routing counts or slots changed",
        )
    return "unsupported_intervention", "unknown intervention type"


def _evaluate_cycles(
    model: nn.Module,
    tokenizer: TextTokenizer,
    examples: tuple[DiagnosticExample, ...],
    config: DiagnosticEvaluationConfig,
) -> dict[str, Any]:
    if isinstance(model, ChunkedEMCModel):
        return {
            "supported": False,
            "reason_code": "architecture_no_cycle_interface",
            "reason": (
                "n1_chunked exposes chunk recurrence and lease telemetry, but "
                "does not expose independently repeatable EMC cycles or a "
                "cycle-limit intervention"
            ),
            "configured_num_cycles": model.config.num_cycles,
            "available_temporal_telemetry": "lease_temporal_analysis",
            "unavailable_fields": [
                "routing_changes_between_cycles",
                "module_persistence_between_cycles",
                "latent_state_change_per_cycle",
                "integrator_contribution_by_cycle",
                "shared_state_convergence_by_cycle",
            ],
        }
    if not isinstance(model, EMCModel):
        return {
            "supported": False,
            "reason_code": "unsupported_model_type",
            "reason": "model does not expose EMC cycle traces",
        }
    if model.config.num_cycles <= 1:
        return {
            "supported": False,
            "reason_code": "single_cycle_architecture",
            "reason": "model is configured to execute exactly one EMC cycle",
            "configured_num_cycles": model.config.num_cycles,
        }
    sample_limit = min(
        32 if config.deep_diagnostics else 8,
        config.examples_per_capability,
    )
    rows: dict[str, dict[str, dict[str, float]]] = {}
    full_cycle_outputs: list[EMCOutput] = []
    for capability in CAPABILITIES:
        selected = [
            example
            for example in examples
            if example.diagnostic_metadata.capability == capability
        ][:sample_limit]
        rows[capability] = {}
        for cycle in range(1, model.config.num_cycles + 1):
            evaluated = [
                _evaluate_example(
                    model,
                    tokenizer,
                    example,
                    config,
                    return_trace=cycle == model.config.num_cycles,
                    cycle_limit=cycle,
                )
                for example in selected
            ]
            metrics = [item[0] for item in evaluated]
            if cycle == model.config.num_cycles:
                full_cycle_outputs.extend(
                    item[1] for item in evaluated if item[1] is not None
                )
            valid = [row for row in metrics if not row["skipped"]]
            rows[capability][str(cycle)] = {
                "exact_accuracy": _mean(row["exact_match"] for row in valid),
                "token_accuracy": _mean(row["token_accuracy"] for row in valid),
                "loss": _mean(row["loss"] for row in valid),
            }
    return {
        "supported": True,
        "configured_cycles": model.config.num_cycles,
        "sampled_examples_per_capability": sample_limit,
        "capability_by_cycle": rows,
        "telemetry": _cycle_trace_telemetry(full_cycle_outputs),
        "limitations": {
            "latent_state_change_per_cycle": (
                "unavailable: cycle traces expose latent shape, not latent values"
            ),
            "shared_state_convergence_by_cycle": (
                "unavailable: token-cycle architecture has no exposed shared-state "
                "convergence scalar"
            ),
        },
    }


def _cycle_trace_telemetry(outputs: Iterable[EMCOutput]) -> dict[str, Any]:
    changes: dict[int, list[float]] = defaultdict(list)
    persistence: dict[int, list[float]] = defaultdict(list)
    contributions: dict[int, list[float]] = defaultdict(list)
    executed_counts: Counter[int] = Counter()
    for output in outputs:
        previous: EMCCycleTrace | None = None
        for trace in output.trace:
            executed_counts[trace.cycle] += 1
            if trace.integrator_trace is not None:
                contributions[trace.cycle].append(
                    float(
                        trace.integrator_trace.proposal_contributions.float()
                        .mean()
                        .item()
                    )
                )
            if previous is not None:
                current_indices = trace.selected_indices
                previous_indices = previous.selected_indices
                if (
                    current_indices is not None
                    and previous_indices is not None
                    and current_indices.shape == previous_indices.shape
                ):
                    changes[trace.cycle].append(
                        float(
                            (current_indices != previous_indices)
                            .float()
                            .mean()
                            .item()
                        )
                    )
                current_modules = set(trace.selected_modules)
                previous_modules = set(previous.selected_modules)
                union = current_modules | previous_modules
                persistence[trace.cycle].append(
                    len(current_modules & previous_modules) / len(union)
                    if union
                    else 1.0
                )
            previous = trace
    return {
        "cycles_executed_observations": {
            str(cycle): count for cycle, count in sorted(executed_counts.items())
        },
        "routing_slot_change_fraction_from_previous_cycle": {
            str(cycle): _mean(values) for cycle, values in sorted(changes.items())
        },
        "module_set_jaccard_from_previous_cycle": {
            str(cycle): _mean(values)
            for cycle, values in sorted(persistence.items())
        },
        "mean_integrator_contribution_by_cycle": {
            str(cycle): _mean(values)
            for cycle, values in sorted(contributions.items())
        },
    }


def _summarize_trace(output: EMCOutput, model: nn.Module) -> _TraceSummary:
    summary = _empty_trace()
    if output.trace:
        for cycle in output.trace:
            if cycle.selected_indices is None:
                continue
            selected = cycle.selected_indices.long()
            _observe_routing(
                summary,
                selected,
                cycle.router_scores,
                cycle.router_weights,
                model.module_families,
            )
            if cycle.integrator_trace is not None:
                _observe_integrator(
                    summary, selected, cycle.integrator_trace, "token"
                )
            summary.executed_module_sets.append(cycle.selected_modules)
        summary.requested_pairs = sum(summary.module_counts.values())
        summary.executed_pairs = sum(
            len(cycle.selected_modules) for cycle in output.trace
        )
        summary.discarded_pairs = 0
        summary.actual_unique_executed_per_chunk = [
            len(cycle.selected_modules) for cycle in output.trace
        ]
        summary.population_fraction_touched = (
            len(summary.unique_modules) / model.config.num_modules
        )
    if isinstance(output.chunk_trace, ChunkedExecutionTrace):
        chunked = output.chunk_trace
        summary.request_pool_counts.update(
            chunked.request_pool.module_indices.reshape(-1).tolist()
        )
        for chunk in chunked.chunks:
            selected = chunk.active_modules.long()
            _observe_routing(
                summary,
                selected,
                chunk.routing_scores,
                chunk.routing_weights,
                model.module_families,
            )
            summary.lease_ages.extend(
                chunk.lease_ages[chunk.lease_ages > 0].float().tolist()
            )
            summary.switch_rates.append(float(chunk.switch_rate))
            summary.continuation_rates.append(float(chunk.retained_rate))
            summary.balancing_biases.append(chunk.balance_bias.float().tolist())
            summary.lease_state_norms.append(float(chunk.lease_state_norm))
            summary.lease_state_changes.append(float(chunk.lease_state_change))
            summary.state_reset_count += int(chunk.state_reset_count)
            _observe_integrator(
                summary, selected, chunk.token_integrator_trace, "token"
            )
            _observe_integrator(
                summary, selected, chunk.state_integrator_trace, "state"
            )
            summary.requested_pairs += int(
                chunk.computed_chunk_module_pairs
            )
            summary.executed_pairs += int(
                chunk.retained_chunk_module_pairs
            )
            summary.discarded_pairs += int(
                chunk.computed_chunk_module_pairs
                - chunk.retained_chunk_module_pairs
            )
            summary.actual_unique_executed_per_chunk.append(
                len(chunk.executed_modules)
            )
            summary.executed_module_sets.append(chunk.executed_modules)
        summary.population_fraction_touched = float(
            chunked.population_fraction_touched
        )
    if isinstance(output.chunk_trace, N2ExecutionTrace):
        n2_trace = output.chunk_trace
        summary.requested_pairs = n2_trace.actual_node_executions
        summary.executed_pairs = n2_trace.actual_node_executions
        summary.discarded_pairs = 0
        summary.actual_unique_executed_per_chunk = [
            len(n2_trace.executed_node_ids)
        ]
        summary.executed_module_sets = [n2_trace.executed_node_ids]
        summary.population_fraction_touched = (
            len(n2_trace.executed_node_ids) / model.config.num_modules
        )
        for diagnostics in n2_trace.node_diagnostics:
            if diagnostics.average_lease_length is not None:
                summary.lease_ages.append(diagnostics.average_lease_length)
            if diagnostics.continuation_probability is not None:
                summary.continuation_rates.append(
                    diagnostics.continuation_probability
                )
            if diagnostics.state_change_magnitude is not None:
                summary.lease_state_changes.append(
                    diagnostics.state_change_magnitude
                )
            summary.state_reset_count += diagnostics.state_resets
    return summary


def _empty_trace() -> _TraceSummary:
    return _TraceSummary(
        module_counts=Counter(),
        family_counts=Counter(),
        module_routing_unit_counts=Counter(),
        family_routing_unit_counts=Counter(),
        module_slot_counts=defaultdict(Counter),
        routing_units=0,
        request_pool_counts=Counter(),
        score_sums=defaultdict(float),
        score_observations=defaultdict(int),
        probability_sums=defaultdict(float),
        probability_observations=defaultdict(int),
        selected_weight_sums=defaultdict(float),
        selected_weight_observations=defaultdict(int),
        pre_top_k_entropies=[],
        post_top_k_entropies=[],
        unique_modules=set(),
        integrator={
            name: defaultdict(list)
            for name in (
                "acceptance",
                "token_contribution",
                "state_contribution",
                "proposal_norm",
            )
        },
        proposal_similarities=[],
        gate_magnitudes=[],
        balancing_biases=[],
        lease_ages=[],
        switch_rates=[],
        continuation_rates=[],
        lease_state_norms=[],
        lease_state_changes=[],
        state_reset_count=0,
        requested_pairs=0,
        executed_pairs=0,
        discarded_pairs=0,
        actual_unique_executed_per_chunk=[],
        executed_module_sets=[],
        population_fraction_touched=0.0,
    )


def _observe_routing(
    summary: _TraceSummary,
    selected: Tensor,
    scores: Tensor,
    selected_weights: Tensor,
    families: tuple[str, ...],
) -> None:
    selected_rows = selected.reshape(-1, selected.size(-1))
    weight_rows = selected_weights.float().reshape(
        -1, selected_weights.size(-1)
    )
    summary.routing_units += selected_rows.size(0)
    for row, weights in zip(selected_rows, weight_rows, strict=True):
        values = row.tolist()
        summary.module_counts.update(values)
        summary.family_counts.update(families[index] for index in values)
        summary.module_routing_unit_counts.update(set(values))
        summary.family_routing_unit_counts.update(
            {families[index] for index in values}
        )
        summary.unique_modules.update(values)
        for slot, (module_index, weight) in enumerate(
            zip(values, weights.tolist(), strict=True)
        ):
            summary.module_slot_counts[module_index][slot] += 1
            summary.selected_weight_sums[module_index] += float(weight)
            summary.selected_weight_observations[module_index] += 1
    _observe_scores(summary, scores)
    probabilities = torch.softmax(scores.float(), dim=-1)
    finite_rows = probabilities.reshape(-1, probabilities.size(-1))
    for index in range(finite_rows.size(-1)):
        values = finite_rows[:, index]
        finite = values[torch.isfinite(values)]
        if finite.numel():
            summary.probability_sums[index] += float(finite.sum().item())
            summary.probability_observations[index] += int(finite.numel())
    pre_entropy = -(
        probabilities * probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1)
    post_entropy = -(
        selected_weights.float()
        * selected_weights.float().clamp_min(1e-12).log()
    ).sum(dim=-1)
    summary.pre_top_k_entropies.extend(pre_entropy.reshape(-1).tolist())
    summary.post_top_k_entropies.extend(post_entropy.reshape(-1).tolist())


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
        mask = ~torch.eye(
            similarity.size(-1), dtype=torch.bool, device=similarity.device
        )
        summary.proposal_similarities.extend(similarity[..., mask].reshape(-1).tolist())
    summary.gate_magnitudes.extend(trace.gate_magnitude.float().reshape(-1).tolist())


def _trace_to_dict(summary: _TraceSummary) -> dict[str, Any]:
    return {
        "module_counts": {
            str(key): value for key, value in summary.module_counts.items()
        },
        "family_counts": dict(summary.family_counts),
        "module_routing_unit_counts": {
            str(key): value
            for key, value in summary.module_routing_unit_counts.items()
        },
        "family_routing_unit_counts": dict(
            summary.family_routing_unit_counts
        ),
        "module_slot_counts": {
            str(module): {
                str(slot): count for slot, count in slots.items()
            }
            for module, slots in summary.module_slot_counts.items()
        },
        "routing_units": summary.routing_units,
        "request_pool_counts": {
            str(key): value
            for key, value in summary.request_pool_counts.items()
        },
        "mean_router_scores": {
            str(key): summary.score_sums[key] / summary.score_observations[key]
            for key in summary.score_observations
        },
        "mean_router_probabilities": {
            str(key): (
                summary.probability_sums[key]
                / summary.probability_observations[key]
            )
            for key in summary.probability_observations
        },
        "mean_selected_weights": {
            str(key): (
                summary.selected_weight_sums[key]
                / summary.selected_weight_observations[key]
            )
            for key in summary.selected_weight_observations
        },
        "mean_pre_top_k_entropy": _mean(summary.pre_top_k_entropies),
        "mean_post_top_k_entropy": _mean(summary.post_top_k_entropies),
        "mean_routing_entropy": _mean(summary.post_top_k_entropies),
        "unique_modules": sorted(summary.unique_modules),
        "integrator": {
            metric: {
                str(index): _mean(values) for index, values in modules.items()
            }
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
        "actual_unique_executed_per_chunk": (
            summary.actual_unique_executed_per_chunk
        ),
        "executed_module_sets": [
            list(indices) for indices in summary.executed_module_sets
        ],
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


def _aggregate_routing(
    records: list[dict[str, Any]], model: nn.Module
) -> dict[str, Any]:
    families = tuple(dict.fromkeys(model.module_families))
    by_capability = _routing_matrix(records, "capability", families)
    by_surface = _routing_matrix(records, "surface_format", families)
    family_metrics = _family_routing_metrics(records, model, by_capability)
    return {
        "routing_variable": (
            "selected family per top-k slot; request and routing-unit presence "
            "are reported separately"
        ),
        "routing_frequency_by_capability": by_capability,
        "routing_frequency_by_surface": by_surface,
        "module_frequency_by_capability": _module_routing_matrix(
            records, "capability", model.config.num_modules
        ),
        "module_frequency_by_surface": _module_routing_matrix(
            records, "surface_format", model.config.num_modules
        ),
        "router_scores_by_capability": _router_score_matrix(
            records, "capability", model.config.num_modules
        ),
        "router_probability_by_capability": _module_trace_metric_matrix(
            records,
            "capability",
            "mean_router_probabilities",
            model.config.num_modules,
        ),
        "normalized_selected_weight_by_capability": (
            _module_trace_metric_matrix(
                records,
                "capability",
                "mean_selected_weights",
                model.config.num_modules,
            )
        ),
        "request_pool_by_capability": _request_pool_matrix(
            records, "capability", model.config.num_modules
        ),
        "pre_top_k_entropy_by_capability": _trace_scalar_by(
            records, "capability", "mean_pre_top_k_entropy"
        ),
        "post_top_k_entropy_by_capability": _trace_scalar_by(
            records, "capability", "mean_post_top_k_entropy"
        ),
        "routing_entropy_by_capability": _trace_scalar_by(
            records, "capability", "mean_post_top_k_entropy"
        ),
        "family_metrics": family_metrics,
        "average_unique_modules_per_request": _mean(
            len((row.get("trace") or {}).get("unique_modules", []))
            for row in records
            if row.get("trace")
        ),
        "population_fraction_touched_by_capability": _trace_scalar_by(
            records, "capability", "population_fraction_touched"
        ),
        "balancing_bias_by_capability": _vector_by(
            records, "capability", "mean_balance_bias"
        ),
    }


def _family_routing_metrics(
    records: list[dict[str, Any]],
    model: nn.Module,
    by_capability: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, Any]]:
    families = tuple(dict.fromkeys(model.module_families))
    module_indices = {
        family: [
            index
            for index, module_family in enumerate(model.module_families)
            if module_family == family
        ]
        for family in families
    }
    traces = [row["trace"] for row in records if row.get("trace")]
    total_selections = sum(
        sum(trace.get("family_counts", {}).values()) for trace in traces
    )
    total_routing_units = sum(trace.get("routing_units", 0) for trace in traces)
    result: dict[str, dict[str, Any]] = {}
    for family in families:
        indices = module_indices[family]
        selection_count = sum(
            trace.get("family_counts", {}).get(family, 0) for trace in traces
        )
        selected_requests = sum(
            bool(trace.get("family_counts", {}).get(family, 0))
            for trace in traces
        )
        selected_units = sum(
            trace.get("family_routing_unit_counts", {}).get(family, 0)
            for trace in traces
        )
        slot_counts: Counter[int] = Counter()
        for trace in traces:
            for index in indices:
                slot_counts.update(
                    {
                        int(slot): count
                        for slot, count in trace.get(
                            "module_slot_counts", {}
                        )
                        .get(str(index), {})
                        .items()
                    }
                )
        probability = _mean(
            sum(
                float(
                    trace.get("mean_router_probabilities", {}).get(
                        str(index), 0.0
                    )
                )
                for index in indices
            )
            for trace in traces
        )
        selected_weight = _mean(
            trace.get("mean_selected_weights", {}).get(str(index))
            for trace in traces
            for index in indices
        )
        acceptance = _mean(
            trace.get("integrator", {})
            .get("acceptance", {})
            .get(str(index))
            for trace in traces
            for index in indices
        )
        contribution = _mean(
            trace.get("integrator", {})
            .get("token_contribution", {})
            .get(str(index))
            for trace in traces
            for index in indices
        )
        never_used_categories = sum(
            metrics.get(family, 0.0) < 0.01
            for metrics in by_capability.values()
        )
        result[family] = {
            "selection_frequency": (
                selection_count / total_selections if total_selections else 0.0
            ),
            "request_selection_fraction": (
                selected_requests / len(traces) if traces else 0.0
            ),
            "routing_unit_selection_fraction": (
                selected_units / total_routing_units
                if total_routing_units
                else 0.0
            ),
            "chunk_selection_fraction": (
                selected_units / total_routing_units
                if total_routing_units
                and isinstance(model, ChunkedEMCModel)
                else None
            ),
            "selection_slot_distribution": {
                str(slot): count / selection_count if selection_count else 0.0
                for slot, count in sorted(slot_counts.items())
            },
            "mean_router_probability_before_top_k": probability,
            "mean_normalized_selected_weight": selected_weight,
            "mean_integrator_acceptance": acceptance,
            "mean_integrator_token_contribution": contribution,
            "effectively_never_used_capability_fraction": (
                never_used_categories / len(by_capability)
                if by_capability
                else None
            ),
        }
    return result


def _aggregate_integrator(records: list[dict[str, Any]], model: nn.Module) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in ("acceptance", "token_contribution", "state_contribution", "proposal_norm"):
        output[f"{metric}_by_capability"] = _integrator_matrix(records, "capability", metric, model)
        output[f"{metric}_by_surface"] = _integrator_matrix(records, "surface_format", metric, model)
    output["proposal_similarity_by_capability"] = _trace_scalar_by(records, "capability", "mean_proposal_similarity")
    output["gate_magnitude_by_capability"] = _trace_scalar_by(records, "capability", "mean_gate_magnitude")
    output["nexus_vs_integrator_disagreement"] = _nexus_integrator_disagreement(records, model)
    return output


def _aggregate_interventions(
    results: list[InterventionResult], model: nn.Module
) -> dict[str, Any]:
    matrices: dict[str, dict[str, dict[str, Any]]] = {}
    intervention_types = sorted(
        {result.intervention_type for result in results}
    )
    for intervention_type in intervention_types:
        matrices[intervention_type] = {}
        relevant = [
            result
            for result in results
            if result.intervention_type == intervention_type
        ]
        targets = sorted({result.target for result in relevant})
        for capability in CAPABILITIES:
            matrices[intervention_type][capability] = {}
            for target in targets:
                rows = [
                    result
                    for result in relevant
                    if result.capability == capability
                    and result.target == target
                ]
                if rows:
                    matrices[intervention_type][capability][target] = (
                        _intervention_metrics(rows)
                    )
    family_matrix = matrices.get("disable_family", {})
    specialization = _specialization_analysis(family_matrix)
    expert_names = tuple(
        getattr(
            model,
            "expert_names",
            tuple(f"m{index}" for index in range(model.config.num_modules)),
        )
    )
    module_matrix = matrices.get("disable_module", {})
    named_module_matrix = {
        capability: {
            expert_names[int(target)]: metrics
            for target, metrics in values.items()
        }
        for capability, values in module_matrix.items()
    }
    node_specialization = _specialization_analysis(named_module_matrix)
    active = [
        result for result in results if result.status == "active_intervention"
    ]
    worsened = sum(_intervention_effect(result) == "worsened" for result in active)
    status_counts = Counter(result.status for result in results)
    return {
        "sign_conventions": {
            "exact_accuracy_delta": (
                "intervened minus baseline; negative means ablation hurt"
            ),
            "token_accuracy_delta": (
                "intervened minus baseline; negative means ablation hurt"
            ),
            "token_accuracy_degradation": (
                "baseline minus intervened; positive means ablation hurt"
            ),
            "loss_increase": (
                "intervened cross-entropy minus baseline; positive means "
                "ablation hurt"
            ),
            "perplexity_increase": (
                "intervened minus baseline; positive means ablation hurt"
            ),
        },
        "effect_thresholds": {
            "loss_absolute": 1e-4,
            "token_accuracy_absolute": 1e-6,
            "precedence": (
                "loss increase determines direction when measurable; token "
                "accuracy is the fallback"
            ),
        },
        "matrices": matrices,
        "performance_drop_when_family_removed": family_matrix,
        "family_capability_causal_impact": specialization["impact_matrix"],
        "most_important_family_by_capability": specialization[
            "most_important_family_by_capability"
        ],
        "specialization_status": specialization["status"],
        "specialization_statement": specialization["statement"],
        "specialization_criteria": specialization["criteria"],
        "node_capability_causal_impact": node_specialization["impact_matrix"],
        "most_important_node_by_capability": node_specialization[
            "most_important_family_by_capability"
        ],
        "same_family_specialization_status": node_specialization["status"],
        "same_family_specialization_statement": node_specialization["statement"],
        "same_family_specialization_criteria": node_specialization["criteria"],
        "expert_names": list(expert_names),
        "statistical_confidence": (
            "not estimated; sampled interventions are descriptive and no "
            "independence or variance assumptions are imposed"
        ),
        "intervention_status_counts": dict(status_counts),
        "active_interventions_evaluated": len(active),
        "active_interventions_measurably_worsened": worsened,
        "summary": (
            f"{len(active)} active interventions evaluated; {worsened} "
            "measurably worsened loss or token accuracy"
        ),
        "module_families": list(model.module_families),
    }


def _intervention_metrics(rows: list[InterventionResult]) -> dict[str, Any]:
    active = [row for row in rows if row.status == "active_intervention"]
    exact_deltas = [
        row.intervened_exact - row.baseline_exact for row in active
    ]
    token_deltas = [
        row.intervened_token_accuracy - row.baseline_token_accuracy
        for row in active
    ]
    loss_increases = [
        row.intervened_loss - row.baseline_loss for row in active
    ]
    perplexity_increases = [
        math.exp(min(row.intervened_loss, 20.0))
        - math.exp(min(row.baseline_loss, 20.0))
        for row in active
    ]
    effects = [_intervention_effect(row) for row in active]
    mean_exact = _mean(exact_deltas)
    mean_token = _mean(token_deltas)
    return {
        "attempted_interventions": len(rows),
        "evaluated_interventions": len(active),
        "examples": len(active),
        "status_counts": dict(Counter(row.status for row in rows)),
        "baseline_exact_accuracy": _mean(
            row.baseline_exact for row in active
        ),
        "intervened_exact_accuracy": _mean(
            row.intervened_exact for row in active
        ),
        "mean_exact_accuracy_delta": mean_exact,
        "median_exact_accuracy_delta": _median(exact_deltas),
        "exact_accuracy_delta": mean_exact,
        "accuracy_delta_points": (
            100 * mean_exact if mean_exact is not None else None
        ),
        "baseline_token_accuracy": _mean(
            row.baseline_token_accuracy for row in active
        ),
        "intervened_token_accuracy": _mean(
            row.intervened_token_accuracy for row in active
        ),
        "mean_token_accuracy_delta": mean_token,
        "median_token_accuracy_delta": _median(token_deltas),
        "mean_token_accuracy_degradation": (
            -mean_token if mean_token is not None else None
        ),
        "median_token_accuracy_degradation": (
            -_median(token_deltas) if token_deltas else None
        ),
        "token_accuracy_delta_points": (
            100 * mean_token if mean_token is not None else None
        ),
        "baseline_cross_entropy": _mean(
            row.baseline_loss for row in active
        ),
        "intervened_cross_entropy": _mean(
            row.intervened_loss for row in active
        ),
        "mean_loss_increase": _mean(loss_increases),
        "median_loss_increase": _median(loss_increases),
        "mean_perplexity_increase": _mean(perplexity_increases),
        "median_perplexity_increase": _median(perplexity_increases),
        "fraction_measurably_worsened": (
            effects.count("worsened") / len(effects) if effects else None
        ),
        "fraction_measurably_improved": (
            effects.count("improved") / len(effects) if effects else None
        ),
        "fraction_no_measurable_change": (
            effects.count("unchanged") / len(effects) if effects else None
        ),
    }


def _intervention_effect(result: InterventionResult) -> str:
    loss_increase = result.intervened_loss - result.baseline_loss
    if abs(loss_increase) > 1e-4:
        return "worsened" if loss_increase > 0 else "improved"
    token_degradation = (
        result.baseline_token_accuracy - result.intervened_token_accuracy
    )
    if abs(token_degradation) > 1e-6:
        return "worsened" if token_degradation > 0 else "improved"
    return "unchanged"


def _specialization_analysis(
    family_matrix: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    impact_matrix: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    important: dict[str, str | None] = {}
    dominant_signals: list[tuple[str, str, Mapping[str, Any], bool]] = []
    active_cells = 0
    for capability in CAPABILITIES:
        values = family_matrix.get(capability, {})
        usable = {
            family: metrics
            for family, metrics in values.items()
            if metrics.get("evaluated_interventions", 0) > 0
        }
        for family, metrics in usable.items():
            active_cells += 1
            impact_matrix[family][capability] = {
                "mean_loss_increase": metrics.get("mean_loss_increase"),
                "median_loss_increase": metrics.get("median_loss_increase"),
                "mean_token_accuracy_degradation": metrics.get(
                    "mean_token_accuracy_degradation"
                ),
                "median_token_accuracy_degradation": metrics.get(
                    "median_token_accuracy_degradation"
                ),
                "exact_accuracy_delta": metrics.get(
                    "mean_exact_accuracy_delta"
                ),
                "evaluated_interventions": metrics.get(
                    "evaluated_interventions"
                ),
                "fraction_measurably_worsened": metrics.get(
                    "fraction_measurably_worsened"
                ),
            }
        if not usable:
            important[capability] = None
            continue
        ordered = sorted(
            usable.items(),
            key=lambda item: _descriptive_impact(item[1]),
            reverse=True,
        )
        family, top = ordered[0]
        important[capability] = family
        second = ordered[1][1] if len(ordered) > 1 else None
        meaningful = _meaningful_causal_cell(top)
        margin = _meaningful_impact_margin(top, second)
        if meaningful:
            dominant_signals.append((capability, family, top, margin))
    criteria = {
        "meaningful_family_capability_effect": (
            "at least 2 active interventions, at least 50% measurably worsened, "
            "and mean loss increase >=0.02 nats or mean token-accuracy "
            "degradation >=0.02"
        ),
        "weak": (
            "a meaningful capability-specific contrast or differential effect "
            "exists, but family diversity/sample consistency is limited"
        ),
        "moderate": (
            "at least 2 capabilities have meaningful, margin-separated effects "
            "with at least 2 different dominant families and >=3 interventions "
            "per dominant cell"
        ),
        "strong": (
            "at least 3 capabilities have margin-separated effects with at "
            "least 2 dominant families, >=5 interventions per dominant cell, "
            "and >=75% measurable worsening"
        ),
        "insufficient_evidence": (
            "fewer than 8 active family-capability cells or fewer than 2 "
            "capabilities with active family comparisons"
        ),
    }
    comparable_capabilities = sum(
        sum(
            metrics.get("evaluated_interventions", 0) > 0
            for metrics in values.values()
        )
        >= 2
        for values in family_matrix.values()
    )
    if active_cells < 8 or comparable_capabilities < 2:
        status = "insufficient_evidence"
    else:
        separated = [signal for signal in dominant_signals if signal[3]]
        distinct_families = {signal[1] for signal in separated}
        strong = (
            len(separated) >= 3
            and len(distinct_families) >= 2
            and all(
                signal[2].get("evaluated_interventions", 0) >= 5
                and (signal[2].get("fraction_measurably_worsened") or 0.0)
                >= 0.75
                for signal in separated
            )
        )
        moderate = (
            len(separated) >= 2
            and len(distinct_families) >= 2
            and all(
                signal[2].get("evaluated_interventions", 0) >= 3
                for signal in separated
            )
        )
        if strong:
            status = "strong_specialization_signal"
        elif moderate:
            status = "moderate_specialization_signal"
        elif separated or _has_differential_family_effect(impact_matrix):
            status = "weak_specialization_signal"
        else:
            status = "no_detectable_specialization"
    examples = [
        f"{capability}: {family}"
        for capability, family, _, margin in dominant_signals
        if margin
    ][:4]
    if status == "insufficient_evidence":
        statement = (
            "Insufficient active family ablations for a specialization "
            "conclusion."
        )
    elif status == "no_detectable_specialization":
        statement = (
            "No detectable capability-specific family effect under the "
            "documented loss/token criteria."
        )
    else:
        detail = "; ".join(examples) if examples else "differential effects"
        statement = (
            f"{status}: descriptive causal differences ({detail}); sampled "
            "data do not establish statistical significance."
        )
    return {
        "status": status,
        "statement": statement,
        "criteria": criteria,
        "impact_matrix": dict(impact_matrix),
        "most_important_family_by_capability": important,
    }


def _descriptive_impact(metrics: Mapping[str, Any]) -> float:
    return float(metrics.get("mean_loss_increase") or 0.0) + float(
        metrics.get("mean_token_accuracy_degradation") or 0.0
    )


def _meaningful_causal_cell(metrics: Mapping[str, Any]) -> bool:
    return bool(
        metrics.get("evaluated_interventions", 0) >= 2
        and (metrics.get("fraction_measurably_worsened") or 0.0) >= 0.5
        and (
            (metrics.get("mean_loss_increase") or 0.0) >= 0.02
            or (metrics.get("mean_token_accuracy_degradation") or 0.0)
            >= 0.02
        )
    )


def _meaningful_impact_margin(
    top: Mapping[str, Any], second: Mapping[str, Any] | None
) -> bool:
    if second is None:
        return True
    return bool(
        (top.get("mean_loss_increase") or 0.0)
        - (second.get("mean_loss_increase") or 0.0)
        >= 0.02
        or (top.get("mean_token_accuracy_degradation") or 0.0)
        - (second.get("mean_token_accuracy_degradation") or 0.0)
        >= 0.02
    )


def _has_differential_family_effect(
    impact_matrix: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> bool:
    for capabilities in impact_matrix.values():
        losses = [
            float(metrics["mean_loss_increase"])
            for metrics in capabilities.values()
            if metrics.get("mean_loss_increase") is not None
        ]
        tokens = [
            float(metrics["mean_token_accuracy_degradation"])
            for metrics in capabilities.values()
            if metrics.get("mean_token_accuracy_degradation") is not None
        ]
        if (
            losses
            and max(losses) - min(losses) >= 0.03
            or tokens
            and max(tokens) - min(tokens) >= 0.02
        ):
            return True
    return False


def _surface_vs_operation(records: list[dict[str, Any]]) -> dict[str, Any]:
    family_observations: list[tuple[str, str, str, str]] = []
    node_observations: list[tuple[str, str, str, str]] = []
    for row in records:
        trace = row.get("trace")
        if not trace:
            continue
        metadata = row["metadata"]
        prefix = (
            metadata.get("capability", metadata["operation"]),
            metadata["operation"],
            metadata["surface_format"],
        )
        for family, count in trace["family_counts"].items():
            family_observations.extend((*prefix, family) for _ in range(count))
        for module, count in trace.get("module_counts", {}).items():
            node_observations.extend((*prefix, str(module)) for _ in range(count))
    family = _routing_association(family_observations)
    nodes = _routing_association(node_observations)
    relative = family["relative_association"]
    strength = family["association_strength"]
    if family["insufficient_routing_diversity"]:
        statement = (
            f"{relative}, but family-routing diversity is too low for a strong "
            "operation-dependent routing claim."
        )
    else:
        statement = (
            f"{relative}; the larger normalized family association is {strength}, "
            "which is descriptive rather than evidence of causal specialization."
        )
    return {
        "routing_variable": "selected family and individual N1 node per top-k slot",
        "samples": family["samples"],
        "operation_selection_mutual_information_nats": family[
            "operation_mutual_information_nats"
        ],
        "surface_selection_mutual_information_nats": family[
            "surface_mutual_information_nats"
        ],
        "operation_selection_normalized_mutual_information": family[
            "operation_normalized_mutual_information"
        ],
        "surface_selection_normalized_mutual_information": family[
            "surface_normalized_mutual_information"
        ],
        "capability_selection_mutual_information_nats": family[
            "capability_mutual_information_nats"
        ],
        "capability_selection_normalized_mutual_information": family[
            "capability_normalized_mutual_information"
        ],
        "operation_to_surface_ratio": family["operation_to_surface_ratio"],
        "routing_family_entropy_nats": family["target_entropy_nats"],
        "routing_family_normalized_entropy": family["target_normalized_entropy"],
        "effective_routing_family_count": family["effective_target_count"],
        "insufficient_routing_diversity": family[
            "insufficient_routing_diversity"
        ],
        "association_strength": strength,
        "statement": statement,
        "operation_family_matrix": family["operation_target_matrix"],
        "surface_family_matrix": family["surface_target_matrix"],
        "capability_family_matrix": family["capability_target_matrix"],
        "individual_n1_node_analysis": nodes,
        "weak_association_warning": (
            "Individual-node association is too weak to interpret."
            if nodes["association_strength"] == "very_weak"
            else None
        ),
    }


def _routing_association(
    observations: list[tuple[str, str, str, str]],
) -> dict[str, Any]:
    capability_pairs = [
        (capability, target) for capability, _, _, target in observations
    ]
    operation_pairs = [
        (operation, target) for _, operation, _, target in observations
    ]
    surface_pairs = [
        (surface, target) for _, _, surface, target in observations
    ]
    capability_mi = _mutual_information(capability_pairs)
    operation_mi = _mutual_information(operation_pairs)
    surface_mi = _mutual_information(surface_pairs)
    capability_nmi = _normalized_mutual_information(capability_pairs)
    operation_nmi = _normalized_mutual_information(operation_pairs)
    surface_nmi = _normalized_mutual_information(surface_pairs)
    target_counts = Counter(target for _, _, _, target in observations)
    target_entropy = _categorical_entropy(target_counts)
    target_categories = len(target_counts)
    normalized_entropy = (
        target_entropy / math.log(target_categories)
        if target_categories > 1
        else 0.0
    )
    effective_targets = math.exp(target_entropy) if target_counts else 0.0
    insufficient = bool(
        target_categories < 2
        or normalized_entropy < 0.35
        or effective_targets < 1.5
    )
    if operation_mi > surface_mi * 1.1:
        relative = "operation correlates more strongly than surface"
    elif surface_mi > operation_mi * 1.1:
        relative = "surface correlates more strongly than operation"
    else:
        relative = "operation and surface have similar association"
    maximum_nmi = max(capability_nmi, operation_nmi, surface_nmi)
    if maximum_nmi < 0.05:
        strength = "very_weak"
    elif maximum_nmi < 0.15:
        strength = "weak"
    elif maximum_nmi < 0.30:
        strength = "moderate"
    else:
        strength = "strong"
    return {
        "samples": len(observations),
        "capability_mutual_information_nats": capability_mi,
        "operation_mutual_information_nats": operation_mi,
        "surface_mutual_information_nats": surface_mi,
        "capability_normalized_mutual_information": capability_nmi,
        "operation_normalized_mutual_information": operation_nmi,
        "surface_normalized_mutual_information": surface_nmi,
        "operation_to_surface_ratio": (
            operation_mi / surface_mi if surface_mi else None
        ),
        "target_entropy_nats": target_entropy,
        "target_normalized_entropy": normalized_entropy,
        "effective_target_count": effective_targets,
        "insufficient_routing_diversity": insufficient,
        "association_strength": strength,
        "relative_association": relative,
        "capability_target_matrix": _categorical_matrix(observations, 0, 3),
        "operation_target_matrix": _categorical_matrix(observations, 1, 3),
        "surface_target_matrix": _categorical_matrix(observations, 2, 3),
    }


def _collapse_analysis(
    records: list[dict[str, Any]],
    model: nn.Module,
    thresholds: RouterDiagnosticThresholds | None = None,
) -> dict[str, Any]:
    resolved = thresholds or RouterDiagnosticThresholds()
    top_k = (
        model.config.resolved_active_top_k
        if isinstance(model, ChunkedEMCModel)
        else model.config.modules_per_cycle
    )
    overall = _collapse_metrics(
        records, model.config.num_modules, top_k, resolved
    )
    by_capability = {
        capability: _collapse_metrics(
            [
                row
                for row in records
                if row["metadata"]["capability"] == capability
            ],
            model.config.num_modules,
            top_k,
            resolved,
        )
        for capability in CAPABILITIES
    }
    dominant = [
        metrics["top_module"]
        for metrics in by_capability.values()
        if metrics["observations"]
    ]
    same_dominant_fraction = (
        max(Counter(dominant).values()) / len(dominant)
        if dominant
        else 0.0
    )
    near_universal = overall["near_universal_experts"]
    near_dead = overall["near_dead_experts"]
    fixed_set = overall["most_common_request_module_set_fraction"]
    effective = overall["effective_module_count"]
    if (
        len(near_universal) >= top_k
        and fixed_set >= resolved.global_fixed_set_request_fraction
        and effective <= top_k * 1.25
    ):
        status = "global_collapse"
    elif near_universal:
        status = "slot_monopoly"
    elif (
        overall["utilization_gini"] >= resolved.partial_collapse_gini
        or overall["normalized_entropy"] < resolved.low_utilization_entropy
        or len(near_dead) >= max(1, model.config.num_modules // 2)
    ):
        status = "partial_collapse"
    elif (
        overall["utilization_gini"] >= resolved.mild_imbalance_gini
        or near_dead
    ):
        status = "mild_imbalance"
    else:
        status = "healthy"
    universal_text = ", ".join(
        f"{model.module_families[index]}[{index}] selected in "
        f"{100 * overall['request_selection_fraction'][index]:.1f}% of requests"
        for index in near_universal
    )
    dead_text = ", ".join(
        f"{model.module_families[index]}[{index}] at "
        f"{100 * overall['selection_frequency'][index]:.2f}% of slots"
        for index in near_dead
    )
    details = "; ".join(
        detail
        for detail in (
            universal_text,
            f"near-dead: {dead_text}" if dead_text else "",
        )
        if detail
    )
    summary = f"{status}"
    if details:
        summary += f"; {details}"
    return {
        "status": status,
        "summary": summary,
        "overall": overall,
        "by_capability": by_capability,
        "same_dominant_module_fraction": same_dominant_fraction,
        "thresholds": asdict(resolved),
        "status_definitions": {
            "healthy": "no configured imbalance or liveness threshold crossed",
            "mild_imbalance": (
                "utilization Gini crosses the mild threshold or at least one "
                "expert is near-dead"
            ),
            "partial_collapse": (
                "strong utilization imbalance, low entropy, or at least half "
                "the experts are near-dead"
            ),
            "slot_monopoly": (
                "at least one expert appears in the configured near-universal "
                "fraction of requests while other top-k slots may still vary"
            ),
            "global_collapse": (
                "at least top-k experts are near-universal, the same request "
                "set dominates, and effective utilization is close to top-k"
            ),
        },
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


def _sparsity_analysis(
    records: list[dict[str, Any]], model: nn.Module
) -> dict[str, Any]:
    parameter_estimate = parameter_counts(model)
    total_parameters = count_parameters(model)
    module_parameter_counts = [
        count_parameters(module) for module in model.emc_modules
    ]
    total_module_parameters = sum(module_parameter_counts)

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
        selected_population_per_request = [
            sum(
                module_parameter_counts[index]
                for index in trace.get("unique_modules", ())
            )
            for trace in traces
        ]
        executed_parameters_per_unit = [
            sum(module_parameter_counts[index] for index in indices)
            for trace in traces
            for indices in trace.get("executed_module_sets", ())
        ]
        mean_selected_population = _mean(selected_population_per_request)
        mean_executed_parameters = _mean(executed_parameters_per_unit)
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
            "total_model_parameter_count": total_parameters,
            "total_routable_module_parameter_count": total_module_parameters,
            "mean_selected_module_parameter_count_per_request": (
                mean_selected_population
            ),
            "mean_fraction_total_parameters_in_selected_module_population": (
                mean_selected_population / total_parameters
                if mean_selected_population is not None and total_parameters
                else None
            ),
            "mean_actively_executed_module_parameters_per_routing_unit": (
                mean_executed_parameters
            ),
            "mean_fraction_total_parameters_actively_executed_in_modules": (
                mean_executed_parameters / total_parameters
                if mean_executed_parameters is not None and total_parameters
                else None
            ),
            "estimated_module_compute_relative_to_executing_all_modules": (
                mean_executed_parameters / total_module_parameters
                if mean_executed_parameters is not None
                and total_module_parameters
                else None
            ),
            "chunk_module_computations_requested": requested,
            "chunk_module_computations_executed": executed,
            "discarded_module_computations": discarded,
            "requested_executed_discrepancy_fraction": (
                abs(requested - executed) / requested if requested else 0.0
            ),
            "large_discrepancy": bool(
                requested
                and abs(requested - executed) / requested > 0.05
            ),
            "approximate_active_parameter_uses_per_forward": (
                parameter_estimate.approximate_parameter_uses_per_forward
            ),
            "approximate_active_parameter_fraction_per_forward": (
                parameter_estimate.approximate_parameter_uses_per_forward
                / total_parameters
                if total_parameters
                else None
            ),
            "route_weighted_module_parameters_per_selected_slot": (
                route_weighted_module_parameters
            ),
            "active_flops": None,
        }

    return {
        "definitions": {
            "module_population_touched": (
                "fraction of module identities selected at least once; ignores "
                "module size"
            ),
            "selected_parameter_population": (
                "unique parameters belonging to modules selected in a request; "
                "not a FLOP count"
            ),
            "actively_executed_parameters": (
                "parameters in the unique module implementations executed per "
                "routing unit; shared non-module parameters are excluded"
            ),
            "estimated_compute_relative_to_all_modules": (
                "routable module parameter ratio; heterogeneous operators make "
                "this a size proxy rather than a measured FLOP ratio"
            ),
        },
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


def _module_diagnostics(
    model: nn.Module,
    records: list[dict[str, Any]],
    training_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    proposal_norms = _integrator_matrix(
        records, "capability", "proposal_norm", model
    )
    saved_modules = {
        int(row["module"]): row
        for row in (training_diagnostics or {}).get("modules", ())
        if "module" in row
    }
    expert_names = tuple(
        getattr(
            model,
            "expert_names",
            tuple(f"m{item}" for item in range(model.config.num_modules)),
        )
    )
    modules = []
    any_live_gradients = False
    for index, module in enumerate(model.emc_modules):
        saved = saved_modules.get(index)
        if saved is not None:
            gradient_norm = saved.get("gradient_norm")
            update_norm = saved.get("update_norm")
            gradient_source = "training_instrumentation"
        else:
            observed_gradients = [
                parameter.grad.detach()
                for parameter in module.parameters()
                if parameter.grad is not None
            ]
            any_live_gradients = any_live_gradients or bool(
                observed_gradients
            )
            gradient_norm = (
                sum(
                    gradient.float().square().sum().item()
                    for gradient in observed_gradients
                )
                ** 0.5
                if observed_gradients
                else None
            )
            update_norm = None
            gradient_source = (
                "live_gradient_tensors"
                if observed_gradients
                else "unavailable"
            )
        parameter_squared = sum(
            parameter.detach().float().square().sum().item()
            for parameter in module.parameters()
        )
        modules.append(
            {
                "module": index,
                "family": model.module_families[index],
                "expert_name": expert_names[index],
                "gradient_norm": gradient_norm,
                "gradient_source": gradient_source,
                "parameter_norm": parameter_squared**0.5,
                "update_norm": update_norm,
            }
        )
    captured_from_training = bool(saved_modules)
    gradients_captured = captured_from_training or any_live_gradients
    updates_available = any(
        row.get("update_norm") is not None for row in modules
    )
    return {
        "modules": modules,
        "availability": {
            "execution_context": (
                "live_training_instrumentation"
                if captured_from_training
                else (
                    "live_model_with_gradient_tensors"
                    if any_live_gradients
                    else "checkpoint_or_evaluation_only"
                )
            ),
            "gradients_captured": gradients_captured,
            "optimizer_state_available": bool(
                (training_diagnostics or {}).get(
                    "optimizer_state_available", False
                )
            ),
            "update_norms_available": updates_available,
            "note": (
                "Evaluation runs under torch.inference_mode; absent tensors are "
                "reported as unavailable, never as numeric zero."
            ),
        },
        "proposal_norms_by_capability": proposal_norms,
        "stateful_diagnostics": _lease_analysis(records),
        "sampling_policy": (
            "Evaluation forwards retain aggregate trace scalars only. Optional "
            "training instrumentation samples module gradient/update norms at "
            "checkpoint intervals."
        ),
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
                    "loss_increase": (
                        effect.intervened_loss - effect.baseline_loss
                    ),
                    "status": effect.status,
                    "validation": effect.validation,
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


def _module_trace_metric_matrix(
    records: list[dict[str, Any]],
    dimension: str,
    field: str,
    num_modules: int,
) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    for key, rows in _group_records(
        records, lambda row: row["metadata"][dimension]
    ).items():
        output[key] = {
            str(index): _mean(
                (row.get("trace") or {})
                .get(field, {})
                .get(str(index))
                for row in rows
            )
            for index in range(num_modules)
        }
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


def _collapse_metrics(
    records: list[dict[str, Any]],
    num_modules: int,
    top_k: int,
    thresholds: RouterDiagnosticThresholds,
) -> dict[str, Any]:
    counts: Counter[int] = Counter()
    routing_unit_counts: Counter[int] = Counter()
    request_counts: Counter[int] = Counter()
    request_pool: Counter[str] = Counter()
    slot_counts: dict[int, Counter[int]] = defaultdict(Counter)
    unique: list[int] = []
    active_sets: Counter[tuple[int, ...]] = Counter()
    pre_top_k_entropies: list[float] = []
    post_top_k_entropies: list[float] = []
    traces = [row["trace"] for row in records if row.get("trace")]
    for trace in traces:
        counts.update(
            {
                int(key): value
                for key, value in trace.get("module_counts", {}).items()
            }
        )
        routing_unit_counts.update(
            {
                int(key): value
                for key, value in trace.get(
                    "module_routing_unit_counts", {}
                ).items()
            }
        )
        selected = tuple(sorted(trace.get("unique_modules", ())))
        request_counts.update(selected)
        active_sets[selected] += 1
        request_pool.update(trace.get("request_pool_counts", {}))
        unique.append(len(selected))
        if trace.get("mean_pre_top_k_entropy") is not None:
            pre_top_k_entropies.append(trace["mean_pre_top_k_entropy"])
        if trace.get("mean_post_top_k_entropy") is not None:
            post_top_k_entropies.append(trace["mean_post_top_k_entropy"])
        for module, slots in trace.get("module_slot_counts", {}).items():
            slot_counts[int(module)].update(
                {int(slot): count for slot, count in slots.items()}
            )
    total = sum(counts.values())
    routing_units = sum(trace.get("routing_units", 0) for trace in traces)
    distribution = [
        counts[index] / total if total else 0.0
        for index in range(num_modules)
    ]
    request_fractions = [
        request_counts[index] / len(traces) if traces else 0.0
        for index in range(num_modules)
    ]
    routing_unit_fractions = [
        routing_unit_counts[index] / routing_units if routing_units else 0.0
        for index in range(num_modules)
    ]
    entropy = -sum(
        value * math.log(value) for value in distribution if value > 0
    )
    sorted_values = sorted(distribution, reverse=True)
    pool_total = sum(request_pool.values())
    pool_top = (
        max(request_pool.values(), default=0) / pool_total if pool_total else 0.0
    )
    fixed_set_fraction = (
        max(active_sets.values(), default=0) / len(traces) if traces else 0.0
    )
    slot_occupancy = {
        str(slot): [
            (
                slot_counts[module][slot] / routing_units
                if routing_units
                else 0.0
            )
            for module in range(num_modules)
        ]
        for slot in range(top_k)
    }
    gini = _gini_coefficient(distribution)
    near_dead = [
        index
        for index, frequency in enumerate(distribution)
        if frequency < thresholds.near_dead_selection_frequency
    ]
    near_universal = [
        index
        for index, fraction in enumerate(request_fractions)
        if fraction >= thresholds.near_universal_request_fraction
    ]
    return {
        "observations": total,
        "requests": len(traces),
        "routing_units": routing_units,
        "selection_frequency": distribution,
        "distribution": distribution,
        "request_selection_fraction": request_fractions,
        "routing_unit_selection_fraction": routing_unit_fractions,
        "top_k_slot_occupancy": slot_occupancy,
        "normalized_entropy": (
            entropy / math.log(num_modules) if num_modules > 1 else 1.0
        ),
        "post_top_k_utilization_entropy_nats": entropy,
        "effective_module_count": math.exp(entropy) if total else 0.0,
        "utilization_gini": gini,
        "mean_pre_top_k_routing_entropy_nats": _mean(
            pre_top_k_entropies
        ),
        "mean_post_top_k_routing_entropy_nats": _mean(
            post_top_k_entropies
        ),
        "top_module": (
            max(
                range(num_modules),
                key=lambda index: distribution[index],
            )
            if total
            else None
        ),
        "top_1_concentration": (
            sorted_values[0] if sorted_values else 0.0
        ),
        "top_2_concentration": sum(sorted_values[:2]),
        "minimum_utilization": min(distribution) if distribution else 0.0,
        "request_pool_concentration": pool_top,
        "chunk_routing_concentration": (
            sorted_values[0] if sorted_values else 0.0
        ),
        "average_unique_modules_per_request": _mean(unique),
        "most_common_request_module_set": (
            list(active_sets.most_common(1)[0][0]) if active_sets else []
        ),
        "most_common_request_module_set_fraction": fixed_set_fraction,
        "near_dead_experts": near_dead,
        "near_universal_experts": near_universal,
        "number_near_dead_experts": len(near_dead),
        "number_near_universal_experts": len(near_universal),
    }


def _gini_coefficient(values: Iterable[float]) -> float:
    data = [max(0.0, float(value)) for value in values]
    total = sum(data)
    if not data or total == 0:
        return 0.0
    pairwise = sum(abs(left - right) for left in data for right in data)
    return pairwise / (2 * len(data) * total)


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


def _normalized_mutual_information(
    pairs: Iterable[tuple[str, str]],
) -> float:
    observations = list(pairs)
    if not observations:
        return 0.0
    left = Counter(item[0] for item in observations)
    right = Counter(item[1] for item in observations)
    denominator = math.sqrt(
        _categorical_entropy(left) * _categorical_entropy(right)
    )
    return _mutual_information(observations) / denominator if denominator else 0.0


def _categorical_entropy(counts: Mapping[Any, int]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
        if count
    )


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


def _median(values: Iterable[float | int | None]) -> float | None:
    valid = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.median(valid) if valid else None


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


def _diagnostic_integrity_warnings(
    overall_metrics: Mapping[str, Any],
    interventions: list[InterventionResult],
    causal: Mapping[str, Any],
    collapse: Mapping[str, Any],
    surface_analysis: Mapping[str, Any],
    module_diagnostics: Mapping[str, Any],
    cycle_results: Mapping[str, Any],
) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []

    def add(code: str, message: str, severity: str = "warning") -> None:
        warnings.append(
            {"code": code, "severity": severity, "message": message}
        )

    exact = overall_metrics.get("overall_exact_accuracy")
    if exact is not None and exact <= 0.05:
        add(
            "exact_accuracy_too_low_for_ablation",
            (
                "Overall exact accuracy is at or below 5%; exact-match deltas "
                "are unlikely to resolve causal effects. Use loss increase and "
                "token-accuracy degradation."
            ),
        )
    overall_collapse = collapse.get("overall", {})
    if overall_collapse.get("near_dead_experts"):
        add(
            "near_dead_experts",
            (
                f"Experts {overall_collapse['near_dead_experts']} are below "
                "the configured selection-frequency threshold."
            ),
        )
    if overall_collapse.get("near_universal_experts"):
        add(
            "near_universal_experts",
            (
                f"Experts {overall_collapse['near_universal_experts']} exceed "
                "the configured request-presence threshold; inspect slot "
                "monopolization."
            ),
        )
    threshold = collapse.get("thresholds", {}).get(
        "low_utilization_entropy", 0.35
    )
    if overall_collapse.get("normalized_entropy", 1.0) < threshold:
        add(
            "low_routing_entropy",
            "Post-top-k utilization entropy is below the configured threshold.",
        )
    if surface_analysis.get("insufficient_routing_diversity"):
        add(
            "insufficient_routing_diversity_for_mi",
            (
                "Routing diversity constrains operation/surface mutual "
                "information; relative MI does not imply strong routing."
            ),
        )
    availability = module_diagnostics.get("availability", {})
    if not availability.get("gradients_captured"):
        add(
            "gradients_unavailable",
            (
                "No live or training-instrumented gradient tensors are "
                "available for this diagnostic run."
            ),
            "info",
        )
    if not availability.get("update_norms_available"):
        add(
            "update_norms_unavailable",
            "Parameter update norms were not captured during training.",
            "info",
        )
    if not cycle_results.get("supported"):
        add(
            "cycle_telemetry_unavailable",
            str(cycle_results.get("reason", "cycle telemetry unavailable")),
            "info",
        )
    family_cells = [
        metrics
        for capabilities in causal.get(
            "performance_drop_when_family_removed", {}
        ).values()
        for metrics in capabilities.values()
    ]
    if not family_cells or any(
        metrics.get("evaluated_interventions", 0) < 3
        for metrics in family_cells
        if metrics.get("evaluated_interventions", 0)
    ):
        add(
            "small_ablation_sample",
            (
                "At least one family/capability cell has fewer than three "
                "active ablations; specialization evidence is descriptive."
            ),
        )
    inactive = Counter(
        result.status
        for result in interventions
        if result.status != "active_intervention"
    )
    if inactive.get("target_not_selected"):
        add(
            "ablation_target_not_active",
            (
                f"{inactive['target_not_selected']} interventions targeted a "
                "module absent from the baseline active path and were excluded "
                "from causal aggregation."
            ),
            "info",
        )
    if inactive.get("intervention_no_effect"):
        add(
            "intervention_path_unchanged",
            (
                f"{inactive['intervention_no_effect']} interventions did not "
                "change the validated active path."
            ),
        )
    if inactive.get("unsupported_intervention"):
        add(
            "unsupported_interventions",
            (
                f"{inactive['unsupported_intervention']} interventions could "
                "not be validated and were excluded."
            ),
        )
    return warnings


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
        for capability, targets in report["causal_ablations"][
            "performance_drop_when_family_removed"
        ].items():
            for family, metrics in targets.items():
                for metric in (
                    "mean_exact_accuracy_delta",
                    "mean_token_accuracy_delta",
                    "mean_token_accuracy_degradation",
                    "mean_loss_increase",
                    "median_loss_increase",
                    "mean_perplexity_increase",
                    "evaluated_interventions",
                    "fraction_measurably_worsened",
                    "fraction_measurably_improved",
                ):
                    writer.writerow(
                        {
                            "section": f"disable_family_{metric}",
                            "capability": capability,
                            "metric": family,
                            "value": metrics.get(metric),
                        }
                    )


def _write_comparison_csv(comparison: Mapping[str, Any], path: Path) -> None:
    names = [row["name"] for row in comparison["checkpoints"]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("capability", *names))
        writer.writeheader()
        for capability, values in comparison["capability_exact_accuracy"].items():
            writer.writerow({"capability": capability, **values})


def _report_markdown(report: Mapping[str, Any]) -> str:
    summary = report["executive_summary"]
    surface = report["surface_vs_computation"]
    cycle = report["cycle_analysis"]
    causal = report["causal_ablations"]
    lines = [
        "# Executive Summary",
        "",
        f"- Learning: `{summary['overall_learning_status']}`",
        f"- Overall loss: `{_format_value(summary['overall_loss'])}`",
        f"- Language perplexity: `{_format_value(summary['language_perplexity'])}`",
        (
            f"- Strongest / weakest: `{summary['strongest_capability']}` / "
            f"`{summary['weakest_capability']}`"
        ),
        f"- Routing: {summary['routing']}",
        (
            f"- Specialization (`{summary['specialization_status']}`): "
            f"{summary['specialization_evidence']}"
        ),
        f"- Causal diagnostics: {summary['causal_diagnostics']}",
        f"- Surface vs computation: {summary['surface_routing_evidence']}",
        f"- Cycles: {summary['cycles']}",
        "",
        "# Diagnostic Integrity Warnings",
        "",
    ]
    warnings = report.get("diagnostic_integrity_warnings", [])
    if warnings:
        lines.extend(
            f"- `{warning['code']}` ({warning['severity']}): "
            f"{warning['message']}"
            for warning in warnings
        )
    else:
        lines.append("No integrity warnings.")
    lines.extend(
        [
            "",
            "# Overall Metrics",
            "",
            _markdown_table({"overall": report["overall_metrics"]}),
            "",
            "# Capability Results",
            "",
            _markdown_table(
                report["capability_results"],
                (
                    "exact_accuracy",
                    "token_accuracy",
                    "cross_entropy",
                    "perplexity",
                ),
            ),
            "",
            "# Generalization and Difficulty Curves",
            "",
            _markdown_table(
                _difficulty_table_rows(report["difficulty_curves"])
            ),
            "",
            "# Nexus Analysis",
            "",
            "Selection frequency is a slot count; router probability, normalized "
            "selected weight, and Integrator acceptance are distinct metrics.",
            "",
            "## Per-Family Routing Metrics",
            "",
            _markdown_table(report["nexus_analysis"]["family_metrics"]),
            "",
            "## Routing Frequency by Capability",
            "",
            _markdown_table(
                report["nexus_analysis"]["routing_frequency_by_capability"]
            ),
            "",
            "## Mean Router Probability Before Top-K by Capability",
            "",
            _markdown_table(
                report["nexus_analysis"]["router_probability_by_capability"]
            ),
            "",
            "## Mean Normalized Selected Weight by Capability",
            "",
            _markdown_table(
                report["nexus_analysis"][
                    "normalized_selected_weight_by_capability"
                ]
            ),
            "",
            "## Router Collapse",
            "",
            f"Status: `{report['router_collapse']['status']}`",
            "",
            _markdown_table(
                {"overall": report["router_collapse"]["overall"]}
            ),
            "",
            "Configured thresholds:",
            "",
            *[
                f"- `{name}`: `{_format_value(value)}`"
                for name, value in report["router_collapse"][
                    "thresholds"
                ].items()
            ],
            "",
            "Status definitions:",
            "",
            *[
                f"- `{name}`: {description}"
                for name, description in report["router_collapse"][
                    "status_definitions"
                ].items()
            ],
            "",
            "# Integrator Analysis",
            "",
            "## Acceptance by Capability",
            "",
            _markdown_table(
                report["integrator_analysis"]["acceptance_by_capability"]
            ),
            "",
            "## Nexus Selection versus Integrator Acceptance",
            "",
            (
                "Flagged rows: "
                f"{len(report['integrator_analysis']['nexus_vs_integrator_disagreement']['flagged'])}"
            ),
            "",
            "# Causal Ablations",
            "",
            "Sign conventions:",
            "",
            *[
                f"- `{name}`: {description}"
                for name, description in causal["sign_conventions"].items()
            ],
            "",
            "## Family Removal: Mean Loss Increase",
            "",
            _nested_metric_table(
                causal["performance_drop_when_family_removed"],
                "mean_loss_increase",
            ),
            "",
            "## Family Removal: Mean Token-Accuracy Degradation",
            "",
            _nested_metric_table(
                causal["performance_drop_when_family_removed"],
                "mean_token_accuracy_degradation",
            ),
            "",
            "## Family Removal: Mean Exact-Accuracy Delta",
            "",
            _nested_metric_table(
                causal["performance_drop_when_family_removed"],
                "mean_exact_accuracy_delta",
            ),
            "",
            "## Specialization Criteria",
            "",
            *[
                f"- `{name}`: {description}"
                for name, description in causal[
                    "specialization_criteria"
                ].items()
            ],
            "",
            "# Surface-vs-Computation Analysis",
            "",
            surface["statement"],
            "",
            (
                f"Samples: `{surface['samples']}`; routing variable: "
                f"`{surface['routing_variable']}`."
            ),
            (
                f"Operation MI/NMI: "
                f"`{surface['operation_selection_mutual_information_nats']:.6f}` / "
                f"`{surface['operation_selection_normalized_mutual_information']:.6f}`; "
                f"surface MI/NMI: "
                f"`{surface['surface_selection_mutual_information_nats']:.6f}` / "
                f"`{surface['surface_selection_normalized_mutual_information']:.6f}`."
            ),
            "",
            "# Temporal/Lease Analysis",
            "",
            _markdown_table(
                {"overall": report["lease_temporal_analysis"]["overall"]}
            ),
            "",
            "# Sparse Execution",
            "",
            _markdown_table({"overall": report["sparse_execution"]["overall"]}),
            "",
            "# Cycle Analysis",
            "",
            f"Supported: `{cycle.get('supported')}`",
            "",
            f"Reason: {cycle.get('reason', 'cycle telemetry available')}",
            "",
            "# Module Diagnostics",
            "",
            _markdown_table(
                {"availability": report["module_diagnostics"]["availability"]}
            ),
            "",
            _list_table(report["module_diagnostics"]["modules"]),
            "",
            "# Notable Examples",
            "",
        ]
    )
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
        return "unavailable"
    return str(value).replace("|", "\\|")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Rayvan computational capability diagnostics")
    parser.add_argument("--suite", choices=("capability",), default="capability")
    parser.add_argument("--checkpoint", action="append", required=True, help="repeat for comparison mode")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--examples-per-capability", type=int, default=100)
    parser.add_argument("--ablation-examples-per-capability", type=int, default=8)
    parser.add_argument("--diagnostic-smoke", action="store_true")
    parser.add_argument(
        "--deep-diagnostics",
        action="store_true",
        help=(
            "increase sampled ablations/cycle comparisons; default diagnostics "
            "retain streaming aggregates and small samples"
        ),
    )
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
        deep_diagnostics=args.deep_diagnostics,
    )
    if len(args.checkpoint) == 1:
        evaluate_checkpoint(args.checkpoint[0], args.output_dir, config)
    else:
        compare_checkpoints(args.checkpoint, args.output_dir, config)
    print(f"wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
