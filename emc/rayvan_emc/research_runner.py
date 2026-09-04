from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .architecture import architecture_accounting, build_architectures
from .baseline import TransformerConfig, TransformerLanguageModel
from .capability_tasks import CapabilityCorpus, CapabilitySuiteConfig, CapabilityTaskSuite
from .capability_tasks import diagnostic_tokenizer
from .chunked import ChunkedEMCModel
from .diagnostics import parameter_counts
from .evaluate import DiagnosticEvaluationConfig, evaluate_suite, write_report
from .experiments.common import create_emc_model, create_n2_model, load_experiment_corpus
from .model import EMCConfig, EMCModel, SequentialEMCModel
from .projections import projection_payload
from .research_config import ExperimentConfig
from .serial import HeterogeneousSerialModel
from .training import TrainingCancelledError, TrainingConfig, TrainingMetrics, train_model


EVENT_SCHEMA_VERSION = 2


class EventWriter:
    def __init__(self, run_directory: Path) -> None:
        self.run_directory = run_directory
        self.metrics_path = run_directory / "metrics.jsonl"
        self.routing_path = run_directory / "routing.jsonl"
        self.log_path = run_directory / "logs.txt"

    def emit(self, event_type: str, **payload: Any) -> dict[str, Any]:
        event = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **_json_safe(payload),
        }
        line = json.dumps(event, separators=(",", ":"), allow_nan=False)
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
        if event_type in {"routing_metrics", "expert_metrics"}:
            with self.routing_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        print(line, flush=True)
        return event

    def log(self, text: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(text.rstrip() + "\n")
        print(text, file=sys.stderr, flush=True)


def estimate_experiment(config: ExperimentConfig) -> dict[str, Any]:
    vocab_size = diagnostic_tokenizer().vocab_size if config.suite == "capability_10" else 50_257
    model = _build_model(config, vocab_size)
    accounting = architecture_accounting(model, sequence_length=config.model.context_length)
    counts = parameter_counts(model)
    return {
        **asdict(accounting),
        "expert_count": len(config.expert_families),
        "expert_families": list(config.expert_families),
        "approximate_active_parameters_per_cycle": counts.approximate_active_per_cycle,
    }


def run_experiment(
    config: ExperimentConfig,
    *,
    runs_directory: str | Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    resolved_id = run_id or _run_id()
    if not all(character.isalnum() or character in "-_" for character in resolved_id):
        raise ValueError("run_id contains unsafe characters")
    run_directory = Path(runs_directory).resolve() / resolved_id
    if run_directory.exists():
        unexpected = [path for path in run_directory.iterdir() if path.name != "cancel.requested"]
        if unexpected:
            raise FileExistsError(f"run directory already contains artifacts: {run_directory}")
    run_directory.mkdir(parents=True, exist_ok=True)
    writer = EventWriter(run_directory)
    started_at = datetime.now(timezone.utc).isoformat()
    metadata = _metadata(resolved_id, started_at)
    _write_json(run_directory / "config.json", config.to_dict())
    _write_json(run_directory / "metadata.json", metadata)
    _write_json(run_directory / "status.json", {"status": "initializing", "updated_at": started_at})
    latest: dict[str, Any] = {}
    measured: list[tuple[float, float]] = []
    measured_perplexity: list[tuple[float, float]] = []
    prediction_records: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    throughput_history: list[float] = []
    routing_history: list[list[float]] = []
    writer.emit("run_started", run_id=resolved_id, config=config.to_dict(), metadata=metadata)

    try:
        writer.emit("state_changed", run_id=resolved_id, state="initializing")
        corpus = _build_corpus(config)
        model = _build_model(config, corpus.tokenizer.vocab_size)
        accounting = architecture_accounting(model, sequence_length=config.model.context_length)
        counts = parameter_counts(model)
        model_info = {
            **asdict(accounting),
            "expert_names": list(getattr(model, "expert_names", ())),
            "module_families": list(getattr(model, "module_families", ())),
            "approximate_active_parameters_per_cycle": counts.approximate_active_per_cycle,
        }
        _write_json(run_directory / "model.json", model_info)
        writer.emit("config", run_id=resolved_id, model=model_info)
        training_config = TrainingConfig(
            steps=None,
            train_tokens=config.training.tokens,
            batch_size=config.training.batch_size,
            sequence_length=config.model.context_length,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            evaluation_interval=config.training.evaluation_interval,
            evaluation_batches=config.training.evaluation_batches,
            gradient_accumulation_steps=config.training.gradient_accumulation,
            router_balance_coefficient=config.routing.balance_coefficient,
            router_balance_entropy_floor=config.routing.balance_entropy_floor,
            precision=config.training.precision,
            checkpoint_directory=str(run_directory / "checkpoints"),
            checkpoint_prefix="model",
            seed=config.training.seed,
            device=config.training.device,
            collect_module_diagnostics=True,
            diagnostic_milestones=_milestones(config.training.tokens),
            collect_milestone_telemetry=isinstance(model, (EMCModel, ChunkedEMCModel)),
            retain_milestone_checkpoints=False,
        )
        _write_json(run_directory / "training-config.json", asdict(training_config))
        _set_status(run_directory, "running")
        writer.emit("state_changed", run_id=resolved_id, state="running")

        def cancelled() -> bool:
            return (run_directory / "cancel.requested").exists()

        def progress(step: int, current_model: nn.Module, values: dict[str, Any]) -> None:
            del current_model
            latest.update(values)
            if step % config.training.telemetry_interval:
                return
            system = _system_metrics(config.training.device)
            writer.emit("training_step", run_id=resolved_id, **values, system=system)
            throughput = values.get("tokens_per_second")
            if isinstance(throughput, (int, float)):
                if len(throughput_history) >= 5:
                    baseline = sorted(throughput_history[-5:])[2]
                    if throughput < baseline * 0.8:
                        writer.emit("diagnostic_result", run_id=resolved_id, warnings=[{"code": "throughput_regression", "severity": "warning", "message": "Step throughput is more than 20% below the recent median."}])
                throughput_history.append(float(throughput))
            gradient_norm = values.get("gradient_norm")
            if isinstance(gradient_norm, (int, float)) and gradient_norm > 10:
                writer.emit("diagnostic_result", run_id=resolved_id, warnings=[{"code": "exploding_gradients", "severity": "critical", "message": "Pre-clipping gradient norm exceeded 10."}])
            routing = values.get("routing")
            if routing:
                writer.emit("routing_metrics", run_id=resolved_id, step=step, tokens_processed=values["tokens_processed"], **routing)
                warnings = _routing_warnings(routing)
                if warnings:
                    writer.emit("diagnostic_result", run_id=resolved_id, warnings=warnings)
                probabilities = [float(value) for value in routing.get("mean_probabilities", [])]
                routing_history.append(probabilities)
                if len(routing_history) >= 5 and probabilities:
                    recent = routing_history[-5:]
                    movement = max(max(row[index] for row in recent) - min(row[index] for row in recent) for index in range(len(probabilities)))
                    if movement < 1e-5:
                        writer.emit("diagnostic_result", run_id=resolved_id, warnings=[{"code": "static_routing", "severity": "warning", "message": "Router probabilities changed by less than 1e-5 across five telemetry samples."}])
            writer.emit("system_metrics", run_id=resolved_id, step=step, tokens_processed=values["tokens_processed"], **system)

        def evaluation(step: int, current_model: nn.Module, metrics: TrainingMetrics) -> None:
            del current_model
            row = asdict(metrics)
            latest.update(row)
            writer.emit("validation", run_id=resolved_id, **row)
            measured.append((float(metrics.tokens_processed), float(metrics.validation_loss)))
            measured_perplexity.append((float(metrics.tokens_processed), float(metrics.validation_perplexity)))
            projection = _projection_update(measured, config.training.tokens, config.projection_targets)
            if projection["fits"]:
                projection_record = {
                    "made_at_tokens": metrics.tokens_processed,
                    **projection,
                    "perplexity_fits": projection_payload(
                        measured_perplexity,
                        (
                            target
                            for target in sorted({*config.projection_targets, config.training.tokens})
                            if target > metrics.tokens_processed
                        ),
                        metric="validation_perplexity",
                    )["fits"],
                    "runtime_estimates": _runtime_estimates(metrics.tokens_per_second, metrics.tokens_processed, config.projection_targets),
                }
                prediction_records.append(projection_record)
                writer.emit("projection_update", run_id=resolved_id, **projection_record)
            elif len(measured) < 4:
                writer.emit("diagnostic_result", run_id=resolved_id, warnings=[{"code": "insufficient_projection_points", "severity": "info", "message": f"Projection needs at least four validation checkpoints; {len(measured)} available."}])
            if len(measured) >= 2 and metrics.validation_loss > measured[-2][1] * 1.05:
                writer.emit("diagnostic_result", run_id=resolved_id, warnings=[{"code": "validation_regression", "severity": "warning", "message": "Validation loss increased by more than 5% from the previous checkpoint."}])
            _calibrate_predictions(prediction_records, calibration, metrics.tokens_processed, metrics.validation_loss)
            _write_json(run_directory / "projections.json", {"predictions": prediction_records, "calibration": calibration})

        result = train_model(
            model,
            corpus,
            training_config,
            print_progress=False,
            evaluation_callback=evaluation,
            progress_callback=progress,
            progress_callback_interval=config.training.telemetry_interval,
            cancellation_callback=cancelled,
        )
        diagnostics: dict[str, Any] | None = None
        if config.suite == "capability_10":
            _set_status(run_directory, "diagnostics")
            writer.emit("state_changed", run_id=resolved_id, state="diagnostics")
            diagnostics = evaluate_suite(
                model,
                corpus.tokenizer,
                DiagnosticEvaluationConfig(
                    seed=config.training.seed,
                    examples_per_capability=config.training.diagnostic_examples_per_capability,
                    ablation_examples_per_capability=0,
                    run_module_ablations=False,
                    run_family_ablations=False,
                    run_zero_proposal=False,
                    run_forced_alternatives=False,
                    device=config.training.device,
                    precision=config.training.precision,
                    smoke=True,
                    max_notable_examples=8,
                ),
                checkpoint=result.latest_checkpoint,
                checkpoint_training_config=asdict(training_config),
                checkpoint_training_diagnostics=result.module_diagnostics,
            )
            write_report(diagnostics, run_directory / "diagnostics")
            writer.emit(
                "diagnostic_result",
                run_id=resolved_id,
                tasks=diagnostics.get("capability_results", {}),
                overall=diagnostics.get("overall_metrics", {}),
                warnings=diagnostics.get("diagnostic_integrity_warnings", []),
            )

        summary = {
            "run_id": resolved_id,
            "status": "completed",
            "name": config.name or resolved_id,
            "suite": config.suite,
            "architecture": config.architecture,
            "experts": dict(config.experts),
            "tags": list(config.tags),
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "model": model_info,
            "training_result": _json_safe(asdict(result)),
            "headline": _headline(asdict(result), diagnostics),
            "warnings": _final_warnings(result, latest),
            "git": metadata["git"],
        }
        _write_json(run_directory / "summary.json", summary)
        _set_status(run_directory, "completed")
        writer.emit("run_completed", run_id=resolved_id, summary=summary)
        return summary
    except TrainingCancelledError as error:
        summary = _terminal_summary(config, resolved_id, started_at, "cancelled", latest, str(error), metadata)
        _write_json(run_directory / "summary.json", summary)
        _set_status(run_directory, "cancelled")
        writer.emit("run_cancelled", run_id=resolved_id, summary=summary)
        return summary
    except BaseException as error:
        detail = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        writer.log(detail)
        summary = _terminal_summary(config, resolved_id, started_at, "failed", latest, str(error), metadata)
        _write_json(run_directory / "summary.json", summary)
        _set_status(run_directory, "failed")
        writer.emit("run_failed", run_id=resolved_id, error=str(error), summary=summary)
        raise


def _build_corpus(config: ExperimentConfig):
    if config.suite == "capability_10":
        return CapabilityCorpus(CapabilityTaskSuite(CapabilitySuiteConfig(seed=config.training.seed)))
    return load_experiment_corpus(
        "tinystories",
        train_stories=max(100, min(100_000, config.training.tokens // max(config.model.context_length, 1))),
        validation_stories=1_000,
    )


def _build_model(config: ExperimentConfig, vocab_size: int) -> nn.Module:
    families = config.expert_families
    parallel_top_k = config.routing.top_k or 2
    if config.architecture.startswith("n2_"):
        return create_n2_model(
            vocab_size,
            "quick" if config.model.preset == "custom" else config.model.preset,
            maximum_sequence_length=config.model.context_length,
            seed=config.training.seed,
            population=config.architecture.removeprefix("n2_"),
            top_k=parallel_top_k,
            tie_embeddings=config.model.tie_embeddings,
            n1_depth=config.model.n1_depth,
        )
    if config.model.fairness_mode != "custom" and config.architecture in {"heterogeneous_serial", "homogeneous_serial", "old_emc"}:
        build = build_architectures(
            (config.architecture,),
            vocab_size=vocab_size,
            model_preset="quick" if config.model.preset == "custom" else config.model.preset,
            maximum_sequence_length=config.model.context_length,
            seed=config.training.seed,
            fairness_mode=config.model.fairness_mode,
            heterogeneous_order=families,
            tie_embeddings=config.model.tie_embeddings,
            top_k=parallel_top_k,
            n1_depth=config.model.n1_depth,
        )
        return build.models[config.architecture]
    if config.architecture == "homogeneous_serial":
        return TransformerLanguageModel(
            TransformerConfig(
                vocab_size=vocab_size,
                latent_dim=config.model.latent_dim,
                num_layers=max(1, len(families)),
                attention_heads=config.model.attention_heads,
                feed_forward_dim=config.model.module_hidden_dim,
                max_sequence_length=config.model.context_length,
                tie_embeddings=config.model.tie_embeddings,
            )
        )
    stage = (
        "n1"
        if config.architecture == "old_emc"
        else "n1_chunked"
        if config.architecture == "legacy_parallel_emc"
        else "n1_sequential"
    )
    template = create_emc_model(
        vocab_size,
        "quick" if config.model.preset == "custom" else config.model.preset,
        maximum_sequence_length=config.model.context_length,
        seed=config.training.seed,
        tie_embeddings=config.model.tie_embeddings,
        n1_stage=("n1_chunked" if stage == "n1_sequential" else stage),
        module_population="mixed",
    )
    values = asdict(template.config)
    values.update(
        {
            "latent_dim": config.model.latent_dim,
            "num_modules": len(families),
            "modules_per_cycle": (1 if stage == "n1_sequential" else parallel_top_k),
            "num_cycles": (
                config.routing.trajectory_steps
                if stage == "n1_sequential"
                else 1 if stage == "n1_chunked"
                else config.routing.cycles
            ),
            "trajectory_steps": (
                config.routing.trajectory_steps
                if stage == "n1_sequential" else None
            ),
            "vocab_size": vocab_size,
            "max_sequence_length": config.model.context_length,
            "module_hidden_dim": config.model.module_hidden_dim,
            "attention_heads": config.model.attention_heads,
            "tie_embeddings": config.model.tie_embeddings,
            "module_families": families,
            "router_type": config.routing.router_type,
            "integrator_type": config.routing.integrator_type,
            "integrator_heads": config.model.integrator_heads,
            "architecture_stage": (
                "n1_chunked" if stage == "n1_chunked"
                else "n1_sequential" if stage == "n1_sequential"
                else "token"
            ),
            "chunk_size": config.model.chunk_size,
            "shared_state_slots": config.model.shared_state_slots,
            "request_pool_size": len(families),
            "active_top_k": (None if stage == "n1_sequential" else parallel_top_k),
            "switch_cost": config.routing.switch_cost,
            "persistence_bonus": config.routing.persistence_bonus,
            "minimum_lease_chunks": config.routing.minimum_lease_chunks,
            "loss_free_balance_enabled": config.routing.loss_free_balance_enabled,
            "balance_bias_lr": config.routing.balance_bias_lr,
            "balance_bias_limit": config.routing.balance_bias_limit,
            "balance_warmup_chunks": config.routing.balance_warmup_chunks,
            "refractory_enabled": config.routing.refractory_enabled,
            "refractory_strength": config.routing.refractory_strength,
            "refractory_decay": config.routing.refractory_decay,
        }
    )
    if config.model.preset == "custom":
        values.update(
            {
                "state_space_dim": config.model.latent_dim,
                "recurrent_dim": config.model.latent_dim,
                "router_descriptor_dim": config.model.latent_dim,
                "shared_core_hidden_dim": config.model.latent_dim,
                "delta_internal_dim": config.model.latent_dim,
                "delta_heads": config.model.attention_heads,
                "delta_ffn_dim": config.model.module_hidden_dim,
            }
        )
    torch.manual_seed(config.training.seed)
    emc_config = EMCConfig(**values)
    if config.architecture == "heterogeneous_serial":
        return HeterogeneousSerialModel(emc_config)
    if stage == "n1_chunked":
        return ChunkedEMCModel(emc_config)
    if stage == "n1_sequential":
        return SequentialEMCModel(emc_config)
    return EMCModel(emc_config)


def _projection_update(measured: list[tuple[float, float]], budget: int, configured_targets: tuple[int, ...]) -> dict[str, Any]:
    end = measured[-1][0]
    targets = sorted({*configured_targets, budget})
    return projection_payload(measured, (target for target in targets if target > end), metric="validation_loss")


def _runtime_estimates(tokens_per_second: float, measured_tokens: int, targets: tuple[int, ...]) -> list[dict[str, Any]]:
    return [
        {
            "target_tokens": target,
            "estimated_total_seconds": target / max(tokens_per_second, 1e-12),
            "estimated_remaining_seconds": max(0, target - measured_tokens) / max(tokens_per_second, 1e-12),
            "model": "recent_measured_average_throughput",
            "confidence": "exploratory",
        }
        for target in targets
    ]


def _calibrate_predictions(records: list[dict[str, Any]], calibration: list[dict[str, Any]], tokens: int, actual: float) -> None:
    existing = {(row["made_at_tokens"], row["prediction_target"]) for row in calibration}
    for record in records:
        for fit in record.get("fits", []):
            key = (record["made_at_tokens"], fit["prediction_target"])
            if key in existing or fit["prediction_target"] > tokens:
                continue
            calibration.append(
                {
                    "made_at_tokens": record["made_at_tokens"],
                    "prediction_target": fit["prediction_target"],
                    "predicted_value": fit["predicted_value"],
                    "actual_value": actual,
                    "absolute_error": abs(fit["predicted_value"] - actual),
                    "relative_error": abs(fit["predicted_value"] - actual) / max(abs(actual), 1e-12),
                }
            )


def _system_metrics(device_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {"gpu_utilization_percent": None, "vram_used_bytes": None, "vram_total_bytes": None}
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        return result
    device = torch.device(device_name)
    result["vram_used_bytes"] = int(torch.cuda.memory_allocated(device))
    try:
        _, total = torch.cuda.mem_get_info(device)
        result["vram_total_bytes"] = int(total)
    except (RuntimeError, TypeError):
        pass
    try:
        result["gpu_utilization_percent"] = float(torch.cuda.utilization(device))
    except (AttributeError, ImportError, ModuleNotFoundError, RuntimeError, TypeError):
        pass
    return result


def _routing_warnings(routing: Mapping[str, Any]) -> list[dict[str, str]]:
    utilization = [float(value) for value in routing.get("utilization", [])]
    warnings = []
    dead = [str(index) for index, value in enumerate(utilization) if value == 0]
    rare = [str(index) for index, value in enumerate(utilization) if 0 < value < 0.01]
    if dead:
        warnings.append({"code": "expert_starvation", "severity": "warning", "message": f"Experts never selected in this sample: {', '.join(dead)}"})
    if rare:
        warnings.append({"code": "rare_experts", "severity": "warning", "message": f"Experts below the backend 1% near-dead threshold: {', '.join(rare)}"})
    if utilization and max(utilization) >= 0.95:
        warnings.append({"code": "collapsed_routing", "severity": "critical", "message": "One expert received at least 95% of sampled routes."})
    return warnings


def _final_warnings(result: Any, latest: Mapping[str, Any]) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    if not all(torch.isfinite(torch.tensor(value)) for value in (result.final_training_loss, result.final_validation_loss)):
        warnings.append({"code": "non_finite", "severity": "critical", "message": "Run ended with a non-finite loss."})
    routing = latest.get("routing")
    if isinstance(routing, Mapping):
        warnings.extend(_routing_warnings(routing))
    return warnings


def _headline(result: Mapping[str, Any], diagnostics: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "validation_loss": result.get("final_validation_loss"),
        "perplexity": result.get("final_validation_perplexity"),
        "tokens_per_second": result.get("tokens_per_second"),
        "runtime_seconds": result.get("elapsed_seconds"),
        "tokens_processed": result.get("tokens_processed"),
        "peak_vram_bytes": result.get("gpu_peak_memory_bytes"),
        "task_overall": diagnostics.get("overall_metrics") if diagnostics else None,
    }


def _terminal_summary(config: ExperimentConfig, run_id: str, started_at: str, status: str, latest: Mapping[str, Any], error: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": status,
        "name": config.name or run_id,
        "suite": config.suite,
        "architecture": config.architecture,
        "experts": dict(config.experts),
        "tags": list(config.tags),
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "headline": dict(latest),
        "error": error,
        "git": metadata["git"],
    }


def _metadata(run_id: str, started_at: str) -> dict[str, Any]:
    commit, dirty = None, None
    try:
        root = Path(__file__).resolve().parents[2]
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=3, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=3, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    cuda_device = None
    if torch.cuda.is_available():
        cuda_device = {"index": torch.cuda.current_device(), "name": torch.cuda.get_device_name(torch.cuda.current_device()), "cuda": torch.version.cuda}
    return {
        "run_id": run_id,
        "started_at": started_at,
        "software": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "event_schema": EVENT_SCHEMA_VERSION},
        "hardware": {"processor": platform.processor(), "machine": platform.machine(), "cuda_device": cuda_device},
        "git": {"commit": commit, "dirty": dirty},
    }


def _milestones(budget: int) -> tuple[int, ...]:
    return tuple(value for value in (50_000, 100_000, 250_000, 500_000, 1_000_000, 10_000_000) if value <= budget)


def _set_status(directory: Path, status: str) -> None:
    _write_json(directory / "status.json", {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()})


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not (float("-inf") < value < float("inf")):
        return None
    return value


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
