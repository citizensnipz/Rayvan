from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .architecture import (
    DEFAULT_HETEROGENEOUS_ORDER,
    architecture_accounting,
    architecture_mismatch_report,
    build_architectures,
    config_manifest,
    deterministic_stream_fingerprint,
)
from .chunked import ChunkedEMCModel
from .model import EMCModel
from .experiments.common import load_experiment_corpus
from .training import TrainingConfig, train_model


TOKEN_PRESETS = {"smoke": 25_000, "standard": 1_000_000, "follow-up": 5_000_000}
DEFAULT_MILESTONES = (100_000, 250_000, 500_000, 750_000, 1_000_000)
LARGE_WORKLOAD_TOKENS = 100_000
COMPARISON_MODELS = (
    "homogeneous_serial",
    "heterogeneous_serial",
    "emc",
    "old_emc",
    "n2_mixed",
    "n2_gpt4",
    "n2_ssm4",
    "n2_recurrent4",
    "n2_delta4",
)
DEFAULT_EXPERIMENT_MATRIX = (
    "homogeneous_serial",
    "old_emc",
    "n2_mixed",
    "n2_gpt4",
    "n2_ssm4",
    "n2_recurrent4",
    "n2_delta4",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reproducible three-way Rayvan architecture comparison"
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--models",
        default="homogeneous_serial,heterogeneous_serial,emc",
        help="comma-separated architecture names",
    )
    model_group.add_argument(
        "--model",
        choices=COMPARISON_MODELS,
    )
    parser.add_argument(
        "--budget", choices=tuple(TOKEN_PRESETS), default="smoke"
    )
    parser.add_argument("--train-tokens", type=int)
    parser.add_argument("--dataset", choices=("tiny", "tinystories"), default="tinystories")
    parser.add_argument("--model-preset", choices=("quick", "research"), default="quick")
    parser.add_argument("--fairness-mode", choices=("capacity", "compute"), default="capacity")
    parser.add_argument(
        "--heterogeneous-order",
        default=",".join(DEFAULT_HETEROGENEOUS_ORDER),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--evaluation-batches", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--n1-depth", type=int, default=3)
    parser.add_argument("--diagnostic-checkpoints")
    parser.add_argument("--top-k-schedule")
    parser.add_argument("--milestone-causal-diagnostics", action="store_true")
    parser.add_argument("--causal-sample-size", type=int, default=2)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--train-stories", type=int, default=100_000)
    parser.add_argument("--validation-stories", type=int, default=1_000)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--output-dir", default="architecture-results")
    parser.add_argument("--run-name", default="three-way")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-large-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)
    if args.train_tokens is not None and args.train_tokens <= 0:
        parser.error("--train-tokens must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.n1_depth < 2:
        parser.error("--n1-depth must be at least two")
    return args


def main(argv: Sequence[str] | None = None) -> dict[str, Any] | None:
    args = parse_args(argv)
    names = (args.model,) if args.model else _csv_strings(args.models)
    seeds = tuple(int(value) for value in _csv_strings(args.seeds))
    if not seeds:
        raise ValueError("at least one seed is required")
    train_tokens = args.train_tokens or TOKEN_PRESETS[args.budget]
    milestones = _milestones(args.diagnostic_checkpoints, train_tokens)
    top_k_schedule = _top_k_schedule(
        args.top_k_schedule or f"0:{args.top_k}"
    )
    jobs = len(names) * len(seeds)
    if train_tokens * jobs > LARGE_WORKLOAD_TOKENS and not args.allow_large_run:
        raise SystemExit(
            "Refusing a large comparison workload. Re-run with --allow-large-run "
            f"after reviewing {jobs} jobs × {train_tokens:,} tokens."
        )

    corpus = load_experiment_corpus(
        args.dataset,
        train_stories=args.train_stories,
        validation_stories=args.validation_stories,
        tokenizer_identifier=args.tokenizer,
    )
    first_build = build_architectures(
        names,
        vocab_size=corpus.tokenizer.vocab_size,
        model_preset=args.model_preset,
        maximum_sequence_length=args.sequence_length,
        seed=seeds[0],
        fairness_mode=args.fairness_mode,
        heterogeneous_order=_csv_strings(args.heterogeneous_order),
        tie_embeddings=args.dataset == "tinystories",
        top_k=args.top_k,
        n1_depth=args.n1_depth,
    )
    accounting = {
        name: architecture_accounting(
            model, sequence_length=args.sequence_length
        )
        for name, model in first_build.models.items()
    }
    planning_config = TrainingConfig(
        steps=None,
        train_tokens=train_tokens,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        gradient_accumulation_steps=args.gradient_accumulation,
        evaluation_interval=max(
            1,
            math.ceil(train_tokens / (args.batch_size * args.sequence_length))
            // max(1, len(milestones)),
        ),
    )
    _print_preflight(
        names=names,
        train_tokens=train_tokens,
        steps=planning_config.planned_steps,
        seeds=seeds,
        precision=args.precision,
        milestones=milestones,
        causal=args.milestone_causal_diagnostics,
        jobs=jobs,
        fairness_mode=args.fairness_mode,
        accounting=accounting,
        top_k_schedule=top_k_schedule,
    )
    if args.dry_run:
        return None

    destination = Path(args.output_dir) / args.run_name
    destination.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    for seed in seeds:
        build = build_architectures(
            names,
            vocab_size=corpus.tokenizer.vocab_size,
            model_preset=args.model_preset,
            maximum_sequence_length=args.sequence_length,
            seed=seed,
            fairness_mode=args.fairness_mode,
            heterogeneous_order=_csv_strings(args.heterogeneous_order),
            tie_embeddings=args.dataset == "tinystories",
            top_k=args.top_k,
            n1_depth=args.n1_depth,
        )
        for name in names:
            model = build.models[name]
            model_accounting = architecture_accounting(
                model, sequence_length=args.sequence_length
            )
            checkpoint_directory = destination / name / f"seed-{seed}"
            latest = checkpoint_directory / "latest.pt"
            training_config = TrainingConfig(
                steps=None,
                train_tokens=train_tokens,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                gradient_accumulation_steps=args.gradient_accumulation,
                learning_rate=args.learning_rate,
                evaluation_interval=max(
                    1,
                    planning_config.planned_steps // max(1, len(milestones)),
                ),
                evaluation_batches=args.evaluation_batches,
                precision=args.precision,
                checkpoint_directory=str(checkpoint_directory),
                checkpoint_prefix="",
                resume_from=str(latest) if args.resume and latest.exists() else None,
                seed=seed,
                retain_best_checkpoint=False,
                device=args.device,
                diagnostic_milestones=milestones,
                retain_milestone_checkpoints=True,
                collect_module_diagnostics=isinstance(
                    model, (EMCModel, ChunkedEMCModel)
                ),
                collect_milestone_telemetry=isinstance(
                    model, (EMCModel, ChunkedEMCModel)
                ),
                milestone_causal_diagnostics=(
                    args.milestone_causal_diagnostics
                    and isinstance(model, (EMCModel, ChunkedEMCModel))
                ),
                milestone_causal_sample_size=args.causal_sample_size,
                top_k_schedule=(
                    top_k_schedule
                    if isinstance(model, (EMCModel, ChunkedEMCModel))
                    else None
                ),
            )
            stream_fingerprint = deterministic_stream_fingerprint(
                corpus, training_config
            )
            print(f"\nTraining {name} seed={seed}")
            result = train_model(model, corpus, training_config)
            latency = _inference_latency_seconds(
                model,
                corpus,
                training_config,
            )
            row = {
                "architecture": name,
                "seed": seed,
                "stream_fingerprint": stream_fingerprint,
                "manifest": config_manifest(model),
                "accounting": asdict(model_accounting),
                "fairness_mode": args.fairness_mode,
                "final_training_loss": result.final_training_loss,
                "final_validation_loss": result.final_validation_loss,
                "final_validation_perplexity": result.final_validation_perplexity,
                "final_training_token_accuracy": result.final_training_token_accuracy,
                "final_validation_token_accuracy": result.final_validation_token_accuracy,
                "tokens_processed": result.tokens_processed,
                "tokens_per_second": result.tokens_per_second,
                "wall_clock_seconds": result.elapsed_seconds,
                "peak_vram_bytes": result.gpu_peak_memory_bytes,
                "peak_ram_bytes": _peak_process_memory_bytes(),
                "peak_ram_measurement": (
                    "process-lifetime high-water mark; because jobs run "
                    "sequentially in one process, later jobs can inherit an "
                    "earlier job's peak"
                ),
                "inference_latency_seconds": latency,
                "approximate_training_flops": (
                    model_accounting.approximate_flops_per_token
                    * result.tokens_processed
                ),
                "module_computation_count": (
                    model_accounting.module_computations_per_forward
                ),
                "routing": (
                    asdict(result.routing) if result.routing is not None else None
                ),
                "module_diagnostics": result.module_diagnostics,
                "n1_nodes_actually_selected": (
                    int(getattr(model, "active_top_k"))
                    if getattr(model.config, "architecture_stage", None) == "n2"
                    else None
                ),
                "active_compute_relative_to_all_n1s": (
                    model_accounting.approximate_flops_per_token
                    / model_accounting.theoretical_all_nodes_flops_per_token
                    if model_accounting.theoretical_all_nodes_flops_per_token
                    else None
                ),
                "history": [
                    {
                        **asdict(metrics),
                        "approximate_cumulative_flops": (
                            model_accounting.approximate_flops_per_token
                            * metrics.tokens_processed
                        ),
                    }
                    for metrics in result.history
                ],
                "milestone_checkpoints": list(result.milestone_checkpoints),
                "telemetry_path": result.telemetry_path,
                "routed_emc_path": isinstance(model, (EMCModel, ChunkedEMCModel)),
                "serial_execution_order": list(
                    getattr(model, "last_execution_order", ())
                ),
            }
            run_rows.append(row)
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            (checkpoint_directory / "comparison-result.json").write_text(
                json.dumps(row, indent=2, sort_keys=True), encoding="utf-8"
            )

    fingerprints = {row["stream_fingerprint"] for row in run_rows}
    if len(fingerprints) != len(seeds):
        raise RuntimeError(
            "architectures did not receive one identical token stream per seed"
        )
    report = _comparison_report(
        run_rows,
        accounting={name: asdict(row) for name, row in accounting.items()},
        mismatches=architecture_mismatch_report(accounting),
        args=args,
        train_tokens=train_tokens,
        milestones=milestones,
    )
    _write_report(destination, report)
    print(f"\nWrote comparison to {destination}")
    return report


def _comparison_report(
    rows: list[dict[str, Any]],
    *,
    accounting: dict[str, Any],
    mismatches: dict[str, Any],
    args: argparse.Namespace,
    train_tokens: int,
    milestones: tuple[int, ...],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["architecture"], []).append(row)
    metric_names = (
        "final_training_loss",
        "final_validation_loss",
        "final_validation_perplexity",
        "final_training_token_accuracy",
        "final_validation_token_accuracy",
        "tokens_per_second",
        "wall_clock_seconds",
        "peak_vram_bytes",
        "peak_ram_bytes",
        "inference_latency_seconds",
        "approximate_training_flops",
    )
    aggregate: dict[str, dict[str, Any]] = {}
    for architecture, architecture_rows in grouped.items():
        aggregate[architecture] = {}
        for metric in metric_names:
            values = [float(row[metric]) for row in architecture_rows]
            aggregate[architecture][metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
    return {
        "schema_version": 1,
        "scientific_interpretation": (
            "No architecture is preferred by the harness. Quality and causal "
            "usefulness take precedence over balanced utilization."
        ),
        "configuration": {
            "models": sorted(grouped),
            "train_tokens": train_tokens,
            "seeds": sorted({row["seed"] for row in rows}),
            "fairness_mode": args.fairness_mode,
            "milestones": list(milestones),
            "dataset": args.dataset,
            "tokenizer": args.tokenizer,
            "precision": args.precision,
            "top_k_schedule": args.top_k_schedule,
            "top_k": args.top_k,
            "n1_depth": args.n1_depth,
            "causal_diagnostics": args.milestone_causal_diagnostics,
        },
        "accounting": accounting,
        "fairness_mismatches": mismatches,
        "runs": rows,
        "aggregate": aggregate,
    }


def _write_report(destination: Path, report: dict[str, Any]) -> None:
    (destination / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (destination / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "architecture",
            "seed",
            "final_training_loss",
            "final_validation_loss",
            "final_validation_perplexity",
            "final_validation_token_accuracy",
            "tokens_per_second",
            "wall_clock_seconds",
            "peak_vram_bytes",
            "peak_ram_bytes",
            "inference_latency_seconds",
            "approximate_training_flops",
            "stream_fingerprint",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["runs"]:
            writer.writerow({field: row.get(field) for field in fields})
    lines = [
        "# Three-Way Architecture Comparison",
        "",
        report["scientific_interpretation"],
        "",
        "| architecture | validation loss mean±std | perplexity mean±std | token accuracy mean±std | tokens/s mean±std |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in report["aggregate"].items():
        lines.append(
            f"| {name} | {_mean_std(metrics['final_validation_loss'])} | "
            f"{_mean_std(metrics['final_validation_perplexity'])} | "
            f"{_mean_std(metrics['final_validation_token_accuracy'])} | "
            f"{_mean_std(metrics['tokens_per_second'])} |"
        )
    lines.extend(
        [
            "",
            "## Fairness mismatches",
            "",
            "Exact parameter, active-parameter, and approximate-compute ratios are in `comparison.json` under `fairness_mismatches`. Approximate FLOPs are documented lower bounds, not hidden profiler claims.",
            "",
            "## Developmental diagnostics",
            "",
            "EMC milestone routing/update telemetry is stored under `emc/seed-*/telemetry.json`; optional causal reports are under `milestone-diagnostics/<tokens>/`.",
        ]
    )
    (destination / "comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _print_preflight(**summary: Any) -> None:
    print("Architecture comparison preflight")
    print(f"  architectures: {', '.join(summary['names'])}")
    print(f"  token budget/job: {summary['train_tokens']:,}")
    print(f"  estimated steps/job: {summary['steps']:,}")
    print(f"  seeds: {list(summary['seeds'])}")
    print(f"  estimated training jobs: {summary['jobs']}")
    print(f"  precision: {summary['precision']}")
    print(f"  fairness mode: {summary['fairness_mode']}")
    print(f"  checkpoints: {list(summary['milestones'])}")
    print(f"  deep causal diagnostics: {summary['causal']}")
    print(f"  EMC top-K schedule: {list(summary['top_k_schedule'])}")
    for name, row in summary["accounting"].items():
        print(
            f"  {name}: parameters={row.total_parameters:,}, "
            f"active≈{row.approximate_active_parameters:,}, "
            f"FLOPs/token≈{row.approximate_flops_per_token:,}"
        )


def _inference_latency_seconds(
    model: nn.Module,
    corpus: Any,
    config: TrainingConfig,
) -> float:
    generator = torch.Generator().manual_seed(config.seed + 4)
    inputs, _ = corpus.sample_batch(
        "validation",
        1,
        config.sequence_length,
        generator=generator,
        device=config.device,
    )
    was_training = model.training
    model.eval()
    samples: list[float] = []
    with torch.no_grad():
        for _ in range(3):
            if str(config.device).startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            model(inputs)
            if str(config.device).startswith("cuda"):
                torch.cuda.synchronize()
            samples.append(time.perf_counter() - started)
    model.train(was_training)
    return statistics.median(samples)


def _peak_process_memory_bytes() -> int:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            ):
                return int(counters.PeakWorkingSetSize)
        except (AttributeError, OSError):
            return 0
        return 0
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak * (1 if os.uname().sysname == "Darwin" else 1024))
    except (ImportError, OSError):
        return 0


def _milestones(value: str | None, train_tokens: int) -> tuple[int, ...]:
    requested = (
        tuple(int(item) for item in _csv_strings(value))
        if value
        else DEFAULT_MILESTONES
    )
    within_budget = tuple(item for item in requested if item <= train_tokens)
    return within_budget if within_budget else (train_tokens,)


def _top_k_schedule(value: str) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(start), int(top_k))
        for item in _csv_strings(value)
        for start, top_k in (item.split(":", 1),)
    )


def _csv_strings(value: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in (value or "").split(",") if item.strip())


def _mean_std(metric: dict[str, Any]) -> str:
    return f"{metric['mean']:.6g}±{metric['std']:.3g}"


if __name__ == "__main__":
    main()
