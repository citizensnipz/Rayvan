from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .capability_tasks import (
    DIAGNOSTIC_CHECKPOINTS,
    CapabilityCorpus,
    CapabilitySuiteConfig,
    CapabilityTaskSuite,
    token_budget_for_preset,
)
from .diagnostics import parameter_counts
from .evaluate import (
    DiagnosticEvaluationConfig,
    evaluate_suite,
    git_commit,
    write_report,
)
from .experiments.common import MODULE_POPULATIONS, N1_STAGES, create_emc_model
from .training import TrainingConfig, TrainingMetrics, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the cheap mixed computational-capability curriculum"
    )
    parser.add_argument(
        "--budget", choices=("smoke", "quick", "standard", "extended"), default="smoke"
    )
    parser.add_argument(
        "--train-tokens",
        type=int,
        help="required for extended; overrides a finite preset when provided",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--n1-stage", choices=N1_STAGES, default="n1_chunked")
    parser.add_argument("--module-population", choices=MODULE_POPULATIONS, default="mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default="checkpoints/capability")
    parser.add_argument("--resume")
    parser.add_argument("--evaluation-interval", type=int)
    parser.add_argument("--evaluation-batches", type=int, default=4)
    parser.add_argument(
        "--diagnostic-checkpoints",
        help="comma-separated token thresholds; defaults to preset thresholds",
    )
    parser.add_argument("--checkpoint-diagnostic-examples", type=int, default=20)
    parser.add_argument("--skip-checkpoint-diagnostics", action="store_true")
    parser.add_argument(
        "--collect-module-diagnostics",
        action="store_true",
        help=(
            "sample module gradient/update norms at checkpoint intervals; "
            "disabled by default"
        ),
    )
    parser.add_argument(
        "--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.budget == "extended" and args.train_tokens is None:
        parser.error("extended requires an explicit --train-tokens value")
    return args


def main() -> None:
    args = parse_args()
    train_tokens = (
        args.train_tokens
        if args.train_tokens is not None
        else token_budget_for_preset(args.budget)
    )
    suite = CapabilityTaskSuite(CapabilitySuiteConfig(seed=args.seed))
    corpus = CapabilityCorpus(suite)
    model = create_emc_model(
        corpus.tokenizer.vocab_size,
        "quick",
        maximum_sequence_length=max(1_024, args.sequence_length),
        seed=args.seed,
        n1_stage=args.n1_stage,
        module_population=args.module_population,
    )
    tokens_per_step = (
        args.batch_size * args.sequence_length * args.gradient_accumulation
    )
    checkpoint_thresholds = _checkpoint_thresholds(args)
    interval = args.evaluation_interval or max(
        1,
        min(
            (threshold + tokens_per_step - 1) // tokens_per_step
            for threshold in checkpoint_thresholds or (train_tokens,)
        ),
    )
    training_config = TrainingConfig(
        steps=None,
        train_tokens=train_tokens,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        evaluation_interval=interval,
        evaluation_batches=args.evaluation_batches,
        gradient_accumulation_steps=args.gradient_accumulation,
        checkpoint_directory=args.checkpoint_dir,
        checkpoint_prefix=f"capability-{args.budget}",
        resume_from=args.resume,
        seed=args.seed,
        precision=args.precision,
        device=args.device,
        collect_module_diagnostics=args.collect_module_diagnostics,
    )
    pending = list(checkpoint_thresholds)
    diagnostics_root = Path(args.checkpoint_dir) / "diagnostics"

    def checkpoint_diagnostics(
        step: int, current_model: torch.nn.Module, metrics: TrainingMetrics
    ) -> None:
        del step
        if args.skip_checkpoint_diagnostics:
            return
        reached = [threshold for threshold in pending if metrics.tokens_processed >= threshold]
        for threshold in reached:
            report = evaluate_suite(
                current_model,
                corpus.tokenizer,
                DiagnosticEvaluationConfig(
                    seed=args.seed,
                    examples_per_capability=args.checkpoint_diagnostic_examples,
                    ablation_examples_per_capability=0,
                    run_module_ablations=False,
                    run_family_ablations=False,
                    run_zero_proposal=False,
                    run_forced_alternatives=False,
                    device=args.device,
                    precision=args.precision,
                    smoke=True,
                ),
                checkpoint=str(
                    Path(args.checkpoint_dir) / f"capability-{args.budget}-latest.pt"
                ),
                checkpoint_training_config=asdict(training_config),
            )
            write_report(report, diagnostics_root / f"tokens-{threshold}")
            pending.remove(threshold)

    result = train_model(
        model,
        corpus,
        training_config,
        evaluation_callback=checkpoint_diagnostics,
    )
    counts = parameter_counts(model)
    summary = {
        "code_version": git_commit(),
        "budget": args.budget,
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
        "runtime": {
            "wall_time_seconds": result.elapsed_seconds,
            "tokens_per_second": result.tokens_per_second,
            "training_tokens": result.tokens_processed,
            "parameter_count": counts.total,
            "approximate_active_parameters_per_cycle": (
                counts.approximate_active_per_cycle
            ),
            "approximate_parameter_uses_per_forward": (
                counts.approximate_parameter_uses_per_forward
            ),
            "active_flops": None,
            "peak_vram_bytes": result.gpu_peak_memory_bytes,
        },
        "training_result": asdict(result),
        "suite": {
            "seed": args.seed,
            "mixture_weights": dict(suite.config.mixture_weights),
            "held_out_combinations": list(suite.config.held_out_combinations),
        },
        "diagnostic_checkpoints": list(checkpoint_thresholds),
    }
    destination = Path(args.checkpoint_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "training-summary.json").write_text(
        json.dumps(summary, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    print(f"latest checkpoint: {result.latest_checkpoint}")


def _checkpoint_thresholds(args: argparse.Namespace) -> tuple[int, ...]:
    if args.diagnostic_checkpoints:
        values = tuple(
            sorted(
                {
                    int(value.strip())
                    for value in args.diagnostic_checkpoints.split(",")
                    if value.strip()
                }
            )
        )
        if any(value <= 0 for value in values):
            raise ValueError("diagnostic checkpoints must be positive token counts")
        return values
    if args.budget == "extended":
        return ()
    return DIAGNOSTIC_CHECKPOINTS[args.budget]


if __name__ == "__main__":
    main()
