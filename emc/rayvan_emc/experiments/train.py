from __future__ import annotations

import argparse
import math

import torch

from ..checkpoint import load_model_checkpoint
from ..generation import generate_text
from ..model import EMCModel
from ..training import (
    TrainingConfig,
    TrainingMetrics,
    collect_routing_report,
    evaluate_emc_cycles,
    train_model,
)
from .common import (
    MODULE_POPULATIONS,
    N1_STAGES,
    create_baseline_model,
    create_emc_model,
    load_experiment_corpus,
    print_parameter_summary,
    print_routing_report,
    token_budget_for_preset,
)

STORY_PROMPTS = (
    "Once upon a time",
    "There was a little girl named",
    "One day, a boy went",
    "The dog was very",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one local language model")
    parser.add_argument("--model", choices=("emc", "baseline"), default="emc")
    parser.add_argument("--dataset", choices=("tiny", "tinystories"), default="tinystories")
    parser.add_argument("--preset", choices=("quick", "research"), default="quick")
    parser.add_argument("--n1-stage", choices=N1_STAGES, default="n1")
    parser.add_argument(
        "--module-population",
        choices=MODULE_POPULATIONS,
        default="mixed",
    )
    parser.add_argument("--budget", choices=("quick", "medium", "research"), default="quick")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--train-tokens", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--balance-coefficient", type=float, default=0.01)
    parser.add_argument("--balance-entropy-floor", type=float, default=0.75)
    parser.add_argument("--train-stories", type=int)
    parser.add_argument("--validation-stories", type=int, default=1_000)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--evaluation-interval", type=int, default=100)
    parser.add_argument("--evaluation-batches", type=int, default=8)
    parser.add_argument("--sample-every-evaluations", type=int, default=2)
    parser.add_argument("--sample-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.steps is not None and args.train_tokens is not None:
        parser.error("--steps and --train-tokens are mutually exclusive")
    return args


def main() -> None:
    args = parse_args()
    train_tokens = (
        args.train_tokens
        if args.train_tokens is not None
        else (None if args.steps is not None else token_budget_for_preset(args.budget))
    )
    default_story_counts = {"quick": 10_000, "medium": 50_000, "research": 100_000}
    train_stories = args.train_stories or default_story_counts[args.budget]
    corpus = load_experiment_corpus(
        args.dataset,
        train_stories=train_stories,
        validation_stories=args.validation_stories,
        tokenizer_identifier=args.tokenizer,
    )
    model_arguments = {
        "maximum_sequence_length": args.sequence_length,
        "seed": args.seed,
        "tie_embeddings": args.dataset == "tinystories",
    }
    if args.model == "emc":
        model = create_emc_model(
            corpus.tokenizer.vocab_size,
            args.preset,
            n1_stage=args.n1_stage,
            module_population=args.module_population,
            **model_arguments,
        )
    else:
        model = create_baseline_model(
            corpus.tokenizer.vocab_size,
            args.preset,
            **model_arguments,
        )
    config = TrainingConfig(
        steps=args.steps,
        train_tokens=train_tokens,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        evaluation_interval=args.evaluation_interval,
        evaluation_batches=args.evaluation_batches,
        router_balance_coefficient=args.balance_coefficient,
        router_balance_entropy_floor=args.balance_entropy_floor,
        precision=args.precision,
        checkpoint_directory=args.checkpoint_dir,
        checkpoint_prefix=args.model,
        resume_from=args.resume,
        seed=args.seed,
        device=args.device,
    )

    print(
        f"stage={args.n1_stage} population={args.module_population} | "
        f"tokenizer={corpus.tokenizer.identifier} "
        f"vocab={corpus.tokenizer.vocab_size:,} | "
        f"context={args.sequence_length} | planned_steps={config.planned_steps:,} | "
        f"planned_tokens="
        f"{config.planned_steps * args.batch_size * args.sequence_length * args.gradient_accumulation:,}"
    )
    print_parameter_summary(args.model.upper(), model)
    evaluation_count = 0

    def periodic_samples(
        _step: int, sampled_model: torch.nn.Module, _metrics: TrainingMetrics
    ) -> None:
        nonlocal evaluation_count
        evaluation_count += 1
        if (
            args.sample_every_evaluations <= 0
            or evaluation_count % args.sample_every_evaluations != 0
        ):
            return
        print("samples:")
        for prompt in STORY_PROMPTS:
            print(
                generate_text(
                    sampled_model,
                    corpus.tokenizer,
                    prompt,
                    max_new_tokens=args.sample_tokens,
                    temperature=args.temperature,
                    greedy=args.greedy,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
            )

    result = train_model(
        model,
        corpus,
        config,
        evaluation_callback=periodic_samples,
    )
    print(
        f"final: tokens={result.tokens_processed:,} | "
        f"loss={result.final_validation_loss:.4f} | "
        f"ppl={result.final_validation_perplexity:.2f} | "
        f"{result.tokens_per_second:,.0f} tok/s | {result.elapsed_seconds:.1f}s"
    )
    if args.device.startswith("cuda"):
        print(
            f"GPU memory: current={result.gpu_memory_used_bytes / 2**30:.2f} GiB | "
            f"peak={result.gpu_peak_memory_bytes / 2**30:.2f} GiB"
        )
    if args.model == "emc":
        print(
            f"balance: raw={result.average_router_balance_loss:.6f} | "
            f"weighted={result.average_weighted_balance_contribution:.6f}"
        )
    if result.routing is not None:
        print("Training-run routing:")
        print_routing_report(result.routing)

    if result.best_checkpoint is None:
        print("No checkpoint directory configured; final model remains in memory.")
        best_model = model
        best_tokenizer = corpus.tokenizer
        best_tokens = result.tokens_processed
        best_loss = result.best_validation_loss
    else:
        loaded = load_model_checkpoint(result.best_checkpoint, device=args.device)
        best_model = loaded.model
        best_tokenizer = loaded.tokenizer
        best_tokens = loaded.progress.tokens_processed
        best_loss = loaded.progress.best_validation_loss
        print(
            f"loaded best checkpoint={result.best_checkpoint} | "
            f"tokens={best_tokens:,} | validation={best_loss:.4f} | "
            f"ppl={math.exp(min(best_loss, 20.0)):.2f}"
        )
        print("Best-checkpoint quality by fixed cycle count:")
        for cycle_metrics in evaluate_emc_cycles(best_model, corpus, config):
            print(
                f"  cycle {cycle_metrics.cycle}: "
                f"loss={cycle_metrics.validation_loss:.4f} | "
                f"ppl={cycle_metrics.perplexity:.2f} | "
                f"{cycle_metrics.seconds_per_batch:.4f}s/batch"
            )

    if isinstance(best_model, EMCModel):
        print("Best-checkpoint validation routing:")
        best_routing = collect_routing_report(
            best_model,
            corpus,
            config,
            batches=args.evaluation_batches,
        )
        print_routing_report(
            best_routing,
            include_training_signals=False,
        )

    print("\nBest-checkpoint samples:")
    for prompt in STORY_PROMPTS:
        print(f"\nPrompt: {prompt}")
        print(
            generate_text(
                best_model,
                best_tokenizer,
                prompt,
                max_new_tokens=args.sample_tokens,
                temperature=args.temperature,
                greedy=args.greedy,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        )


if __name__ == "__main__":
    main()
