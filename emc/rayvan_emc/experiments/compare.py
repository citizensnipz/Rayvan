from __future__ import annotations

import argparse

import torch

from ..generation import generate_text
from ..training import TrainingConfig, train_model
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare EMC with a plain transformer")
    parser.add_argument("--dataset", choices=("tiny", "tinystories"), default="tiny")
    parser.add_argument("--preset", choices=("quick", "research"), default="quick")
    parser.add_argument("--n1-stage", choices=N1_STAGES, default="n1")
    parser.add_argument(
        "--module-population",
        choices=MODULE_POPULATIONS,
        default="mixed",
    )
    parser.add_argument("--budget", choices=("quick", "medium", "research"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--train-tokens", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--balance-coefficient", type=float, default=0.01)
    parser.add_argument("--balance-entropy-floor", type=float, default=0.75)
    parser.add_argument("--train-stories", type=int, default=10_000)
    parser.add_argument("--validation-stories", type=int, default=1_000)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--evaluation-interval", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    duration_options = sum(
        value is not None
        for value in (args.steps, args.train_tokens, args.budget)
    )
    if duration_options > 1:
        parser.error("use only one of --steps, --train-tokens, or --budget")
    return args


def main() -> None:
    args = parse_args()
    sequence_length = args.sequence_length or (
        256 if args.dataset == "tinystories" else 32
    )
    steps: int | None = args.steps
    train_tokens = args.train_tokens
    if steps is None and train_tokens is None and args.budget is None:
        steps = 100
    if args.budget is not None:
        train_tokens = token_budget_for_preset(args.budget)
        steps = None
    elif train_tokens is not None:
        steps = None

    corpus = load_experiment_corpus(
        args.dataset,
        train_stories=args.train_stories,
        validation_stories=args.validation_stories,
        tokenizer_identifier=args.tokenizer,
    )
    tie_embeddings = args.dataset == "tinystories"
    emc = create_emc_model(
        corpus.tokenizer.vocab_size,
        args.preset,
        maximum_sequence_length=sequence_length,
        seed=args.seed,
        tie_embeddings=tie_embeddings,
        n1_stage=args.n1_stage,
        module_population=args.module_population,
    )
    baseline = create_baseline_model(
        corpus.tokenizer.vocab_size,
        args.preset,
        maximum_sequence_length=sequence_length,
        seed=args.seed,
        tie_embeddings=tie_embeddings,
    )
    planned_steps = TrainingConfig(
        steps=steps,
        train_tokens=train_tokens,
        batch_size=args.batch_size,
        sequence_length=sequence_length,
        gradient_accumulation_steps=args.gradient_accumulation,
    ).planned_steps
    config = TrainingConfig(
        steps=steps,
        train_tokens=train_tokens,
        batch_size=args.batch_size,
        sequence_length=sequence_length,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        evaluation_interval=args.evaluation_interval
        or max(1, planned_steps // 5),
        evaluation_batches=4,
        router_balance_coefficient=args.balance_coefficient,
        router_balance_entropy_floor=args.balance_entropy_floor,
        precision=args.precision,
        seed=args.seed,
        device=args.device,
    )

    print(
        f"stage={args.n1_stage} population={args.module_population} | "
        f"tokenizer={corpus.tokenizer.identifier} "
        f"vocab={corpus.tokenizer.vocab_size:,} | "
        f"context={sequence_length} | steps={config.planned_steps:,} | "
        f"tokens="
        f"{config.planned_steps * args.batch_size * sequence_length * args.gradient_accumulation:,}"
    )
    print_parameter_summary("EMC", emc)
    print_parameter_summary("Baseline", baseline)
    print("\nTraining EMC")
    emc_result = train_model(emc, corpus, config)
    print("\nTraining baseline")
    baseline_result = train_model(baseline, corpus, config)

    print("\nComparison")
    print("model      val loss   perplexity   tokens/s   elapsed")
    print(
        f"EMC        {emc_result.final_validation_loss:8.4f}   "
        f"{emc_result.final_validation_perplexity:10.2f}   "
        f"{emc_result.tokens_per_second:8.0f}   {emc_result.elapsed_seconds:7.1f}s"
    )
    print(
        f"Baseline   {baseline_result.final_validation_loss:8.4f}   "
        f"{baseline_result.final_validation_perplexity:10.2f}   "
        f"{baseline_result.tokens_per_second:8.0f}   "
        f"{baseline_result.elapsed_seconds:7.1f}s"
    )
    print(
        "EMC balance: "
        f"average_raw={emc_result.average_router_balance_loss:.6f} | "
        f"average_weighted={emc_result.average_weighted_balance_contribution:.6f}"
    )

    for prompt in ("Once upon a time", "There was a little girl named"):
        emc_text = generate_text(
            emc,
            corpus.tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            greedy=True,
        )
        baseline_text = generate_text(
            baseline,
            corpus.tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            greedy=True,
        )
        print(f"\nprompt {prompt!r}")
        print(f"  EMC:      {emc_text!r}")
        print(f"  Baseline: {baseline_text!r}")

    if emc_result.routing is not None:
        print("\nEMC routing")
        print_routing_report(emc_result.routing)


if __name__ == "__main__":
    main()
