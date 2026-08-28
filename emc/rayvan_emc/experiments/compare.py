from __future__ import annotations

import argparse

from ..generation import generate_text
from ..training import TrainingConfig, train_model
from .common import (
    create_baseline_model,
    create_emc_model,
    load_experiment_corpus,
    print_parameter_summary,
    print_routing_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare EMC with a plain transformer")
    parser.add_argument("--dataset", choices=("tiny", "tinystories"), default="tiny")
    parser.add_argument("--preset", choices=("quick", "research"), default="quick")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--train-stories", type=int, default=2_000)
    parser.add_argument("--validation-stories", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = load_experiment_corpus(
        args.dataset,
        train_stories=args.train_stories,
        validation_stories=args.validation_stories,
    )
    emc = create_emc_model(
        corpus.tokenizer.vocab_size,
        args.preset,
        maximum_sequence_length=args.sequence_length,
        seed=args.seed,
    )
    baseline = create_baseline_model(
        corpus.tokenizer.vocab_size,
        args.preset,
        maximum_sequence_length=args.sequence_length,
        seed=args.seed,
    )
    config = TrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        evaluation_interval=max(1, args.steps // 5),
        evaluation_batches=4,
        seed=args.seed,
        device=args.device,
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

    for prompt in ("the ", "a ", "we "):
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
