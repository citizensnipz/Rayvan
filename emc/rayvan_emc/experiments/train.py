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
    parser = argparse.ArgumentParser(description="Train one local language model")
    parser.add_argument("--model", choices=("emc", "baseline"), default="emc")
    parser.add_argument("--dataset", choices=("tiny", "tinystories"), default="tinystories")
    parser.add_argument("--preset", choices=("quick", "research"), default="quick")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--balance-coefficient", type=float, default=0.01)
    parser.add_argument("--balance-entropy-floor", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-stories", type=int, default=2_000)
    parser.add_argument("--validation-stories", type=int, default=200)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--greedy", action="store_true")
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
    create_model = create_emc_model if args.model == "emc" else create_baseline_model
    model = create_model(
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
        evaluation_interval=max(1, args.steps // 10),
        evaluation_batches=5,
        router_balance_coefficient=args.balance_coefficient,
        router_balance_entropy_floor=args.balance_entropy_floor,
        seed=args.seed,
        device=args.device,
    )

    print_parameter_summary(args.model.upper(), model)
    result = train_model(model, corpus, config)
    print(
        f"final: loss={result.final_validation_loss:.4f} | "
        f"ppl={result.final_validation_perplexity:.2f} | "
        f"{result.tokens_per_second:,.0f} tok/s | {result.elapsed_seconds:.1f}s"
    )
    if args.model == "emc":
        print(
            f"balance: raw={result.average_router_balance_loss:.6f} | "
            f"weighted={result.average_weighted_balance_contribution:.6f} | "
            f"coefficient={args.balance_coefficient:.4f}"
        )
    if result.routing is not None:
        print_routing_report(result.routing)
    print("\nSample:")
    print(
        generate_text(
            model,
            corpus.tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            greedy=args.greedy,
        )
    )


if __name__ == "__main__":
    main()
