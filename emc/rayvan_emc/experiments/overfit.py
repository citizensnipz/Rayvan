from __future__ import annotations

import argparse

import torch

from ..generation import generate_text
from ..training import TrainingConfig, evaluate_model, train_model
from .common import create_emc_model, print_parameter_summary, print_routing_report
from ..data import tiny_overfit_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit EMC on a tiny local corpus")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--balance-coefficient", type=float, default=0.01)
    parser.add_argument("--balance-entropy-floor", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = tiny_overfit_corpus()
    model = create_emc_model(
        corpus.tokenizer.vocab_size,
        "quick",
        maximum_sequence_length=args.sequence_length,
        seed=args.seed,
    )
    training = TrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        evaluation_interval=max(1, args.steps // 5),
        evaluation_batches=4,
        router_balance_coefficient=args.balance_coefficient,
        router_balance_entropy_floor=args.balance_entropy_floor,
        seed=args.seed,
    )

    print_parameter_summary("EMC", model)
    initial_loss, initial_perplexity = evaluate_model(model, corpus, training)
    print(f"initial validation loss={initial_loss:.4f} ppl={initial_perplexity:.2f}")
    result = train_model(model, corpus, training)
    print(
        f"loss reduction: {initial_loss:.4f} -> {result.final_validation_loss:.4f}"
    )
    print(
        f"balance: raw={result.average_router_balance_loss:.6f} | "
        f"weighted={result.average_weighted_balance_contribution:.6f} | "
        f"coefficient={args.balance_coefficient:.4f}"
    )
    if result.routing is not None:
        print_routing_report(result.routing)

    print("\nGreedy memorization samples:")
    for prompt in ("the ", "a ", "we "):
        generated = generate_text(
            model,
            corpus.tokenizer,
            prompt,
            max_new_tokens=60,
            greedy=True,
        )
        print(f"  {generated!r}")


if __name__ == "__main__":
    main()
