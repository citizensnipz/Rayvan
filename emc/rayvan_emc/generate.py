from __future__ import annotations

import argparse
import math

import torch

from .checkpoint import load_model_checkpoint
from .generation import generate_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue text from a local EMC checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_model_checkpoint(args.checkpoint, device=args.device)
    print(
        f"checkpoint tokens={loaded.progress.tokens_processed:,} | "
        f"best validation={loaded.progress.best_validation_loss:.4f} | "
        f"ppl={math.exp(min(loaded.progress.best_validation_loss, 20.0)):.2f} | "
        f"tokenizer={loaded.tokenizer.identifier}"
    )
    print(
        generate_text(
            loaded.model,
            loaded.tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            greedy=args.greedy,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    )


if __name__ == "__main__":
    main()
