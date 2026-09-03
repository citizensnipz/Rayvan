"""Prepare deterministic GPT-2 TinyStories streams for the native runtime.

This is offline data preparation. Native EMC never imports or invokes Python.
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import requests

from rayvan_emc.tokenization import HuggingFaceTokenizer

from export_token_stream import VERSION, fingerprint

MAGIC = b"RVTOKEN1"


def write_stream(path: Path, tokens: list[int], split_id: str) -> None:
    encoded_split = split_id.encode("utf-8")
    digest = fingerprint(tokens, split_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(
            struct.pack(
                "<8sIQQI", MAGIC, VERSION, len(tokens), digest, len(encoded_split)
            )
        )
        stream.write(encoded_split)
        stream.write(struct.pack(f"<{len(tokens)}i", *tokens))
    print(f"{path}: {len(tokens)} tokens, fingerprint={digest:016x}")


def fetch_tokens(split: str, count: int, tokenizer: HuggingFaceTokenizer) -> list[int]:
    tokens: list[int] = []
    endpoint = "https://datasets-server.huggingface.co/rows"
    for offset in range(0, count, 100):
        length = min(100, count - offset)
        parameters = {
            "dataset": "roneneldan/TinyStories",
            "config": "default",
            "split": split,
            "offset": offset,
            "length": length,
        }
        for attempt in range(6):
            response = requests.get(endpoint, params=parameters, timeout=60)
            if response.status_code != 429:
                break
            delay = 10 * (attempt + 1)
            print(f"{split}: rate limited at row {offset}; retrying in {delay}s")
            time.sleep(delay)
        response.raise_for_status()
        rows = response.json()["rows"]
        if len(rows) != length:
            raise RuntimeError(f"expected {length} {split} rows at {offset}, got {len(rows)}")
        for item in rows:
            tokens.extend(tokenizer.encode(str(item["row"]["text"])))
            tokens.append(tokenizer.eos_token_id)
        if offset == 0 or offset + length == count or (offset + length) % 1_000 == 0:
            print(f"{split}: {offset + length}/{count} stories, {len(tokens)} tokens")
        time.sleep(0.5)
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--train-stories", type=int, default=10_000)
    parser.add_argument("--validation-stories", type=int, default=1_000)
    parser.add_argument("--tokenizer", default="gpt2")
    args = parser.parse_args()
    tokenizer = HuggingFaceTokenizer.from_pretrained(args.tokenizer)
    train_tokens = fetch_tokens("train", args.train_stories, tokenizer)
    validation_tokens = fetch_tokens("validation", args.validation_stories, tokenizer)
    prefix = (
        f"roneneldan/TinyStories/stream-head/"
        f"{args.train_stories}-{args.validation_stories}/{args.tokenizer}"
    )
    write_stream(
        args.destination / "train.rvtok",
        train_tokens,
        prefix + "/train",
    )
    write_stream(
        args.destination / "validation.rvtok",
        validation_tokens,
        prefix + "/validation",
    )


if __name__ == "__main__":
    main()
