"""Export the unchanged CapabilityTaskSuite for the native EMC runtime."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "emc"))

from rayvan_emc.capability_tasks import (  # noqa: E402
    CAPABILITIES,
    CAPABILITY_GENERATOR_VERSION,
    CapabilitySuiteConfig,
    CapabilityTaskSuite,
    diagnostic_tokenizer,
)

TOKEN_MAGIC = b"RVTOKEN1"
EVALUATION_MAGIC = b"RVCAPEV1"
VERSION = 1
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211


def fingerprint(tokens: list[int], split_id: str) -> int:
    value = FNV_OFFSET
    for byte in split_id.encode("utf-8") + b"\0":
        value = ((value ^ byte) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    for token in tokens:
        for byte in struct.pack("<i", token):
            value = ((value ^ byte) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def packed_tokens(examples: tuple[object, ...], tokenizer: object) -> list[int]:
    tokens: list[int] = []
    for example in examples:
        tokens.extend(tokenizer.encode(example.model_text))
        tokens.append(tokenizer.eos_token_id)
    return tokens


def write_token_stream(path: Path, tokens: list[int], split_id: str) -> dict[str, object]:
    encoded_split = split_id.encode("utf-8")
    digest = fingerprint(tokens, split_id)
    with path.open("wb") as stream:
        stream.write(struct.pack("<8sIQQI", TOKEN_MAGIC, VERSION, len(tokens), digest, len(encoded_split)))
        stream.write(encoded_split)
        stream.write(struct.pack(f"<{len(tokens)}i", *tokens))
    return {"path": str(path), "tokens": len(tokens), "fingerprint": f"{digest:016x}"}


def write_evaluation(
    path: Path,
    capabilities: tuple[str, ...],
    examples: tuple[object, ...],
    tokenizer: object,
    *,
    group_by_surface: bool = False,
) -> dict[str, object]:
    group_counts = {capability: 0 for capability in capabilities}
    with path.open("wb") as stream:
        stream.write(struct.pack("<8sIII", EVALUATION_MAGIC, VERSION, tokenizer.vocab_size, len(capabilities)))
        for capability in capabilities:
            encoded = capability.encode("utf-8")
            stream.write(struct.pack("<I", len(encoded)))
            stream.write(encoded)
        stream.write(struct.pack("<Q", len(examples)))
        for example in examples:
            prompt = tokenizer.encode(example.prompt)
            target = tokenizer.encode(example.target)
            group = example.diagnostic_metadata.capability
            if group_by_surface:
                group += "/" + example.diagnostic_metadata.surface_format
            capability = capabilities.index(group)
            group_counts[group] += 1
            stream.write(struct.pack("<III", capability, len(prompt), len(target)))
            stream.write(struct.pack(f"<{len(prompt)}i", *prompt))
            stream.write(struct.pack(f"<{len(target)}i", *target))
    return {"path": str(path), "examples": len(examples), "group_counts": group_counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-examples", type=int, default=8_000)
    parser.add_argument("--validation-examples", type=int, default=1_000)
    parser.add_argument("--evaluation-examples-per-capability", type=int, default=100)
    args = parser.parse_args()

    suite = CapabilityTaskSuite(CapabilitySuiteConfig(seed=args.seed))
    tokenizer = diagnostic_tokenizer()
    train_examples = suite.mixed_examples(args.train_examples, split="train")
    validation_examples = suite.mixed_examples(args.validation_examples, split="validation")
    evaluation_examples = suite.balanced_evaluation(args.evaluation_examples_per_capability)
    surface_groups = tuple(
        dict.fromkeys(
            example.diagnostic_metadata.capability
            + "/"
            + example.diagnostic_metadata.surface_format
            for example in evaluation_examples
        )
    )
    train_tokens = packed_tokens(train_examples, tokenizer)
    validation_tokens = packed_tokens(validation_examples, tokenizer)

    args.destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generator_version": CAPABILITY_GENERATOR_VERSION,
        "seed": args.seed,
        "tokenizer": tokenizer.to_config(),
        "vocab_size": tokenizer.vocab_size,
        "capabilities": list(CAPABILITIES),
        "mixture_weights": dict(suite.config.mixture_weights),
        "held_out_combinations": [list(value) for value in suite.config.held_out_combinations],
        "train": write_token_stream(args.destination / "train.rvtok", train_tokens, "capability-v2/train"),
        "validation": write_token_stream(args.destination / "validation.rvtok", validation_tokens, "capability-v2/validation"),
        "evaluation": write_evaluation(args.destination / "evaluation.rvcap", CAPABILITIES, evaluation_examples, tokenizer),
        "evaluation_by_surface": write_evaluation(
            args.destination / "evaluation-by-surface.rvcap",
            surface_groups,
            evaluation_examples,
            tokenizer,
            group_by_surface=True,
        ),
    }
    (args.destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
