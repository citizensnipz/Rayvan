"""Export deterministic pretokenized IDs to the native `.rvtok` format."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch

MAGIC = b"RVTOKEN1"
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


def load_tokens(path: Path, key: str | None) -> list[int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if key is not None:
        if not isinstance(payload, dict):
            raise TypeError("--key requires a dictionary payload")
        payload = payload[key]
    if isinstance(payload, torch.Tensor):
        values = payload.reshape(-1).to(torch.int64).tolist()
    elif isinstance(payload, (list, tuple)):
        values = [int(value) for value in payload]
    else:
        raise TypeError("input must be a tensor/list or a dictionary entry containing one")
    if len(values) < 2:
        raise ValueError("token stream must contain at least two IDs")
    if any(value < -(2**31) or value >= 2**31 for value in values):
        raise ValueError("token ID is outside int32 range")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="torch-saved tensor/list/dictionary")
    parser.add_argument("output", type=Path)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--key")
    args = parser.parse_args()
    tokens = load_tokens(args.input, args.key)
    split = args.split_id.encode("utf-8")
    digest = fingerprint(tokens, args.split_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(struct.pack("<8sIQQI", MAGIC, VERSION, len(tokens), digest, len(split)))
        stream.write(split)
        stream.write(struct.pack(f"<{len(tokens)}i", *tokens))
    print(json.dumps({"path": str(args.output), "split_id": args.split_id, "tokens": len(tokens), "fingerprint_fnv1a64": f"{digest:016x}"}))


if __name__ == "__main__":
    main()
