"""Measure model-only Python reference loading in a fresh process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import psutil
import torch

from export_reference import config
from rayvan_emc.n2 import N2EMCModel


def bundle(path: Path) -> dict[str, torch.Tensor]:
    module = torch.jit.load(str(path), map_location="cpu")
    return {
        name.replace("__DOT__", "."): value
        for name, value in module.named_buffers(recurse=True)
        if name != "_anchor"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu else "cuda:0")
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    model = N2EMCModel(config())
    model.load_state_dict(bundle(args.fixture / "weights.pt"), strict=True)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    memory = psutil.Process().memory_info()
    print(
        json.dumps(
            {
                "runtime": "python",
                "checkpoint_load_rss_bytes": memory.rss,
                "checkpoint_load_peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
                "cuda_allocated_bytes": torch.cuda.memory_allocated(device) if device.type == "cuda" else 0,
                "cuda_reserved_bytes": torch.cuda.memory_reserved(device) if device.type == "cuda" else 0,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
