"""Benchmark the Python reference on the same supported fixture as C++."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import psutil
import torch
from torch.nn import functional as F

from export_reference import config
from rayvan_emc.n2 import N2EMCModel


def bundle(path: Path) -> dict[str, torch.Tensor]:
    module = torch.jit.load(str(path), map_location="cpu")
    return {
        name.replace("__DOT__", "."): value
        for name, value in module.named_buffers(recurse=True)
        if name != "_anchor"
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed(device: torch.device, iterations: int, operation) -> float:
    synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        operation()
    synchronize(device)
    return time.perf_counter() - started


def loss_for(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu else "cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.bf16 and device.type != "cuda":
        raise RuntimeError("BF16 benchmark requires CUDA")

    torch.set_num_threads(1)
    torch.manual_seed(20260902)
    process = psutil.Process()
    model = N2EMCModel(config())
    model.load_state_dict(bundle(args.fixture / "weights.pt"), strict=True)
    model_cpu_memory = sum(
        parameter.numel() * parameter.element_size()
        for parameter in {id(value): value for value in model.parameters()}.values()
    )
    model.to(device)
    fixture = bundle(args.fixture / "forward.pt")
    tokens = fixture["tokens"].to(device)
    targets = fixture["targets"].to(device)
    warmup = 10
    iterations = 100
    dtype = torch.bfloat16 if args.bf16 else None

    def autocast():
        return torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=dtype is not None,
        )

    model.eval()
    with torch.no_grad(), autocast():
        for _ in range(warmup):
            model(tokens)
    cpu_before = sum(process.cpu_times()[:2])
    with torch.no_grad(), autocast():
        forward_seconds = elapsed(device, iterations, lambda: model(tokens))

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    def forward_backward() -> None:
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            loss_for(model(tokens), targets).backward()

    for _ in range(warmup):
        forward_backward()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_backward_seconds = elapsed(device, iterations, forward_backward)
    optimizer_seconds = elapsed(device, iterations, optimizer.step)

    def train_step() -> None:
        forward_backward()
        optimizer.step()

    train_seconds = elapsed(device, iterations, train_step)
    cpu_after = sum(process.cpu_times()[:2])
    wall = forward_seconds + forward_backward_seconds + optimizer_seconds + train_seconds
    memory = process.memory_info()
    optimizer_bytes = sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    report = {
        "runtime": "python",
        "device": device.type,
        "precision": "bf16" if args.bf16 else "fp32",
        "iterations": iterations,
        "forward_ms": forward_seconds * 1000 / iterations,
        "forward_backward_ms": forward_backward_seconds * 1000 / iterations,
        "optimizer_step_ms": optimizer_seconds * 1000 / iterations,
        "train_step_ms": train_seconds * 1000 / iterations,
        "tokens_per_second": tokens.numel() * iterations / train_seconds,
        "cpu_utilization_percent": 100 * (cpu_after - cpu_before) / max(wall, 1e-9) / (os.cpu_count() or 1),
        "process_rss_bytes": memory.rss,
        "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
        "model_cpu_memory_bytes": model_cpu_memory,
        "parameter_bytes": model_cpu_memory,
        "optimizer_bytes": optimizer_bytes,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(device) if device.type == "cuda" else 0,
        "cuda_reserved_bytes": torch.cuda.memory_reserved(device) if device.type == "cuda" else 0,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        "gpu_utilization_percent": None,
        "cuda_synchronization_count": None,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
