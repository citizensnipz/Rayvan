"""Profile the Python AdamW path on the small EMC CUDA fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from benchmark_reference import bundle
from export_reference import config
from rayvan_emc.n2 import N2EMCModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    device = torch.device("cuda:0")
    model = N2EMCModel(config()).train()
    model.load_state_dict(bundle(args.fixture / "weights.pt"), strict=True)
    model.to(device)
    fixture = bundle(args.fixture / "forward.pt")
    tokens = fixture["tokens"].to(device)
    targets = fixture["targets"].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    def forward_backward() -> None:
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).backward()

    for _ in range(6):
        forward_backward()
        optimizer.step()
    forward_backward()
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
    ) as profile:
        for _ in range(args.steps):
            optimizer.step()
    torch.cuda.synchronize(device)

    events = profile.key_averages()
    launch_events = sum(event.count for event in events if "cudaLaunchKernel" in event.key)
    cuda_time_us = sum(getattr(event, "device_time_total", 0.0) for event in events)
    positive_cuda_allocations = sum(
        max(0, getattr(event, "self_device_memory_usage", 0)) for event in events
    )
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    try:
        from torch.optim.optimizer import _default_to_fused_or_foreach

        selected_fused, selected_foreach = _default_to_fused_or_foreach(
            parameters, differentiable=False, use_fused=False
        )
    except (ImportError, TypeError):
        selected_fused, selected_foreach = None, None

    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "requested_foreach": optimizer.param_groups[0].get("foreach"),
                "requested_fused": optimizer.param_groups[0].get("fused"),
                "selected_foreach": selected_foreach,
                "selected_fused": selected_fused,
                "parameter_tensors": len(parameters),
                "steps": args.steps,
                "cuda_launches_total": launch_events,
                "cuda_launches_per_step": launch_events / args.steps,
                "cuda_execution_us_per_step": cuda_time_us / args.steps,
                "positive_cuda_allocations_bytes_per_step": positive_cuda_allocations / args.steps,
                "peak_temporary_cuda_bytes": torch.cuda.max_memory_allocated(device) - before,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
