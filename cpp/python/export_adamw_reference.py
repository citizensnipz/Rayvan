"""Export deterministic Python AdamW states for the native foreach test."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from export_reference import save_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    parameters = [
        torch.linspace(-0.5, 0.5, 17).requires_grad_(),
        torch.linspace(-0.25, 0.75, 35).reshape(7, 5).requires_grad_(),
        torch.full((), 0.125).requires_grad_(),
        torch.linspace(-1.0, 1.0, 129).requires_grad_(),
    ]
    options = {
        "lr": 3e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
    }
    optimizer = torch.optim.AdamW(parameters, **options)
    snapshots: dict[str, torch.Tensor] = {}

    for step in range(1, 101):
        for index, parameter in enumerate(parameters):
            gradient = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            parameter.grad = gradient.mul(1e-3).add(step * 1e-4 + index * 1e-2)
        optimizer.step()

        if step in (1, 10, 100):
            loss = torch.zeros(())
            for index, parameter in enumerate(parameters):
                state = optimizer.state[parameter]
                snapshots[f"step{step}.parameter{index}"] = parameter.detach().clone()
                snapshots[f"step{step}.exp_avg{index}"] = state["exp_avg"].detach().clone()
                snapshots[f"step{step}.exp_avg_sq{index}"] = state["exp_avg_sq"].detach().clone()
                snapshots[f"step{step}.state_step{index}"] = state["step"].detach().clone()
                loss = loss + parameter.square().sum()
            snapshots[f"step{step}.loss"] = loss.detach().clone()

    save_bundle(args.destination, snapshots)


if __name__ == "__main__":
    main()
