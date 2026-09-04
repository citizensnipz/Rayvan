from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .baseline import TransformerConfig, TransformerLanguageModel
from .chunked import ChunkedEMCModel
from .model import EMCConfig, EMCModel, SequentialEMCModel
from .n2 import N2Config, N2EMCModel
from .serial import HeterogeneousSerialModel
from .tokenization import TextTokenizer, tokenizer_from_config


CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CheckpointProgress:
    step: int
    tokens_processed: int
    validation_loss: float
    best_validation_loss: float
    train_generator_state: torch.Tensor | None
    evaluation_generator_state: torch.Tensor | None


@dataclass(frozen=True)
class LoadedModelCheckpoint:
    model: nn.Module
    tokenizer: TextTokenizer
    progress: CheckpointProgress
    training_config: dict[str, Any]
    training_diagnostics: dict[str, Any]


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    tokenizer: TextTokenizer,
    step: int,
    tokens_processed: int,
    validation_loss: float,
    best_validation_loss: float,
    training_config: dict[str, Any],
    train_generator_state: torch.Tensor | None = None,
    evaluation_generator_state: torch.Tensor | None = None,
    training_diagnostics: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_type": _model_type(model),
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "runtime_routing": _runtime_routing(model),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "tokens_processed": tokens_processed,
        "validation_loss": validation_loss,
        "best_validation_loss": best_validation_loss,
        "training_config": training_config,
        "training_diagnostics": dict(training_diagnostics or {}),
        "tokenizer": tokenizer.to_config(),
        "train_generator_state": train_generator_state,
        "evaluation_generator_state": evaluation_generator_state,
        "torch_rng_state": torch.random.get_rng_state(),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
) -> CheckpointProgress:
    payload = _load_payload(path, device)
    expected_type = _model_type(model)
    if payload["model_type"] != expected_type:
        raise ValueError(
            f"checkpoint contains {payload['model_type']}, expected {expected_type}"
        )
    model.load_state_dict(payload["model_state"])
    _restore_runtime_routing(model, payload.get("runtime_routing", {}))
    optimizer.load_state_dict(payload["optimizer_state"])
    if payload.get("torch_rng_state") is not None:
        torch.random.set_rng_state(payload["torch_rng_state"].cpu())
    return _progress_from_payload(payload)


def load_model_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedModelCheckpoint:
    payload = _load_payload(path, "cpu")
    model = _create_model(payload["model_type"], payload["model_config"])
    model.load_state_dict(payload["model_state"])
    _restore_runtime_routing(model, payload.get("runtime_routing", {}))
    model.to(device)
    tokenizer = tokenizer_from_config(payload["tokenizer"])
    return LoadedModelCheckpoint(
        model=model,
        tokenizer=tokenizer,
        progress=_progress_from_payload(payload),
        training_config=dict(payload["training_config"]),
        training_diagnostics=dict(payload.get("training_diagnostics", {})),
    )


def _load_payload(path: str | Path, device: torch.device | str) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format: {payload.get('format_version')!r}"
        )
    return payload


def _model_type(model: nn.Module) -> str:
    if isinstance(model, N2EMCModel):
        return "n2_emc"
    if isinstance(model, ChunkedEMCModel):
        return "emc_chunked"
    if isinstance(model, SequentialEMCModel):
        return "emc_sequential"
    if isinstance(model, EMCModel):
        return "emc"
    if isinstance(model, HeterogeneousSerialModel):
        return "heterogeneous_serial"
    if isinstance(model, TransformerLanguageModel):
        return "baseline"
    raise TypeError(f"unsupported checkpoint model type: {type(model).__name__}")


def _create_model(model_type: str, config: dict[str, Any]) -> nn.Module:
    if model_type == "n2_emc":
        return N2EMCModel(N2Config(**config))
    if model_type == "emc":
        return EMCModel(EMCConfig(**config))
    if model_type == "emc_sequential":
        return SequentialEMCModel(EMCConfig(**config))
    if model_type == "emc_chunked":
        return ChunkedEMCModel(EMCConfig(**config))
    if model_type == "baseline":
        return TransformerLanguageModel(TransformerConfig(**config))
    if model_type == "heterogeneous_serial":
        return HeterogeneousSerialModel(EMCConfig(**config))
    raise ValueError(f"unknown checkpoint model type: {model_type!r}")


def _runtime_routing(model: nn.Module) -> dict[str, Any]:
    if isinstance(model, SequentialEMCModel):
        return {"trajectory_steps": model.config.resolved_trajectory_steps}
    active_top_k = getattr(model, "active_top_k", None)
    return {"active_top_k": int(active_top_k)} if active_top_k is not None else {}


def _restore_runtime_routing(
    model: nn.Module, state: dict[str, Any]
) -> None:
    active_top_k = state.get("active_top_k")
    setter = getattr(model, "set_active_top_k", None)
    if active_top_k is not None and setter is not None:
        setter(int(active_top_k))


def _progress_from_payload(payload: dict[str, Any]) -> CheckpointProgress:
    return CheckpointProgress(
        step=int(payload["step"]),
        tokens_processed=int(payload["tokens_processed"]),
        validation_loss=float(payload["validation_loss"]),
        best_validation_loss=float(payload["best_validation_loss"]),
        train_generator_state=payload.get("train_generator_state"),
        evaluation_generator_state=payload.get("evaluation_generator_state"),
    )
