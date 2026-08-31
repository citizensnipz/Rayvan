from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .baseline import TransformerConfig, TransformerLanguageModel
from .chunked import ChunkedEMCModel
from .model import EMCConfig, EMCModel
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
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_type": _model_type(model),
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "tokens_processed": tokens_processed,
        "validation_loss": validation_loss,
        "best_validation_loss": best_validation_loss,
        "training_config": training_config,
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
    optimizer.load_state_dict(payload["optimizer_state"])
    if payload.get("torch_rng_state") is not None:
        torch.random.set_rng_state(payload["torch_rng_state"].cpu())
    return _progress_from_payload(payload)


def load_model_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedModelCheckpoint:
    payload = _load_payload(path, device)
    model = _create_model(payload["model_type"], payload["model_config"])
    model.load_state_dict(payload["model_state"])
    model.to(device)
    tokenizer = tokenizer_from_config(payload["tokenizer"])
    return LoadedModelCheckpoint(
        model=model,
        tokenizer=tokenizer,
        progress=_progress_from_payload(payload),
        training_config=dict(payload["training_config"]),
    )


def _load_payload(path: str | Path, device: torch.device | str) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format: {payload.get('format_version')!r}"
        )
    return payload


def _model_type(model: nn.Module) -> str:
    if isinstance(model, ChunkedEMCModel):
        return "emc_chunked"
    if isinstance(model, EMCModel):
        return "emc"
    if isinstance(model, TransformerLanguageModel):
        return "baseline"
    raise TypeError(f"unsupported checkpoint model type: {type(model).__name__}")


def _create_model(model_type: str, config: dict[str, Any]) -> nn.Module:
    if model_type == "emc":
        return EMCModel(EMCConfig(**config))
    if model_type == "emc_chunked":
        return ChunkedEMCModel(EMCConfig(**config))
    if model_type == "baseline":
        return TransformerLanguageModel(TransformerConfig(**config))
    raise ValueError(f"unknown checkpoint model type: {model_type!r}")


def _progress_from_payload(payload: dict[str, Any]) -> CheckpointProgress:
    return CheckpointProgress(
        step=int(payload["step"]),
        tokens_processed=int(payload["tokens_processed"]),
        validation_loss=float(payload["validation_loss"]),
        best_validation_loss=float(payload["best_validation_loss"]),
        train_generator_state=payload.get("train_generator_state"),
        evaluation_generator_state=payload.get("evaluation_generator_state"),
    )
