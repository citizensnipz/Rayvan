from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import LanguageCorpus
from .diagnostics import EMCDiagnostics, RoutingReport
from .model import EMCModel, EMCOutput


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 100
    batch_size: int = 16
    sequence_length: int = 32
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    evaluation_interval: int = 25
    evaluation_batches: int = 4
    gradient_clip_norm: float = 1.0
    router_balance_coefficient: float = 0.01
    router_balance_entropy_floor: float = 0.75
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive_fields = {
            "steps": self.steps,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "learning_rate": self.learning_rate,
            "evaluation_interval": self.evaluation_interval,
            "evaluation_batches": self.evaluation_batches,
            "gradient_clip_norm": self.gradient_clip_norm,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.router_balance_coefficient < 0:
            raise ValueError("router_balance_coefficient cannot be negative")
        if not 0.0 <= self.router_balance_entropy_floor <= 1.0:
            raise ValueError(
                "router_balance_entropy_floor must be between zero and one"
            )


@dataclass(frozen=True)
class TrainingMetrics:
    step: int
    training_loss: float
    validation_loss: float
    validation_perplexity: float
    router_balance_loss: float
    weighted_balance_contribution: float
    tokens_per_second: float
    elapsed_seconds: float


@dataclass(frozen=True)
class TrainingResult:
    history: tuple[TrainingMetrics, ...]
    final_training_loss: float
    final_validation_loss: float
    final_validation_perplexity: float
    average_router_balance_loss: float
    average_weighted_balance_contribution: float
    tokens_per_second: float
    elapsed_seconds: float
    routing: RoutingReport | None


def next_token_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def evaluate_model(
    model: nn.Module,
    corpus: LanguageCorpus,
    config: TrainingConfig,
    *,
    split: str = "validation",
    generator: torch.Generator | None = None,
) -> tuple[float, float]:
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")
    device = torch.device(config.device)
    evaluation_generator = generator or torch.Generator().manual_seed(config.seed + 1)
    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(config.evaluation_batches):
            inputs, targets = corpus.sample_batch(
                split,
                config.batch_size,
                config.sequence_length,
                generator=evaluation_generator,
                device=device,
            )
            output = model(inputs)
            logits = output.logits if isinstance(output, EMCOutput) else output
            losses.append(next_token_loss(logits, targets).item())
    model.train(was_training)
    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(min(mean_loss, 20.0))


def train_model(
    model: nn.Module,
    corpus: LanguageCorpus,
    config: TrainingConfig,
    *,
    print_progress: bool = True,
) -> TrainingResult:
    device = torch.device(config.device)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_generator = torch.Generator().manual_seed(config.seed)
    evaluation_generator = torch.Generator().manual_seed(config.seed + 1)
    emc_diagnostics = EMCDiagnostics(model) if isinstance(model, EMCModel) else None
    history: list[TrainingMetrics] = []
    started = time.perf_counter()
    tokens_seen = 0
    evaluation_seconds = 0.0
    cumulative_balance_loss = 0.0

    for step in range(1, config.steps + 1):
        inputs, targets = corpus.sample_batch(
            "train",
            config.batch_size,
            config.sequence_length,
            generator=train_generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, EMCModel):
            output = model(
                inputs,
                return_trace=True,
                balance_entropy_floor=config.router_balance_entropy_floor,
            )
            if not isinstance(output, EMCOutput):
                raise RuntimeError("trace-enabled EMC forward did not return EMCOutput")
            if output.router_balance_loss is None:
                raise RuntimeError("EMC forward did not return router balance loss")
            logits = output.logits
            balance_loss = output.router_balance_loss
            if emc_diagnostics is not None:
                emc_diagnostics.observe_trace(output.trace)
        else:
            logits = model(inputs)
            balance_loss = logits.new_zeros(())
        language_model_loss = next_token_loss(logits, targets)
        weighted_balance = config.router_balance_coefficient * balance_loss
        total_loss = language_model_loss + weighted_balance
        total_loss.backward()
        if emc_diagnostics is not None:
            emc_diagnostics.observe_router_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        tokens_seen += targets.numel()
        cumulative_balance_loss += balance_loss.detach().item()

        if step % config.evaluation_interval == 0 or step == config.steps:
            evaluation_started = time.perf_counter()
            training_loss, _ = evaluate_model(
                model,
                corpus,
                config,
                split="train",
                generator=evaluation_generator,
            )
            validation_loss, validation_perplexity = evaluate_model(
                model,
                corpus,
                config,
                split="validation",
                generator=evaluation_generator,
            )
            evaluation_seconds += time.perf_counter() - evaluation_started
            elapsed = time.perf_counter() - started
            training_seconds = elapsed - evaluation_seconds
            metrics = TrainingMetrics(
                step=step,
                training_loss=training_loss,
                validation_loss=validation_loss,
                validation_perplexity=validation_perplexity,
                router_balance_loss=cumulative_balance_loss / step,
                weighted_balance_contribution=(
                    config.router_balance_coefficient
                    * cumulative_balance_loss
                    / step
                ),
                tokens_per_second=tokens_seen / training_seconds,
                elapsed_seconds=elapsed,
            )
            history.append(metrics)
            if print_progress:
                print(
                    f"step {step:>5} | train_lm {training_loss:.4f} | "
                    f"balance {metrics.router_balance_loss:.5f} | "
                    f"weighted {metrics.weighted_balance_contribution:.5f} | "
                    f"validation {validation_loss:.4f} | "
                    f"ppl {validation_perplexity:.2f} | "
                    f"{metrics.tokens_per_second:,.0f} tok/s | {elapsed:.1f}s"
                )

    final_metrics = history[-1]
    routing = emc_diagnostics.report(model) if emc_diagnostics is not None else None
    return TrainingResult(
        history=tuple(history),
        final_training_loss=final_metrics.training_loss,
        final_validation_loss=final_metrics.validation_loss,
        final_validation_perplexity=final_metrics.validation_perplexity,
        average_router_balance_loss=final_metrics.router_balance_loss,
        average_weighted_balance_contribution=(
            final_metrics.weighted_balance_contribution
        ),
        tokens_per_second=final_metrics.tokens_per_second,
        elapsed_seconds=final_metrics.elapsed_seconds,
        routing=routing,
    )
