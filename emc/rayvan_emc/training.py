from __future__ import annotations

from copy import deepcopy
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .checkpoint import load_training_checkpoint, save_training_checkpoint
from .chunked import ChunkedEMCModel
from .data import LanguageCorpus
from .diagnostics import EMCDiagnostics, RoutingReport
from .model import EMCModel, EMCOutput
from .telemetry import (
    ModuleTelemetry,
    attach_causal_report,
    write_developmental_record,
)


@dataclass(frozen=True)
class TrainingConfig:
    steps: int | None = 100
    train_tokens: int | None = None
    batch_size: int = 16
    sequence_length: int = 32
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    evaluation_interval: int = 25
    evaluation_batches: int = 4
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    router_balance_coefficient: float = 0.01
    router_balance_entropy_floor: float = 0.75
    precision: str = "fp32"
    checkpoint_directory: str | None = None
    checkpoint_prefix: str = "emc"
    retain_best_checkpoint: bool = True
    resume_from: str | None = None
    seed: int = 42
    device: str = "cpu"
    collect_module_diagnostics: bool = False
    diagnostic_milestones: tuple[int, ...] = (
        100_000,
        250_000,
        500_000,
        750_000,
        1_000_000,
    )
    retain_milestone_checkpoints: bool = True
    collect_milestone_telemetry: bool = False
    milestone_causal_diagnostics: bool = False
    milestone_causal_sample_size: int = 2
    top_k_schedule: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        positive_fields = {
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "learning_rate": self.learning_rate,
            "evaluation_interval": self.evaluation_interval,
            "evaluation_batches": self.evaluation_batches,
            "gradient_clip_norm": self.gradient_clip_norm,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.steps is None and self.train_tokens is None:
            raise ValueError("steps or train_tokens must be provided")
        if self.steps is not None and self.steps <= 0:
            raise ValueError("steps must be positive when provided")
        if self.train_tokens is not None and self.train_tokens <= 0:
            raise ValueError("train_tokens must be positive when provided")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.router_balance_coefficient < 0:
            raise ValueError("router_balance_coefficient cannot be negative")
        if not 0.0 <= self.router_balance_entropy_floor <= 1.0:
            raise ValueError(
                "router_balance_entropy_floor must be between zero and one"
            )
        if self.precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("precision must be auto, fp32, fp16, or bf16")
        if any(milestone <= 0 for milestone in self.diagnostic_milestones):
            raise ValueError("diagnostic milestones must be positive token counts")
        if tuple(sorted(set(self.diagnostic_milestones))) != self.diagnostic_milestones:
            raise ValueError("diagnostic milestones must be sorted and unique")
        if self.milestone_causal_sample_size <= 0:
            raise ValueError("milestone causal sample size must be positive")
        if self.milestone_causal_diagnostics and not self.retain_milestone_checkpoints:
            raise ValueError(
                "milestone causal diagnostics require retained milestone checkpoints"
            )
        if self.top_k_schedule is not None:
            starts = tuple(start for start, _ in self.top_k_schedule)
            if not starts or starts[0] != 0:
                raise ValueError("top-K schedule must begin at token zero")
            if starts != tuple(sorted(set(starts))):
                raise ValueError("top-K schedule transitions must be sorted and unique")
            if any(top_k <= 0 for _, top_k in self.top_k_schedule):
                raise ValueError("scheduled top-K values must be positive")
        if (
            (self.collect_milestone_telemetry or self.milestone_causal_diagnostics)
            and self.checkpoint_directory is None
        ):
            raise ValueError(
                "milestone telemetry and causal diagnostics require a checkpoint directory"
            )
        if self.milestone_causal_diagnostics and not self.collect_milestone_telemetry:
            raise ValueError(
                "milestone causal diagnostics require milestone telemetry"
            )

    @property
    def planned_steps(self) -> int:
        if self.train_tokens is not None:
            return steps_for_token_budget(
                self.train_tokens,
                self.batch_size,
                self.sequence_length,
                self.gradient_accumulation_steps,
            )
        if self.steps is None:
            raise RuntimeError("training configuration has no duration")
        return self.steps


@dataclass(frozen=True)
class TrainingMetrics:
    step: int
    tokens_processed: int
    training_loss: float
    validation_loss: float
    validation_perplexity: float
    training_token_accuracy: float
    validation_token_accuracy: float
    router_balance_loss: float
    weighted_balance_contribution: float
    tokens_per_second: float
    elapsed_seconds: float
    gpu_memory_used_bytes: int
    gpu_peak_memory_bytes: int


@dataclass(frozen=True)
class TrainingResult:
    history: tuple[TrainingMetrics, ...]
    steps_completed: int
    tokens_processed: int
    final_training_loss: float
    final_validation_loss: float
    final_validation_perplexity: float
    best_validation_loss: float
    final_training_token_accuracy: float
    final_validation_token_accuracy: float
    average_router_balance_loss: float
    average_weighted_balance_contribution: float
    tokens_per_second: float
    elapsed_seconds: float
    gpu_memory_used_bytes: int
    gpu_peak_memory_bytes: int
    routing: RoutingReport | None
    latest_checkpoint: str | None
    best_checkpoint: str | None
    module_diagnostics: dict[str, Any] | None


    milestone_checkpoints: tuple[str, ...]
    telemetry_path: str | None
@dataclass(frozen=True)
class CycleEvaluationMetrics:
    cycle: int
    validation_loss: float
    perplexity: float
    seconds_per_batch: float


EvaluationCallback = Callable[[int, nn.Module, TrainingMetrics], None]


def steps_for_token_budget(
    train_tokens: int,
    batch_size: int,
    sequence_length: int,
    gradient_accumulation_steps: int = 1,
) -> int:
    if (
        train_tokens <= 0
        or batch_size <= 0
        or sequence_length <= 0
        or gradient_accumulation_steps <= 0
    ):
        raise ValueError(
            "token budget, batch size, sequence length, and accumulation must be positive"
        )
    tokens_per_step = (
        batch_size * sequence_length * gradient_accumulation_steps
    )
    return (train_tokens + tokens_per_step - 1) // tokens_per_step


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
    loss, perplexity, _ = evaluate_model_metrics(
        model, corpus, config, split=split, generator=generator
    )
    return loss, perplexity


def evaluate_model_metrics(
    model: nn.Module,
    corpus: LanguageCorpus,
    config: TrainingConfig,
    *,
    split: str = "validation",
    generator: torch.Generator | None = None,
) -> tuple[float, float, float]:
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")
    device = torch.device(config.device)
    precision = _resolved_precision(config.precision, device)
    evaluation_generator = generator or torch.Generator().manual_seed(config.seed + 1)
    was_training = model.training
    model.eval()
    losses: list[float] = []
    correct = 0
    tokens = 0
    with torch.no_grad():
        for _ in range(config.evaluation_batches):
            inputs, targets = corpus.sample_batch(
                split,
                config.batch_size,
                config.sequence_length,
                generator=evaluation_generator,
                device=device,
            )
            with _autocast_context(device, precision):
                output = model(inputs)
                logits = output.logits if isinstance(output, EMCOutput) else output
                losses.append(next_token_loss(logits, targets).item())
            correct += int((logits.argmax(dim=-1) == targets).sum().item())
            tokens += targets.numel()
    model.train(was_training)
    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(min(mean_loss, 20.0)), correct / tokens


def evaluate_emc_cycles(
    model: EMCModel,
    corpus: LanguageCorpus,
    config: TrainingConfig,
) -> tuple[CycleEvaluationMetrics, ...]:
    device = torch.device(config.device)
    precision = _resolved_precision(config.precision, device)
    was_training = model.training
    model.eval()
    results: list[CycleEvaluationMetrics] = []
    try:
        for cycle_limit in range(1, model.config.num_cycles + 1):
            generator = torch.Generator().manual_seed(config.seed + 3)
            losses: list[float] = []
            started = time.perf_counter()
            with torch.no_grad():
                for _ in range(config.evaluation_batches):
                    inputs, targets = corpus.sample_batch(
                        "validation",
                        config.batch_size,
                        config.sequence_length,
                        generator=generator,
                        device=device,
                    )
                    with _autocast_context(device, precision):
                        logits = model(
                            inputs,
                            evaluation_cycle_limit=cycle_limit,
                        )
                        if not isinstance(logits, Tensor):
                            raise RuntimeError(
                                "cycle-limited EMC evaluation returned diagnostics"
                            )
                        losses.append(next_token_loss(logits, targets).item())
            mean_loss = sum(losses) / len(losses)
            elapsed = time.perf_counter() - started
            results.append(
                CycleEvaluationMetrics(
                    cycle=cycle_limit,
                    validation_loss=mean_loss,
                    perplexity=math.exp(min(mean_loss, 20.0)),
                    seconds_per_batch=elapsed / config.evaluation_batches,
                )
            )
    finally:
        model.train(was_training)
    return tuple(results)


def collect_routing_report(
    model: EMCModel,
    corpus: LanguageCorpus,
    config: TrainingConfig,
    *,
    batches: int = 4,
) -> RoutingReport:
    if batches <= 0:
        raise ValueError("routing evaluation batches must be positive")
    device = torch.device(config.device)
    precision = _resolved_precision(config.precision, device)
    diagnostics = EMCDiagnostics(model)
    generator = torch.Generator().manual_seed(config.seed + 2)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            inputs, _ = corpus.sample_batch(
                "validation",
                config.batch_size,
                config.sequence_length,
                generator=generator,
                device=device,
            )
            with _autocast_context(device, precision):
                output = model(inputs, return_trace=True)
            if not isinstance(output, EMCOutput):
                raise RuntimeError("trace-enabled EMC forward did not return EMCOutput")
            diagnostics.observe_trace(output.trace)
    model.train(was_training)
    return diagnostics.report(model)
def train_model(
    model: nn.Module,
    corpus: LanguageCorpus,
    config: TrainingConfig,
    *,
    print_progress: bool = True,
    evaluation_callback: EvaluationCallback | None = None,
) -> TrainingResult:
    device = torch.device(config.device)
    precision = _resolved_precision(config.precision, device)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    use_fp16_scaler = device.type == "cuda" and precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    train_generator = torch.Generator().manual_seed(config.seed)
    evaluation_generator = torch.Generator().manual_seed(config.seed + 1)
    start_step = 0
    tokens_processed = 0
    best_validation_loss = math.inf

    if config.resume_from is not None:
        progress = load_training_checkpoint(
            config.resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        start_step = progress.step
        tokens_processed = progress.tokens_processed
        best_validation_loss = progress.best_validation_loss
        if progress.train_generator_state is not None:
            train_generator.set_state(progress.train_generator_state.cpu())
        if progress.evaluation_generator_state is not None:
            evaluation_generator.set_state(
                progress.evaluation_generator_state.cpu()
            )

    total_steps = config.planned_steps
    if start_step >= total_steps:
        raise ValueError(
            f"checkpoint is already at step {start_step}; configure a larger budget"
        )

    emc_diagnostics = EMCDiagnostics(model) if isinstance(model, EMCModel) else None
    telemetry = (
        ModuleTelemetry(model)
        if config.collect_milestone_telemetry
        and isinstance(model, (EMCModel, ChunkedEMCModel))
        else None
    )
    if config.collect_milestone_telemetry and telemetry is None:
        raise ValueError("milestone module telemetry is available only for EMC models")
    history: list[TrainingMetrics] = []
    started = time.perf_counter()
    initial_tokens = tokens_processed
    evaluation_seconds = 0.0
    cumulative_balance_loss = 0.0
    completed_steps_this_run = 0
    latest_checkpoint: Path | None = None
    best_checkpoint: Path | None = None
    checkpoint_directory: Path | None = None
    milestone_checkpoints: list[str] = []
    telemetry_path: str | None = None
    latest_module_diagnostics: dict[str, Any] = {}
    if config.checkpoint_directory is not None:
        checkpoint_directory = Path(config.checkpoint_directory)
        latest_checkpoint = checkpoint_directory / _checkpoint_filename(
            config.checkpoint_prefix, "latest"
        )
        best_checkpoint = (
            checkpoint_directory / _checkpoint_filename(
                config.checkpoint_prefix, "best"
            )
            if config.retain_best_checkpoint
            else None
        )
    pending_milestones = tuple(
        milestone
        for milestone in config.diagnostic_milestones
        if milestone > tokens_processed
    )
    planned_step_tokens = (
        config.batch_size
        * config.sequence_length
        * config.gradient_accumulation_steps
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(start_step + 1, total_steps + 1):
        _apply_top_k_schedule(model, config.top_k_schedule, tokens_processed)
        crossed_milestones = tuple(
            milestone
            for milestone in pending_milestones
            if tokens_processed < milestone <= tokens_processed + planned_step_tokens
        )
        evaluation_due = bool(
            step % config.evaluation_interval == 0
            or step == total_steps
            or crossed_milestones
        )
        optimizer.zero_grad(set_to_none=True)
        step_balance_loss = 0.0
        step_tokens = 0
        capture_module_diagnostics = bool(
            (config.collect_module_diagnostics or telemetry is not None)
            and evaluation_due
            and isinstance(model, (EMCModel, ChunkedEMCModel))
        )
        parameter_snapshot: list[list[Tensor]] | None = None
        gradient_norms: list[float | None] | None = None
        try:
            for _ in range(config.gradient_accumulation_steps):
                inputs, targets = corpus.sample_batch(
                    "train",
                    config.batch_size,
                    config.sequence_length,
                    generator=train_generator,
                    device=device,
                )
                with _autocast_context(device, precision):
                    if isinstance(model, (EMCModel, ChunkedEMCModel)):
                        output = model(
                            inputs,
                            return_trace=True,
                            balance_entropy_floor=(
                                config.router_balance_entropy_floor
                            ),
                        )
                        if not isinstance(output, EMCOutput):
                            raise RuntimeError(
                                "trace-enabled EMC forward did not return EMCOutput"
                            )
                        if output.router_balance_loss is None:
                            raise RuntimeError(
                                "EMC forward did not return router balance loss"
                            )
                        logits = output.logits
                        balance_loss = output.router_balance_loss
                        if emc_diagnostics is not None:
                            emc_diagnostics.observe_trace(output.trace)
                        if telemetry is not None:
                            telemetry.observe(output, step)
                    else:
                        logits = model(inputs)
                        balance_loss = logits.new_zeros(())
                    language_model_loss = next_token_loss(logits, targets)
                    weighted_balance = (
                        config.router_balance_coefficient * balance_loss
                    )
                    total_loss = (
                        language_model_loss + weighted_balance
                    ) / config.gradient_accumulation_steps

                if not bool(torch.isfinite(total_loss.detach()).item()):
                    raise FloatingPointError(
                        "training produced a non-finite loss before backward; "
                        "the optimizer step was not applied"
                    )
                scaler.scale(total_loss).backward()
                step_balance_loss += balance_loss.detach().item()
                step_tokens += targets.numel()

            scaler.unscale_(optimizer)
            if capture_module_diagnostics:
                parameter_snapshot = _snapshot_module_parameters(model)
                gradient_norms = _module_gradient_norms(model)
            if emc_diagnostics is not None:
                emc_diagnostics.observe_router_gradients(model)
                emc_diagnostics.observe_module_gradients(model)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            if parameter_snapshot is not None and gradient_norms is not None:
                latest_module_diagnostics = _module_update_diagnostics(
                    model,
                    parameter_snapshot,
                    gradient_norms,
                    step,
                )
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                raise RuntimeError(
                    "CUDA out of memory during EMC training; reduce --batch-size "
                    "or --sequence-length and resume from the latest checkpoint"
                ) from error
            raise

        tokens_processed += step_tokens
        completed_steps_this_run += 1
        cumulative_balance_loss += (
            step_balance_loss / config.gradient_accumulation_steps
        )

        if evaluation_due:
            evaluation_started = time.perf_counter()
            training_loss, _, training_token_accuracy = evaluate_model_metrics(
                model,
                corpus,
                config,
                split="train",
                generator=evaluation_generator,
            )
            (
                validation_loss,
                validation_perplexity,
                validation_token_accuracy,
            ) = evaluate_model_metrics(
                model,
                corpus,
                config,
                split="validation",
                generator=evaluation_generator,
            )
            evaluation_seconds += time.perf_counter() - evaluation_started
            elapsed = time.perf_counter() - started
            training_seconds = max(elapsed - evaluation_seconds, 1e-12)
            average_balance = cumulative_balance_loss / completed_steps_this_run
            gpu_memory_used = (
                torch.cuda.memory_allocated(device)
                if device.type == "cuda"
                else 0
            )
            gpu_peak_memory = (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0
            )
            metrics = TrainingMetrics(
                step=step,
                tokens_processed=tokens_processed,
                training_loss=training_loss,
                validation_loss=validation_loss,
                validation_perplexity=validation_perplexity,
                training_token_accuracy=training_token_accuracy,
                validation_token_accuracy=validation_token_accuracy,
                router_balance_loss=average_balance,
                weighted_balance_contribution=(
                    config.router_balance_coefficient * average_balance
                ),
                tokens_per_second=(tokens_processed - initial_tokens)
                / training_seconds,
                elapsed_seconds=elapsed,
                gpu_memory_used_bytes=gpu_memory_used,
                gpu_peak_memory_bytes=gpu_peak_memory,
            )
            history.append(metrics)
            is_best = validation_loss < best_validation_loss
            if is_best:
                best_validation_loss = validation_loss

            if telemetry is not None and crossed_milestones:
                if checkpoint_directory is None:
                    raise RuntimeError("telemetry checkpoint directory disappeared")
                base_record = telemetry.snapshot(
                    model,
                    milestone_tokens=crossed_milestones[0],
                    observed_tokens=tokens_processed,
                    step=step,
                    active_top_k=_active_top_k(model),
                    module_signals=latest_module_diagnostics,
                )
                for milestone in crossed_milestones:
                    record = deepcopy(base_record)
                    record["milestone_tokens"] = milestone
                    record["metrics"] = asdict(metrics)
                    record["checkpoint"] = (
                        str(
                            checkpoint_directory
                            / _checkpoint_filename(
                                config.checkpoint_prefix, str(milestone)
                            )
                        )
                        if config.retain_milestone_checkpoints
                        else None
                    )
                    telemetry_path = str(
                        write_developmental_record(checkpoint_directory, record)
                    )

            if latest_checkpoint is not None:
                checkpoint_arguments = {
                    "model": model,
                    "optimizer": optimizer,
                    "tokenizer": corpus.tokenizer,
                    "step": step,
                    "tokens_processed": tokens_processed,
                    "validation_loss": validation_loss,
                    "best_validation_loss": best_validation_loss,
                    "training_config": asdict(config),
                    "train_generator_state": train_generator.get_state(),
                    "evaluation_generator_state": evaluation_generator.get_state(),
                    "training_diagnostics": latest_module_diagnostics,
                }
                save_training_checkpoint(latest_checkpoint, **checkpoint_arguments)
                if is_best and best_checkpoint is not None:
                    save_training_checkpoint(best_checkpoint, **checkpoint_arguments)
                if config.retain_milestone_checkpoints:
                    if checkpoint_directory is None:
                        raise RuntimeError("milestone checkpoint directory disappeared")
                    for milestone in crossed_milestones:
                        milestone_path = checkpoint_directory / _checkpoint_filename(
                            config.checkpoint_prefix, str(milestone)
                        )
                        save_training_checkpoint(
                            milestone_path, **checkpoint_arguments
                        )
                        milestone_checkpoints.append(str(milestone_path))
                        if config.milestone_causal_diagnostics:
                            _run_milestone_causal_diagnostics(
                                milestone_path,
                                checkpoint_directory,
                                milestone,
                                config,
                            )
            if print_progress:
                gpu_suffix = (
                    f" | gpu {gpu_memory_used / 2**30:.2f} GiB "
                    f"(peak {gpu_peak_memory / 2**30:.2f} GiB)"
                    if device.type == "cuda"
                    else ""
                )
                print(
                    f"step {step:>6} | tokens {tokens_processed:>12,} | "
                    f"train_lm {training_loss:.4f} | "
                    f"balance {metrics.router_balance_loss:.5f} | "
                    f"weighted {metrics.weighted_balance_contribution:.5f} | "
                    f"validation {validation_loss:.4f} | "
                    f"ppl {validation_perplexity:.2f} | "
                    f"{metrics.tokens_per_second:,.0f} tok/s | {elapsed:.1f}s"
                    f"{gpu_suffix}"
                )
            if evaluation_callback is not None:
                evaluation_callback(step, model, metrics)

    final_metrics = history[-1]
    routing = emc_diagnostics.report(model) if emc_diagnostics is not None else None
    return TrainingResult(
        history=tuple(history),
        steps_completed=final_metrics.step,
        tokens_processed=final_metrics.tokens_processed,
        final_training_loss=final_metrics.training_loss,
        final_validation_loss=final_metrics.validation_loss,
        final_validation_perplexity=final_metrics.validation_perplexity,
        final_training_token_accuracy=final_metrics.training_token_accuracy,
        final_validation_token_accuracy=final_metrics.validation_token_accuracy,
        best_validation_loss=best_validation_loss,
        average_router_balance_loss=final_metrics.router_balance_loss,
        average_weighted_balance_contribution=(
            final_metrics.weighted_balance_contribution
        ),
        tokens_per_second=final_metrics.tokens_per_second,
        elapsed_seconds=final_metrics.elapsed_seconds,
        gpu_memory_used_bytes=final_metrics.gpu_memory_used_bytes,
        gpu_peak_memory_bytes=final_metrics.gpu_peak_memory_bytes,
        routing=routing,
        latest_checkpoint=(str(latest_checkpoint) if latest_checkpoint else None),
        best_checkpoint=(str(best_checkpoint) if best_checkpoint else None),
        module_diagnostics=latest_module_diagnostics or None,
        milestone_checkpoints=tuple(milestone_checkpoints),
        telemetry_path=telemetry_path,
    )



def _snapshot_module_parameters(model: nn.Module) -> list[list[Tensor]]:
    return [
        [parameter.detach().clone() for parameter in module.parameters()]
        for module in model.emc_modules
    ]


def _module_gradient_norms(model: nn.Module) -> list[float | None]:
    norms: list[float | None] = []
    for module in model.emc_modules:
        gradients = [
            parameter.grad.detach()
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        norms.append(
            sum(
                gradient.float().square().sum().item()
                for gradient in gradients
            )
            ** 0.5
            if gradients
            else None
        )
    return norms


def _module_update_diagnostics(
    model: nn.Module,
    before: list[list[Tensor]],
    gradient_norms: list[float | None],
    step: int,
) -> dict[str, Any]:
    modules = []
    for index, (module, previous, gradient_norm) in enumerate(
        zip(model.emc_modules, before, gradient_norms, strict=True)
    ):
        current = list(module.parameters())
        update_squared = sum(
            (parameter.detach().float() - old.float()).square().sum().item()
            for parameter, old in zip(current, previous, strict=True)
        )
        modules.append(
            {
                "module": index,
                "family": model.module_families[index],
                "gradient_norm": gradient_norm,
                "parameter_norm": sum(
                    parameter.detach().float().square().sum().item()
                    for parameter in current
                )
                ** 0.5,
                "update_norm": update_squared**0.5,
            }
        )
    return {
        "source": "live_training",
        "step": step,
        "optimizer_state_available": True,
        "sampling": (
            "single unscaled gradient and optimizer update norm sampled at "
            "checkpoint/evaluation interval; no gradient history retained"
        ),
        "modules": modules,
    }


def _checkpoint_filename(prefix: str, label: str) -> str:
    return f"{prefix}-{label}.pt" if prefix else f"{label}.pt"


def _active_top_k(model: nn.Module) -> int | None:
    value = getattr(model, "active_top_k", None)
    return int(value) if value is not None else None


def _apply_top_k_schedule(
    model: nn.Module,
    schedule: tuple[tuple[int, int], ...] | None,
    tokens_processed: int,
) -> None:
    if schedule is None:
        return
    setter = getattr(model, "set_active_top_k", None)
    if setter is None:
        raise ValueError("top-K schedules require an EMC model")
    active = schedule[0][1]
    for transition_tokens, top_k in schedule:
        if tokens_processed < transition_tokens:
            break
        active = top_k
    setter(active)


def _run_milestone_causal_diagnostics(
    checkpoint: Path,
    checkpoint_directory: Path,
    milestone: int,
    config: TrainingConfig,
) -> None:
    from .evaluate import DiagnosticEvaluationConfig, evaluate_checkpoint

    diagnostic_directory = (
        checkpoint_directory / "milestone-diagnostics" / str(milestone)
    )
    random_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )
    try:
        report = evaluate_checkpoint(
            checkpoint,
            diagnostic_directory,
            DiagnosticEvaluationConfig(
                seed=config.seed,
                examples_per_capability=max(
                    4, config.milestone_causal_sample_size
                ),
                ablation_examples_per_capability=(
                    config.milestone_causal_sample_size
                ),
                run_module_ablations=True,
                run_family_ablations=True,
                run_zero_proposal=False,
                run_forced_alternatives=False,
                max_notable_examples=0,
                device=config.device,
                precision=config.precision,
                smoke=True,
            ),
        )
        attach_causal_report(checkpoint_directory, milestone, report)
    finally:
        torch.random.set_rng_state(random_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

def _resolved_precision(precision: str, device: torch.device) -> str:
    if precision == "auto":
        return "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            "fp16" if device.type == "cuda" else "fp32"
        )
    if device.type != "cuda" and precision != "fp32":
        raise ValueError("fp16 and bf16 precision require a CUDA device")
    return precision


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)
