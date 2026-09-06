"""Alternating snapshot, development, shared-state, and router learning blocks."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import time

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .checkpoint import load_training_checkpoint, save_training_checkpoint
from .model import EMCOutput
from .value_routing import CounterfactualValueEMC, StateBatch, center


@contextmanager
def trainable_only(model: nn.Module, parameters):
    selected = {id(p) for p in parameters}
    flags = [(p, p.requires_grad) for p in model.parameters()]
    try:
        for p, _ in flags:
            p.requires_grad_(id(p) in selected)
        yield
    finally:
        for p, enabled in flags:
            p.requires_grad_(enabled)


def frozen_snapshot(model: CounterfactualValueEMC) -> CounterfactualValueEMC:
    # A distinct copy is crucial when the inserted expert is selected again later.
    snapshot = deepcopy(model).eval().requires_grad_(False)
    for p in snapshot.parameters():
        p.grad = None
    return snapshot


def sample_probe_pairs(batch_size: int, steps: int, rate: float, budget: int,
                       generator: torch.Generator) -> list[tuple[int, int]]:
    # Pick the depth before budget truncation; shuffle requests instead of taking
    # the first surviving indices. No preferential allocation to cycle zero.
    depths = torch.randint(steps, (batch_size,), generator=generator)
    eligible = torch.rand(batch_size, generator=generator) < rate
    order = torch.randperm(batch_size, generator=generator)
    return [(int(i), int(depths[i])) for i in order[eligible[order]][:budget]]


def curriculum_probabilities(snapshot: CounterfactualValueEMC, states: list[StateBatch],
                             common_fraction: float, temperature: float) -> Tensor:
    with torch.no_grad():
        log_responsibility = torch.cat([
            torch.log_softmax(-snapshot.router(s.latent, snapshot.config.resolved_trajectory_steps - s.depth).float() / temperature, -1)
            for s in states
        ])
        # Normalize in log space: an incumbent's large advantage must not make
        # a challenger's whole curriculum underflow to zero probability.
        specialist = torch.softmax(log_responsibility, dim=0)
        return common_fraction / log_responsibility.size(0) + (1 - common_fraction) * specialist


def insertion_loss(live_expert: nn.Module, snapshot: CounterfactualValueEMC,
                   batch: StateBatch) -> Tensor:
    """One live insertion; frozen suffix parameters still differentiate w.r.t. state."""
    prestate = batch.latent.detach()
    proposal = live_expert(prestate).unsqueeze(2)
    integrated = snapshot.integrator(prestate, proposal, prestate.new_ones(*prestate.shape[:2], 1))
    # DO NOT put this continuation inside no_grad/inference_mode.
    terminal = snapshot.continue_state(integrated, batch.depth + 1)
    return F.cross_entropy(snapshot.read_endpoint(terminal).float(), batch.targets)


class ValueOptimizers:
    """Checkpoint all optimizer clocks, sampling RNG and opportunity counters."""

    def __init__(self, model: CounterfactualValueEMC, learning_rate: float, weight_decay: float, seed: int):
        expert_ids = {id(p) for p in model.emc_modules.parameters()}
        router_ids = {id(p) for p in model.router.parameters()}
        self.shared_parameters = [p for p in model.parameters() if id(p) not in expert_ids | router_ids]
        self.router_parameters = list(model.router.parameters())
        self.expert_parameters = [list(e.parameters()) for e in model.emc_modules]
        make = lambda ps: torch.optim.AdamW(ps, lr=learning_rate, weight_decay=weight_decay)
        self.shared = make(self.shared_parameters)
        self.router = make(self.router_parameters)
        self.experts = [make(ps) for ps in self.expert_parameters]
        self.generator = torch.Generator().manual_seed(seed + 701)
        self.version = 0
        self.total_probes = 0
        self.reference_router_state = None
        self.loss_sums = torch.zeros(model.config.resolved_trajectory_steps, model.config.num_modules, dtype=torch.float64)
        self.loss_counts = torch.zeros(model.config.resolved_trajectory_steps, dtype=torch.long)
        self.updates = [0] * len(self.experts)
        self.items = [0] * len(self.experts)
        self.costs = dict(collection_expert_items=0, probe_expert_items=0, development_expert_items=0,
                          shared_expert_items=0, audit_expert_items=0, evaluation_expert_items=0, input_tokens=0, supervised_targets=0)

    def state_dict(self):
        return dict(kind="counterfactual_value_v1", shared=self.shared.state_dict(), router=self.router.state_dict(),
                    experts=[o.state_dict() for o in self.experts], generator=self.generator.get_state(),
                    total_probes=self.total_probes, reference_router_state=self.reference_router_state,
                    loss_sums=self.loss_sums, loss_counts=self.loss_counts,
                    version=self.version, updates=self.updates, items=self.items, costs=self.costs)

    def load_state_dict(self, state):
        if state.get("kind") != "counterfactual_value_v1" or len(state["experts"]) != len(self.experts):
            raise ValueError("incompatible value optimizer checkpoint")
        self.total_probes = state.get("total_probes", 0)
        self.loss_sums = state.get("loss_sums", self.loss_sums).cpu()
        self.loss_counts = state.get("loss_counts", self.loss_counts).cpu()
        self.reference_router_state = state.get("reference_router_state")
        self.shared.load_state_dict(state["shared"])
        self.router.load_state_dict(state["router"])
        for optimizer, saved in zip(self.experts, state["experts"], strict=True):
            optimizer.load_state_dict(saved)
        self.generator.set_state(state["generator"].cpu())
        self.version, self.updates, self.items, self.costs = state["version"], state["updates"], state["items"], state["costs"]


def checked_step(loss: Tensor, optimizer, parameters: list, clip: float) -> float:
    if not torch.isfinite(loss.detach()):
        raise FloatingPointError("non-finite value-training loss")
    loss.backward()
    norm = nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def train_block(model: CounterfactualValueEMC, optimizers: ValueOptimizers, inputs: Tensor, targets: Tensor,
                step: int, clip: float, reference: CounterfactualValueEMC | None = None) -> dict:
    c = model.config
    T, E, B = c.resolved_trajectory_steps, c.num_modules, inputs.size(0)
    endpoint_targets = targets[:, -1]
    # A fresh snapshot per block: states, labels and all insertion suffixes share
    # one immutable version. No old latents are reused after shared weights move.
    snapshot = reference if reference is not None else frozen_snapshot(model)
    calibrating = step <= c.value_calibration_steps or optimizers.total_probes < c.value_calibration_min_probes
    exploration = 1.0 if calibrating or reference is not None else c.value_exploration_rate
    states = snapshot.collect_states(inputs, endpoint_targets, generator=optimizers.generator,
                                     exploration=exploration)
    optimizers.costs["collection_expert_items"] += B * (T - 1)
    pairs = sample_probe_pairs(B, T, 1.0 if calibrating else c.value_probe_rate, c.value_probe_budget, optimizers.generator)
    probes = []
    for request, depth in pairs:
        s = states[depth]
        batch = StateBatch(s.latent[request:request+1], s.targets[request:request+1], depth)
        labels = snapshot.counterfactual_losses(batch, target=c.value_target)
        probes.append((batch, labels))
        optimizers.total_probes += 1
        optimizers.loss_sums[depth] += labels.detach().double().cpu().sum(0)
        optimizers.loss_counts[depth] += labels.size(0)
        optimizers.costs["probe_expert_items"] += E * (T - depth if c.value_target == "suffix" else 1)
    grad_norms = [0.0] * E
    parameter_changes = [0.0] * E
    development_losses = [None] * E
    model.zero_grad(set_to_none=True)
    if c.value_expert_training == "controlled" and step % c.value_development_interval == 0:
        rho = 1.0 if step <= c.value_warmup_steps else c.value_common_fraction
        probabilities = curriculum_probabilities(snapshot, states, rho, c.value_specialist_temperature)
        for e, (expert, optimizer) in enumerate(zip(model.emc_modules, optimizers.experts, strict=True)):
            chosen = torch.multinomial(probabilities[:, e].cpu(), c.value_development_batch_size,
                                       replacement=True, generator=optimizers.generator).to(inputs.device)
            optimizer.zero_grad(set_to_none=True)
            total = inputs.new_zeros((), dtype=torch.float32)
            before = [p.detach().clone() for p in expert.parameters()]
            with trainable_only(model, optimizers.expert_parameters[e]):
                for depth, state in enumerate(states):
                    rows = chosen[chosen // B == depth] % B
                    if not rows.numel():
                        continue
                    batch = StateBatch(state.latent[rows], state.targets[rows], depth)
                    loss = insertion_loss(expert, snapshot, batch) * rows.numel() / chosen.numel()
                    total = total + loss
                    optimizers.costs["development_expert_items"] += rows.numel() * (T - depth)
                grad_norms[e] = checked_step(total, optimizer, optimizers.expert_parameters[e], clip)
            parameter_changes[e] = math.sqrt(sum(float((p.detach() - old).float().square().sum())
                                                for p, old in zip(expert.parameters(), before, strict=True)))
            optimizers.updates[e] += 1
            optimizers.items[e] += chosen.numel()
            development_losses[e] = float(total.detach())
            optimizer.zero_grad(set_to_none=True)
    # Shared embedding/readout/gate see exploratory candidate paths. In the
    # controlled arm this graph cannot accumulate expert parameter gradients.
    shared_norm = 0.0
    optimizers.shared.zero_grad(set_to_none=True)
    shared_params = optimizers.shared_parameters
    ordinary = c.value_expert_training == "ordinary"
    active = shared_params + (list(model.emc_modules.parameters()) if ordinary else [])
    with trainable_only(model, active if c.value_expert_training != "frozen" else []):
        output = model.endpoint(inputs, return_trace=True, exploration=exploration,
                                generator=optimizers.generator)
        loss = F.cross_entropy(output.logits[:, -1].float(), endpoint_targets)
        if c.value_expert_training != "frozen":
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError("non-finite shared loss")
            loss.backward()
            shared_norm = float(nn.utils.clip_grad_norm_(active, clip, error_if_nonfinite=True))
            optimizers.shared.step()
            if ordinary:
                for e, optimizer in enumerate(optimizers.experts):
                    if any(p.grad is not None for p in optimizers.expert_parameters[e]):
                        optimizer.step()
                        optimizers.updates[e] += 1
                        optimizers.items[e] += sum(int((trace.selected_indices == e).sum()) for trace in output.trace)
    optimizers.costs["shared_expert_items"] += B * T
    model.zero_grad(set_to_none=True)
    calibration = None
    if probes:
        optimizers.router.zero_grad(set_to_none=True)
        with trainable_only(model, optimizers.router_parameters):
            fit = torch.stack([model.router.calibration_loss(s.latent, T - s.depth, labels) for s, labels in probes]).mean()
            checked_step(fit, optimizers.router, optimizers.router_parameters, clip)
            calibration = float(fit.detach())
    model.zero_grad(set_to_none=True)
    optimizers.costs["input_tokens"] += inputs.numel()
    optimizers.costs["supervised_targets"] += B
    label_version = optimizers.version
    optimizers.version += 1
    return dict(output=output, loss=float(loss.detach()), gradient_norm=shared_norm,
                calibration_mse=calibration, training_probe_count=len(probes), label_snapshot_version=label_version,
                router_phase="calibration" if calibrating else "learning",
                total_training_probes=optimizers.total_probes, collection_exploration=exploration,
                fixed_reference=reference is not None,
                expert_update_batches=list(optimizers.updates), expert_training_items=list(optimizers.items),
                expert_gradient_norms=grad_norms, expert_parameter_change_norms=parameter_changes,
                expert_development_losses=development_losses, costs=dict(optimizers.costs))


@torch.no_grad()
def audit_values(model: CounterfactualValueEMC, inputs: Tensor, targets: Tensor,
                 reference: CounterfactualValueEMC | None = None,
                 constant_choices: list[int] | None = None) -> dict:
    """Held-out full-candidate suffix audit, never reused as router training labels."""
    snapshot = reference if reference is not None else frozen_snapshot(model)
    states = snapshot.collect_states(inputs, targets[:, -1], generator=torch.Generator().manual_seed(0),
                                     exploration=1.0 if reference is not None else 0.0)
    rows = []
    for s in states:
        losses = snapshot.counterfactual_losses(s, target="suffix")
        predicted = model.router(s.latent, model.config.resolved_trajectory_steps - s.depth).float()
        chosen = predicted.argmin(-1)
        chosen_loss = losses.gather(1, chosen[:, None]).squeeze(1)
        gaps = losses[:, :, None] - losses[:, None, :]
        errors = (predicted[:, :, None] - predicted[:, None, :]) - gaps
        rows.append(dict(depth=s.depth, probe_count=inputs.size(0),
                         regret=float((chosen_loss - losses.min(-1).values).mean()),
                         gap_rmse=float(errors.square().mean().sqrt()),
                         advantage_rmse=float((predicted - center(losses)).square().mean().sqrt()),
                         constant_regret=(float((losses[:, constant_choices[s.depth]] - losses.min(-1).values).mean())
                                          if constant_choices is not None else None),
                         constant_expert=constant_choices[s.depth] if constant_choices is not None else None,
                         uniform_regret=float((losses.mean(-1) - losses.min(-1).values).mean()),
                         expert_mean_losses=losses.mean(0).tolist(),
                         oracle_loss=float(losses.min(-1).values.mean()),
                         chosen_loss=float(chosen_loss.mean()),
                         near_tie_accuracy=float((chosen_loss <= losses.min(-1).values + 1e-3).float().mean()),
                         route_counts=torch.bincount(chosen, minlength=model.config.num_modules).tolist()))
    return dict(by_depth=rows, mean_regret=sum(r["regret"] for r in rows) / len(rows),
                gap_rmse=math.sqrt(sum(r["gap_rmse"]**2 for r in rows) / len(rows)),
                constant_regret=(sum(r["constant_regret"] for r in rows) / len(rows) if constant_choices is not None else None),
                uniform_regret=sum(r["uniform_regret"] for r in rows) / len(rows),
                probe_count=inputs.size(0) * len(rows),
                target="fixed_reference_suffix" if reference is not None else "held_out_suffix")


def train_value_model(model, corpus, config, *, print_progress=True, evaluation_callback=None,
                      progress_callback=None, progress_callback_interval=1, cancellation_callback=None):
    from .training import TrainingCancelledError, TrainingMetrics, TrainingResult, _autocast_context, _live_routing_snapshot
    if progress_callback_interval <= 0:
        raise ValueError("progress_callback_interval must be positive")
    if config.top_k_schedule is not None or config.milestone_causal_diagnostics:
        raise ValueError("value trainer uses latest/best checkpoints and suffix audits; disable legacy milestone/Top-K controls")
    if config.precision == "fp16":
        raise ValueError("value training supports fp32/bf16; fp16 skipped updates would confound controlled opportunity")
    device = torch.device(config.device)
    precision = config.precision
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 value training requires a CUDA device with bf16 support; choose fp32")
    model.to(device).train()
    opts = ValueOptimizers(model, config.learning_rate, config.weight_decay, config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    evaluation_generator = torch.Generator().manual_seed(config.seed + 1)
    start_step, tokens, best = 0, 0, math.inf
    if config.resume_from:
        progress = load_training_checkpoint(config.resume_from, model=model, optimizer=opts, device=device)
        start_step, tokens, best = progress.step, progress.tokens_processed, progress.best_validation_loss
        generator.set_state(progress.train_generator_state.cpu())
        evaluation_generator.set_state(progress.evaluation_generator_state.cpu())
    reference = None
    if model.config.value_expert_training == "frozen" and model.config.value_fixed_reference:
        reference = frozen_snapshot(model)
        source = getattr(model, "_value_reference_router", model.router)
        reference.router.load_state_dict(opts.reference_router_state or source.state_dict())
        reference.router.to(device).eval().requires_grad_(False)
        opts.reference_router_state = {k: v.detach().cpu().clone() for k, v in reference.router.state_dict().items()}
    # Resolve/compile before throughput timing, rather than charging build time
    # to the first training block. Automatic fallback is visible in telemetry.
    from .ssm_scan import resolve_backend
    ssm_backends = {}
    for name, expert in zip(model.expert_names, model.emc_modules):
        if hasattr(expert, "ssm_backend"):
            ssm_backends[name] = resolve_backend(expert.ssm_backend, device)
    initial_audit = None
    if reference is not None:
        x, y = corpus.sample_batch("validation", config.batch_size * config.evaluation_batches,
                                  config.sequence_length, generator=torch.Generator().manual_seed(config.seed + 1901), device=device)
        with torch.no_grad(), _autocast_context(device, precision):
            initial_audit = audit_values(model, x, y, reference)
    # For this objective a training token is one supervised endpoint, not one
    # token merely read as context. Accumulation grows the block's state pool.
    batch_size = config.batch_size * config.gradient_accumulation_steps
    total = math.ceil(config.train_tokens / batch_size) if config.train_tokens else config.steps
    if start_step >= total:
        raise ValueError("checkpoint already exhausted the configured endpoint budget")
    history, started, initial_tokens = [], time.perf_counter(), tokens
    latest_path = best_path = None
    if config.checkpoint_directory:
        root = Path(config.checkpoint_directory)
        latest_path = root / f"{config.checkpoint_prefix}-latest.pt"
        best_path = root / f"{config.checkpoint_prefix}-best.pt" if config.retain_best_checkpoint else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    last = {}
    audit = None
    for step in range(start_step + 1, total + 1):
        if cancellation_callback and cancellation_callback():
            raise TrainingCancelledError("value experiment cancelled at a completed alternating-block boundary")
        step_started = time.perf_counter()
        count = min(batch_size, config.train_tokens - tokens) if config.train_tokens else batch_size
        inputs, targets = corpus.sample_batch("train", count, config.sequence_length, generator=generator, device=device)
        with _autocast_context(device, precision):
            block = train_block(model, opts, inputs, targets, step, config.gradient_clip_norm, reference)
        tokens += count
        elapsed = time.perf_counter() - started
        output = block.pop("output")
        route = _live_routing_snapshot(output, model)
        block["initial_held_out"] = initial_audit
        block["ssm_backends"] = ssm_backends
        route["value_routing"] = {k: v for k, v in block.items() if k != "loss"}
        if audit is not None:
            route["value_routing"]["held_out"] = audit
        values = dict(step=step, tokens_processed=tokens, target_tokens=config.train_tokens,
                      objective="prefix_endpoint", throughput_unit="endpoint/s",
                      endpoints_per_second=(tokens-initial_tokens)/max(elapsed, 1e-9),
                      context_tokens_per_second=(tokens-initial_tokens)*config.sequence_length/max(elapsed, 1e-9),
                      context_length=config.sequence_length, training_loss=block["loss"],
                      gradient_norm=block["gradient_norm"], learning_rate=config.learning_rate,
                      step_time_seconds=time.perf_counter() - step_started,
                      tokens_per_second=(tokens-initial_tokens)/max(elapsed, 1e-9), elapsed_seconds=elapsed,
                      gpu_memory_used_bytes=torch.cuda.memory_allocated(device) if device.type == "cuda" else 0,
                      gpu_peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                      routing=route, **{k: v for k, v in block.items() if k not in {"loss", "gradient_norm"}})
        evaluation_due = step % config.evaluation_interval == 0 or step == total
        if progress_callback and step % progress_callback_interval == 0 and not evaluation_due:
            progress_callback(step, model, values)
        last = block
        if not evaluation_due:
            continue
        model.eval()
        measured = {}
        with torch.no_grad(), _autocast_context(device, precision):
            for split in ("train", "validation"):
                losses, correct, n = [], 0, 0
                for _ in range(config.evaluation_batches):
                    x, y = corpus.sample_batch(split, config.batch_size, config.sequence_length,
                                               generator=evaluation_generator, device=device)
                    logits = model.endpoint(x)[:, -1].float()
                    losses.append(float(F.cross_entropy(logits, y[:, -1])))
                    correct += int((logits.argmax(-1) == y[:, -1]).sum())
                    n += x.size(0)
                measured[split] = (sum(losses)/len(losses), correct/n)
            # Fixed held-out raw prefixes, freshly encoded at each checkpoint.
            # Audit targets never enter router training or specialist sampling.
            x, y = corpus.sample_batch("validation", config.batch_size * config.evaluation_batches,
                                       config.sequence_length,
                                       generator=torch.Generator().manual_seed(config.seed + 1901), device=device)
            constant_choices = (opts.loss_sums.argmin(-1).tolist()
                                if reference is not None and bool((opts.loss_counts > 0).all()) else None)
            audit = audit_values(model, x, y, reference, constant_choices)
        T = model.config.resolved_trajectory_steps
        opts.costs["evaluation_expert_items"] += 2 * config.evaluation_batches * config.batch_size * T
        opts.costs["audit_expert_items"] += x.size(0) * (T - 1 + model.config.num_modules*T*(T+1)//2)
        audit["snapshot_version"] = opts.version
        last["held_out"] = audit
        last["costs"] = dict(opts.costs)
        values["evaluation_completed"] = True
        values["routing"]["value_routing"].update(held_out=audit, costs=dict(opts.costs))
        elapsed = time.perf_counter() - started
        rate = (tokens - initial_tokens) / max(elapsed, 1e-9)
        values.update(elapsed_seconds=elapsed, tokens_per_second=rate, endpoints_per_second=rate,
                      context_tokens_per_second=rate * config.sequence_length,
                      step_time_seconds=time.perf_counter() - step_started)
        # Emit fresh audit even when validation and telemetry cadences differ.
        if progress_callback:
            progress_callback(step, model, values)
        train_loss, train_acc = measured["train"]
        val_loss, val_acc = measured["validation"]
        elapsed = time.perf_counter() - started
        metrics = TrainingMetrics(step, tokens, train_loss, val_loss, math.exp(min(val_loss, 20)), train_acc, val_acc,
                                  0., 0., (tokens-initial_tokens)/max(elapsed,1e-9), elapsed,
                                  values["gpu_memory_used_bytes"], values["gpu_peak_memory_bytes"])
        history.append(metrics)
        improved = val_loss < best
        best = min(best, val_loss)
        if latest_path:
            args = dict(model=model, optimizer=opts, tokenizer=corpus.tokenizer, step=step, tokens_processed=tokens,
                        validation_loss=val_loss, best_validation_loss=best,
                        training_config={**asdict(config), "objective": "prefix_endpoint"},
                        train_generator_state=generator.get_state(), evaluation_generator_state=evaluation_generator.get_state(),
                        training_diagnostics={"value_routing": last})
            save_training_checkpoint(latest_path, **args)
            if improved and best_path:
                save_training_checkpoint(best_path, **args)
        if evaluation_callback:
            evaluation_callback(step, model, metrics)
        if print_progress:
            print(f"block {step} | endpoints {tokens} | validation {val_loss:.4f} | suffix regret {audit['mean_regret']:.5f}", flush=True)
        model.train()
    m = history[-1]
    return TrainingResult(tuple(history), m.step, tokens, m.training_loss, m.validation_loss, m.validation_perplexity,
                          best, m.training_token_accuracy, m.validation_token_accuracy, 0., 0., m.tokens_per_second,
                          m.elapsed_seconds, m.gpu_memory_used_bytes, m.gpu_peak_memory_bytes, None,
                          str(latest_path) if latest_path else None, str(best_path) if best_path else None,
                          {"value_routing": last, "objective": "prefix_endpoint"}, (), None)
