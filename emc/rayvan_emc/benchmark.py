from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn

from .diagnostics import EMCDiagnostics, count_parameters, parameter_counts
from .chunked import ChunkedEMCModel
from .integrator import Integrator
from .model import EMCConfig, EMCModel, EMCOutput
from .modules import create_emc_module
from .nexus import ModuleAwareNexusRouter, NexusRouter
from .training import next_token_loss
from .experiments.common import create_emc_model, create_n2_model


@dataclass(frozen=True)
class TimingResult:
    milliseconds: float
    tokens_per_second: float


class SyntheticBatchSource:
    def __init__(self, vocab_size: int, length: int = 1_000_000) -> None:
        generator = torch.Generator().manual_seed(123)
        self.tokens = torch.randint(
            0, vocab_size, (length,), generator=generator, dtype=torch.long
        )
        self.generator = torch.Generator().manual_seed(456)

    def sample(
        self,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        maximum_start = self.tokens.numel() - sequence_length - 1
        starts = torch.randint(
            maximum_start + 1,
            (batch_size,),
            generator=self.generator,
        ).tolist()
        inputs = torch.stack(
            [self.tokens[start : start + sequence_length] for start in starts]
        )
        targets = torch.stack(
            [
                self.tokens[start + 1 : start + sequence_length + 1]
                for start in starts
            ]
        )
        return inputs.to(device), targets.to(device)


class NvidiaSampler:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self) -> NvidiaSampler:
        try:
            self.process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                    "--format=csv,noheader,nounits",
                    "--loop-ms=100",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            self.process = None
        return self

    def __exit__(self, *_args: object) -> None:
        if self.process is not None:
            self.process.terminate()

    def finish(self) -> dict[str, float]:
        if self.process is None:
            return {}
        self.process.terminate()
        output, _ = self.process.communicate(timeout=5)
        samples: list[tuple[float, float, float]] = []
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 3:
                try:
                    samples.append(tuple(float(part) for part in parts))
                except ValueError:
                    continue
        if not samples:
            return {}
        gpu = [sample[0] for sample in samples]
        memory = [sample[1] for sample in samples]
        vram = [sample[2] for sample in samples]
        return {
            "samples": float(len(samples)),
            "gpu_utilization_average": sum(gpu) / len(gpu),
            "gpu_utilization_peak": max(gpu),
            "memory_utilization_average": sum(memory) / len(memory),
            "memory_utilization_peak": max(memory),
            "vram_used_average_mib": sum(vram) / len(vram),
            "vram_used_peak_mib": max(vram),
        }


class ComponentEventRecorder:
    def __init__(self, model: EMCModel) -> None:
        self.model = model
        self.events: list[tuple[str, int, torch.cuda.Event, torch.cuda.Event]] = []
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.cycle = -1
        self._starts: dict[int, list[tuple[int, torch.cuda.Event]]] = defaultdict(list)

    def reset(self) -> None:
        self.events.clear()
        self.cycle = -1
        self._starts.clear()

    def install(self) -> None:
        self._hook(self.model.token_embedding, "embedding/token")
        self._hook(self.model.position_embedding, "embedding/position")
        self._hook(self.model.router, "nexus", advances_cycle=True)
        for index, module in enumerate(self.model.emc_modules):
            self._hook(module, f"module/{index}/{module.family}")
        self._hook(self.model.integrator, "integrator")
        self._hook(self.model.output_norm, "output/norm")
        self._hook(self.model.output_projection, "output/projection")

    def _hook(
        self, module: nn.Module, label: str, *, advances_cycle: bool = False
    ) -> None:
        identity = id(module)

        def before(_module: nn.Module, _inputs: tuple[object, ...]) -> None:
            if advances_cycle:
                self.cycle += 1
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._starts[identity].append((self.cycle, event))

        def after(
            _module: nn.Module, _inputs: tuple[object, ...], _output: object
        ) -> None:
            cycle, start = self._starts[identity].pop()
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.events.append((label, cycle, start, end))

        self.handles.append(module.register_forward_pre_hook(before))
        self.handles.append(module.register_forward_hook(after))

    def totals(self) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for label, cycle, start, end in self.events:
            if label.startswith(("nexus", "module/", "integrator")):
                microbatch = cycle // self.model.config.num_cycles + 1
                cycle_in_microbatch = cycle % self.model.config.num_cycles + 1
                key = (
                    f"micro_{microbatch}/cycle_{cycle_in_microbatch}/{label}"
                )
            else:
                key = label
            totals[key] += start.elapsed_time(end)
        return dict(totals)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class TrainingStepper:
    def __init__(
        self,
        model: EMCModel | ChunkedEMCModel,
        *,
        batch_size: int,
        sequence_length: int,
        accumulation: int,
        precision: str,
        device: torch.device,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.accumulation = accumulation
        self.precision = precision
        self.device = device
        self.source = SyntheticBatchSource(model.config.vocab_size)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and precision == "fp16"
        )
        self.diagnostics = (
            EMCDiagnostics(model) if isinstance(model, EMCModel) else None
        )

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.sequence_length * self.accumulation

    def step(self, *, phase_timing: bool = False) -> dict[str, float]:
        phases: dict[str, float] = defaultdict(float)
        self.optimizer.zero_grad(set_to_none=True)
        balance_values: list[Tensor] = []
        for _ in range(self.accumulation):
            inputs, targets = _phase(
                "data_loading",
                phases,
                self.device,
                phase_timing,
                lambda: self.source.sample(
                    self.batch_size, self.sequence_length, self.device
                ),
            )
            output = _phase(
                "forward",
                phases,
                self.device,
                phase_timing,
                lambda: self._forward(inputs),
            )
            _phase(
                "trace_diagnostics",
                phases,
                self.device,
                phase_timing,
                lambda: (
                    self.diagnostics.observe_trace(output.trace)
                    if self.diagnostics is not None
                    else None
                ),
            )
            loss = _phase(
                "loss",
                phases,
                self.device,
                phase_timing,
                lambda: next_token_loss(output.logits, targets),
            )
            if output.router_balance_loss is None:
                raise RuntimeError("benchmark EMC output has no balance loss")
            balance_values.append(output.router_balance_loss)
            total = (
                loss + 0.01 * output.router_balance_loss
            ) / self.accumulation
            _phase(
                "backward",
                phases,
                self.device,
                phase_timing,
                lambda: self.scaler.scale(total).backward(),
            )

        def gradients() -> None:
            self.scaler.unscale_(self.optimizer)
            if self.diagnostics is not None:
                self.diagnostics.observe_router_gradients(self.model)
                self.diagnostics.observe_module_gradients(self.model)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        _phase(
            "gradient_diagnostics_and_clip",
            phases,
            self.device,
            phase_timing,
            gradients,
        )

        def optimizer_step() -> None:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        _phase(
            "optimizer",
            phases,
            self.device,
            phase_timing,
            optimizer_step,
        )
        _phase(
            "scalar_bookkeeping",
            phases,
            self.device,
            phase_timing,
            lambda: sum(value.detach().item() for value in balance_values),
        )
        return dict(phases)

    def _forward(self, inputs: Tensor) -> EMCOutput:
        with _autocast(self.device, self.precision):
            output = self.model(inputs, return_trace=True)
        if not isinstance(output, EMCOutput):
            raise RuntimeError("benchmark forward did not return EMCOutput")
        return output


def _phase(
    name: str,
    phases: dict[str, float],
    device: torch.device,
    enabled: bool,
    function: Callable[[], object],
):
    if not enabled:
        return function()
    _synchronize(device)
    started = time.perf_counter()
    result = function()
    _synchronize(device)
    phases[name] += (time.perf_counter() - started) * 1_000
    return result


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_iterations(
    function: Callable[[], None],
    *,
    iterations: int,
    tokens: int,
    device: torch.device,
) -> TimingResult:
    _synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    _synchronize(device)
    elapsed = time.perf_counter() - started
    milliseconds = elapsed * 1_000 / iterations
    return TimingResult(milliseconds, tokens / (milliseconds / 1_000))


def benchmark_module_families(
    config: EMCConfig,
    *,
    batch_size: int,
    sequence_length: int,
    precision: str,
    device: torch.device,
    iterations: int,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for family in ("gpt", "ssm", "recurrent"):
        module = create_emc_module(config, family).to(device)
        latent = torch.randn(
            batch_size,
            sequence_length,
            config.latent_dim,
            device=device,
            requires_grad=True,
        )

        def forward() -> None:
            with _autocast(device, precision):
                module(latent)

        def forward_backward() -> None:
            module.zero_grad(set_to_none=True)
            if latent.grad is not None:
                latent.grad = None
            with _autocast(device, precision):
                output = module(latent)
                loss = output.float().square().mean()
            loss.backward()

        for _ in range(2):
            forward_backward()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            baseline_memory = torch.cuda.memory_allocated(device)
        else:
            baseline_memory = 0
        forward_timing = _time_iterations(
            forward,
            iterations=iterations,
            tokens=batch_size * sequence_length,
            device=device,
        )
        combined_timing = _time_iterations(
            forward_backward,
            iterations=max(2, iterations // 2),
            tokens=batch_size * sequence_length,
            device=device,
        )
        peak_increment = (
            torch.cuda.max_memory_allocated(device) - baseline_memory
            if device.type == "cuda"
            else 0
        )
        utilization = _sample_utilization(
            forward_backward,
            device=device,
            minimum_seconds=0.75,
        )
        results[family] = {
            "parameters": float(count_parameters(module)),
            "forward_ms": forward_timing.milliseconds,
            "forward_tokens_per_second": forward_timing.tokens_per_second,
            "forward_backward_ms": combined_timing.milliseconds,
            "forward_backward_tokens_per_second": (
                combined_timing.tokens_per_second
            ),
            "estimated_flops": float(
                _estimate_module_flops(
                    family, config, batch_size, sequence_length
                )
            ),
            "peak_increment_mib": peak_increment / 2**20,
            **utilization,
        }
        del module, latent
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def _estimate_module_flops(
    family: str,
    config: EMCConfig,
    batch: int,
    sequence: int,
) -> int:
    d = config.latent_dim
    if family == "gpt":
        hidden = config.resolved_module_hidden_dim
        return batch * (
            8 * sequence * d * d
            + 4 * sequence * sequence * d
            + 4 * sequence * d * hidden
        )
    if family == "ssm":
        hidden = config.resolved_state_space_dim
        return batch * sequence * (
            4 * d * hidden + 6 * hidden * hidden + 12 * hidden
        )
    hidden = config.resolved_recurrent_dim
    return batch * sequence * (
        4 * d * hidden + 12 * hidden * hidden
    )


def benchmark_integrator_and_nexus(
    config: EMCConfig,
    *,
    batch_size: int,
    sequence_length: int,
    precision: str,
    device: torch.device,
    iterations: int,
) -> dict[str, dict[str, float]]:
    latent = torch.randn(
        batch_size,
        sequence_length,
        config.latent_dim,
        device=device,
        requires_grad=True,
    )
    proposals = torch.randn(
        batch_size,
        sequence_length,
        config.modules_per_cycle,
        config.latent_dim,
        device=device,
        requires_grad=True,
    )
    weights = torch.softmax(
        torch.randn(
            batch_size,
            sequence_length,
            config.modules_per_cycle,
            device=device,
        ),
        dim=-1,
    )
    results: dict[str, dict[str, float]] = {}
    for name, component, operation in (
        (
            "integrator",
            Integrator(config).to(device),
            lambda component: component(latent, proposals, weights),
        ),
        (
            "fixed_index_nexus",
            NexusRouter(config).to(device),
            lambda component: component(latent),
        ),
        (
            "descriptor_nexus",
            ModuleAwareNexusRouter(config).to(device),
            lambda component: component(latent),
        ),
    ):
        def forward(component: nn.Module = component) -> None:
            with _autocast(device, precision):
                operation(component)

        def forward_backward(component: nn.Module = component) -> None:
            component.zero_grad(set_to_none=True)
            latent.grad = None
            proposals.grad = None
            with _autocast(device, precision):
                result = operation(component)
                value = result.scores if hasattr(result, "scores") else result
                if isinstance(value, tuple):
                    value = value[0]
                loss = value.float().square().mean()
            loss.backward()

        for _ in range(3):
            forward()
        timing = _time_iterations(
            forward,
            iterations=iterations,
            tokens=batch_size * sequence_length,
            device=device,
        )
        combined_timing = _time_iterations(
            forward_backward,
            iterations=max(2, iterations // 2),
            tokens=batch_size * sequence_length,
            device=device,
        )
        results[name] = {
            "parameters": float(count_parameters(component)),
            "forward_ms": timing.milliseconds,
            "tokens_per_second": timing.tokens_per_second,
            "forward_backward_ms": combined_timing.milliseconds,
            "forward_backward_tokens_per_second": (
                combined_timing.tokens_per_second
            ),
        }
    return results


def _sample_utilization(
    function: Callable[[], None],
    *,
    device: torch.device,
    minimum_seconds: float,
) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    sampler = NvidiaSampler()
    sampler.__enter__()
    _synchronize(device)
    started = time.perf_counter()
    while time.perf_counter() - started < minimum_seconds:
        function()
    _synchronize(device)
    return {
        f"benchmark_{name}": value
        for name, value in sampler.finish().items()
    }


def routing_waste(
    model: EMCModel | ChunkedEMCModel, inputs: Tensor
) -> list[dict[str, object]]:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(inputs, return_trace=True)
    finally:
        model.train(was_training)
    if not isinstance(output, EMCOutput):
        raise RuntimeError("routing waste forward did not return trace")
    results: list[dict[str, object]] = []
    if output.chunk_trace is not None:
        return [
            {
                "chunk": trace.chunk_index,
                "active_modules": trace.active_modules.unique().tolist(),
                "computed_chunk_module_pairs": trace.computed_chunk_module_pairs,
                "retained_chunk_module_pairs": trace.retained_chunk_module_pairs,
                "retained_fraction": (
                    trace.retained_chunk_module_pairs
                    / trace.computed_chunk_module_pairs
                ),
                "wasted_fraction": 0.0,
            }
            for trace in output.chunk_trace.chunks
        ]
    token_count = inputs.numel()
    for trace in output.trace:
        if trace.selected_indices is None:
            continue
        selected = trace.selected_indices
        active = selected.unique().tolist()
        retained = selected.numel()
        computed = len(active) * token_count
        per_module = {
            int(index): int((selected == index).sum().item()) / token_count
            for index in active
        }
        results.append(
            {
                "cycle": trace.cycle,
                "active_modules": active,
                "computed_token_module_pairs": computed,
                "retained_token_module_pairs": retained,
                "retained_fraction": retained / computed,
                "wasted_fraction": 1.0 - retained / computed,
                "per_module_retained_fraction": per_module,
            }
        )
    return results


def benchmark_cycle_costs(
    model: EMCModel,
    inputs: Tensor,
    *,
    precision: str,
    device: torch.device,
    iterations: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    results: dict[str, float] = {}
    try:
        for cycles in range(1, model.config.num_cycles + 1):
            def forward() -> None:
                with torch.no_grad(), _autocast(device, precision):
                    model(inputs, evaluation_cycle_limit=cycles)

            for _ in range(2):
                forward()
            timing = _time_iterations(
                forward,
                iterations=iterations,
                tokens=inputs.numel(),
                device=device,
            )
            results[f"{cycles}_cycles_forward_ms"] = timing.milliseconds
    finally:
        model.train(was_training)
    return results


def benchmark_n2_forced_components(
    model: EMCModel,
    inputs: Tensor,
    *,
    precision: str,
    device: torch.device,
    iterations: int,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for pair in ((0, 1), (2, 3)):
        forced = torch.tensor(pair, device=device)
        recorder = ComponentEventRecorder(model)
        recorder.install()

        def forward() -> None:
            with torch.no_grad(), _autocast(device, precision):
                model(inputs, diagnostic_forced_modules=forced)

        try:
            for _ in range(2):
                forward()
            _synchronize(device)
            recorder.reset()
            timing = _time_iterations(
                forward,
                iterations=iterations,
                tokens=inputs.numel(),
                device=device,
            )
            component_times: dict[str, float] = defaultdict(float)
            for key, value in recorder.totals().items():
                label = key.split("/", 2)[-1] if key.startswith("micro_") else key
                component_times[label] += value / iterations
        finally:
            recorder.remove()
        results["-".join(map(str, pair))] = {
            "forward_ms": timing.milliseconds,
            **component_times,
        }
    return results


def _profiler_summary(profiler: torch.profiler.profile) -> dict[str, object]:
    events = profiler.events()
    cuda_events = [
        event
        for event in events
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    synchronization_events = [
        event.name
        for event in events
        if any(
            marker in event.name.lower()
            for marker in ("synchronize", "dtoh", "aten::item", "aten::_local_scalar")
        )
    ]
    allocation_names = {
        "aten::empty",
        "aten::empty_like",
        "aten::empty_strided",
        "aten::new_empty",
        "aten::new_zeros",
        "aten::zeros",
        "aten::zeros_like",
    }
    allocation_events = sum(event.name in allocation_names for event in events)
    return {
        "cuda_kernel_launches": len(cuda_events),
        "synchronization_events": len(synchronization_events),
        "synchronization_event_names": dict(
            sorted(
                (
                    name,
                    synchronization_events.count(name),
                )
                for name in set(synchronization_events)
            )
        ),
        "temporary_allocation_events": allocation_events,
        "total_event_count": len(events),
    }


def run_profiler(
    stepper: TrainingStepper,
    output_directory: Path,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if stepper.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    sample_inputs, _ = stepper.source.sample(
        stepper.batch_size, stepper.sequence_length, stepper.device
    )
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as forward_profiler:
        stepper._forward(sample_inputs)
    forward_trace_path = output_directory / "n2-forward.json"
    forward_profiler.export_chrome_trace(str(forward_trace_path))

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as step_profiler:
        stepper.step()
    step_trace_path = output_directory / "n2-training-step.json"
    step_profiler.export_chrome_trace(str(step_trace_path))
    sort_key = (
        "self_cuda_time_total"
        if stepper.device.type == "cuda"
        else "self_cpu_time_total"
    )
    table = step_profiler.key_averages().table(
        sort_by=sort_key,
        row_limit=30,
    )
    print("\nTorch profiler top operators")
    print(table)
    return {
        "forward_trace": str(forward_trace_path),
        "training_step_trace": str(step_trace_path),
        "operator_table": table,
        "forward": _profiler_summary(forward_profiler),
        "training_step": _profiler_summary(step_profiler),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile EMC N1 without changing training behavior"
    )
    parser.add_argument("--preset", choices=("quick", "research"), default="research")
    parser.add_argument(
        "--n1-stage",
        choices=(
            "baseline",
            "integrator",
            "heterogeneous",
            "n1",
            "n1_chunked",
            "n2",
        ),
        default="n1",
    )
    parser.add_argument("--module-population", default="mixed")
    parser.add_argument(
        "--n2-execution-mode", choices=("sparse", "dense"), default="sparse"
    )
    parser.add_argument("--n2-cuda-streams", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--benchmark-steps", type=int, default=5)
    parser.add_argument("--component-iterations", type=int, default=10)
    parser.add_argument("--skip-components", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output-dir", default="benchmark-results")
    parser.add_argument("--json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA benchmark requested but this Python environment has no CUDA PyTorch"
        )
    if args.n1_stage == "n2":
        model = create_n2_model(
            50_257,
            args.preset,
            maximum_sequence_length=args.sequence_length,
            seed=42,
            population=args.module_population,
            top_k=2,
            tie_embeddings=True,
            n1_depth=3,
            execution_mode=args.n2_execution_mode,
            use_cuda_streams=args.n2_cuda_streams,
        ).to(device)
    else:
        model = create_emc_model(
            50_257,
            args.preset,
            maximum_sequence_length=args.sequence_length,
            seed=42,
            tie_embeddings=True,
            n1_stage=args.n1_stage,
            module_population=args.module_population,
        ).to(device)
    if isinstance(model, ChunkedEMCModel) or args.n1_stage == "n2":
        args.skip_components = True
    stepper = TrainingStepper(
        model,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        accumulation=args.gradient_accumulation,
        precision=args.precision,
        device=device,
    )
    print("EMC N1 benchmark configuration")
    print(asdict(model.config))
    print(
        {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "gradient_accumulation": args.gradient_accumulation,
            "tokens_per_step": stepper.tokens_per_step,
            "precision": args.precision,
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "compile": False,
            "optimizer": "AdamW",
            "parameter_counts": asdict(parameter_counts(model)),
        }
    )
    if device.type == "cuda":
        print(
            {
                "gpu": torch.cuda.get_device_name(device),
                "capability": torch.cuda.get_device_capability(device),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
                "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            }
        )

    for _ in range(args.warmup_steps):
        stepper.step()
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    process_started = time.process_time()
    wall_started = time.perf_counter()
    sampler = NvidiaSampler() if device.type == "cuda" else None
    if sampler is not None:
        sampler.__enter__()
    stable = _time_iterations(
        lambda: stepper.step(),
        iterations=args.benchmark_steps,
        tokens=stepper.tokens_per_step,
        device=device,
    )
    utilization = sampler.finish() if sampler is not None else {}
    wall_elapsed = time.perf_counter() - wall_started
    cpu_elapsed = time.process_time() - process_started
    cpu_percent = 100 * cpu_elapsed / max(wall_elapsed * (os.cpu_count() or 1), 1e-9)

    recorder = ComponentEventRecorder(model) if device.type == "cuda" else None
    if recorder is not None:
        recorder.install()
        recorder.reset()
    phase_timings = stepper.step(phase_timing=True)
    component_timings = recorder.totals() if recorder is not None else {}
    if recorder is not None:
        recorder.remove()

    sample_inputs, _ = stepper.source.sample(
        args.batch_size, args.sequence_length, device
    )
    forced_n2_components = (
        benchmark_n2_forced_components(
            model,
            sample_inputs,
            precision=args.precision,
            device=device,
            iterations=max(3, args.component_iterations // 2),
        )
        if args.n1_stage == "n2" and device.type == "cuda"
        else {}
    )
    waste = [] if args.n1_stage == "n2" else routing_waste(model, sample_inputs)
    if args.skip_components:
        cycle_costs: dict[str, float] = {}
        module_results: dict[str, dict[str, float]] = {}
        component_results: dict[str, dict[str, float]] = {}
    else:
        cycle_costs = benchmark_cycle_costs(
            model,
            sample_inputs,
            precision=args.precision,
            device=device,
            iterations=max(3, args.component_iterations // 2),
        )
        module_results = benchmark_module_families(
            model.config,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            precision=args.precision,
            device=device,
            iterations=args.component_iterations,
        )
        component_results = benchmark_integrator_and_nexus(
            model.config,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            precision=args.precision,
            device=device,
            iterations=args.component_iterations,
        )

    report: dict[str, object] = {
        "stable_training": {
            "average_step_ms": stable.milliseconds,
            "tokens_per_second": stable.tokens_per_second,
            "cpu_percent_of_system": cpu_percent,
            "gpu_peak_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda"
                else 0
            ),
        },
        "utilization": utilization,
        "phase_timings_ms": phase_timings,
        "component_cuda_timings_ms": component_timings,
        "forced_n2_component_cuda_timings_ms": forced_n2_components,
        "routing_waste": waste,
        "cycle_costs": cycle_costs,
        "module_benchmarks": module_results,
        "component_benchmarks": component_results,
    }
    if args.profile:
        report["profiler"] = run_profiler(
            stepper, Path(args.output_dir)
        )

    print("\nBenchmark report")
    print(json.dumps(report, indent=2, default=str))
    if args.json:
        destination = Path(args.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"wrote {destination}")


if __name__ == "__main__":
    main()
