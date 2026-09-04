from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, replace
from typing import Literal, Mapping

import torch
from torch import nn

from .baseline import TransformerConfig, TransformerLanguageModel
from .chunked import ChunkedEMCModel
from .data import LanguageCorpus
from .diagnostics import count_parameters
from .experiments.common import create_emc_model, create_n2_model
from .model import EMCConfig, EMCModel, SequentialEMCModel
from .n2 import N2EMCModel
from .serial import HeterogeneousSerialModel
from .training import TrainingConfig


ArchitectureName = Literal[
    "homogeneous_serial",
    "heterogeneous_serial",
    "emc",
    "sequential_module_aware_emc",
    "legacy_parallel_emc",
    "old_emc",
    "n2_mixed",
    "n2_gpt4",
    "n2_ssm4",
    "n2_recurrent4",
    "n2_delta4",
]
FairnessMode = Literal["capacity", "compute"]
DEFAULT_HETEROGENEOUS_ORDER = ("gpt", "ssm", "recurrent", "delta")
N2_ARCHITECTURES = {
    "n2_mixed": "mixed",
    "n2_gpt4": "gpt4",
    "n2_ssm4": "ssm4",
    "n2_recurrent4": "recurrent4",
    "n2_delta4": "delta4",
}


@dataclass(frozen=True)
class ArchitectureAccounting:
    architecture: str
    total_parameters: int
    routable_parameters: int
    approximate_active_parameters: int
    approximate_parameter_uses_per_forward: int
    approximate_flops_per_token: int
    module_computations_per_forward: float
    sequence_length: int
    method: str
    limitations: tuple[str, ...]
    node_parameters: tuple[int, ...] = ()
    node_flops_per_invocation: tuple[int, ...] = ()
    blocks_per_n1: int | None = None
    nexus_parameters: int = 0
    integrator_parameters: int = 0
    embedding_parameters: int = 0
    shared_parameters: int = 0
    selected_nodes_per_event: int | None = None
    theoretical_all_nodes_flops_per_token: int | None = None


@dataclass(frozen=True)
class ArchitectureBuild:
    models: Mapping[str, nn.Module]
    emc_config: EMCConfig
    fairness_mode: str
    homogeneous_layers: int
    target_parameters: int
    target_active_parameters: int


def build_architectures(
    names: tuple[str, ...],
    *,
    vocab_size: int,
    model_preset: str,
    maximum_sequence_length: int,
    seed: int,
    fairness_mode: FairnessMode = "capacity",
    heterogeneous_order: tuple[str, ...] = DEFAULT_HETEROGENEOUS_ORDER,
    tie_embeddings: bool = True,
    top_k: int = 2,
    n1_depth: int = 3,
) -> ArchitectureBuild:
    supported = {
        "homogeneous_serial",
        "heterogeneous_serial",
        "emc",
        "sequential_module_aware_emc",
        "legacy_parallel_emc",
        "old_emc",
        *N2_ARCHITECTURES,
    }
    unknown = set(names) - supported
    if unknown:
        raise ValueError(f"unknown architectures: {sorted(unknown)}")
    if not heterogeneous_order:
        raise ValueError("heterogeneous serial order cannot be empty")
    template = create_emc_model(
        vocab_size,
        model_preset,
        maximum_sequence_length=maximum_sequence_length,
        seed=seed,
        tie_embeddings=tie_embeddings,
        n1_stage="n1_chunked",
        module_population="mixed",
    )
    if not isinstance(template, ChunkedEMCModel):
        raise RuntimeError("n1_chunked factory did not create the real ChunkedEMCModel")
    legacy_config = replace(
        template.config,
        module_families=heterogeneous_order,
        num_modules=len(heterogeneous_order),
        request_pool_size=len(heterogeneous_order),
        modules_per_cycle=top_k,
        active_top_k=top_k,
    )
    emc_config = replace(
        legacy_config,
        modules_per_cycle=1,
        active_top_k=None,
        num_cycles=3,
        trajectory_steps=3,
        architecture_stage="n1_sequential",
        router_type="geometric",
        integrator_type="acceptance_gate",
        loss_free_balance_enabled=False,
    )
    torch.manual_seed(seed)
    emc = SequentialEMCModel(emc_config)
    emc_accounting = architecture_accounting(
        emc, sequence_length=maximum_sequence_length
    )
    target = (
        emc_accounting.total_parameters
        if fairness_mode == "capacity"
        else emc_accounting.approximate_active_parameters
    )
    homogeneous_layers = _matching_transformer_layers(
        emc_config,
        target,
        fairness_mode=fairness_mode,
    )
    transformer_config = TransformerConfig(
        vocab_size=vocab_size,
        latent_dim=emc_config.latent_dim,
        num_layers=homogeneous_layers,
        attention_heads=emc_config.attention_heads,
        feed_forward_dim=emc_config.resolved_module_hidden_dim,
        max_sequence_length=maximum_sequence_length,
        tie_embeddings=tie_embeddings,
    )
    all_models: dict[str, nn.Module] = {}
    for name in names:
        torch.manual_seed(seed)
        if name == "homogeneous_serial":
            all_models[name] = TransformerLanguageModel(transformer_config)
        elif name == "heterogeneous_serial":
            all_models[name] = HeterogeneousSerialModel(legacy_config)
        elif name == "emc":
            all_models[name] = emc
        elif name == "sequential_module_aware_emc":
            all_models[name] = SequentialEMCModel(
                replace(
                    emc_config,
                    router_type="module_aware",
                    integrator_type="proposal_attention",
                )
            )
        elif name == "legacy_parallel_emc":
            all_models[name] = ChunkedEMCModel(legacy_config)
        elif name == "old_emc":
            all_models[name] = EMCModel(
                replace(legacy_config, architecture_stage="token")
            )
        else:
            all_models[name] = create_n2_model(
                vocab_size,
                model_preset,
                maximum_sequence_length=maximum_sequence_length,
                seed=seed,
                population=N2_ARCHITECTURES[name],
                top_k=top_k,
                tie_embeddings=tie_embeddings,
                n1_depth=n1_depth,
            )
    return ArchitectureBuild(
        models=all_models,
        emc_config=emc_config,
        fairness_mode=fairness_mode,
        homogeneous_layers=homogeneous_layers,
        target_parameters=emc_accounting.total_parameters,
        target_active_parameters=emc_accounting.approximate_active_parameters,
    )


def architecture_accounting(
    model: nn.Module, *, sequence_length: int
) -> ArchitectureAccounting:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    total = count_parameters(model)
    embedding_parameters = _unique_parameters(
        tuple(
            module
            for module in (
                getattr(model, "token_embedding", None),
                getattr(model, "position_embedding", None),
            )
            if module is not None
        )
    )
    output_parameters = _unique_parameters(
        tuple(
            module
            for module in (
                getattr(model, "output_norm", None),
                getattr(model, "output_projection", None),
            )
            if module is not None
        )
    )
    limitations = [
        "FLOPs are an explicit parameter-use proxy, not profiler measurements.",
        "Embedding lookup table storage is excluded; tied output weights remain compute.",
        "Backend scan, elementwise, normalization, indexing, and memory traffic are excluded.",
    ]
    node_parameters: tuple[int, ...] = ()
    node_flops: tuple[int, ...] = ()
    blocks_per_n1: int | None = None
    nexus_parameters = 0
    integrator_parameters = 0
    selected_nodes: int | None = None
    all_nodes_flops_per_token: int | None = None
    shared_parameters = total

    if isinstance(model, N2EMCModel):
        node_parameters = tuple(
            count_parameters(node) for node in model.n1_nodes
        )
        node_flops = tuple(
            node.approximate_flops(sequence_length) for node in model.n1_nodes
        )
        top_k = model.active_top_k
        selected_nodes = top_k
        expected_node_parameters = (
            top_k / len(node_parameters) * sum(node_parameters)
        )
        nexus_parameters = count_parameters(model.nexus)
        integrator_parameters = count_parameters(model.integrator)
        active_overhead = _unique_parameters(
            (model.nexus, model.integrator, model.output_norm, model.output_projection)
        )
        active = int(active_overhead + expected_node_parameters)
        parameter_uses = active * sequence_length
        expected_node_flops = top_k / len(node_flops) * sum(node_flops)
        overhead_flops = 2 * active_overhead * sequence_length
        flops = int(expected_node_flops + overhead_flops)
        all_nodes_flops_per_token = max(
            1, int((sum(node_flops) + overhead_flops) // sequence_length)
        )
        module_computations = float(top_k)
        routable = sum(node_parameters)
        architecture = f"n2_{model.config.n2_population}"
        blocks_per_n1 = model.config.n1_depth
        shared_parameters = total - routable
        method = (
            "Each selected N1 uses a 2*parameter*token proxy; GPT N1s also "
            "include explicit chunked attention score/value products. Nexus, "
            "proposal-attention Integrator, and output projection are included."
        )
        limitations.append(
            "Active N1 cost uses the population mean; realized routing can select "
            "a more or less expensive family."
        )
    elif isinstance(model, ChunkedEMCModel):
        module_counts = [count_parameters(module) for module in model.emc_modules]
        top_k = model.active_top_k
        expected_modules = top_k / len(module_counts) * sum(module_counts)
        overhead = _unique_parameters(
            (model.shared_core, model.router, model.integrator)
        )
        active = int(output_parameters + overhead + expected_modules)
        chunks = math.ceil(sequence_length / model.config.chunk_size)
        module_computations = chunks * top_k
        parameter_uses = active * sequence_length
        gpt_fraction = model.module_families.count("gpt") / len(module_counts)
        attention_extra = int(
            chunks
            * top_k
            * gpt_fraction
            * 4
            * model.config.chunk_size
            * model.config.chunk_size
            * model.config.latent_dim
        )
        flops = 2 * parameter_uses + attention_extra
        routable = sum(module_counts)
        architecture = "legacy_parallel_emc"
        method = (
            "router, both Integrators, SharedCore, and output projection are "
            "included in a once-per-token parameter-use proxy. Attention "
            "score/value products add 4*C*D per GPT execution and token."
        )
        limitations.append(
            "Heterogeneous expert cost uses the population mean before routing trajectories exist."
        )
    elif isinstance(model, EMCModel):
        module_counts = [count_parameters(module) for module in model.emc_modules]
        top_k = model.config.modules_per_cycle
        expected_modules = top_k / len(module_counts) * sum(module_counts)
        overhead = _unique_parameters((model.router, model.integrator))
        active = int(output_parameters + overhead + expected_modules)
        module_computations = float(model.config.num_cycles * top_k)
        parameter_uses = (
            output_parameters
            + model.config.num_cycles * (overhead + expected_modules)
        ) * sequence_length
        flops = 2 * parameter_uses
        routable = sum(module_counts)
        sequential = model.config.architecture_stage == "n1_sequential"
        architecture = (
            "emc"
            if sequential and model.config.router_type == "geometric"
            else "sequential_module_aware_emc"
            if sequential
            else "old_emc"
        )
        method = (
            "Sequential EMC estimate includes one selected expert, Nexus, the "
            "single-proposal Integrator gate, and output projection for every "
            "trajectory step."
            if sequential else
            "Legacy token-routed EMC estimate includes selected modules, router, Integrator, and output projection for every configured cycle."
        )
        limitations.append(
            "Parameter-use accounting is a proxy without attention-score products."
        )
    elif isinstance(model, HeterogeneousSerialModel):
        module_counts = [count_parameters(module) for module in model.emc_modules]
        shared = _unique_parameters((model.shared_core,))
        active = total
        chunks = math.ceil(sequence_length / model.config.chunk_size)
        module_computations = chunks * len(module_counts)
        parameter_uses = (
            output_parameters + shared + sum(module_counts)
        ) * sequence_length
        gpt_executions = chunks * model.module_families.count("gpt")
        attention_extra = int(
            gpt_executions
            * 4
            * model.config.chunk_size
            * model.config.chunk_size
            * model.config.latent_dim
        )
        flops = 2 * parameter_uses + attention_extra
        routable = 0
        architecture = "heterogeneous_serial"
        method = (
            "Every configured heterogeneous module executes once per chunk in order; "
            "SharedCore and output projection are included. Attention score/value "
            "products add 4*C*D per GPT execution."
        )
    elif isinstance(model, TransformerLanguageModel):
        block_parameters = sum(count_parameters(block) for block in model.blocks)
        active = total
        module_computations = float(len(model.blocks))
        parameter_uses = (
            output_parameters + block_parameters
        ) * sequence_length
        attention_extra = (
            len(model.blocks)
            * 4
            * sequence_length
            * sequence_length
            * model.config.latent_dim
        )
        flops = 2 * parameter_uses + attention_extra
        routable = 0
        architecture = "homogeneous_serial"
        method = (
            "All decoder blocks and the output projection execute for every token; "
            "attention score/value products add 4*L*D per block and token."
        )
    else:
        raise TypeError(f"unsupported benchmark model: {type(model).__name__}")

    return ArchitectureAccounting(
        architecture=architecture,
        total_parameters=total,
        routable_parameters=routable,
        approximate_active_parameters=active,
        approximate_parameter_uses_per_forward=parameter_uses,
        approximate_flops_per_token=max(1, flops // sequence_length),
        module_computations_per_forward=float(module_computations),
        sequence_length=sequence_length,
        method=method,
        limitations=tuple(limitations),
        node_parameters=node_parameters,
        node_flops_per_invocation=node_flops,
        blocks_per_n1=blocks_per_n1,
        nexus_parameters=nexus_parameters,
        integrator_parameters=integrator_parameters,
        embedding_parameters=embedding_parameters,
        shared_parameters=shared_parameters,
        selected_nodes_per_event=selected_nodes,
        theoretical_all_nodes_flops_per_token=all_nodes_flops_per_token,
    )


def architecture_mismatch_report(
    accounting: Mapping[str, ArchitectureAccounting], *, reference: str = "emc"
) -> dict[str, dict[str, float | int]]:
    if reference not in accounting:
        reference = next(iter(accounting))
    target = accounting[reference]
    output: dict[str, dict[str, float | int]] = {}
    for name, row in accounting.items():
        output[name] = {
            "total_parameter_difference": row.total_parameters
            - target.total_parameters,
            "total_parameter_ratio": row.total_parameters
            / target.total_parameters,
            "active_parameter_difference": row.approximate_active_parameters
            - target.approximate_active_parameters,
            "active_parameter_ratio": row.approximate_active_parameters
            / target.approximate_active_parameters,
            "flops_per_token_difference": row.approximate_flops_per_token
            - target.approximate_flops_per_token,
            "flops_per_token_ratio": row.approximate_flops_per_token
            / target.approximate_flops_per_token,
        }
    return output


def deterministic_stream_fingerprint(
    corpus: LanguageCorpus,
    config: TrainingConfig,
    *,
    max_steps: int | None = None,
) -> str:
    generator = torch.Generator().manual_seed(config.seed)
    digest = hashlib.sha256()
    steps = config.planned_steps if max_steps is None else min(config.planned_steps, max_steps)
    for _ in range(steps):
        for _ in range(config.gradient_accumulation_steps):
            inputs, targets = corpus.sample_batch(
                "train",
                config.batch_size,
                config.sequence_length,
                generator=generator,
                device="cpu",
            )
            digest.update(inputs.numpy().tobytes())
            digest.update(targets.numpy().tobytes())
    return digest.hexdigest()


def config_manifest(model: nn.Module) -> dict[str, object]:
    return {
        "class": type(model).__name__,
        "config": asdict(model.config),
        "module_order": list(getattr(model, "module_families", ())),
        "expert_names": list(getattr(model, "expert_names", ())),
    }


def _matching_transformer_layers(
    emc_config: EMCConfig,
    target: int,
    *,
    fairness_mode: FairnessMode,
) -> int:
    one = TransformerLanguageModel(
        TransformerConfig(
            vocab_size=emc_config.vocab_size,
            latent_dim=emc_config.latent_dim,
            num_layers=1,
            attention_heads=emc_config.attention_heads,
            feed_forward_dim=emc_config.resolved_module_hidden_dim,
            max_sequence_length=emc_config.max_sequence_length,
            tie_embeddings=emc_config.tie_embeddings,
        )
    )
    block = count_parameters(one.blocks[0])
    if fairness_mode == "capacity":
        fixed = count_parameters(one) - block
    else:
        fixed = _unique_parameters((one.output_norm, one.output_projection))
    return max(1, round((target - fixed) / block))


def _unique_parameters(modules: tuple[nn.Module, ...]) -> int:
    seen: set[int] = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                total += parameter.numel()
    return total
