from __future__ import annotations

from dataclasses import asdict
from typing import Literal

import torch
from torch import nn

from ..baseline import TransformerConfig, TransformerLanguageModel
from ..chunked import ChunkedEMCModel
from ..data import LanguageCorpus, load_tinystories, tiny_overfit_corpus
from ..diagnostics import (
    RoutingReport,
    format_parameter_breakdown,
    parameter_breakdown,
    parameter_counts,
)
from ..model import EMCConfig, EMCModel
from ..n2 import N2_POPULATIONS, N2Config, N2EMCModel
from ..tokenization import DEFAULT_TOKENIZER_IDENTIFIER

Preset = Literal["quick", "research"]
BudgetPreset = Literal["quick", "medium", "research"]

N1_STAGES = (
    "baseline",
    "integrator",
    "heterogeneous",
    "n1",
    "n1_chunked",
)
MODULE_POPULATIONS = (
    "gpt-only",
    "ssm-only",
    "recurrent-only",
    "delta-only",
    "gpt-ssm",
    "gpt-recurrent",
    "gpt-delta",
    "ssm-recurrent",
    "ssm-delta",
    "recurrent-delta",
    "mixed",
)
N2_POPULATION_PRESETS = tuple(N2_POPULATIONS)
DatasetName = Literal["tiny", "tinystories"]

TOKEN_BUDGETS: dict[BudgetPreset, int] = {
    "quick": 1_000_000,
    "medium": 10_000_000,
    "research": 25_000_000,
}

MODEL_PRESET_DIMENSIONS: dict[str, dict[str, int]] = {
    "quick": {
        "latent_dim": 64,
        "module_hidden_dim": 128,
        "attention_heads": 4,
        "state_space_dim": 96,
        "recurrent_dim": 64,
        "router_descriptor_dim": 64,
        "integrator_heads": 4,
        "chunk_size": 16,
        "delta_internal_dim": 64,
        "delta_heads": 4,
        "delta_ffn_dim": 128,
    },
    "research": {
        "latent_dim": 256,
        "module_hidden_dim": 6_144,
        "attention_heads": 8,
        "state_space_dim": 960,
        "recurrent_dim": 704,
        "router_descriptor_dim": 128,
        "integrator_heads": 8,
        "chunk_size": 64,
        "delta_internal_dim": 512,
        "delta_heads": 8,
        "delta_ffn_dim": 4_096,
    },
}


def load_experiment_corpus(
    dataset: DatasetName,
    *,
    train_stories: int = 10_000,
    validation_stories: int = 1_000,
    tokenizer_identifier: str = DEFAULT_TOKENIZER_IDENTIFIER,
) -> LanguageCorpus:
    if dataset == "tiny":
        return tiny_overfit_corpus()
    if dataset == "tinystories":
        return load_tinystories(
            max_train_stories=train_stories,
            max_validation_stories=validation_stories,
            tokenizer_identifier=tokenizer_identifier,
        )
    raise ValueError(f"unknown dataset: {dataset}")


def create_emc_model(
    vocab_size: int,
    preset: Preset,
    *,
    maximum_sequence_length: int,
    seed: int,
    tie_embeddings: bool = False,
    n1_stage: str = "n1",
    module_population: str = "mixed",
) -> EMCModel | ChunkedEMCModel:
    torch.manual_seed(seed)
    if n1_stage not in N1_STAGES:
        raise ValueError(f"unknown N1 stage: {n1_stage!r}")
    if module_population not in MODULE_POPULATIONS:
        raise ValueError(f"unknown module population: {module_population!r}")
    chunked = n1_stage == "n1_chunked"
    heterogeneous = n1_stage in {"heterogeneous", "n1", "n1_chunked"}
    families = (
        module_families_for_population(
            module_population, include_delta=chunked
        )
        if heterogeneous
        else ("gpt",) * 4
    )
    if not chunked and "delta" in families:
        raise ValueError("DeltaNet populations require n1_stage='n1_chunked'")
    integrator_type = (
        "proposal_attention"
        if n1_stage in {"integrator", "heterogeneous", "n1", "n1_chunked"}
        else "weighted_average"
    )
    router_type = (
        "module_aware" if n1_stage in {"n1", "n1_chunked"} else "fixed_index"
    )
    if preset not in MODEL_PRESET_DIMENSIONS:
        raise ValueError(f"unknown preset: {preset}")
    dimensions = MODEL_PRESET_DIMENSIONS[preset]
    config = EMCConfig(
        **dimensions,
        num_modules=4,
        modules_per_cycle=2,
        num_cycles=2,
        vocab_size=vocab_size,
        max_sequence_length=maximum_sequence_length,
        tie_embeddings=(True if preset == "research" else tie_embeddings),
        module_families=families,
        router_type=router_type,
        integrator_type=integrator_type,
        architecture_stage=("n1_chunked" if chunked else "token"),
        shared_state_slots=4,
        request_pool_size=4,
        recurrent_precision="fp16",
    )
    if chunked:
        return ChunkedEMCModel(config)
    return EMCModel(config)


def create_n2_model(
    vocab_size: int,
    preset: Preset,
    *,
    maximum_sequence_length: int,
    seed: int,
    population: str = "mixed",
    top_k: int = 2,
    tie_embeddings: bool = False,
    n1_depth: int = 3,
    execution_mode: str = "sparse",
    use_cuda_streams: bool = False,
) -> N2EMCModel:
    if population not in N2_POPULATIONS:
        raise ValueError(f"unknown N2 population: {population!r}")
    torch.manual_seed(seed)
    template = create_emc_model(
        vocab_size,
        preset,
        maximum_sequence_length=maximum_sequence_length,
        seed=seed,
        tie_embeddings=tie_embeddings,
        n1_stage="n1_chunked",
        module_population="mixed",
    )
    values = asdict(template.config)
    values.update(
        {
            "architecture_stage": "n2",
            "num_cycles": 1,
            "modules_per_cycle": top_k,
            "active_top_k": top_k,
            "num_modules": len(N2_POPULATIONS[population]),
            "module_families": N2_POPULATIONS[population],
            "n2_population": population,
            "n1_depth": n1_depth,
            "n2_execution_mode": execution_mode,
            "n2_use_cuda_streams": use_cuda_streams,
            "loss_free_balance_enabled": False,
        }
    )
    if preset == "quick":
        values.update(
            {
                "module_hidden_dim": 192,
                "state_space_dim": 88,
                "recurrent_dim": 66,
                "delta_internal_dim": 48,
                "delta_ffn_dim": 144,
            }
        )
    else:
        values["delta_ffn_dim"] = 5_120
    torch.manual_seed(seed)
    return N2EMCModel(N2Config(**values))


def create_baseline_model(
    vocab_size: int,
    preset: Preset,
    *,
    maximum_sequence_length: int,
    seed: int,
    tie_embeddings: bool = False,
) -> TransformerLanguageModel:
    torch.manual_seed(seed)
    if preset == "quick":
        config = TransformerConfig(
            vocab_size=vocab_size,
            latent_dim=64,
            num_layers=4,
            attention_heads=4,
            feed_forward_dim=128,
            max_sequence_length=maximum_sequence_length,
            tie_embeddings=tie_embeddings,
        )
    elif preset == "research":
        config = TransformerConfig(
            vocab_size=vocab_size,
            latent_dim=320,
            num_layers=8,
            attention_heads=8,
            feed_forward_dim=1_280,
            max_sequence_length=maximum_sequence_length,
            tie_embeddings=True,
        )
    else:
        raise ValueError(f"unknown preset: {preset}")
    return TransformerLanguageModel(config)


def token_budget_for_preset(preset: BudgetPreset) -> int:
    return TOKEN_BUDGETS[preset]


def module_families_for_population(
    population: str,
    *,
    include_delta: bool = False,
) -> tuple[str, str, str, str]:
    populations = {
        "gpt-only": ("gpt", "gpt", "gpt", "gpt"),
        "ssm-only": ("ssm", "ssm", "ssm", "ssm"),
        "recurrent-only": (
            "recurrent",
            "recurrent",
            "recurrent",
            "recurrent",
        ),
        "delta-only": ("delta", "delta", "delta", "delta"),
        "gpt-ssm": ("gpt", "ssm", "gpt", "ssm"),
        "gpt-recurrent": ("gpt", "recurrent", "gpt", "recurrent"),
        "gpt-delta": ("gpt", "delta", "gpt", "delta"),
        "ssm-recurrent": ("ssm", "recurrent", "ssm", "recurrent"),
        "ssm-delta": ("ssm", "delta", "ssm", "delta"),
        "recurrent-delta": (
            "recurrent",
            "delta",
            "recurrent",
            "delta",
        ),
        "mixed": (
            ("gpt", "ssm", "recurrent", "delta")
            if include_delta
            else ("gpt", "ssm", "recurrent", "gpt")
        ),
    }
    try:
        return populations[population]
    except KeyError as error:
        raise ValueError(f"unknown module population: {population!r}") from error


def print_parameter_summary(name: str, model: nn.Module) -> None:
    counts = parameter_counts(model)
    if isinstance(model, (EMCModel, ChunkedEMCModel)):
        print(format_parameter_breakdown(parameter_breakdown(model)))
    print(
        f"{name} parameters: total={counts.total:,} | "
        f"theoretical active/token-cycle={counts.approximate_active_per_cycle:,} | "
        f"theoretical parameter uses/token-forward="
        f"{counts.approximate_parameter_uses_per_forward:,}"
    )


def print_routing_report(
    report: RoutingReport, *, include_training_signals: bool = True
) -> None:
    labels = report.expert_names or tuple(
        f"m{index}" for index in range(len(report.traffic_fraction))
    )
    traffic = ", ".join(
        f"{label}={fraction:.1%}"
        for label, fraction in zip(labels, report.traffic_fraction, strict=True)
    )
    print(f"EMC module traffic: {traffic}")
    family_traffic = ", ".join(
        f"{family}={fraction:.1%}"
        for family, fraction in report.family_traffic_fraction
    )
    print(f"EMC family traffic: {family_traffic}")
    for cycle, (distribution, entropy) in enumerate(
        zip(
            report.routing_distribution_per_cycle,
            report.mean_router_entropy,
            strict=True,
        ),
        start=1,
    ):
        formatted = ", ".join(f"{value:.1%}" for value in distribution)
        print(f"  cycle {cycle}: [{formatted}] entropy={entropy:.3f}")
    print(
        "EMC concentration: "
        f"top1={report.top_1_traffic_share:.1%} | "
        f"top2={report.top_2_traffic_share:.1%} | "
        f"minimum={report.minimum_module_share:.1%} | "
        f"normalized_entropy={report.normalized_routing_entropy:.3f} | "
        f"effective_modules={report.effective_active_modules:.2f} | "
        f"severe_collapse={report.severe_collapse}"
    )
    print(
        "EMC Integrator: "
        f"proposal_similarity={report.mean_proposal_similarity:.3f} | "
        f"update_norm={report.mean_integrated_update_norm:.3f} | "
        f"gate={report.mean_gate_magnitude:.3f}"
    )
    for index, (
        probability,
        acceptance,
        proposal_norm,
        contribution,
        parameter_count,
    ) in enumerate(
        zip(
            report.average_routing_probability,
            report.average_integrator_acceptance,
            report.average_proposal_norm,
            report.average_proposal_contribution,
            report.module_parameter_counts,
            strict=True,
        )
    ):
        print(
            f"  {labels[index]}: route_p={probability:.3f} | "
            f"accept={acceptance:.3f} | proposal_norm={proposal_norm:.3f} | "
            f"contribution={contribution:.3f} | params={parameter_count:,}"
        )
    family_acceptance = ", ".join(
        f"{family}={fraction:.1%}"
        for family, fraction in report.family_integrator_acceptance
    )
    print(f"EMC family acceptance: {family_acceptance}")
    if include_training_signals:
        print(
            "EMC diagnostics: "
            f"all_modules_used={report.all_modules_used} | "
            f"top_modules={list(report.dominant_modules)} | "
            f"varies_by_input={report.routing_differs_across_inputs} | "
            f"varies_by_cycle={report.routing_differs_across_cycles} | "
            f"max_router_grad={report.maximum_router_gradient_norm:.3e} | "
            f"module_updates_diverged={report.module_updates_diverged}"
        )
        print(
            "EMC module update norms: "
            + ", ".join(
                f"m{index}={norm:.3e}"
                for index, norm in enumerate(report.module_update_norms)
            )
        )
        print(
            "EMC module gradient norms: "
            + ", ".join(
                f"m{index}={norm:.3e}"
                for index, norm in enumerate(report.module_gradient_norms)
            )
        )
