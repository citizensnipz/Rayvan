from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from ..baseline import TransformerConfig, TransformerLanguageModel
from ..data import LanguageCorpus, load_tinystories, tiny_overfit_corpus
from ..diagnostics import RoutingReport, parameter_counts
from ..model import EMCConfig, EMCModel

Preset = Literal["quick", "research"]
DatasetName = Literal["tiny", "tinystories"]


def load_experiment_corpus(
    dataset: DatasetName,
    *,
    train_stories: int = 2_000,
    validation_stories: int = 200,
) -> LanguageCorpus:
    if dataset == "tiny":
        return tiny_overfit_corpus()
    if dataset == "tinystories":
        return load_tinystories(
            max_train_stories=train_stories,
            max_validation_stories=validation_stories,
        )
    raise ValueError(f"unknown dataset: {dataset}")


def create_emc_model(
    vocab_size: int,
    preset: Preset,
    *,
    maximum_sequence_length: int,
    seed: int,
) -> EMCModel:
    torch.manual_seed(seed)
    if preset == "quick":
        config = EMCConfig(
            latent_dim=64,
            num_modules=4,
            modules_per_cycle=2,
            num_cycles=2,
            vocab_size=vocab_size,
            max_sequence_length=maximum_sequence_length,
            module_hidden_dim=128,
            attention_heads=4,
        )
    elif preset == "research":
        config = EMCConfig(
            latent_dim=512,
            num_modules=8,
            modules_per_cycle=2,
            num_cycles=3,
            vocab_size=vocab_size,
            max_sequence_length=maximum_sequence_length,
            module_hidden_dim=2_048,
            attention_heads=8,
        )
    else:
        raise ValueError(f"unknown preset: {preset}")
    return EMCModel(config)


def create_baseline_model(
    vocab_size: int,
    preset: Preset,
    *,
    maximum_sequence_length: int,
    seed: int,
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
        )
    elif preset == "research":
        config = TransformerConfig(
            vocab_size=vocab_size,
            latent_dim=512,
            num_layers=8,
            attention_heads=8,
            feed_forward_dim=2_048,
            max_sequence_length=maximum_sequence_length,
        )
    else:
        raise ValueError(f"unknown preset: {preset}")
    return TransformerLanguageModel(config)


def print_parameter_summary(name: str, model: nn.Module) -> None:
    counts = parameter_counts(model)
    print(
        f"{name} parameters: total={counts.total:,} | "
        f"theoretical active/token-cycle={counts.approximate_active_per_cycle:,} | "
        f"theoretical parameter uses/token-forward="
        f"{counts.approximate_parameter_uses_per_forward:,}"
    )


def print_routing_report(report: RoutingReport) -> None:
    traffic = ", ".join(
        f"m{index}={fraction:.1%}"
        for index, fraction in enumerate(report.traffic_fraction)
    )
    print(f"EMC module traffic: {traffic}")
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
        "EMC diagnostics: "
        f"all_modules_used={report.all_modules_used} | "
        f"collapsed={report.routing_collapsed} | "
        f"top_modules={list(report.dominant_modules)} "
        f"({report.dominant_traffic_fraction:.1%} traffic) | "
        f"varies_by_input={report.routing_differs_across_inputs} | "
        f"varies_by_cycle={report.routing_differs_across_cycles} | "
        f"max_router_grad={report.maximum_router_gradient_norm:.3e} | "
        f"module_updates_diverged={report.module_updates_diverged}"
    )
    print(
        "EMC module update norms: "
        + ", ".join(f"m{index}={norm:.3e}" for index, norm in enumerate(report.module_update_norms))
    )
