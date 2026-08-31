from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RoutingDecision:
    scores: Tensor
    selected_indices: Tensor
    selected_weights: Tensor


class NexusRouter(nn.Module):
    """Reproducible fixed-index MoE router baseline."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.modules_per_cycle = config.modules_per_cycle
        self.input_norm = nn.LayerNorm(config.latent_dim)
        self.score_projection = nn.Linear(config.latent_dim, config.num_modules)

    def forward(
        self,
        latent: Tensor,
        *,
        availability_mask: Tensor | None = None,
        module_descriptors: Tensor | None = None,
    ) -> RoutingDecision:
        if module_descriptors is not None:
            raise ValueError("fixed-index Nexus does not accept module descriptors")
        scores = self.score_projection(self.input_norm(latent))
        scores = _mask_unavailable(
            scores, availability_mask, self.modules_per_cycle
        )
        selected_scores, selected_indices = torch.topk(
            scores, k=self.modules_per_cycle, dim=-1
        )
        selected_weights = torch.softmax(selected_scores, dim=-1)
        return RoutingDecision(scores, selected_indices, selected_weights)


class ModuleAwareNexusRouter(nn.Module):
    """Scores latent-derived queries against learned module descriptor keys."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.modules_per_cycle = config.modules_per_cycle
        self.descriptor_dim = config.resolved_router_descriptor_dim
        self.input_norm = nn.LayerNorm(config.latent_dim)
        self.query_projection = nn.Linear(config.latent_dim, self.descriptor_dim)
        self.module_descriptors = nn.Parameter(
            torch.empty(config.num_modules, self.descriptor_dim)
        )
        nn.init.normal_(self.module_descriptors, mean=0.0, std=0.02)

    def forward(
        self,
        latent: Tensor,
        *,
        availability_mask: Tensor | None = None,
        module_descriptors: Tensor | None = None,
    ) -> RoutingDecision:
        descriptors = (
            self.module_descriptors
            if module_descriptors is None
            else module_descriptors
        )
        if descriptors.ndim != 2 or descriptors.size(1) != self.descriptor_dim:
            raise ValueError(
                "module descriptors must have shape [modules, descriptor_dim]"
            )
        query = self.query_projection(self.input_norm(latent))
        scores = torch.einsum("bsd,md->bsm", query, descriptors)
        scores = scores / math.sqrt(self.descriptor_dim)
        scores = _mask_unavailable(
            scores, availability_mask, self.modules_per_cycle
        )
        selected_scores, selected_indices = torch.topk(
            scores, k=self.modules_per_cycle, dim=-1
        )
        selected_weights = torch.softmax(selected_scores, dim=-1)
        return RoutingDecision(scores, selected_indices, selected_weights)


def _mask_unavailable(
    scores: Tensor,
    availability_mask: Tensor | None,
    modules_per_cycle: int,
) -> Tensor:
    if availability_mask is None:
        return scores
    if availability_mask.ndim != 1 or availability_mask.numel() != scores.size(-1):
        raise ValueError("availability_mask must contain one value per module")
    available = availability_mask.to(device=scores.device, dtype=torch.bool)
    if int(available.sum().item()) < modules_per_cycle:
        raise ValueError("availability_mask leaves fewer modules than top-K requires")
    return scores.masked_fill(~available, -torch.inf)
