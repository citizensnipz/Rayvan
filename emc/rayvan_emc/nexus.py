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
    raw_scores: Tensor | None = None
    pre_inhibition_scores: Tensor | None = None
    refractory_penalty: Tensor | None = None


class NexusRouter(nn.Module):
    """Reproducible fixed-index MoE router baseline."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.modules_per_cycle = config.modules_per_cycle
        self.input_norm = nn.LayerNorm(config.latent_dim)
        self.score_projection = nn.Linear(config.latent_dim, config.num_modules)
        self.register_buffer(
            "balance_bias", torch.zeros(config.num_modules), persistent=True
        )

    def forward(
        self,
        latent: Tensor,
        *,
        availability_mask: Tensor | None = None,
        module_descriptors: Tensor | None = None,
        top_k: int | None = None,
    ) -> RoutingDecision:
        if module_descriptors is not None:
            raise ValueError("fixed-index Nexus does not accept module descriptors")
        active_top_k = top_k or self.modules_per_cycle
        scores = self.score_projection(self.input_norm(latent))
        scores = _mask_unavailable(
            scores, availability_mask, active_top_k
        )
        selected_scores, selected_indices = torch.topk(
            scores, k=active_top_k, dim=-1
        )
        selected_weights = torch.softmax(selected_scores, dim=-1)
        return RoutingDecision(scores, selected_indices, selected_weights)

    def route_one(
        self,
        latent: Tensor,
        *,
        availability_mask: Tensor | None = None,
        module_descriptors: Tensor | None = None,
        score_adjustment: Tensor | None = None,
        refractory_penalty: Tensor | None = None,
    ) -> RoutingDecision:
        if module_descriptors is not None:
            raise ValueError("fixed-index Nexus does not accept module descriptors")
        raw = self.score_projection(self.input_norm(latent))
        return _route_one(
            raw, self.balance_bias, availability_mask,
            score_adjustment, refractory_penalty,
        )

    def update_balance_bias(self, selected: Tensor) -> None:
        _update_balance_bias(self, selected)


class ModuleAwareNexusRouter(nn.Module):
    """Scores latent-derived queries against learned module descriptor keys."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.modules_per_cycle = config.modules_per_cycle
        self.descriptor_dim = config.resolved_router_descriptor_dim
        self.input_norm = nn.LayerNorm(config.latent_dim)
        self.query_projection = nn.Linear(config.latent_dim, self.descriptor_dim)
        self.module_descriptors = nn.Parameter(
            torch.empty(config.num_modules, self.descriptor_dim)
        )
        nn.init.normal_(self.module_descriptors, mean=0.0, std=0.02)
        self.register_buffer(
            "balance_bias", torch.zeros(config.num_modules), persistent=True
        )

    def forward(
        self,
        latent: Tensor,
        *,
        availability_mask: Tensor | None = None,
        module_descriptors: Tensor | None = None,
        top_k: int | None = None,
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
        active_top_k = top_k or self.modules_per_cycle
        query = self.query_projection(self.input_norm(latent))
        scores = torch.einsum("bsd,md->bsm", query, descriptors)
        scores = scores / math.sqrt(self.descriptor_dim)
        scores = _mask_unavailable(
            scores, availability_mask, active_top_k
        )
        selected_scores, selected_indices = torch.topk(
            scores, k=active_top_k, dim=-1
        )
        selected_weights = torch.softmax(selected_scores, dim=-1)
        return RoutingDecision(scores, selected_indices, selected_weights)

    def route_one(
        self,
        latent: Tensor,
        *,
        availability_mask: Tensor | None = None,
        module_descriptors: Tensor | None = None,
        score_adjustment: Tensor | None = None,
        refractory_penalty: Tensor | None = None,
    ) -> RoutingDecision:
        descriptors = (
            self.module_descriptors
            if module_descriptors is None
            else module_descriptors
        )
        query = self.query_projection(self.input_norm(latent))
        raw = torch.einsum("bsd,md->bsm", query, descriptors)
        raw = raw / math.sqrt(self.descriptor_dim)
        return _route_one(
            raw, self.balance_bias, availability_mask,
            score_adjustment, refractory_penalty,
        )

    def update_balance_bias(self, selected: Tensor) -> None:
        _update_balance_bias(self, selected)


def _route_one(
    raw_scores: Tensor,
    balance_bias: Tensor,
    availability_mask: Tensor | None,
    score_adjustment: Tensor | None,
    refractory_penalty: Tensor | None,
) -> RoutingDecision:
    pre_inhibition = raw_scores + balance_bias
    if score_adjustment is not None:
        pre_inhibition = pre_inhibition + score_adjustment
    effective = pre_inhibition
    if refractory_penalty is not None:
        effective = effective - refractory_penalty
    effective = _mask_unavailable(effective, availability_mask, 1)
    selected = effective.argmax(dim=-1, keepdim=True)
    return RoutingDecision(
        scores=effective,
        selected_indices=selected,
        selected_weights=torch.ones_like(selected, dtype=effective.dtype),
        raw_scores=raw_scores,
        pre_inhibition_scores=pre_inhibition,
        refractory_penalty=refractory_penalty,
    )


@torch.no_grad()
def _update_balance_bias(router: nn.Module, selected: Tensor) -> None:
    config = router.config
    if not getattr(config, "loss_free_balance_enabled", False):
        return
    counts = torch.bincount(
        selected.reshape(-1), minlength=config.num_modules
    ).to(router.balance_bias.dtype)
    observed = counts / counts.sum().clamp_min(1)
    if config.balance_target_utilization is None:
        target = torch.full_like(observed, 1.0 / config.num_modules)
    else:
        target = observed.new_tensor(config.balance_target_utilization)
        target = target / target.sum()
    router.balance_bias.add_(config.balance_bias_lr * (target - observed))
    router.balance_bias.clamp_(
        -config.balance_bias_limit, config.balance_bias_limit
    )


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
