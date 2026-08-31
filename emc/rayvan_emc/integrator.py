from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class IntegratorTrace:
    proposal_acceptance: Tensor
    proposal_norms: Tensor
    proposal_similarity: Tensor
    proposal_contributions: Tensor
    integrated_update_norm: Tensor
    gate_magnitude: Tensor


class WeightedAverageIntegrator(nn.Module):
    """Reproducible pre-N1 Integrator baseline."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        integration_dim = config.latent_dim * 2
        self.latent_norm = nn.LayerNorm(config.latent_dim)
        self.update_projection = nn.Linear(integration_dim, config.latent_dim)
        self.gate_projection = nn.Linear(integration_dim, config.latent_dim)

    def forward(
        self,
        latent: Tensor,
        module_updates: Tensor,
        routing_weights: Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, IntegratorTrace]:
        weighted_update = torch.einsum(
            "bsk,bskd->bsd", routing_weights, module_updates
        )
        integration_input = torch.cat(
            (self.latent_norm(latent), weighted_update), dim=-1
        )
        candidate_update = self.update_projection(integration_input)
        update_gate = torch.sigmoid(self.gate_projection(integration_input))
        gated_update = update_gate * candidate_update
        next_latent = latent + gated_update
        if not return_diagnostics:
            return next_latent

        contributions = routing_weights * module_updates.norm(dim=-1)
        return next_latent, _trace(
            module_updates,
            routing_weights,
            contributions,
            candidate_update,
            update_gate,
        )


class Integrator(nn.Module):
    """N1 proposal-aware multi-head set Integrator with vector gating."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.latent_dim = config.latent_dim
        self.num_heads = config.integrator_heads
        if self.latent_dim % self.num_heads != 0:
            raise ValueError("latent_dim must be divisible by integrator_heads")
        self.head_dim = self.latent_dim // self.num_heads
        self.latent_norm = nn.LayerNorm(self.latent_dim)
        self.proposal_norm = nn.LayerNorm(self.latent_dim)
        self.query_projection = nn.Linear(self.latent_dim, self.latent_dim)
        self.key_projection = nn.Linear(self.latent_dim, self.latent_dim)
        self.value_projection = nn.Linear(self.latent_dim, self.latent_dim)
        self.attention_output = nn.Linear(self.latent_dim, self.latent_dim)
        self.routing_prior_scale = nn.Parameter(torch.tensor(0.5))
        integration_dim = self.latent_dim * 4
        self.update_projection = nn.Linear(integration_dim, self.latent_dim)
        self.gate_projection = nn.Linear(integration_dim, self.latent_dim)

    def forward(
        self,
        latent: Tensor,
        module_updates: Tensor,
        routing_weights: Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, IntegratorTrace]:
        batch, sequence, selected, latent_dim = module_updates.shape
        normalized_proposals = self.proposal_norm(module_updates)
        query = self.query_projection(self.latent_norm(latent)).reshape(
            batch, sequence, self.num_heads, self.head_dim
        )
        keys = self.key_projection(normalized_proposals).reshape(
            batch, sequence, selected, self.num_heads, self.head_dim
        ).permute(0, 1, 3, 2, 4)
        values = self.value_projection(normalized_proposals).reshape(
            batch, sequence, selected, self.num_heads, self.head_dim
        ).permute(0, 1, 3, 2, 4)

        attention_scores = torch.einsum("bshd,bshkd->bshk", query, keys)
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        routing_prior = routing_weights.clamp_min(1e-9).log().unsqueeze(2)
        attention_scores = (
            attention_scores + self.routing_prior_scale * routing_prior
        )
        head_acceptance = torch.softmax(attention_scores, dim=-1)
        attended_heads = torch.einsum(
            "bshk,bshkd->bshd", head_acceptance, values
        )
        attended = self.attention_output(
            attended_heads.reshape(batch, sequence, latent_dim)
        )

        proposal_mean = module_updates.mean(dim=2)
        proposal_variance = module_updates.var(dim=2, unbiased=False)
        integration_input = torch.cat(
            (
                self.latent_norm(latent),
                attended,
                proposal_mean,
                proposal_variance,
            ),
            dim=-1,
        )
        candidate_update = self.update_projection(integration_input)
        update_gate = torch.sigmoid(self.gate_projection(integration_input))
        gated_update = update_gate * candidate_update
        next_latent = latent + gated_update
        if not return_diagnostics:
            return next_latent

        acceptance = head_acceptance.mean(dim=2)
        per_proposal_values = (
            head_acceptance.unsqueeze(-1) * values
        ).permute(0, 1, 3, 2, 4)
        contributions = per_proposal_values.flatten(start_dim=3).norm(dim=-1)
        return next_latent, _trace(
            module_updates,
            acceptance,
            contributions,
            candidate_update,
            update_gate,
        )


def _trace(
    proposals: Tensor,
    acceptance: Tensor,
    contributions: Tensor,
    integrated_update: Tensor,
    gate: Tensor,
) -> IntegratorTrace:
    normalized = torch.nn.functional.normalize(proposals, dim=-1)
    similarity = torch.einsum("bskd,bsjd->bskj", normalized, normalized)
    return IntegratorTrace(
        proposal_acceptance=acceptance.detach().cpu(),
        proposal_norms=proposals.norm(dim=-1).detach().cpu(),
        proposal_similarity=similarity.detach().cpu(),
        proposal_contributions=contributions.detach().cpu(),
        integrated_update_norm=integrated_update.norm(dim=-1).detach().cpu(),
        gate_magnitude=gate.abs().mean(dim=-1).detach().cpu(),
    )
