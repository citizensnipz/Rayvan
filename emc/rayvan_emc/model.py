from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class EMCConfig:
    latent_dim: int = 32
    num_modules: int = 6
    modules_per_cycle: int = 2
    num_cycles: int = 3
    vocab_size: int = 256
    module_hidden_dim: int | None = None
    attention_heads: int = 4

    def __post_init__(self) -> None:
        positive_fields = {
            "latent_dim": self.latent_dim,
            "num_modules": self.num_modules,
            "modules_per_cycle": self.modules_per_cycle,
            "num_cycles": self.num_cycles,
            "vocab_size": self.vocab_size,
            "attention_heads": self.attention_heads,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.modules_per_cycle > self.num_modules:
            raise ValueError("modules_per_cycle cannot exceed num_modules")
        if self.latent_dim % self.attention_heads != 0:
            raise ValueError("latent_dim must be divisible by attention_heads")
        if self.module_hidden_dim is not None and self.module_hidden_dim <= 0:
            raise ValueError("module_hidden_dim must be positive when provided")

    @property
    def resolved_module_hidden_dim(self) -> int:
        return self.module_hidden_dim or self.latent_dim * 4


@dataclass(frozen=True)
class RoutingDecision:
    scores: Tensor
    selected_indices: Tensor
    selected_weights: Tensor


@dataclass(frozen=True)
class EMCCycleTrace:
    cycle: int
    selected_modules: tuple[int, ...]
    router_scores: Tensor
    router_weights: Tensor
    latent_shape: tuple[int, ...]


@dataclass(frozen=True)
class EMCOutput:
    logits: Tensor
    trace: tuple[EMCCycleTrace, ...]


class NexusRouter(nn.Module):
    """Selects one sparse module set for the current latent batch."""

    def __init__(self, config: EMCConfig) -> None:
        super().__init__()
        self.modules_per_cycle = config.modules_per_cycle
        self.input_norm = nn.LayerNorm(config.latent_dim)
        self.score_projection = nn.Linear(config.latent_dim, config.num_modules)

    def forward(self, latent: Tensor) -> RoutingDecision:
        pooled_latent = latent.mean(dim=(0, 1))
        scores = self.score_projection(self.input_norm(pooled_latent))
        selected_scores, selected_indices = torch.topk(
            scores, k=self.modules_per_cycle, dim=-1
        )
        selected_weights = torch.softmax(selected_scores, dim=-1)
        return RoutingDecision(scores, selected_indices, selected_weights)


class EMCModule(nn.Module):
    """An independent transformer-style processor returning a latent update."""

    def __init__(self, config: EMCConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.latent_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.latent_dim,
            num_heads=config.attention_heads,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(config.latent_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.latent_dim, config.resolved_module_hidden_dim),
            nn.GELU(),
            nn.Linear(config.resolved_module_hidden_dim, config.latent_dim),
        )

    def forward(self, latent: Tensor) -> Tensor:
        normalized = self.attention_norm(latent)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        state = latent + attended
        processed = state + self.feed_forward(self.feed_forward_norm(state))
        return processed - latent


class Integrator(nn.Module):
    """Combines routed updates and learns how strongly to change shared state."""

    def __init__(self, config: EMCConfig) -> None:
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
    ) -> Tensor:
        weighted_update = torch.einsum(
            "k,kbsd->bsd", routing_weights, module_updates
        )
        integration_input = torch.cat(
            (self.latent_norm(latent), weighted_update), dim=-1
        )
        candidate_update = self.update_projection(integration_input)
        update_gate = torch.sigmoid(self.gate_projection(integration_input))
        return latent + update_gate * candidate_update


class EMCModel(nn.Module):
    def __init__(self, config: EMCConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.latent_dim)
        self.router = NexusRouter(config)
        self.emc_modules = nn.ModuleList(
            EMCModule(config) for _ in range(config.num_modules)
        )
        self.integrator = Integrator(config)
        self.output_norm = nn.LayerNorm(config.latent_dim)
        self.output_projection = nn.Linear(config.latent_dim, config.vocab_size)

    def execute_selected_modules(
        self, latent: Tensor, selected_indices: Tensor
    ) -> Tensor:
        """Local execution boundary; every selected module receives the same state."""
        updates = [
            self.emc_modules[index](latent) for index in selected_indices.tolist()
        ]
        return torch.stack(updates, dim=0)

    def forward(
        self, token_ids: Tensor, *, return_trace: bool = False
    ) -> Tensor | EMCOutput:
        latent = self.token_embedding(token_ids)
        cycle_traces: list[EMCCycleTrace] = []

        for cycle in range(self.config.num_cycles):
            routing = self.router(latent)
            module_updates = self.execute_selected_modules(
                latent, routing.selected_indices
            )
            latent = self.integrator(
                latent, module_updates, routing.selected_weights
            )

            if return_trace:
                cycle_traces.append(
                    EMCCycleTrace(
                        cycle=cycle + 1,
                        selected_modules=tuple(routing.selected_indices.tolist()),
                        router_scores=routing.scores.detach().cpu(),
                        router_weights=routing.selected_weights.detach().cpu(),
                        latent_shape=tuple(latent.shape),
                    )
                )

        logits = self.output_projection(self.output_norm(latent))
        if return_trace:
            return EMCOutput(logits=logits, trace=tuple(cycle_traces))
        return logits
