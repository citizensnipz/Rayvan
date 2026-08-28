from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    latent_dim: int = 128
    num_layers: int = 4
    attention_heads: int = 4
    feed_forward_dim: int | None = None
    max_sequence_length: int = 128

    def __post_init__(self) -> None:
        positive_fields = {
            "vocab_size": self.vocab_size,
            "latent_dim": self.latent_dim,
            "num_layers": self.num_layers,
            "attention_heads": self.attention_heads,
            "max_sequence_length": self.max_sequence_length,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.latent_dim % self.attention_heads != 0:
            raise ValueError("latent_dim must be divisible by attention_heads")
        if self.feed_forward_dim is not None and self.feed_forward_dim <= 0:
            raise ValueError("feed_forward_dim must be positive when provided")

    @property
    def resolved_feed_forward_dim(self) -> int:
        return self.feed_forward_dim or self.latent_dim * 4


class DecoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.latent_dim)
        self.attention = nn.MultiheadAttention(
            config.latent_dim, config.attention_heads, batch_first=True
        )
        self.feed_forward_norm = nn.LayerNorm(config.latent_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.latent_dim, config.resolved_feed_forward_dim),
            nn.GELU(),
            nn.Linear(config.resolved_feed_forward_dim, config.latent_dim),
        )

    def forward(self, latent: Tensor) -> Tensor:
        normalized = self.attention_norm(latent)
        sequence_length = latent.size(1)
        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=latent.device,
        ).triu(diagonal=1)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        latent = latent + attended
        return latent + self.feed_forward(self.feed_forward_norm(latent))


class TransformerLanguageModel(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.latent_dim)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.latent_dim
        )
        self.blocks = nn.ModuleList(
            DecoderBlock(config) for _ in range(config.num_layers)
        )
        self.output_norm = nn.LayerNorm(config.latent_dim)
        self.output_projection = nn.Linear(config.latent_dim, config.vocab_size)

    def forward(self, token_ids: Tensor) -> Tensor:
        sequence_length = token_ids.size(1)
        if sequence_length > self.config.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds configured maximum "
                f"{self.config.max_sequence_length}"
            )
        positions = torch.arange(sequence_length, device=token_ids.device)
        latent = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            latent = block(latent)
        return self.output_projection(self.output_norm(latent))
