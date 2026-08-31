from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ModuleFamily = Literal["gpt", "ssm", "recurrent"]


class EMCModuleBase(nn.Module, ABC):
    family: ModuleFamily

    @abstractmethod
    def forward(self, latent: Tensor) -> Tensor:
        """Return one proposal with the same [batch, sequence, latent] shape."""
        raise NotImplementedError


class EMCModule(EMCModuleBase):
    """Existing GPT-style causal attention and FFN module."""

    family: ModuleFamily = "gpt"

    def __init__(self, config: Any) -> None:
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
        state = latent + attended
        processed = state + self.feed_forward(self.feed_forward_norm(state))
        return processed - latent


class StateSpaceEMCModule(EMCModuleBase):
    """Pure-PyTorch selective diagonal state-space module.

    State recurs only across tokens inside one forward call. It resets for every
    EMC cycle, batch, and inference request; no persistent memory is introduced.
    """

    family: ModuleFamily = "ssm"

    def __init__(self, config: Any) -> None:
        super().__init__()
        width = config.resolved_state_space_dim
        self.input_norm = nn.LayerNorm(config.latent_dim)
        self.input_adapter = nn.Linear(config.latent_dim, width)
        self.causal_convolution = nn.Conv1d(
            width,
            width,
            kernel_size=config.state_space_kernel_size,
            groups=width,
        )
        self.delta_projection = nn.Linear(width, width)
        self.input_projection = nn.Linear(width, width)
        self.gate_projection = nn.Linear(width, width)
        self.log_decay = nn.Parameter(torch.zeros(width))
        self.output_adapter = nn.Linear(width, config.latent_dim)
        self.kernel_size = config.state_space_kernel_size

    def forward(self, latent: Tensor) -> Tensor:
        internal = self.input_adapter(self.input_norm(latent))
        convolved = self.causal_convolution(
            F.pad(internal.transpose(1, 2), (self.kernel_size - 1, 0))
        ).transpose(1, 2)
        decay_rate = F.softplus(self.log_decay).unsqueeze(0)
        state = convolved.new_zeros(convolved.size(0), convolved.size(-1))
        outputs: list[Tensor] = []
        for token_state in convolved.unbind(dim=1):
            delta = F.softplus(self.delta_projection(token_state))
            decay = torch.exp(-decay_rate * delta)
            candidate = torch.tanh(self.input_projection(token_state))
            state = decay * state + (1.0 - decay) * candidate
            output = torch.sigmoid(self.gate_projection(token_state)) * state
            outputs.append(output)
        return self.output_adapter(torch.stack(outputs, dim=1))


class RecurrentEMCModule(EMCModuleBase):
    """GRU proposal module with sequence-local, non-persistent hidden state."""

    family: ModuleFamily = "recurrent"

    def __init__(self, config: Any) -> None:
        super().__init__()
        width = config.resolved_recurrent_dim
        self.input_norm = nn.LayerNorm(config.latent_dim)
        self.input_adapter = nn.Linear(config.latent_dim, width)
        self.recurrent = nn.GRU(width, width, batch_first=True)
        self.output_adapter = nn.Linear(width, config.latent_dim)

    def forward(self, latent: Tensor) -> Tensor:
        internal = self.input_adapter(self.input_norm(latent))
        recurrent_output, _ = self.recurrent(internal)
        return self.output_adapter(recurrent_output)


def create_emc_module(config: Any, family: str) -> EMCModuleBase:
    if family == "gpt":
        return EMCModule(config)
    if family in {"ssm", "mamba"}:
        return StateSpaceEMCModule(config)
    if family in {"recurrent", "gru"}:
        return RecurrentEMCModule(config)
    raise ValueError(f"unknown EMC module family: {family!r}")
