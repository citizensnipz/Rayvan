from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .chunk_contracts import ChunkMetadata, ModuleInput
from .chunk_modules import ChunkGatedDeltaNetModule

ModuleFamily = Literal["gpt", "ssm", "recurrent", "delta"]


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
    """Selective diagonal state-space module with fused/parallel recurrence.

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
        self.ssm_backend = getattr(config, "ssm_backend", "auto")
        self.last_backend = "uninitialized"

    def forward(self, latent: Tensor) -> Tensor:
        internal = self.input_adapter(self.input_norm(latent))
        convolved = self.causal_convolution(
            F.pad(internal.transpose(1, 2), (self.kernel_size - 1, 0))
        ).transpose(1, 2)
        from .ssm_scan import affine_scan, resolve_backend
        self.last_backend = resolve_backend(self.ssm_backend, latent.device)
        # Project all tokens with GEMMs, instead of three tiny GEMMs per token.
        delta = F.softplus(self.delta_projection(convolved))
        candidate = torch.tanh(self.input_projection(convolved))
        gate = torch.sigmoid(self.gate_projection(convolved))
        dtype = torch.float64 if convolved.dtype == torch.float64 else torch.float32
        decay = torch.exp(-F.softplus(self.log_decay).to(dtype) * delta.to(dtype))
        state = affine_scan(decay, (1 - decay) * candidate.to(dtype), self.last_backend)
        return self.output_adapter((gate * state).to(convolved.dtype))


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
        # Frozen evaluation suffixes still need derivatives w.r.t. their input.
        # cuDNN inference forward does not retain the reserve needed by backward.
        # Select native autograd BEFORE forward only for this differentiable eval
        # path. Keep evaluation semantics, frozen parameters and ordinary cuDNN
        # training/no-grad inference unchanged; restore backend flags on exit.
        differentiable_eval = not self.recurrent.training and torch.is_grad_enabled() and internal.requires_grad
        with torch.backends.cudnn.flags(enabled=False) if differentiable_eval else nullcontext():
            recurrent_output, _ = self.recurrent(internal)
        return self.output_adapter(recurrent_output)


class DeltaEMCModule(EMCModuleBase):
    """Token-trajectory adapter for the existing Gated DeltaNet N1."""

    family: ModuleFamily = "delta"

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.module = ChunkGatedDeltaNetModule(config)
        self.shared_slots = config.shared_state_slots

    def forward(self, latent: Tensor) -> Tensor:
        shared = latent.mean(dim=1, keepdim=True).expand(
            -1, self.shared_slots, -1
        )
        batch = latent.size(0)
        output = self.module.forward_chunk(
            ModuleInput(
                chunk_latent=latent,
                shared_state=shared,
                lease_state=self.module.begin_lease(shared),
                metadata=ChunkMetadata(
                    request_indices=torch.arange(batch, device=latent.device),
                    chunk_index=0,
                    lease_ages=torch.ones(
                        batch, dtype=torch.long, device=latent.device
                    ),
                    module_index=0,
                    lease_ids=torch.zeros(
                        batch, 3, dtype=torch.long, device=latent.device
                    ),
                    continuing_lease=torch.zeros(
                        batch, dtype=torch.bool, device=latent.device
                    ),
                ),
            )
        )
        return output.token_proposal


def create_emc_module(config: Any, family: str) -> EMCModuleBase:
    if family == "gpt":
        return EMCModule(config)
    if family in {"ssm", "mamba"}:
        return StateSpaceEMCModule(config)
    if family in {"recurrent", "gru"}:
        return RecurrentEMCModule(config)
    if family in {"delta", "deltanet"}:
        return DeltaEMCModule(config)
    raise ValueError(f"unknown EMC module family: {family!r}")

