from __future__ import annotations

import math
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .chunk_contracts import (
    ModuleCapabilities,
    ModuleInput,
    ModuleLeaseState,
    ModuleOutput,
)


class ChunkEMCModuleBase(nn.Module, ABC):
    family: str

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.latent_dim = config.latent_dim
        self.shared_slots = config.shared_state_slots
        self.shared_condition = nn.Linear(config.latent_dim, config.latent_dim)
        self.state_output = nn.Linear(config.latent_dim * 2, config.latent_dim)

    @property
    @abstractmethod
    def capabilities(self) -> ModuleCapabilities:
        raise NotImplementedError

    @abstractmethod
    def begin_lease(self, shared_state: Tensor) -> ModuleLeaseState:
        raise NotImplementedError

    @abstractmethod
    def forward_chunk(self, module_input: ModuleInput) -> ModuleOutput:
        raise NotImplementedError

    def end_lease(self, _lease_state: ModuleLeaseState) -> None:
        return None

    def _condition_chunk(self, chunk: Tensor, shared_state: Tensor) -> Tensor:
        context = self.shared_condition(shared_state.mean(dim=1)).unsqueeze(1)
        return chunk + context

    def _state_proposal(
        self, token_proposal: Tensor, shared_state: Tensor
    ) -> Tensor:
        pooled = token_proposal.mean(dim=1, keepdim=True).expand(
            -1, shared_state.size(1), -1
        )
        return self.state_output(torch.cat((shared_state, pooled), dim=-1))


class ChunkGPTModule(ChunkEMCModuleBase):
    family = "gpt"

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.attention_norm = nn.LayerNorm(config.latent_dim)
        self.shared_norm = nn.LayerNorm(config.latent_dim)
        self.attention = nn.MultiheadAttention(
            config.latent_dim, config.attention_heads, batch_first=True
        )
        self.feed_forward_norm = nn.LayerNorm(config.latent_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.latent_dim, config.resolved_module_hidden_dim),
            nn.GELU(),
            nn.Linear(config.resolved_module_hidden_dim, config.latent_dim),
        )
        attention_length = config.chunk_size + config.shared_state_slots
        self.register_buffer(
            "_causal_mask",
            torch.ones(attention_length, attention_length, dtype=torch.bool).triu(
                diagonal=1
            ),
            persistent=False,
        )
        self._capabilities = ModuleCapabilities(
            family=self.family,
            internal_width=config.resolved_module_hidden_dim,
            state_elements_per_request=0,
            preferred_precision="model",
            backend="torch_mha",
        )

    @property
    def capabilities(self) -> ModuleCapabilities:
        return self._capabilities

    def begin_lease(self, shared_state: Tensor) -> ModuleLeaseState:
        return ModuleLeaseState()

    def forward_chunk(self, module_input: ModuleInput) -> ModuleOutput:
        conditioned = self._condition_chunk(
            module_input.chunk_latent, module_input.shared_state
        )
        memory = self.shared_norm(module_input.shared_state)
        combined = torch.cat((memory, conditioned), dim=1)
        normalized = self.attention_norm(combined)
        length = combined.size(1)
        causal_mask = self._causal_mask[:length, :length]
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        state = combined + attended
        processed = state + self.feed_forward(self.feed_forward_norm(state))
        token_state = processed[:, memory.size(1) :]
        token_proposal = token_state - conditioned
        return ModuleOutput(
            token_proposal=token_proposal,
            state_proposal=self._state_proposal(
                token_proposal, module_input.shared_state
            ),
            new_lease_state=ModuleLeaseState(),
        )


class ChunkStateSpaceModule(ChunkEMCModuleBase):
    family = "ssm"

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        width = config.resolved_state_space_dim
        self.width = width
        self.kernel_size = config.state_space_kernel_size
        self.input_norm = nn.LayerNorm(config.latent_dim)
        self.input_adapter = nn.Linear(config.latent_dim, width)
        self.causal_convolution = nn.Conv1d(
            width, width, kernel_size=self.kernel_size, groups=width
        )
        self.delta_projection = nn.Linear(width, width)
        self.input_projection = nn.Linear(width, width)
        self.gate_projection = nn.Linear(width, width)
        self.log_decay = nn.Parameter(torch.zeros(width))
        self.output_adapter = nn.Linear(width, config.latent_dim)
        self.state_initializer = nn.Linear(config.latent_dim, width)
        self.register_buffer(
            "_scan_causal_mask",
            torch.ones(config.chunk_size, config.chunk_size, dtype=torch.bool).tril(),
            persistent=False,
        )
        self._capabilities = ModuleCapabilities(
            family=self.family,
            internal_width=width,
            state_elements_per_request=width + (self.kernel_size - 1) * width,
            preferred_precision="fp32_scan",
            backend=config.ssm_backend,
        )

    @property
    def capabilities(self) -> ModuleCapabilities:
        return self._capabilities

    def begin_lease(self, shared_state: Tensor) -> ModuleLeaseState:
        state = self.state_initializer(shared_state.mean(dim=1))
        history = state.new_zeros(
            state.size(0), self.kernel_size - 1, self.width
        )
        return ModuleLeaseState({"state": state, "conv_history": history})

    def forward_chunk(self, module_input: ModuleInput) -> ModuleOutput:
        conditioned = self._condition_chunk(
            module_input.chunk_latent, module_input.shared_state
        )
        internal = self.input_adapter(self.input_norm(conditioned))
        history = module_input.lease_state.tensors["conv_history"].to(
            internal.dtype
        )
        convolution_input = torch.cat((history, internal), dim=1)
        convolved = self.causal_convolution(
            convolution_input.transpose(1, 2)
        ).transpose(1, 2)
        delta = F.softplus(self.delta_projection(convolved)).float()
        log_decay = -F.softplus(self.log_decay).reshape(1, 1, -1) * delta
        candidate = torch.tanh(self.input_projection(convolved)).float()
        initial_state = module_input.lease_state.tensors["state"].float()
        states = _parallel_diagonal_scan(
            log_decay,
            candidate,
            initial_state,
            self._scan_causal_mask,
        )
        gate = torch.sigmoid(self.gate_projection(convolved)).float()
        token_proposal = self.output_adapter(
            (gate * states).to(internal.dtype)
        ).to(module_input.chunk_latent.dtype)
        new_history = convolution_input[:, -(self.kernel_size - 1) :]
        return ModuleOutput(
            token_proposal=token_proposal,
            state_proposal=self._state_proposal(
                token_proposal, module_input.shared_state
            ),
            new_lease_state=ModuleLeaseState(
                {
                    "state": states[:, -1],
                    "conv_history": new_history,
                }
            ),
        )


class ChunkRecurrentModule(ChunkEMCModuleBase):
    family = "recurrent"

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.width = config.resolved_recurrent_dim
        self.precision = config.recurrent_precision
        self.input_norm = nn.LayerNorm(config.latent_dim)
        self.input_adapter = nn.Linear(config.latent_dim, self.width)
        self.recurrent = nn.GRU(self.width, self.width, batch_first=True)
        self.output_adapter = nn.Linear(self.width, config.latent_dim)
        self.state_initializer = nn.Linear(config.latent_dim, self.width)
        self._capabilities = ModuleCapabilities(
            family=self.family,
            internal_width=self.width,
            state_elements_per_request=self.width,
            preferred_precision=self.precision,
            backend=config.recurrent_backend,
        )

    @property
    def capabilities(self) -> ModuleCapabilities:
        return self._capabilities

    def begin_lease(self, shared_state: Tensor) -> ModuleLeaseState:
        hidden = self.state_initializer(shared_state.mean(dim=1))
        return ModuleLeaseState({"hidden": hidden})

    def forward_chunk(self, module_input: ModuleInput) -> ModuleOutput:
        conditioned = self._condition_chunk(
            module_input.chunk_latent, module_input.shared_state
        )
        precision_context = _module_precision_context(
            conditioned.device, self.precision
        )
        with precision_context:
            internal = self.input_adapter(self.input_norm(conditioned))
            hidden = module_input.lease_state.tensors["hidden"].to(
                internal.dtype
            ).unsqueeze(0)
            recurrent_output, new_hidden = self.recurrent(internal, hidden)
            token_proposal = self.output_adapter(recurrent_output)
        token_proposal = token_proposal.to(module_input.chunk_latent.dtype)
        return ModuleOutput(
            token_proposal=token_proposal,
            state_proposal=self._state_proposal(
                token_proposal, module_input.shared_state
            ),
            new_lease_state=ModuleLeaseState(
                {"hidden": new_hidden.squeeze(0)}
            ),
        )


class ChunkGatedDeltaNetModule(ChunkEMCModuleBase):
    """Chunkwise gated delta-rule associative memory.

    Implements the gated delta recurrence from Yang et al., arXiv:2412.06464:
    S_t = S_{t-1}[alpha_t(I - beta_t k_t k_t^T)] + beta_t v_t k_t^T.
    Prefix states are evaluated with an O(log C) affine associative scan.
    """

    family = "delta"

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.width = config.resolved_delta_internal_dim
        self.heads = config.delta_heads
        self.head_dim = self.width // self.heads
        self.input_norm = nn.LayerNorm(config.latent_dim)
        self.query_projection = nn.Linear(config.latent_dim, self.width)
        self.key_projection = nn.Linear(config.latent_dim, self.width)
        self.value_projection = nn.Linear(config.latent_dim, self.width)
        self.alpha_projection = nn.Linear(config.latent_dim, self.heads)
        self.beta_projection = nn.Linear(config.latent_dim, self.heads)
        self.output_gate = nn.Linear(config.latent_dim, self.width)
        self.output_adapter = nn.Linear(self.width, config.latent_dim)
        self.post_norm = nn.LayerNorm(config.latent_dim)
        self.post_ffn = nn.Sequential(
            nn.Linear(config.latent_dim, config.resolved_delta_ffn_dim),
            nn.GELU(),
            nn.Linear(config.resolved_delta_ffn_dim, config.latent_dim),
        )
        self.initial_key = nn.Linear(config.latent_dim, self.width)
        self.initial_value = nn.Linear(config.latent_dim, self.width)
        state_elements = self.heads * self.head_dim * self.head_dim
        self.max_transition_bytes = config.delta_max_transition_bytes
        self.register_buffer(
            "_identity",
            torch.eye(self.head_dim, dtype=torch.float32).reshape(
                1, 1, 1, self.head_dim, self.head_dim
            ),
            persistent=False,
        )
        self._capabilities = ModuleCapabilities(
            family=self.family,
            internal_width=self.width,
            state_elements_per_request=state_elements,
            preferred_precision="fp32_scan",
            backend=config.delta_backend,
        )

    @property
    def capabilities(self) -> ModuleCapabilities:
        return self._capabilities

    def begin_lease(self, shared_state: Tensor) -> ModuleLeaseState:
        summary = shared_state.mean(dim=1)
        key = self.initial_key(summary).reshape(
            summary.size(0), self.heads, self.head_dim
        )
        value = self.initial_value(summary).reshape(
            summary.size(0), self.heads, self.head_dim
        )
        memory = torch.einsum("bhv,bhk->bhvk", value, key)
        memory = memory / math.sqrt(self.head_dim)
        return ModuleLeaseState({"memory": memory})

    def forward_chunk(self, module_input: ModuleInput) -> ModuleOutput:
        conditioned = self._condition_chunk(
            module_input.chunk_latent, module_input.shared_state
        )
        normalized = self.input_norm(conditioned)
        transition_bytes = (
            conditioned.size(0)
            * conditioned.size(1)
            * self.heads
            * self.head_dim
            * self.head_dim
            * 4
        )
        if transition_bytes > self.max_transition_bytes:
            raise RuntimeError(
                "DeltaNet transition tensor exceeds the configured safety limit: "
                f"{transition_bytes / 2**20:.1f} MiB > "
                f"{self.max_transition_bytes / 2**20:.1f} MiB"
            )
        shape = (
            conditioned.size(0),
            conditioned.size(1),
            self.heads,
            self.head_dim,
        )
        query = F.normalize(self.query_projection(normalized).reshape(shape), dim=-1)
        key = F.normalize(self.key_projection(normalized).reshape(shape), dim=-1)
        value = torch.tanh(self.value_projection(normalized).reshape(shape))
        alpha = torch.sigmoid(self.alpha_projection(normalized)).float()
        beta = torch.sigmoid(self.beta_projection(normalized)).float()
        eye = self._identity
        key_float = key.float()
        transition = alpha[..., None, None] * (
            eye
            - beta[..., None, None]
            * torch.einsum("bchk,bchl->bchkl", key_float, key_float)
        )
        write = beta[..., None, None] * torch.einsum(
            "bchv,bchk->bchvk", value.float(), key_float
        )
        prefix_transition, prefix_write = _parallel_affine_scan(
            transition, write
        )
        initial_memory = module_input.lease_state.tensors["memory"].float()
        memory_states = torch.einsum(
            "bhvk,bchkl->bchvl", initial_memory, prefix_transition
        ) + prefix_write
        output = torch.einsum(
            "bchvk,bchk->bchv", memory_states, query.float()
        ).reshape(conditioned.size(0), conditioned.size(1), self.width)
        output = output * torch.sigmoid(self.output_gate(normalized)).float()
        token_proposal = self.output_adapter(output.to(conditioned.dtype))
        token_proposal = token_proposal + self.post_ffn(
            self.post_norm(token_proposal)
        )
        token_proposal = token_proposal.to(module_input.chunk_latent.dtype)
        return ModuleOutput(
            token_proposal=token_proposal,
            state_proposal=self._state_proposal(
                token_proposal, module_input.shared_state
            ),
            new_lease_state=ModuleLeaseState(
                {"memory": memory_states[:, -1]}
            ),
        )


def create_chunk_module(config: Any, family: str) -> ChunkEMCModuleBase:
    aliases = {"mamba": "ssm", "gru": "recurrent", "deltanet": "delta"}
    resolved = aliases.get(family, family)
    if resolved == "gpt":
        return ChunkGPTModule(config)
    if resolved == "ssm":
        return ChunkStateSpaceModule(config)
    if resolved == "recurrent":
        return ChunkRecurrentModule(config)
    if resolved == "delta":
        return ChunkGatedDeltaNetModule(config)
    raise ValueError(f"unknown chunk EMC module family: {family!r}")


def _parallel_diagonal_scan(
    log_decay: Tensor,
    candidate: Tensor,
    initial_state: Tensor,
    causal_mask: Tensor | None = None,
) -> Tensor:
    decay = torch.exp(log_decay)
    write = (1.0 - decay) * candidate
    log_prefix = torch.cumsum(log_decay, dim=1)
    prefix_by_dimension = log_prefix.transpose(1, 2)
    log_coefficients = (
        prefix_by_dimension.unsqueeze(-1)
        - prefix_by_dimension.unsqueeze(-2)
    )
    length = log_decay.size(1)
    if causal_mask is None:
        causal = torch.ones(
            length, length, dtype=torch.bool, device=log_decay.device
        ).tril()
    else:
        causal = causal_mask[:length, :length]
    masked_log_coefficients = log_coefficients.masked_fill(
        ~causal.reshape(1, 1, length, length), -torch.inf
    )
    coefficients = torch.exp(masked_log_coefficients)
    accumulated = torch.einsum("bhtj,bjh->bth", coefficients, write)
    initial_contribution = torch.exp(log_prefix) * initial_state.unsqueeze(1)
    return accumulated + initial_contribution


def _parallel_affine_scan(
    transition: Tensor, write: Tensor
) -> tuple[Tensor, Tensor]:
    prefix_transition = transition
    prefix_write = write
    offset = 1
    length = transition.size(1)
    while offset < length:
        left_transition = prefix_transition[:, :-offset]
        right_transition = prefix_transition[:, offset:]
        left_write = prefix_write[:, :-offset]
        right_write = prefix_write[:, offset:]
        composed_transition = torch.matmul(
            left_transition, right_transition
        )
        composed_write = torch.matmul(
            left_write, right_transition
        ) + right_write
        prefix_transition = torch.cat(
            (prefix_transition[:, :offset], composed_transition), dim=1
        )
        prefix_write = torch.cat(
            (prefix_write[:, :offset], composed_write), dim=1
        )
        offset *= 2
    return prefix_transition, prefix_write


def _module_precision_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "model":
        return nullcontext()
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]
    if dtype == torch.float32:
        return torch.autocast(device_type="cuda", enabled=False)
    return torch.autocast(device_type="cuda", dtype=dtype)
