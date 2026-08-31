from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor


@dataclass(frozen=True)
class ModuleCapabilities:
    family: str
    internal_width: int
    state_elements_per_request: int
    preferred_precision: str
    backend: str


@dataclass(frozen=True)
class ChunkMetadata:
    request_indices: Tensor
    chunk_index: int
    lease_ages: Tensor
    module_index: int
    lease_ids: Tensor
    continuing_lease: Tensor


@dataclass
class ModuleLeaseState:
    """Opaque module-private tensors with batch dimension first."""

    tensors: dict[str, Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class ModuleInput:
    chunk_latent: Tensor
    shared_state: Tensor
    lease_state: ModuleLeaseState
    metadata: ChunkMetadata


@dataclass(frozen=True)
class ModuleOutput:
    token_proposal: Tensor
    state_proposal: Tensor
    new_lease_state: ModuleLeaseState
