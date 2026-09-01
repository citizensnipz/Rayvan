from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .chunk_contracts import ChunkMetadata, ModuleInput, ModuleLeaseState
from .chunk_modules import ChunkEMCModuleBase, create_chunk_module
from .chunked import SharedCore
from .model import EMCConfig


class HeterogeneousSerialModel(nn.Module):
    """Serial composition of the same chunk module implementations used by EMC.

    Each configured family executes exactly once per chunk, in
    ``config.module_families`` order. Module token and state proposals are applied
    as residual updates. There is no Nexus, sparse selection, or Integrator.
    """

    def __init__(
        self,
        config: EMCConfig,
        *,
        modules: Sequence[ChunkEMCModuleBase] | None = None,
    ) -> None:
        super().__init__()
        if config.architecture_stage != "n1_chunked":
            raise ValueError(
                "HeterogeneousSerialModel requires architecture_stage=n1_chunked"
            )
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.latent_dim)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.latent_dim
        )
        self.shared_core = SharedCore(config)
        resolved_modules = list(modules) if modules is not None else [
            create_chunk_module(config, family)
            for family in config.resolved_module_families
        ]
        if len(resolved_modules) != config.num_modules:
            raise ValueError("serial module count must equal config.num_modules")
        self.emc_modules = nn.ModuleList(resolved_modules)
        self.output_norm = nn.LayerNorm(config.latent_dim)
        self.output_projection = nn.Linear(config.latent_dim, config.vocab_size)
        if config.tie_embeddings:
            self.output_projection.weight = self.token_embedding.weight
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.output_projection.bias)
        self.last_execution_order: tuple[str, ...] = ()

    @property
    def module_families(self) -> tuple[str, ...]:
        return tuple(module.family for module in self.emc_modules)

    def forward(self, token_ids: Tensor) -> Tensor:
        batch, sequence_length = token_ids.shape
        if sequence_length > self.config.max_sequence_length:
            raise ValueError("sequence exceeds configured maximum")
        positions = torch.arange(sequence_length, device=token_ids.device)
        embedded = self.token_embedding(token_ids) + self.position_embedding(positions)
        shared_state = self.shared_core.initialize_shared_state(embedded[:, 0])
        lease_states: list[ModuleLeaseState | None] = [
            None for _ in self.emc_modules
        ]
        request_indices = torch.arange(batch, device=token_ids.device)
        output_chunks: list[Tensor] = []
        execution_order: list[str] = []

        for chunk_index, start in enumerate(
            range(0, sequence_length, self.config.chunk_size)
        ):
            end = min(start + self.config.chunk_size, sequence_length)
            chunk = self.shared_core.prepare_chunk(
                embedded[:, start:end], shared_state
            )
            for module_index, module in enumerate(self.emc_modules):
                lease_state = lease_states[module_index]
                if lease_state is None:
                    lease_state = module.begin_lease(shared_state)
                module_output = module.forward_chunk(
                    ModuleInput(
                        chunk_latent=chunk,
                        shared_state=shared_state,
                        lease_state=lease_state,
                        metadata=ChunkMetadata(
                            request_indices=request_indices,
                            chunk_index=chunk_index,
                            lease_ages=torch.full(
                                (batch,),
                                chunk_index + 1,
                                device=token_ids.device,
                                dtype=torch.long,
                            ),
                            module_index=module_index,
                            lease_ids=torch.stack(
                                (
                                    request_indices,
                                    torch.full_like(request_indices, module_index),
                                    torch.zeros_like(request_indices),
                                ),
                                dim=-1,
                            ),
                            continuing_lease=torch.full(
                                (batch,),
                                chunk_index > 0,
                                device=token_ids.device,
                                dtype=torch.bool,
                            ),
                        ),
                    )
                )
                chunk = chunk + module_output.token_proposal
                shared_state = shared_state + module_output.state_proposal
                lease_states[module_index] = module_output.new_lease_state
                execution_order.append(module.family)
            output_chunks.append(chunk)

        self.last_execution_order = tuple(execution_order)
        latent = torch.cat(output_chunks, dim=1)
        return self.output_projection(self.output_norm(latent))
