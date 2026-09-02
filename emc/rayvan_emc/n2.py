from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from .balancing import router_balance_loss
from .chunk_contracts import ChunkMetadata, ModuleInput, ModuleLeaseState
from .chunk_modules import (
    ChunkEMCModuleBase,
    ChunkGPTModule,
    ChunkGatedDeltaNetModule,
    ChunkRecurrentModule,
    ChunkStateSpaceModule,
)
from .integrator import Integrator, IntegratorTrace
from .model import EMCConfig, EMCCycleTrace, EMCModel, EMCOutput


N2_POPULATIONS: Mapping[str, tuple[str, ...]] = {
    "mixed": ("gpt", "ssm", "recurrent", "delta"),
    "supported": ("gpt", "ssm", "recurrent"),
    "gpt4": ("gpt", "gpt", "gpt", "gpt"),
    "ssm4": ("ssm", "ssm", "ssm", "ssm"),
    "recurrent4": ("recurrent", "recurrent", "recurrent", "recurrent"),
    "delta4": ("delta", "delta", "delta", "delta"),
}


@dataclass(frozen=True)
class N2Config(EMCConfig):
    """Configuration for one sparse N2 routing/integration event."""

    num_modules: int = 4
    modules_per_cycle: int = 2
    num_cycles: int = 1
    architecture_stage: str = "n2"
    n2_population: str = "mixed"
    n1_depth: int = 3
    n2_execution_mode: str = "sparse"
    n2_use_cuda_streams: bool = False
    loss_free_balance_enabled: bool = False

    def __post_init__(self) -> None:
        if self.n2_population not in N2_POPULATIONS:
            raise ValueError(f"unknown N2 population: {self.n2_population!r}")
        expected = N2_POPULATIONS[self.n2_population]
        if self.module_families is None:
            object.__setattr__(self, "module_families", expected)
        elif tuple(self.resolved_module_families) != expected:
            raise ValueError(
                "module_families must exactly match the selected N2 population preset"
            )
        super().__post_init__()
        if self.num_modules != len(expected):
            raise ValueError("num_modules must match the selected N2 population")
        if self.num_cycles != 1:
            raise ValueError("initial N2 experiments support exactly one N2 cycle")
        if self.n1_depth < 2:
            raise ValueError("n1_depth must be at least two substantial local blocks")
        if self.n2_execution_mode not in {"sparse", "dense"}:
            raise ValueError("n2_execution_mode must be sparse or dense")


@dataclass(frozen=True)
class N1PersistentState:
    block_states: tuple[ModuleLeaseState, ...]
    shared_state: Tensor


@dataclass(frozen=True)
class N1Input:
    shared_latent: Tensor
    shared_state: Tensor | None = None
    local_state: N1PersistentState | None = None
    request_indices: Tensor | None = None


@dataclass(frozen=True)
class N1Diagnostics:
    node_id: int
    node_name: str
    family: str
    blocks_executed: int
    chunks_per_block: int
    block_invocations: int
    parameters: int
    approximate_flops: int
    output_latent_size: int
    stateful: bool
    state_resets: int
    continuation_probability: float | None
    average_lease_length: float | None
    state_change_magnitude: Tensor | float | None


@dataclass(frozen=True)
class N1Output:
    proposal: Tensor
    local_state: N1PersistentState | None
    diagnostics: N1Diagnostics


class HomogeneousN1Node(nn.Module, ABC):
    """Common N2 boundary around a stack of one mathematical block family."""

    family: str
    block_type: type[ChunkEMCModuleBase]
    stateful: bool = False

    def __init__(self, config: N2Config, *, node_id: int, node_name: str) -> None:
        super().__init__()
        self.config = config
        self.node_id = node_id
        self.node_name = node_name
        self.blocks = nn.ModuleList(
            self.block_type(config) for _ in range(config.n1_depth)
        )
        self.state_initializer = nn.Linear(
            config.latent_dim, config.shared_state_slots * config.latent_dim
        )
        self._parameter_count = sum(
            parameter.numel() for parameter in self.parameters()
        )
        self._execution_count = 0

    @property
    def execution_count(self) -> int:
        return self._execution_count

    @property
    def parameter_count(self) -> int:
        return self._parameter_count

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    def approximate_flops(self, sequence_length: int) -> int:
        """Parameter-use proxy plus explicit chunked attention products for GPT."""
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        flops = 2 * self.parameter_count * sequence_length
        if self.family == "gpt":
            chunks = math.ceil(sequence_length / self.config.chunk_size)
            chunk_length = min(sequence_length, self.config.chunk_size)
            attended_length = chunk_length + self.config.shared_state_slots
            flops += (
                self.block_count
                * chunks
                * 4
                * attended_length
                * attended_length
                * self.config.latent_dim
            )
        return int(flops)

    def forward(self, node_input: N1Input) -> N1Output:
        latent = node_input.shared_latent
        if latent.ndim != 3 or latent.size(-1) != self.config.latent_dim:
            raise ValueError("shared_latent must have shape [batch, sequence, latent_dim]")
        batch, sequence_length, _ = latent.shape
        if batch == 0:
            return N1Output(
                proposal=latent,
                local_state=None,
                diagnostics=self._diagnostics(
                    sequence_length,
                    batch=0,
                    state_change=None,
                    state_reset=False,
                ),
            )
        request_indices = (
            node_input.request_indices.to(device=latent.device, dtype=torch.long)
            if node_input.request_indices is not None
            else torch.arange(batch, device=latent.device)
        )
        if tuple(request_indices.shape) != (batch,):
            raise ValueError("request_indices must contain one ID per request")
        if node_input.local_state is not None:
            if len(node_input.local_state.block_states) != self.block_count:
                raise ValueError("local_state must contain one lease state per N1 block")
            local_shared = node_input.local_state.shared_state
            incoming_states = node_input.local_state.block_states
        else:
            local_shared = (
                node_input.shared_state
                if node_input.shared_state is not None
                else self.state_initializer(latent.mean(dim=1)).reshape(
                    batch, self.config.shared_state_slots, self.config.latent_dim
                )
            )
            incoming_states = tuple(
                block.begin_lease(local_shared) for block in self.blocks
            )
        if tuple(local_shared.shape) != (
            batch,
            self.config.shared_state_slots,
            self.config.latent_dim,
        ):
            raise ValueError(
                "shared_state must have shape [batch, shared_state_slots, latent_dim]"
            )

        self._execution_count += 1
        residual_scale = 1.0 / math.sqrt(self.block_count)
        current = latent
        next_states: list[ModuleLeaseState] = []
        initial_shared = local_shared
        chunks = math.ceil(sequence_length / self.config.chunk_size)
        node_ids = torch.full_like(request_indices, self.node_id)
        lease_ids = tuple(
            torch.stack(
                (
                    request_indices,
                    node_ids,
                    torch.full_like(request_indices, block_index),
                ),
                dim=-1,
            )
            for block_index in range(self.block_count)
        )
        lease_ages = tuple(
            torch.full(
                (batch,),
                chunk_index + 1,
                device=latent.device,
                dtype=torch.long,
            )
            for chunk_index in range(chunks)
        )
        continuing_leases = tuple(
            torch.full(
                (batch,),
                chunk_index > 0,
                device=latent.device,
                dtype=torch.bool,
            )
            for chunk_index in range(chunks)
        )
        for block_index, (block, initial_state) in enumerate(
            zip(self.blocks, incoming_states, strict=True)
        ):
            lease_state = initial_state
            layer_chunks: list[Tensor] = []
            for chunk_index, start in enumerate(
                range(0, sequence_length, self.config.chunk_size)
            ):
                end = min(start + self.config.chunk_size, sequence_length)
                chunk = current[:, start:end]
                output = block.forward_chunk(
                    ModuleInput(
                        chunk_latent=chunk,
                        shared_state=local_shared,
                        lease_state=lease_state,
                        metadata=ChunkMetadata(
                            request_indices=request_indices,
                            chunk_index=chunk_index,
                            lease_ages=lease_ages[chunk_index],
                            module_index=block_index,
                            lease_ids=lease_ids[block_index],
                            continuing_lease=continuing_leases[chunk_index],
                        ),
                    )
                )
                layer_chunks.append(chunk + residual_scale * output.token_proposal)
                local_shared = (
                    local_shared + residual_scale * output.state_proposal
                )
                lease_state = output.new_lease_state
            current = torch.cat(layer_chunks, dim=1)
            next_states.append(lease_state)

        state_change = (
            (local_shared - initial_shared).float().norm().detach()
            if self.stateful
            else None
        )
        persistent_state = (
            N1PersistentState(tuple(next_states), local_shared)
            if self.stateful
            else None
        )
        return N1Output(
            proposal=current - latent,
            local_state=persistent_state,
            diagnostics=self._diagnostics(
                sequence_length,
                batch=batch,
                state_change=state_change,
                state_reset=node_input.local_state is None,
            ),
        )

    def _diagnostics(
        self,
        sequence_length: int,
        *,
        batch: int,
        state_change: Tensor | None,
        state_reset: bool,
    ) -> N1Diagnostics:
        chunks = math.ceil(sequence_length / self.config.chunk_size)
        return N1Diagnostics(
            node_id=self.node_id,
            node_name=self.node_name,
            family=self.family,
            blocks_executed=self.block_count,
            chunks_per_block=chunks,
            block_invocations=self.block_count * chunks,
            parameters=self.parameter_count,
            approximate_flops=self.approximate_flops(sequence_length),
            output_latent_size=self.config.latent_dim,
            stateful=self.stateful,
            state_resets=batch if self.stateful and state_reset else 0,
            continuation_probability=(
                (chunks - 1) / chunks if self.stateful and chunks else None
            ),
            average_lease_length=float(chunks) if self.stateful else None,
            state_change_magnitude=state_change,
        )


class GPTN1Node(HomogeneousN1Node):
    family = "gpt"
    block_type = ChunkGPTModule


class SSMN1Node(HomogeneousN1Node):
    family = "ssm"
    block_type = ChunkStateSpaceModule
    stateful = True


class RecurrentN1Node(HomogeneousN1Node):
    family = "recurrent"
    block_type = ChunkRecurrentModule
    stateful = True


class DeltaN1Node(HomogeneousN1Node):
    family = "delta"
    block_type = ChunkGatedDeltaNetModule
    stateful = True


_N1_TYPES: Mapping[str, type[HomogeneousN1Node]] = {
    "gpt": GPTN1Node,
    "ssm": SSMN1Node,
    "recurrent": RecurrentN1Node,
    "delta": DeltaN1Node,
}


def n1_node_names(families: tuple[str, ...]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    names: list[str] = []
    for family in families:
        ordinal = counts.get(family, 0)
        counts[family] = ordinal + 1
        suffix = chr(ord("A") + ordinal)
        display_family = {
            "gpt": "GPT",
            "ssm": "SSM",
            "recurrent": "Recurrent",
            "delta": "Delta",
        }[family]
        names.append(f"{display_family}-N1-{suffix}")
    return tuple(names)


def create_n1_node(
    config: N2Config, family: str, *, node_id: int, node_name: str
) -> HomogeneousN1Node:
    try:
        node_type = _N1_TYPES[family]
    except KeyError as error:
        raise ValueError(f"unsupported N1 family: {family!r}") from error
    return node_type(config, node_id=node_id, node_name=node_name)


@dataclass(frozen=True)
class N2RoutingDecision:
    scores: Tensor
    pre_top_k_probabilities: Tensor
    selected_indices: Tensor
    selected_weights: Tensor
    selected_slots: Tensor


@dataclass(frozen=True)
class N2DispatchPlan:
    """CUDA-resident permutation metadata for request-level expert batches."""

    expert_ids: Tensor
    source_indices: Tensor
    slot_indices: Tensor
    permutation: Tensor
    inverse_permutation: Tensor
    sorted_expert_ids: Tensor
    sorted_source_indices: Tensor
    sorted_slot_indices: Tensor
    expert_counts: Tensor
    expert_offsets: Tensor

    @classmethod
    def from_routing(
        cls, selected_indices: Tensor, *, num_experts: int
    ) -> N2DispatchPlan:
        if selected_indices.ndim != 2:
            raise ValueError("selected_indices must have shape [batch, selected]")
        batch, selected = selected_indices.shape
        expert_ids = selected_indices.reshape(-1)
        source_indices = (
            torch.arange(batch, device=selected_indices.device)
            .unsqueeze(1)
            .expand(-1, selected)
            .reshape(-1)
        )
        slot_indices = (
            torch.arange(selected, device=selected_indices.device)
            .unsqueeze(0)
            .expand(batch, -1)
            .reshape(-1)
        )
        permutation = torch.argsort(expert_ids, stable=True)
        assignment_indices = torch.arange(
            expert_ids.numel(), device=selected_indices.device
        )
        inverse_permutation = torch.empty_like(permutation).scatter(
            0, permutation, assignment_indices
        )
        expert_counts = torch.bincount(expert_ids, minlength=num_experts)
        expert_offsets = torch.cat(
            (
                expert_counts.new_zeros(1),
                expert_counts.cumsum(dim=0),
            )
        )
        return cls(
            expert_ids=expert_ids,
            source_indices=source_indices,
            slot_indices=slot_indices,
            permutation=permutation,
            inverse_permutation=inverse_permutation,
            sorted_expert_ids=expert_ids.index_select(0, permutation),
            sorted_source_indices=source_indices.index_select(0, permutation),
            sorted_slot_indices=slot_indices.index_select(0, permutation),
            expert_counts=expert_counts,
            expert_offsets=expert_offsets,
        )


class N2Nexus(nn.Module):
    """Request-level learned router over complete homogeneous N1 subsystems."""

    def __init__(self, config: N2Config) -> None:
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.latent_dim)
        self.score_projection = nn.Linear(config.latent_dim, config.num_modules)

    def forward(
        self,
        shared_latent: Tensor,
        *,
        top_k: int,
        availability_mask: Tensor | None = None,
    ) -> N2RoutingDecision:
        pooled = self.input_norm(shared_latent).mean(dim=1)
        scores = self.score_projection(pooled)
        if availability_mask is not None:
            if availability_mask.ndim != 1 or availability_mask.numel() != scores.size(-1):
                raise ValueError("availability_mask must contain one value per N1 node")
            available = availability_mask.to(device=scores.device, dtype=torch.bool)
            if int(available.sum().item()) < top_k:
                raise ValueError("availability_mask leaves fewer N1 nodes than top-K")
            scores = scores.masked_fill(~available, -torch.inf)
        selected_scores, selected_indices = torch.topk(scores, k=top_k, dim=-1)
        selected_weights = torch.softmax(selected_scores, dim=-1)
        slots = torch.arange(top_k, device=scores.device).expand_as(selected_indices)
        return N2RoutingDecision(
            scores=scores,
            pre_top_k_probabilities=torch.softmax(scores, dim=-1),
            selected_indices=selected_indices,
            selected_weights=selected_weights,
            selected_slots=slots,
        )


class N2Integrator(nn.Module):
    """Proposal-attention Integrator adapted to request-routed N1 proposals."""

    def __init__(self, config: N2Config) -> None:
        super().__init__()
        self.proposal_integrator = Integrator(config)

    def forward(
        self,
        shared_latent: Tensor,
        proposals: Tensor,
        selected_weights: Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, IntegratorTrace]:
        if proposals.ndim != 4:
            raise ValueError("proposals must have shape [batch, sequence, selected, latent]")
        routing_weights = selected_weights.unsqueeze(1).expand(
            -1, shared_latent.size(1), -1
        )
        return self.proposal_integrator(
            shared_latent,
            proposals,
            routing_weights,
            return_diagnostics=return_diagnostics,
        )


@dataclass(frozen=True)
class N2State:
    local_states: Mapping[int, N1PersistentState]


@dataclass(frozen=True)
class N2ExecutionTrace:
    selected_node_ids: Tensor
    selected_node_weights: Tensor
    pre_top_k_probabilities: Tensor
    selected_slots: Tensor
    dispatch_permutation: Tensor
    dispatch_inverse_permutation: Tensor
    dispatch_counts: Tensor
    dispatch_offsets: Tensor
    execution_mode: str
    executed_node_ids: tuple[int, ...]
    actual_node_executions: int
    theoretical_all_node_executions: int
    node_diagnostics: tuple[N1Diagnostics, ...]


class N2EMCModel(EMCModel):
    """One N2 event routing requests through substantial homogeneous N1 nodes."""

    def __init__(
        self,
        config: N2Config,
        *,
        nodes: tuple[HomogeneousN1Node, ...] | None = None,
        nexus: N2Nexus | None = None,
        integrator: N2Integrator | None = None,
    ) -> None:
        nn.Module.__init__(self)
        if config.architecture_stage != "n2":
            raise ValueError("N2EMCModel requires architecture_stage=n2")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.latent_dim)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.latent_dim
        )
        self.router = nexus or N2Nexus(config)
        names = n1_node_names(config.resolved_module_families)
        resolved_nodes = nodes or tuple(
            create_n1_node(
                config, family, node_id=index, node_name=names[index]
            )
            for index, family in enumerate(config.resolved_module_families)
        )
        if len(resolved_nodes) != config.num_modules:
            raise ValueError("N2 node count must equal config.num_modules")
        self.n1_nodes = nn.ModuleList(resolved_nodes)
        self.integrator = integrator or N2Integrator(config)
        self.output_norm = nn.LayerNorm(config.latent_dim)
        self.output_projection = nn.Linear(config.latent_dim, config.vocab_size)
        if config.tie_embeddings:
            self.output_projection.weight = self.token_embedding.weight
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.output_projection.bias)
        self._active_top_k = config.modules_per_cycle
        self._cuda_streams: tuple[torch.cuda.Stream, ...] | None = None

    @property
    def nexus(self) -> N2Nexus:
        return self.router

    @property
    def emc_modules(self) -> nn.ModuleList:
        """Diagnostic expert boundary: whole N1 nodes, never their internal blocks."""
        return self.n1_nodes

    @property
    def module_families(self) -> tuple[str, ...]:
        return tuple(node.family for node in self.n1_nodes)

    @property
    def expert_names(self) -> tuple[str, ...]:
        return tuple(node.node_name for node in self.n1_nodes)

    @property
    def active_top_k(self) -> int:
        return self._active_top_k

    def set_active_top_k(self, top_k: int) -> None:
        if not 1 <= top_k <= self.config.num_modules:
            raise ValueError("active top-K must be between one and num_modules")
        self._active_top_k = top_k

    def forward(
        self,
        token_ids: Tensor,
        *,
        return_trace: bool = False,
        return_cycle_logits: bool = False,
        balance_entropy_floor: float = 0.75,
        availability_mask: Tensor | None = None,
        module_descriptors: Tensor | None = None,
        evaluation_cycle_limit: int | None = None,
        diagnostic_forced_modules: Tensor | None = None,
        diagnostic_zero_proposal_mask: Tensor | None = None,
        n2_state: N2State | None = None,
    ) -> Tensor | EMCOutput:
        del module_descriptors
        if evaluation_cycle_limit not in {None, 1}:
            raise ValueError("N2 currently supports exactly one routing/integration cycle")
        sequence_length = token_ids.size(1)
        if sequence_length > self.config.max_sequence_length:
            raise ValueError("sequence exceeds configured maximum")
        positions = torch.arange(sequence_length, device=token_ids.device)
        latent = self.token_embedding(token_ids) + self.position_embedding(positions)
        routing = self.router(
            latent,
            top_k=self.active_top_k,
            availability_mask=availability_mask,
        )
        if self.config.n2_execution_mode == "dense":
            routing = _dense_n2_routing(routing, self.config.num_modules)
        if diagnostic_forced_modules is not None:
            routing = _force_n2_routing(
                routing,
                diagnostic_forced_modules,
                batch=token_ids.size(0),
                top_k=self.active_top_k,
                num_nodes=self.config.num_modules,
            )
        (
            proposals,
            updated_states,
            diagnostics,
            executed,
            dispatch,
        ) = self._execute_selected_nodes(
            latent,
            routing.selected_indices,
            n2_state=n2_state,
        )
        if diagnostic_zero_proposal_mask is not None:
            zero_mask = diagnostic_zero_proposal_mask.to(
                device=proposals.device, dtype=torch.bool
            )
            if zero_mask.ndim != 1 or zero_mask.numel() != self.config.num_modules:
                raise ValueError(
                    "diagnostic_zero_proposal_mask must contain one value per N1 node"
                )
            selected_to_zero = zero_mask[routing.selected_indices]
            proposals = proposals.masked_fill(
                selected_to_zero.unsqueeze(1).unsqueeze(-1), 0
            )
        integrated = self.integrator(
            latent,
            proposals,
            routing.selected_weights,
            return_diagnostics=return_trace,
        )
        if return_trace:
            if not isinstance(integrated, tuple):
                raise RuntimeError("N2 Integrator did not return requested diagnostics")
            latent, integrator_trace = integrated
        else:
            if isinstance(integrated, tuple):
                raise RuntimeError("N2 Integrator unexpectedly returned diagnostics")
            latent = integrated
            integrator_trace = None
        logits = self.output_projection(self.output_norm(latent))
        if not (return_trace or return_cycle_logits):
            return logits

        expanded_indices = routing.selected_indices.unsqueeze(1).expand(
            -1, sequence_length, -1
        )
        expanded_scores = routing.scores.unsqueeze(1).expand(
            -1, sequence_length, -1
        )
        expanded_weights = routing.selected_weights.unsqueeze(1).expand(
            -1, sequence_length, -1
        )
        trace = (
            EMCCycleTrace(
                cycle=1,
                selected_modules=executed,
                router_scores=expanded_scores.detach(),
                router_weights=expanded_weights.detach(),
                latent_shape=tuple(latent.shape),
                selected_indices=expanded_indices.detach(),
                integrator_trace=integrator_trace,
                module_families=self.module_families,
                expert_names=self.expert_names,
                local_diagnostics=tuple(diagnostics),
            ),
        ) if return_trace else ()
        execution_trace = N2ExecutionTrace(
            selected_node_ids=routing.selected_indices.detach(),
            selected_node_weights=routing.selected_weights.detach(),
            pre_top_k_probabilities=routing.pre_top_k_probabilities.detach(),
            selected_slots=routing.selected_slots.detach(),
            executed_node_ids=executed,
            actual_node_executions=routing.selected_indices.numel(),
            theoretical_all_node_executions=token_ids.size(0) * self.config.num_modules,
            node_diagnostics=tuple(diagnostics),
            dispatch_permutation=dispatch.permutation.detach(),
            dispatch_inverse_permutation=dispatch.inverse_permutation.detach(),
            dispatch_counts=dispatch.expert_counts.detach(),
            dispatch_offsets=dispatch.expert_offsets.detach(),
            execution_mode=self.config.n2_execution_mode,
        )
        balance_loss = router_balance_loss(
            expanded_scores,
            expanded_indices,
            entropy_floor=balance_entropy_floor,
        )
        return EMCOutput(
            logits=logits,
            trace=trace,
            router_balance_loss=balance_loss.detach() * 0.0,
            cycle_logits=(logits,) if return_cycle_logits else None,
            chunk_trace=execution_trace,
            n2_state=N2State(updated_states),
        )

    def _execute_selected_nodes(
        self,
        latent: Tensor,
        selected_indices: Tensor,
        *,
        n2_state: N2State | None,
    ) -> tuple[
        Tensor,
        dict[int, N1PersistentState],
        list[N1Diagnostics],
        tuple[int, ...],
        N2DispatchPlan,
    ]:
        batch, sequence, latent_dim = latent.shape
        selected = selected_indices.size(1)
        dispatch = N2DispatchPlan.from_routing(
            selected_indices, num_experts=self.config.num_modules
        )
        node_batches: list[tuple[Tensor, Tensor]] = []
        for node_id in range(self.config.num_modules):
            node_mask = dispatch.sorted_expert_ids == node_id
            request_rows = dispatch.sorted_source_indices[node_mask]
            node_batches.append(
                (
                    latent.index_select(0, request_rows),
                    request_rows,
                )
            )

        if self.config.n2_use_cuda_streams and latent.is_cuda:
            node_outputs = self._execute_node_batches_concurrently(
                node_batches, n2_state=n2_state
            )
        else:
            node_outputs = [
                self._execute_node_batch(
                    node_id,
                    node_batch,
                    request_rows,
                    n2_state=n2_state,
                )
                for node_id, (node_batch, request_rows) in enumerate(node_batches)
            ]

        states: dict[int, N1PersistentState] = {}
        diagnostics: list[N1Diagnostics] = []
        executed: list[int] = []
        grouped_proposals: list[Tensor] = []
        for node_id, node_output in enumerate(node_outputs):
            grouped_proposals.append(node_output.proposal)
            if node_output.proposal.size(0) == 0:
                continue
            if node_output.local_state is not None:
                states[node_id] = node_output.local_state
            diagnostics.append(node_output.diagnostics)
            executed.append(node_id)
        sorted_proposals = torch.cat(grouped_proposals, dim=0)
        flattened = sorted_proposals.index_select(
            0, dispatch.inverse_permutation
        )
        proposals = flattened.reshape(
            batch, selected, sequence, latent_dim
        ).permute(0, 2, 1, 3)
        return proposals, states, diagnostics, tuple(executed), dispatch

    def _execute_node_batch(
        self,
        node_id: int,
        latent: Tensor,
        request_rows: Tensor,
        *,
        n2_state: N2State | None,
    ) -> N1Output:
        return self.n1_nodes[node_id](
            N1Input(
                shared_latent=latent,
                local_state=(
                    n2_state.local_states.get(node_id)
                    if n2_state is not None
                    else None
                ),
                request_indices=request_rows,
            )
        )

    def _execute_node_batches_concurrently(
        self,
        node_batches: list[tuple[Tensor, Tensor]],
        *,
        n2_state: N2State | None,
    ) -> list[N1Output]:
        device = node_batches[0][0].device
        if self._cuda_streams is None:
            self._cuda_streams = tuple(
                torch.cuda.Stream(device=device)
                for _ in range(self.config.num_modules)
            )
        current_stream = torch.cuda.current_stream(device)
        outputs: list[N1Output] = []
        for node_id, ((node_batch, request_rows), stream) in enumerate(
            zip(node_batches, self._cuda_streams, strict=True)
        ):
            stream.wait_stream(current_stream)
            node_batch.record_stream(stream)
            request_rows.record_stream(stream)
            with torch.cuda.stream(stream):
                outputs.append(
                    self._execute_node_batch(
                        node_id,
                        node_batch,
                        request_rows,
                        n2_state=n2_state,
                    )
                )
        for stream, output in zip(self._cuda_streams, outputs, strict=True):
            current_stream.wait_stream(stream)
            output.proposal.record_stream(current_stream)
            if output.local_state is not None:
                output.local_state.shared_state.record_stream(current_stream)
                for block_state in output.local_state.block_states:
                    for tensor in block_state.tensors.values():
                        tensor.record_stream(current_stream)
        return outputs


def _dense_n2_routing(
    routing: N2RoutingDecision, num_nodes: int
) -> N2RoutingDecision:
    batch = routing.scores.size(0)
    selected = torch.arange(
        num_nodes, device=routing.scores.device
    ).expand(batch, -1)
    slots = torch.arange(
        num_nodes, device=routing.scores.device
    ).expand_as(selected)
    return N2RoutingDecision(
        scores=routing.scores,
        pre_top_k_probabilities=routing.pre_top_k_probabilities,
        selected_indices=selected,
        selected_weights=routing.pre_top_k_probabilities,
        selected_slots=slots,
    )


def _force_n2_routing(
    routing: N2RoutingDecision,
    forced_modules: Tensor,
    *,
    batch: int,
    top_k: int,
    num_nodes: int,
) -> N2RoutingDecision:
    forced = forced_modules.to(device=routing.scores.device, dtype=torch.long)
    if forced.ndim == 1:
        forced = forced.reshape(1, -1).expand(batch, -1)
    elif forced.ndim == 3:
        if forced.size(0) != batch or forced.size(2) != top_k:
            raise ValueError("forced N2 routing has incompatible token shape")
        if not bool((forced == forced[:, :1]).all().item()):
            raise ValueError("N2 forced routing must be constant across each request")
        forced = forced[:, 0]
    if tuple(forced.shape) != (batch, top_k):
        raise ValueError("forced N2 routing must provide exactly top-K IDs per request")
    if bool(((forced < 0) | (forced >= num_nodes)).any().item()):
        raise ValueError("forced N2 routing contains an out-of-range node ID")
    if any(torch.unique(row).numel() != top_k for row in forced):
        raise ValueError("forced N2 routing cannot select one node twice")
    forced_scores = torch.gather(routing.scores, -1, forced)
    return N2RoutingDecision(
        scores=routing.scores,
        pre_top_k_probabilities=routing.pre_top_k_probabilities,
        selected_indices=forced,
        selected_weights=torch.softmax(forced_scores, dim=-1),
        selected_slots=torch.arange(top_k, device=forced.device).expand_as(forced),
    )
