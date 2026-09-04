from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .chunk_contracts import (
    ChunkMetadata,
    ModuleInput,
    ModuleLeaseState,
)
from .chunk_modules import ChunkEMCModuleBase, create_chunk_module
from .integrator import Integrator, IntegratorTrace
from .model import EMCConfig, EMCOutput


@dataclass(frozen=True)
class RequestPoolTrace:
    module_indices: Tensor
    scores: Tensor
    family_composition: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ChunkRoutingTrace:
    chunk_index: int
    active_modules: Tensor
    active_families: tuple[tuple[str, ...], ...]
    routing_scores: Tensor
    routing_weights: Tensor
    routing_entropy: Tensor
    continuing_leases: Tensor
    lease_ages: Tensor
    lease_generations: Tensor
    switch_rate: float
    retained_rate: float
    persistence_contribution: Tensor
    switching_contribution: Tensor
    balance_bias: Tensor
    executed_modules: tuple[int, ...]
    computed_chunk_module_pairs: int
    retained_chunk_module_pairs: int
    lease_state_norm: float
    lease_state_change: float
    state_reset_count: int
    token_integrator_trace: IntegratorTrace
    state_integrator_trace: IntegratorTrace


@dataclass(frozen=True)
class ChunkedExecutionTrace:
    request_pool: RequestPoolTrace
    chunks: tuple[ChunkRoutingTrace, ...]
    final_shared_state_shape: tuple[int, ...]
    modules_touched: tuple[int, ...]
    population_fraction_touched: float
    long_window_utilization: Tensor
    final_balance_bias: Tensor
    mean_active_lease_age: float
    overall_switch_rate: float
    overall_retained_rate: float


@dataclass(frozen=True)
class ChunkRoutingDecision:
    scores: Tensor
    selected_indices: Tensor
    selected_weights: Tensor
    persistence_contribution: Tensor
    switching_contribution: Tensor


class SharedCore(nn.Module):
    def __init__(self, config: EMCConfig) -> None:
        super().__init__()
        self.enabled = config.shared_core_enabled
        self.ingress_norm = nn.LayerNorm(config.latent_dim)
        self.shared_condition = nn.Linear(config.latent_dim, config.latent_dim)
        self.initial_condition = nn.Linear(config.latent_dim, config.latent_dim)
        self.initial_slots = nn.Parameter(
            torch.empty(config.shared_state_slots, config.latent_dim)
        )
        nn.init.normal_(self.initial_slots, mean=0.0, std=0.02)
        if self.enabled:
            self.residual = nn.Sequential(
                nn.LayerNorm(config.latent_dim),
                nn.Linear(
                    config.latent_dim,
                    config.resolved_shared_core_hidden_dim,
                ),
                nn.GELU(),
                nn.Linear(
                    config.resolved_shared_core_hidden_dim,
                    config.latent_dim,
                ),
            )
        else:
            self.residual = None

    def initialize_shared_state(self, first_token: Tensor) -> Tensor:
        batch = first_token.size(0)
        slots = self.initial_slots.unsqueeze(0).expand(batch, -1, -1)
        condition = self.initial_condition(first_token).unsqueeze(1)
        return slots + condition

    def prepare_chunk(self, chunk: Tensor, shared_state: Tensor) -> Tensor:
        prepared = self.ingress_norm(chunk)
        prepared = prepared + self.shared_condition(
            shared_state.mean(dim=1)
        ).unsqueeze(1)
        if self.residual is not None:
            prepared = prepared + self.residual(prepared)
        return prepared


class ChunkNexus(nn.Module):
    def __init__(self, config: EMCConfig) -> None:
        super().__init__()
        self.config = config
        descriptor_dim = config.resolved_router_descriptor_dim
        self.context_norm = nn.LayerNorm(config.latent_dim)
        self.request_query = nn.Linear(config.latent_dim, descriptor_dim)
        self.chunk_query = nn.Linear(config.latent_dim, descriptor_dim)
        self.module_descriptors = nn.Parameter(
            torch.empty(config.num_modules, descriptor_dim)
        )
        nn.init.normal_(self.module_descriptors, mean=0.0, std=0.02)
        self.register_buffer(
            "balance_bias", torch.zeros(config.num_modules), persistent=True
        )
        self.register_buffer(
            "routing_totals", torch.zeros(config.num_modules), persistent=True
        )
        self.register_buffer(
            "chunks_observed", torch.zeros((), dtype=torch.long), persistent=True
        )

    @property
    def active_top_k(self) -> int:
        return getattr(
            self, "_active_top_k", self.config.resolved_active_top_k
        )

    def set_active_top_k(self, top_k: int) -> None:
        self._active_top_k = top_k

    def select_request_pool(
        self,
        first_token: Tensor,
        shared_state: Tensor,
        availability_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        top_k = self.active_top_k
        context = first_token + shared_state.mean(dim=1)
        query = self.request_query(self.context_norm(context))
        scores = torch.einsum("bd,md->bm", query, self.module_descriptors)
        scores = scores / math.sqrt(self.module_descriptors.size(1))
        scores = scores + self.balance_bias
        eligible = self._availability_mask(
            first_token.size(0), availability_mask
        )
        eligible_count = int(eligible[0].sum().item())
        if eligible_count < top_k:
            raise ValueError("availability leaves fewer modules than active top-K")
        scores = scores.masked_fill(~eligible, -torch.inf)
        pool = torch.topk(
            scores,
            k=min(self.config.resolved_request_pool_size, eligible_count),
            dim=-1,
        ).indices
        return scores, pool

    def route_chunk(
        self,
        first_token: Tensor,
        shared_state: Tensor,
        pool_mask: Tensor,
        request_scores: Tensor,
        previous_active: Tensor,
        lease_ages: Tensor,
        availability_mask: Tensor | None = None,
    ) -> ChunkRoutingDecision:
        top_k = self.active_top_k
        context = first_token + shared_state.mean(dim=1)
        query = self.chunk_query(self.context_norm(context))
        relevance = torch.einsum(
            "bd,md->bm", query, self.module_descriptors
        ) / math.sqrt(self.module_descriptors.size(1))
        persistence = previous_active.to(relevance.dtype) * self.config.persistence_bonus
        switching = (~previous_active).to(relevance.dtype) * (-self.config.switch_cost)
        if self.config.minimum_lease_chunks:
            protected = previous_active & (
                lease_ages < self.config.minimum_lease_chunks
            )
            persistence = persistence + protected.to(relevance.dtype) * 1e4
        scores = (
            relevance
            + request_scores
            + self.balance_bias
            + persistence
            + switching
        )
        eligible = pool_mask & self._availability_mask(
            first_token.size(0), availability_mask
        )
        if torch.any(eligible.sum(dim=-1) < top_k):
            raise ValueError("request pool leaves fewer modules than active top-K")
        scores = scores.masked_fill(~eligible, -torch.inf)
        selected_scores, selected = torch.topk(
            scores, k=top_k, dim=-1
        )
        weights = torch.softmax(selected_scores, dim=-1)
        return ChunkRoutingDecision(
            scores=scores,
            selected_indices=selected,
            selected_weights=weights,
            persistence_contribution=persistence,
            switching_contribution=switching,
        )

    @torch.no_grad()
    def _update_balance_bias(self, selected: Tensor) -> None:
        counts = torch.bincount(
            selected.reshape(-1), minlength=self.config.num_modules
        ).to(self.balance_bias.dtype)
        self.routing_totals.add_(counts)
        self.chunks_observed.add_(1)
        if not self.config.loss_free_balance_enabled:
            return
        if int(self.chunks_observed.item()) <= self.config.balance_warmup_chunks:
            return
        observed = counts / counts.sum().clamp_min(1)
        if self.config.balance_target_utilization is None:
            target = torch.full_like(observed, 1.0 / self.config.num_modules)
        else:
            target = observed.new_tensor(
                self.config.balance_target_utilization
            )
            target = target / target.sum()
        self.balance_bias.add_(
            self.config.balance_bias_lr * (target - observed)
        )
        self.balance_bias.clamp_(
            -self.config.balance_bias_limit,
            self.config.balance_bias_limit,
        )

    def _availability_mask(
        self, batch: int, availability_mask: Tensor | None
    ) -> Tensor:
        if availability_mask is None:
            return torch.ones(
                batch,
                self.config.num_modules,
                dtype=torch.bool,
                device=self.module_descriptors.device,
            )
        if availability_mask.ndim != 1 or availability_mask.numel() != self.config.num_modules:
            raise ValueError("availability_mask must contain one value per module")
        return availability_mask.to(
            device=self.module_descriptors.device, dtype=torch.bool
        ).unsqueeze(0).expand(batch, -1)


class ChunkIntegrator(nn.Module):
    def __init__(self, config: EMCConfig) -> None:
        super().__init__()
        self.token_integrator = Integrator(config)
        self.state_integrator = Integrator(config)

    def forward(
        self,
        chunk_latent: Tensor,
        shared_state: Tensor,
        token_proposals: Tensor,
        state_proposals: Tensor,
        routing_weights: Tensor,
        *,
        return_diagnostics: bool,
    ) -> tuple[Tensor, Tensor, IntegratorTrace | None, IntegratorTrace | None]:
        token_candidates = token_proposals.permute(0, 2, 1, 3)
        token_weights = routing_weights.unsqueeze(1).expand(
            -1, chunk_latent.size(1), -1
        )
        state_candidates = state_proposals.permute(0, 2, 1, 3)
        state_weights = routing_weights.unsqueeze(1).expand(
            -1, shared_state.size(1), -1
        )
        token_result = self.token_integrator(
            chunk_latent,
            token_candidates,
            token_weights,
            return_diagnostics=return_diagnostics,
        )
        state_result = self.state_integrator(
            shared_state,
            state_candidates,
            state_weights,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            if not isinstance(token_result, tuple) or not isinstance(state_result, tuple):
                raise RuntimeError("chunk Integrator did not return diagnostics")
            updated_chunk, token_trace = token_result
            updated_state, state_trace = state_result
            return updated_chunk, updated_state, token_trace, state_trace
        if isinstance(token_result, tuple) or isinstance(state_result, tuple):
            raise RuntimeError("chunk Integrator unexpectedly returned diagnostics")
        return token_result, state_result, None, None


class ChunkedEMCModel(nn.Module):
    def __init__(
        self,
        config: EMCConfig,
        *,
        modules: Sequence[ChunkEMCModuleBase] | None = None,
        nexus: ChunkNexus | None = None,
    ) -> None:
        super().__init__()
        if config.architecture_stage != "n1_chunked":
            raise ValueError("ChunkedEMCModel requires architecture_stage=n1_chunked")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.latent_dim)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.latent_dim
        )
        self.shared_core = SharedCore(config)
        self.router = nexus or ChunkNexus(config)
        resolved_modules = list(modules) if modules is not None else [
            create_chunk_module(config, family)
            for family in config.resolved_module_families
        ]
        if len(resolved_modules) != config.num_modules:
            raise ValueError("chunk module count must equal config.num_modules")
        self.emc_modules = nn.ModuleList(resolved_modules)
        self.integrator = ChunkIntegrator(config)
        self.output_norm = nn.LayerNorm(config.latent_dim)
        self.output_projection = nn.Linear(config.latent_dim, config.vocab_size)
        if config.tie_embeddings:
            self.output_projection.weight = self.token_embedding.weight
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.output_projection.bias)
        self.last_execution_trace: ChunkedExecutionTrace | None = None
        self._active_top_k = config.resolved_active_top_k

    @property
    def module_families(self) -> tuple[str, ...]:
        return tuple(module.family for module in self.emc_modules)

    @property
    def expert_names(self) -> tuple[str, ...]:
        counts: dict[str, int] = {}
        names = []
        for family in self.module_families:
            counts[family] = counts.get(family, 0) + 1
            names.append(f"{family}-{counts[family]}")
        return tuple(names)

    @property
    def active_top_k(self) -> int:
        return self._active_top_k

    def set_active_top_k(self, top_k: int) -> None:
        if not 1 <= top_k <= self.config.num_modules:
            raise ValueError("active top-K must be between one and num_modules")
        if top_k > self.config.resolved_request_pool_size:
            raise ValueError("active top-K cannot exceed request_pool_size")
        self._active_top_k = top_k
        router_setter = getattr(self.router, "set_active_top_k", None)
        if router_setter is not None:
            router_setter(top_k)

    def forward(
        self,
        token_ids: Tensor,
        *,
        return_trace: bool = False,
        balance_entropy_floor: float = 0.75,
        availability_mask: Tensor | None = None,
        diagnostic_forced_modules: Tensor | None = None,
        diagnostic_zero_proposal_mask: Tensor | None = None,
        **_unused: Any,
    ) -> Tensor | EMCOutput:
        del balance_entropy_floor
        batch, sequence_length = token_ids.shape
        if sequence_length > self.config.max_sequence_length:
            raise ValueError("sequence exceeds configured maximum")
        positions = torch.arange(sequence_length, device=token_ids.device)
        embedded = self.token_embedding(token_ids) + self.position_embedding(positions)
        first_token = embedded[:, 0]
        shared_state = self.shared_core.initialize_shared_state(first_token)
        pool_scores, request_pool = self.router.select_request_pool(
            first_token, shared_state, availability_mask
        )
        if diagnostic_forced_modules is not None:
            request_pool = _include_forced_in_pool(
                request_pool,
                diagnostic_forced_modules,
                num_modules=self.config.num_modules,
            )
        pool_mask = torch.zeros(
            batch,
            self.config.num_modules,
            dtype=torch.bool,
            device=token_ids.device,
        )
        pool_mask.scatter_(1, request_pool, True)
        request_trace = RequestPoolTrace(
            module_indices=request_pool.detach().cpu(),
            scores=pool_scores.detach().cpu(),
            family_composition=tuple(
                tuple(self.module_families[index] for index in row.tolist())
                for row in request_pool.detach().cpu()
            ),
        )
        previous_active = torch.zeros_like(pool_mask)
        lease_valid = torch.zeros_like(pool_mask)
        lease_ages = torch.zeros_like(pool_mask, dtype=torch.long)
        lease_generations = torch.zeros_like(pool_mask, dtype=torch.long)
        lease_states: list[ModuleLeaseState | None] = [
            None for _ in range(self.config.num_modules)
        ]
        output_chunks: list[Tensor] = []
        chunk_traces: list[ChunkRoutingTrace] = []
        modules_touched: set[int] = set()
        forward_selections: list[Tensor] = []

        for chunk_index, start in enumerate(
            range(0, sequence_length, self.config.chunk_size)
        ):
            end = min(start + self.config.chunk_size, sequence_length)
            chunk = self.shared_core.prepare_chunk(
                embedded[:, start:end], shared_state
            )
            decision = self.router.route_chunk(
                chunk[:, 0],
                shared_state,
                pool_mask,
                pool_scores,
                previous_active,
                lease_ages,
                availability_mask,
            )
            if diagnostic_forced_modules is not None:
                decision = _force_chunk_routing(
                    decision,
                    diagnostic_forced_modules,
                    batch=batch,
                    active_top_k=self.active_top_k,
                    num_modules=self.config.num_modules,
                )
            selected_mask = torch.zeros_like(previous_active)
            selected_mask.scatter_(1, decision.selected_indices, True)
            forward_selections.append(decision.selected_indices.detach())
            self._end_inactive_leases(
                lease_states, previous_active, selected_mask
            )
            continuing = selected_mask & previous_active & lease_valid
            next_ages = torch.where(
                selected_mask,
                torch.where(continuing, lease_ages + 1, torch.ones_like(lease_ages)),
                torch.zeros_like(lease_ages),
            )
            next_generations = torch.where(
                selected_mask & ~continuing,
                lease_generations + 1,
                lease_generations,
            )
            previous_lease_states = lease_states
            token_proposals, state_proposals, lease_states, executed = (
                self._execute_sparse_chunk(
                    chunk,
                    shared_state,
                    decision.selected_indices,
                    lease_states,
                    continuing,
                    next_ages,
                    next_generations,
                    chunk_index,
                )
            )
            lease_state_norm, lease_state_change = _lease_state_diagnostics(
                previous_lease_states, lease_states
            )
            if diagnostic_zero_proposal_mask is not None:
                zero_mask = diagnostic_zero_proposal_mask.to(
                    device=token_proposals.device, dtype=torch.bool
                )
                if zero_mask.ndim != 1 or zero_mask.numel() != self.config.num_modules:
                    raise ValueError(
                        "diagnostic_zero_proposal_mask must contain one value per module"
                    )
                selected_to_zero = zero_mask[decision.selected_indices]
                token_proposals = token_proposals.masked_fill(
                    selected_to_zero[:, :, None, None], 0
                )
                state_proposals = state_proposals.masked_fill(
                    selected_to_zero[:, :, None, None], 0
                )
            modules_touched.update(executed)
            integrated = self.integrator(
                chunk,
                shared_state,
                token_proposals,
                state_proposals,
                decision.selected_weights,
                return_diagnostics=return_trace,
            )
            updated_chunk, shared_state, token_trace, state_trace = integrated
            output_chunks.append(updated_chunk)
            lease_valid = selected_mask
            previous_active = selected_mask
            lease_ages = next_ages
            lease_generations = next_generations

            if return_trace:
                if token_trace is None or state_trace is None:
                    raise RuntimeError("chunk trace requested without Integrator trace")
                retained = continuing.sum().item()
                total_active = selected_mask.sum().item()
                routing_entropy = -(
                    decision.selected_weights
                    * decision.selected_weights.clamp_min(1e-9).log()
                ).sum(dim=-1)
                chunk_traces.append(
                    ChunkRoutingTrace(
                        chunk_index=chunk_index,
                        active_modules=decision.selected_indices.detach().cpu(),
                        active_families=tuple(
                            tuple(
                                self.module_families[index]
                                for index in row.tolist()
                            )
                            for row in decision.selected_indices.detach().cpu()
                        ),
                        routing_scores=decision.scores.detach().cpu(),
                        routing_weights=decision.selected_weights.detach().cpu(),
                        routing_entropy=routing_entropy.detach().cpu(),
                        continuing_leases=continuing.detach().cpu(),
                        lease_ages=next_ages.detach().cpu(),
                        lease_generations=next_generations.detach().cpu(),
                        switch_rate=(
                            1.0 - continuing.sum().item() / total_active
                            if total_active
                            else 0.0
                        ),
                        retained_rate=(
                            retained / total_active if total_active else 0.0
                        ),
                        persistence_contribution=(
                            decision.persistence_contribution.detach().cpu()
                        ),
                        switching_contribution=(
                            decision.switching_contribution.detach().cpu()
                        ),
                        balance_bias=self.router.balance_bias.detach().cpu().clone(),
                        executed_modules=executed,
                        computed_chunk_module_pairs=batch * self.active_top_k,
                        retained_chunk_module_pairs=batch * self.active_top_k,
                        lease_state_norm=lease_state_norm,
                        lease_state_change=lease_state_change,
                        state_reset_count=int(
                            (selected_mask & ~continuing).sum().item()
                        ),
                        token_integrator_trace=token_trace,
                        state_integrator_trace=state_trace,
                    )
                )

        if self.training and forward_selections:
            # Update once from the complete forward. Mutating this bias after
            # every chunk made checkpoint evaluation route with a different
            # state than the one which produced the recorded gradients.
            self.router._update_balance_bias(torch.cat(forward_selections, dim=0))
        latent = torch.cat(output_chunks, dim=1)
        logits = self.output_projection(self.output_norm(latent))
        overall_switch_rate = _overall_switch_rate(chunk_traces)
        execution_trace = ChunkedExecutionTrace(
            request_pool=request_trace,
            chunks=tuple(chunk_traces),
            final_shared_state_shape=tuple(shared_state.shape),
            modules_touched=tuple(sorted(modules_touched)),
            population_fraction_touched=(
                len(modules_touched) / self.config.num_modules
            ),
            long_window_utilization=(
                self.router.routing_totals
                / self.router.routing_totals.sum().clamp_min(1)
            ).detach().cpu(),
            final_balance_bias=self.router.balance_bias.detach().cpu().clone(),
            mean_active_lease_age=_mean_active_lease_age(chunk_traces),
            overall_switch_rate=overall_switch_rate,
            overall_retained_rate=1.0 - overall_switch_rate,
        )
        self.last_execution_trace = execution_trace
        if return_trace:
            return EMCOutput(
                logits=logits,
                trace=(),
                router_balance_loss=logits.new_zeros(()),
                chunk_trace=execution_trace,
            )
        return logits

    def _end_inactive_leases(
        self,
        lease_states: list[ModuleLeaseState | None],
        previous_active: Tensor,
        selected_mask: Tensor,
    ) -> None:
        for module_index, module in enumerate(self.emc_modules):
            previous_state = lease_states[module_index]
            if previous_state is None or not previous_state.tensors:
                continue
            ended_requests = (
                previous_active[:, module_index]
                & ~selected_mask[:, module_index]
            ).nonzero(as_tuple=False).reshape(-1)
            if ended_requests.numel():
                module.end_lease(
                    _select_lease_state(previous_state, ended_requests)
                )

    def _execute_sparse_chunk(
        self,
        chunk: Tensor,
        shared_state: Tensor,
        selected_indices: Tensor,
        lease_states: list[ModuleLeaseState | None],
        continuing: Tensor,
        lease_ages: Tensor,
        lease_generations: Tensor,
        chunk_index: int,
    ) -> tuple[
        Tensor,
        Tensor,
        list[ModuleLeaseState | None],
        tuple[int, ...],
    ]:
        batch, chunk_length, latent_dim = chunk.shape
        top_k = selected_indices.size(1)
        token_proposals = chunk.new_zeros(
            batch, top_k, chunk_length, latent_dim
        )
        state_proposals = shared_state.new_zeros(
            batch,
            top_k,
            shared_state.size(1),
            latent_dim,
        )
        next_states: list[ModuleLeaseState | None] = [
            None for _ in range(self.config.num_modules)
        ]
        executed: list[int] = []

        for module_index, module in enumerate(self.emc_modules):
            assignments = (selected_indices == module_index).nonzero(
                as_tuple=False
            )
            if assignments.numel() == 0:
                continue
            executed.append(module_index)
            request_indices = assignments[:, 0]
            slots = assignments[:, 1]
            selected_shared = shared_state.index_select(0, request_indices)
            continuing_rows = continuing[
                request_indices, module_index
            ]
            lease_state = _lease_for_assignments(
                module,
                selected_shared,
                lease_states[module_index],
                request_indices,
                continuing_rows,
            )
            module_output = module.forward_chunk(
                ModuleInput(
                    chunk_latent=chunk.index_select(0, request_indices),
                    shared_state=selected_shared,
                    lease_state=lease_state,
                    metadata=ChunkMetadata(
                        request_indices=request_indices,
                        chunk_index=chunk_index,
                        lease_ages=lease_ages[
                            request_indices, module_index
                        ],
                        module_index=module_index,
                        lease_ids=torch.stack(
                            (
                                request_indices,
                                torch.full_like(request_indices, module_index),
                                lease_generations[
                                    request_indices, module_index
                                ],
                            ),
                            dim=-1,
                        ),
                        continuing_lease=continuing_rows,
                    ),
                )
            )
            token_proposals[request_indices, slots] = (
                module_output.token_proposal.to(dtype=token_proposals.dtype)
            )
            state_proposals[request_indices, slots] = (
                module_output.state_proposal.to(dtype=state_proposals.dtype)
            )
            next_states[module_index] = _scatter_lease_state(
                module_output.new_lease_state,
                request_indices,
                batch,
            )

        return (
            token_proposals,
            state_proposals,
            next_states,
            tuple(executed),
        )




def _include_forced_in_pool(
    request_pool: Tensor, forced_modules: Tensor, *, num_modules: int
) -> Tensor:
    forced = forced_modules.to(device=request_pool.device, dtype=torch.long)
    if forced.ndim != 1:
        raise ValueError("chunked diagnostic_forced_modules must be one-dimensional")
    if forced.numel() > request_pool.size(1):
        raise ValueError("forced modules cannot exceed request pool size")
    if bool(((forced < 0) | (forced >= num_modules)).any().item()):
        raise ValueError("diagnostic_forced_modules contains an invalid module index")
    rows: list[Tensor] = []
    for row in request_pool:
        keep = row[~torch.isin(row, forced)]
        rows.append(torch.cat((forced, keep))[: request_pool.size(1)])
    return torch.stack(rows)


def _force_chunk_routing(
    decision: ChunkRoutingDecision,
    forced_modules: Tensor,
    *,
    batch: int,
    active_top_k: int,
    num_modules: int,
) -> ChunkRoutingDecision:
    forced = forced_modules.to(device=decision.scores.device, dtype=torch.long)
    if forced.ndim == 1:
        forced = forced.unsqueeze(0).expand(batch, -1)
    if tuple(forced.shape) != (batch, active_top_k):
        raise ValueError(
            "chunked diagnostic_forced_modules must provide active_top_k indices"
        )
    if bool(((forced < 0) | (forced >= num_modules)).any().item()):
        raise ValueError("diagnostic_forced_modules contains an invalid module index")
    forced_scores = torch.gather(decision.scores, dim=-1, index=forced)
    if bool(torch.isneginf(forced_scores).any().item()):
        raise ValueError("a forced module is unavailable or outside the request pool")
    return ChunkRoutingDecision(
        scores=decision.scores,
        selected_indices=forced,
        selected_weights=torch.softmax(forced_scores, dim=-1),
        persistence_contribution=decision.persistence_contribution,
        switching_contribution=decision.switching_contribution,
    )


def _lease_for_assignments(
    module: ChunkEMCModuleBase,
    selected_shared: Tensor,
    previous: ModuleLeaseState | None,
    request_indices: Tensor,
    continuing: Tensor,
) -> ModuleLeaseState:
    if previous is not None and bool(continuing.all().item()):
        return _select_lease_state(previous, request_indices)
    if previous is None or not bool(continuing.any().item()):
        return module.begin_lease(selected_shared)

    previous_selected = _select_lease_state(previous, request_indices)
    new_positions = (~continuing).nonzero(as_tuple=False).reshape(-1)
    initialized = module.begin_lease(
        selected_shared.index_select(0, new_positions)
    )
    if not previous_selected.tensors:
        return previous_selected
    merged: dict[str, Tensor] = {}
    for name, previous_tensor in previous_selected.tensors.items():
        merged[name] = previous_tensor.index_copy(
            0,
            new_positions,
            initialized.tensors[name].to(previous_tensor.dtype),
        )
    return ModuleLeaseState(merged)


def _lease_state_diagnostics(
    previous: list[ModuleLeaseState | None],
    current: list[ModuleLeaseState | None],
) -> tuple[float, float]:
    norms: list[Tensor] = []
    changes: list[Tensor] = []
    for old_state, new_state in zip(previous, current, strict=True):
        if new_state is None:
            continue
        for name, value in new_state.tensors.items():
            detached = value.detach().float()
            norms.append(detached.norm())
            if old_state is not None and name in old_state.tensors:
                old_value = old_state.tensors[name].detach().float()
                if old_value.shape == detached.shape:
                    changes.append((detached - old_value).norm())
    mean_norm = torch.stack(norms).mean().item() if norms else 0.0
    mean_change = torch.stack(changes).mean().item() if changes else 0.0
    return mean_norm, mean_change


def _select_lease_state(
    lease_state: ModuleLeaseState, request_indices: Tensor
) -> ModuleLeaseState:
    return ModuleLeaseState(
        {
            name: tensor.index_select(0, request_indices)
            for name, tensor in lease_state.tensors.items()
        }
    )


def _scatter_lease_state(
    selected_state: ModuleLeaseState,
    request_indices: Tensor,
    batch: int,
) -> ModuleLeaseState:
    scattered: dict[str, Tensor] = {}
    for name, tensor in selected_state.tensors.items():
        full = tensor.new_zeros((batch, *tensor.shape[1:]))
        scattered[name] = full.index_copy(0, request_indices, tensor)
    return ModuleLeaseState(scattered)


def _overall_switch_rate(traces: list[ChunkRoutingTrace]) -> float:
    active = sum(trace.active_modules.numel() for trace in traces)
    continuing = sum(
        int(trace.continuing_leases.sum().item()) for trace in traces
    )
    return 1.0 - continuing / active if active else 0.0


def _mean_active_lease_age(traces: list[ChunkRoutingTrace]) -> float:
    ages: list[Tensor] = []
    for trace in traces:
        ages.append(
            trace.lease_ages.gather(1, trace.active_modules).to(torch.float32)
        )
    if not ages:
        return 0.0
    return torch.cat(ages, dim=0).mean().item()
