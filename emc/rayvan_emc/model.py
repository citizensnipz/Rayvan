from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .balancing import router_balance_loss
from .integrator import (
    Integrator,
    IntegratorTrace,
    SequentialAcceptanceIntegrator,
    WeightedAverageIntegrator,
)
from .modules import (
    EMCModule,
    EMCModuleBase,
    RecurrentEMCModule,
    StateSpaceEMCModule,
    create_emc_module,
)
from .nexus import (
    GeometricNexusRouter,
    ModuleAwareNexusRouter,
    NexusRouter,
    RoutingDecision,
)


@dataclass(frozen=True)
class EMCConfig:
    latent_dim: int = 32
    num_modules: int = 6
    modules_per_cycle: int = 2
    num_cycles: int = 3
    trajectory_steps: int | None = None
    refractory_enabled: bool = True
    refractory_strength: float = 0.15
    refractory_decay: float = 0.35
    vocab_size: int = 256
    max_sequence_length: int = 128
    module_hidden_dim: int | None = None
    attention_heads: int = 4
    tie_embeddings: bool = False
    module_families: tuple[str, ...] | None = None
    state_space_dim: int | None = None
    state_space_kernel_size: int = 4
    recurrent_dim: int | None = None
    router_type: str = "fixed_index"
    router_descriptor_dim: int | None = None
    routing_geometry_dim: int | None = None
    competence_prototypes_per_expert: int = 1
    counterfactual_calibration_enabled: bool = True
    counterfactual_probe_fixed_rate: float | None = None
    counterfactual_probe_early_rate: float = 0.08
    counterfactual_probe_stable_rate: float = 0.02
    counterfactual_probe_mature_rate: float = 0.01
    counterfactual_probe_early_steps: int = 1_000
    counterfactual_probe_stable_steps: int = 10_000
    counterfactual_uncertainty_enabled: bool = False
    counterfactual_uncertainty_margin: float = 0.05
    counterfactual_max_probes_per_forward: int = 1
    counterfactual_probe_seed: int = 0
    counterfactual_probe_temperature: float = 0.25
    geometry_temperature: float = 0.25
    geometry_calibration_weight: float = 1.0
    counterfactual_tie_epsilon: float = 1e-3
    integrator_type: str = "weighted_average"
    integrator_heads: int = 4
    architecture_stage: str = "token"
    chunk_size: int = 64
    shared_state_slots: int = 4
    request_pool_size: int | None = None
    active_top_k: int | None = None
    switch_cost: float = 0.05
    persistence_bonus: float = 0.10
    minimum_lease_chunks: int = 0
    loss_free_balance_enabled: bool = True
    balance_bias_lr: float = 0.01
    balance_target_utilization: tuple[float, ...] | None = None
    balance_bias_limit: float = 0.25
    balance_warmup_chunks: int = 0
    shared_core_enabled: bool = True
    shared_core_hidden_dim: int | None = None
    ssm_backend: str = "parallel_scan"
    recurrent_backend: str = "gru"
    recurrent_precision: str = "fp16"
    delta_backend: str = "parallel_delta"
    delta_internal_dim: int | None = None
    delta_heads: int = 4
    delta_ffn_dim: int | None = None
    delta_max_transition_bytes: int = 64 * 2**20

    def __post_init__(self) -> None:
        positive_fields = {
            "latent_dim": self.latent_dim,
            "num_modules": self.num_modules,
            "modules_per_cycle": self.modules_per_cycle,
            "num_cycles": self.num_cycles,
            "vocab_size": self.vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "attention_heads": self.attention_heads,
            "state_space_kernel_size": self.state_space_kernel_size,
            "integrator_heads": self.integrator_heads,
            "chunk_size": self.chunk_size,
            "shared_state_slots": self.shared_state_slots,
            "delta_heads": self.delta_heads,
            "delta_max_transition_bytes": self.delta_max_transition_bytes,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.modules_per_cycle > self.num_modules:
            raise ValueError("modules_per_cycle cannot exceed num_modules")
        if self.latent_dim % self.attention_heads != 0:
            raise ValueError("latent_dim must be divisible by attention_heads")
        if self.latent_dim % self.integrator_heads != 0:
            raise ValueError("latent_dim must be divisible by integrator_heads")
        for name, value in {
            "module_hidden_dim": self.module_hidden_dim,
            "state_space_dim": self.state_space_dim,
            "recurrent_dim": self.recurrent_dim,
            "router_descriptor_dim": self.router_descriptor_dim,
            "routing_geometry_dim": self.routing_geometry_dim,
            "request_pool_size": self.request_pool_size,
            "shared_core_hidden_dim": self.shared_core_hidden_dim,
            "delta_internal_dim": self.delta_internal_dim,
            "delta_ffn_dim": self.delta_ffn_dim,
        }.items():
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")
        if self.active_top_k is not None and self.active_top_k <= 0:
            raise ValueError("active_top_k must be positive when provided")
        if self.resolved_active_top_k > self.num_modules:
            raise ValueError("active_top_k cannot exceed num_modules")
        if self.architecture_stage not in {"token", "n1_sequential", "n1_chunked", "n2"}:
            raise ValueError("architecture_stage must be token, n1_sequential, n1_chunked, or n2")
        if self.trajectory_steps is not None and self.trajectory_steps <= 0:
            raise ValueError("trajectory_steps must be positive when provided")
        if self.refractory_strength < 0:
            raise ValueError("refractory_strength cannot be negative")
        if not 0 <= self.refractory_decay <= 1:
            raise ValueError("refractory_decay must be between zero and one")
        if self.competence_prototypes_per_expert <= 0:
            raise ValueError("competence_prototypes_per_expert must be positive")
        for name, value in {
            "counterfactual_probe_early_rate": self.counterfactual_probe_early_rate,
            "counterfactual_probe_stable_rate": self.counterfactual_probe_stable_rate,
            "counterfactual_probe_mature_rate": self.counterfactual_probe_mature_rate,
        }.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.counterfactual_probe_fixed_rate is not None and not 0 <= self.counterfactual_probe_fixed_rate <= 1:
            raise ValueError("counterfactual_probe_fixed_rate must be between zero and one")
        if self.counterfactual_probe_early_steps < 0 or self.counterfactual_probe_stable_steps < self.counterfactual_probe_early_steps:
            raise ValueError("counterfactual probe schedule steps are invalid")
        if self.counterfactual_max_probes_per_forward < 0:
            raise ValueError("counterfactual_max_probes_per_forward cannot be negative")
        if self.counterfactual_uncertainty_margin < 0:
            raise ValueError("counterfactual_uncertainty_margin cannot be negative")
        if self.counterfactual_probe_temperature <= 0 or self.geometry_temperature <= 0:
            raise ValueError("counterfactual and geometry temperatures must be positive")
        if self.geometry_calibration_weight < 0 or self.counterfactual_tie_epsilon < 0:
            raise ValueError("calibration weight and tie epsilon cannot be negative")
        if not 0 <= self.minimum_lease_chunks:
            raise ValueError("minimum_lease_chunks cannot be negative")
        if not 0 <= self.balance_warmup_chunks:
            raise ValueError("balance_warmup_chunks cannot be negative")
        if self.switch_cost < 0 or self.persistence_bonus < 0:
            raise ValueError("switch and persistence terms cannot be negative")
        if self.balance_bias_lr < 0 or self.balance_bias_limit < 0:
            raise ValueError("loss-free balance settings cannot be negative")
        if self.resolved_request_pool_size < self.resolved_active_top_k:
            raise ValueError("request_pool_size cannot be smaller than active top-K")
        if self.resolved_request_pool_size > self.num_modules:
            raise ValueError("request_pool_size cannot exceed num_modules")
        if self.recurrent_precision not in {"model", "fp32", "fp16", "bf16"}:
            raise ValueError("unsupported recurrent_precision")
        if self.ssm_backend != "parallel_scan":
            raise ValueError("unsupported ssm_backend")
        if self.recurrent_backend != "gru":
            raise ValueError("unsupported recurrent_backend")
        if self.delta_backend != "parallel_delta":
            raise ValueError("unsupported delta_backend")
        if self.resolved_delta_internal_dim % self.delta_heads != 0:
            raise ValueError("delta_internal_dim must be divisible by delta_heads")
        if self.balance_target_utilization is not None:
            if len(self.balance_target_utilization) != self.num_modules:
                raise ValueError(
                    "balance_target_utilization must contain one value per module"
                )
            if any(value < 0 for value in self.balance_target_utilization):
                raise ValueError("balance target values cannot be negative")
            if sum(self.balance_target_utilization) <= 0:
                raise ValueError(
                    "balance target utilization must have positive mass"
                )
        if self.router_type not in {"fixed_index", "module_aware", "geometric"}:
            raise ValueError("router_type must be fixed_index, module_aware, or geometric")
        if self.integrator_type not in {"weighted_average", "proposal_attention", "acceptance_gate"}:
            raise ValueError(
                "integrator_type must be weighted_average, proposal_attention, or acceptance_gate"
            )
        if self.module_families is not None:
            if len(self.module_families) != self.num_modules:
                raise ValueError(
                    "module_families must contain exactly num_modules entries"
                )
            allowed = {
                "gpt",
                "ssm",
                "mamba",
                "recurrent",
                "gru",
                "delta",
                "deltanet",
            }
            unknown = set(self.module_families) - allowed
            if unknown:
                raise ValueError(f"unknown module families: {sorted(unknown)}")

    @property
    def resolved_module_hidden_dim(self) -> int:
        return self.module_hidden_dim or self.latent_dim * 4

    @property
    def resolved_state_space_dim(self) -> int:
        return self.state_space_dim or self.latent_dim * 4

    @property
    def resolved_recurrent_dim(self) -> int:
        return self.recurrent_dim or self.latent_dim * 2

    @property
    def resolved_router_descriptor_dim(self) -> int:
        return self.router_descriptor_dim or self.latent_dim

    @property
    def resolved_routing_geometry_dim(self) -> int:
        return self.routing_geometry_dim or self.router_descriptor_dim or self.latent_dim

    @property
    def resolved_request_pool_size(self) -> int:
        return self.request_pool_size or self.num_modules

    @property
    def resolved_active_top_k(self) -> int:
        return self.active_top_k or self.modules_per_cycle

    @property
    def resolved_trajectory_steps(self) -> int:
        return self.trajectory_steps or self.num_cycles

    @property
    def resolved_shared_core_hidden_dim(self) -> int:
        return self.shared_core_hidden_dim or self.latent_dim

    @property
    def resolved_delta_internal_dim(self) -> int:
        return self.delta_internal_dim or self.latent_dim

    @property
    def resolved_delta_ffn_dim(self) -> int:
        return self.delta_ffn_dim or self.latent_dim * 4

    @property
    def resolved_module_families(self) -> tuple[str, ...]:
        families = self.module_families or ("gpt",) * self.num_modules
        aliases = {
            "mamba": "ssm",
            "gru": "recurrent",
            "deltanet": "delta",
        }
        return tuple(aliases.get(family, family) for family in families)


@dataclass(frozen=True)
class CounterfactualProbeTrace:
    sample_indices: Tensor
    trigger: tuple[str, ...]
    candidate_losses: Tensor
    counterfactual_best: Tensor
    chosen_expert: Tensor
    geometric_winner: Tensor
    top1_correct: Tensor
    top2_correct: Tensor
    routing_regret: Tensor
    best_second_margin: Tensor
    effectively_tied: Tensor


@dataclass(frozen=True)
class EMCCycleTrace:
    cycle: int
    selected_modules: tuple[int, ...]
    router_scores: Tensor
    router_weights: Tensor
    latent_shape: tuple[int, ...]
    selected_indices: Tensor | None = None
    integrator_trace: IntegratorTrace | None = None
    module_families: tuple[str, ...] = ()
    expert_names: tuple[str, ...] = ()
    local_diagnostics: tuple[object, ...] = ()
    raw_router_scores: Tensor | None = None
    pre_inhibition_router_scores: Tensor | None = None
    effective_router_scores: Tensor | None = None
    refractory_penalty: Tensor | None = None
    previous_indices: Tensor | None = None
    base_actions: Tensor | None = None
    effective_actions: Tensor | None = None
    balance_action: Tensor | None = None
    geometric_winner: Tensor | None = None
    action_margin: Tensor | None = None
    need_norm: Tensor | None = None
    need_embeddings: Tensor | None = None
    winning_prototypes: Tensor | None = None
    counterfactual: CounterfactualProbeTrace | None = None


@dataclass(frozen=True)
class EMCOutput:
    logits: Tensor
    trace: tuple[EMCCycleTrace, ...]
    router_balance_loss: Tensor | None = None
    cycle_logits: tuple[Tensor, ...] | None = None
    chunk_trace: object | None = None
    n2_state: object | None = None
    geometry_calibration_loss: Tensor | None = None


class EMCModel(nn.Module):
    def __init__(self, config: EMCConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.latent_dim)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.latent_dim
        )
        if config.router_type == "geometric":
            self.router: NexusRouter | ModuleAwareNexusRouter | GeometricNexusRouter = (
                GeometricNexusRouter(config)
            )
        elif config.router_type == "module_aware":
            self.router = ModuleAwareNexusRouter(config)
        else:
            self.router = NexusRouter(config)
        self.emc_modules = nn.ModuleList(
            create_emc_module(config, family)
            for family in config.resolved_module_families
        )
        if config.integrator_type == "acceptance_gate":
            self.integrator: WeightedAverageIntegrator | Integrator | SequentialAcceptanceIntegrator = (
                SequentialAcceptanceIntegrator(config)
            )
        elif config.integrator_type == "proposal_attention":
            self.integrator = Integrator(
                config
            )
        else:
            self.integrator = WeightedAverageIntegrator(config)
        self.output_norm = nn.LayerNorm(config.latent_dim)
        self.output_projection = nn.Linear(config.latent_dim, config.vocab_size)
        if config.tie_embeddings:
            self.output_projection.weight = self.token_embedding.weight
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.output_projection.bias)
        self._active_top_k = config.modules_per_cycle
        self._geometry_statistics: dict[str, object] = {
            "probe_count": 0,
            "top1_correct": 0,
            "top2_correct": 0,
            "regrets": [],
            "tie_count": 0,
            "best_second_margins": [],
            "counterfactual_wins": [0] * config.num_modules,
            "routing_counts": [0] * config.num_modules,
            "evaluation_routing_counts": [0] * config.num_modules,
            "basin_occupancy": [0] * config.num_modules,
            "confusion": [[0] * config.num_modules for _ in range(config.num_modules)],
            "per_step": {},
            "scheduled_probes": 0,
            "uncertainty_probes": 0,
            "action_margins": [],
            "need_norms": [],
            "base_action_sums": [0.0] * config.num_modules,
            "effective_action_sums": [0.0] * config.num_modules,
            "action_observations": 0,
            "refractory_winner_changes": 0,
        }

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
        self._active_top_k = top_k

    def execute_selected_modules(
        self, latent: Tensor, selected_indices: Tensor
    ) -> Tensor:
        """Run the union of selected modules, then gather each token's top-K proposals."""
        unique_indices = torch.unique(selected_indices, sorted=True)
        computed_updates = torch.stack(
            [
                self.emc_modules[index](latent)
                for index in unique_indices.tolist()
            ],
            dim=2,
        )
        lookup = torch.full(
            (self.config.num_modules,),
            -1,
            dtype=torch.long,
            device=selected_indices.device,
        )
        lookup[unique_indices] = torch.arange(
            unique_indices.numel(), device=selected_indices.device
        )
        update_locations = lookup[selected_indices]
        gather_indices = update_locations.unsqueeze(-1).expand(
            *update_locations.shape, latent.size(-1)
        )
        return torch.gather(computed_updates, dim=2, index=gather_indices)

    def execute_selected_requests(
        self, latent: Tensor, selected_indices: Tensor
    ) -> Tensor:
        """Execute each expert only for requests assigned at this trajectory step."""
        selected = selected_indices.reshape(latent.size(0))
        updates = latent.new_zeros(*latent.shape, 1)
        updates = updates.permute(0, 1, 3, 2)
        for module_index in torch.unique(selected, sorted=True).tolist():
            request_indices = (selected == module_index).nonzero(
                as_tuple=False
            ).reshape(-1)
            proposal = self.emc_modules[module_index](
                latent.index_select(0, request_indices)
            ).to(dtype=updates.dtype).unsqueeze(2)
            updates = updates.index_copy(0, request_indices, proposal)
        return updates

    def forward(
        self,
        token_ids: Tensor,
        *,
        counterfactual_targets: Tensor | None = None,
        training_step: int = 0,
        force_counterfactual_probe: bool = False,
        return_trace: bool = False,
        return_cycle_logits: bool = False,
        balance_entropy_floor: float = 0.75,
        availability_mask: Tensor | None = None,
        module_descriptors: Tensor | None = None,
        evaluation_cycle_limit: int | None = None,
        diagnostic_forced_modules: Tensor | None = None,
        diagnostic_zero_proposal_mask: Tensor | None = None,
    ) -> Tensor | EMCOutput:
        sequence_length = token_ids.size(1)
        if sequence_length > self.config.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds configured maximum "
                f"{self.config.max_sequence_length}"
            )
        if (
            module_descriptors is not None
            and module_descriptors.size(0) != self.config.num_modules
        ):
            raise ValueError(
                "local EMC execution requires one descriptor per configured module"
            )
        if evaluation_cycle_limit is not None:
            if self.training:
                raise ValueError(
                    "evaluation_cycle_limit is available only in evaluation mode"
                )
            if not 1 <= evaluation_cycle_limit <= self.config.num_cycles:
                raise ValueError(
                    "evaluation_cycle_limit must be within configured EMC cycles"
                )
        sequential = self.config.architecture_stage == "n1_sequential"
        maximum_steps = (
            self.config.resolved_trajectory_steps
            if sequential
            else self.config.num_cycles
        )
        cycles_to_run = evaluation_cycle_limit or maximum_steps
        positions = torch.arange(sequence_length, device=token_ids.device)
        latent = self.token_embedding(token_ids) + self.position_embedding(positions)
        cycle_traces: list[EMCCycleTrace] = []
        cycle_balance_losses: list[Tensor] = []
        per_cycle_logits: list[Tensor] = []
        refractory = latent.new_zeros(
            token_ids.size(0), 1, self.config.num_modules
        )
        previous_indices: Tensor | None = None
        trajectory_selections: list[Tensor] = []
        calibration_losses: list[Tensor] = []
        already_probed = torch.zeros(
            token_ids.size(0), dtype=torch.bool, device=token_ids.device
        )
        remaining_probe_budget = self.config.counterfactual_max_probes_per_forward

        for cycle in range(cycles_to_run):
            if sequential:
                adjustment = latent.new_zeros(refractory.shape)
                if previous_indices is not None and not isinstance(
                    self.router, GeometricNexusRouter
                ):
                    previous_mask = torch.zeros_like(adjustment, dtype=torch.bool)
                    previous_mask.scatter_(-1, previous_indices, True)
                    adjustment = adjustment - self.config.switch_cost
                    adjustment = torch.where(
                        previous_mask,
                        adjustment + self.config.switch_cost + self.config.persistence_bonus,
                        adjustment,
                    )
                routing = self.router.route_one(
                    latent.mean(dim=1, keepdim=True),
                    availability_mask=availability_mask,
                    module_descriptors=module_descriptors,
                    score_adjustment=adjustment,
                    refractory_penalty=(
                        refractory if self.config.refractory_enabled else None
                    ),
                )
            else:
                routing = self.router(
                    latent,
                    availability_mask=availability_mask,
                    module_descriptors=module_descriptors,
                    top_k=self.active_top_k,
                )
            if diagnostic_forced_modules is not None:
                routing = _force_token_routing(
                    routing,
                    diagnostic_forced_modules,
                    batch=token_ids.size(0),
                    sequence=(1 if sequential else sequence_length),
                    modules_per_cycle=self.active_top_k,
                    num_modules=self.config.num_modules,
                )
            if return_trace:
                cycle_balance_losses.append(
                    router_balance_loss(
                        routing.scores,
                        routing.selected_indices,
                        entropy_floor=balance_entropy_floor,
                    )
                )
            module_updates = (
                self.execute_selected_requests(latent, routing.selected_indices)
                if sequential
                else self.execute_selected_modules(latent, routing.selected_indices)
            )
            if diagnostic_zero_proposal_mask is not None:
                zero_mask = diagnostic_zero_proposal_mask.to(
                    device=module_updates.device, dtype=torch.bool
                )
                if zero_mask.ndim != 1 or zero_mask.numel() != self.config.num_modules:
                    raise ValueError(
                        "diagnostic_zero_proposal_mask must contain one value per module"
                    )
                selected_to_zero = zero_mask[routing.selected_indices]
                module_updates = module_updates.masked_fill(
                    selected_to_zero.unsqueeze(-1), 0
                )
            integration_weights = routing.selected_weights
            if sequential:
                integration_weights = integration_weights.expand(
                    -1, sequence_length, -1
                )
            pre_routing_latent = latent
            if isinstance(self.integrator, SequentialAcceptanceIntegrator):
                integrated = self.integrator(
                    latent,
                    module_updates,
                    integration_weights,
                    selected_indices=routing.selected_indices,
                    return_diagnostics=return_trace,
                )
            else:
                integrated = self.integrator(
                    latent,
                    module_updates,
                    integration_weights,
                    return_diagnostics=return_trace,
                )
            if return_trace:
                if not isinstance(integrated, tuple):
                    raise RuntimeError(
                        "diagnostic Integrator forward did not return a trace"
                    )
                latent, integrator_trace = integrated
            else:
                if isinstance(integrated, tuple):
                    raise RuntimeError("Integrator unexpectedly returned diagnostics")
                latent = integrated

            counterfactual_trace: CounterfactualProbeTrace | None = None
            if (
                sequential
                and isinstance(self.router, GeometricNexusRouter)
                and (self.training or force_counterfactual_probe)
                and self.config.counterfactual_calibration_enabled
                and counterfactual_targets is not None
                and remaining_probe_budget > 0
            ):
                if force_counterfactual_probe:
                    probe_indices = (
                        (~already_probed).nonzero(as_tuple=False).reshape(-1)[
                            :remaining_probe_budget
                        ]
                        if cycle == training_step % cycles_to_run
                        else already_probed.new_empty((0,), dtype=torch.long)
                    )
                    triggers = tuple("diagnostic" for _ in probe_indices.tolist())
                else:
                    probe_indices, triggers = self._sample_counterfactual_probes(
                        routing,
                        already_probed,
                        remaining_probe_budget,
                        training_step=training_step,
                        trajectory_step=cycle,
                    )
                if probe_indices.numel():
                    calibration_loss, counterfactual_trace = self._counterfactual_probe(
                        pre_routing_latent,
                        counterfactual_targets,
                        routing,
                        probe_indices,
                        triggers,
                    )
                    calibration_losses.append(calibration_loss)
                    already_probed[probe_indices] = True
                    remaining_probe_budget -= probe_indices.numel()
                    if not force_counterfactual_probe:
                        self._observe_counterfactual(counterfactual_trace, cycle)

            if return_cycle_logits:
                per_cycle_logits.append(
                    self.output_projection(self.output_norm(latent))
                )
            if return_trace:
                cycle_traces.append(
                    EMCCycleTrace(
                        cycle=cycle + 1,
                        selected_modules=tuple(
                            torch.unique(
                                routing.selected_indices, sorted=True
                            ).tolist()
                        ),
                        router_scores=routing.scores.detach().cpu(),
                        router_weights=routing.selected_weights.detach().cpu(),
                        latent_shape=tuple(latent.shape),
                        selected_indices=routing.selected_indices.detach().cpu(),
                        integrator_trace=integrator_trace,
                        module_families=self.module_families,
                        raw_router_scores=(
                            routing.raw_scores.detach().cpu()
                            if routing.raw_scores is not None else None
                        ),
                        pre_inhibition_router_scores=(
                            routing.pre_inhibition_scores.detach().cpu()
                            if routing.pre_inhibition_scores is not None else None
                        ),
                        effective_router_scores=routing.scores.detach().cpu(),
                        refractory_penalty=(
                            routing.refractory_penalty.detach().cpu()
                            if routing.refractory_penalty is not None else None
                        ),
                        previous_indices=(
                            previous_indices.detach().cpu()
                            if previous_indices is not None else None
                        ),
                        base_actions=(
                            routing.base_actions.detach().cpu()
                            if routing.base_actions is not None else None
                        ),
                        effective_actions=(
                            routing.effective_actions.detach().cpu()
                            if routing.effective_actions is not None else None
                        ),
                        balance_action=(
                            routing.balance_action.detach().cpu()
                            if routing.balance_action is not None else None
                        ),
                        geometric_winner=(
                            routing.geometric_winner.detach().cpu()
                            if routing.geometric_winner is not None else None
                        ),
                        action_margin=(
                            routing.action_margin.detach().cpu()
                            if routing.action_margin is not None else None
                        ),
                        need_norm=(
                            routing.need_norm.detach().cpu()
                            if routing.need_norm is not None else None
                        ),
                        need_embeddings=(
                            routing.need_embedding.detach().cpu()
                            if routing.need_embedding is not None else None
                        ),
                        winning_prototypes=(
                            routing.winning_prototypes.detach().cpu()
                            if routing.winning_prototypes is not None else None
                        ),
                        counterfactual=counterfactual_trace,
                    )
                )

            if sequential and isinstance(self.router, GeometricNexusRouter):
                self._observe_geometric_routing(routing, evaluation=not self.training)

            if sequential:
                trajectory_selections.append(routing.selected_indices.detach())
                refractory = refractory * self.config.refractory_decay
                refractory.scatter_add_(
                    -1,
                    routing.selected_indices,
                    torch.full_like(
                        routing.selected_indices,
                        self.config.refractory_strength,
                        dtype=refractory.dtype,
                    ),
                )
                previous_indices = routing.selected_indices

        if sequential and self.training and trajectory_selections:
            updater = getattr(self.router, "update_balance_bias", None)
            if updater is not None:
                updater(torch.cat(trajectory_selections, dim=0))

        logits = self.output_projection(self.output_norm(latent))
        if return_trace or return_cycle_logits:
            return EMCOutput(
                logits=logits,
                trace=tuple(cycle_traces),
                router_balance_loss=(
                    torch.stack(cycle_balance_losses).mean()
                    if cycle_balance_losses
                    else None
                ),
                cycle_logits=(
                    tuple(per_cycle_logits) if return_cycle_logits else None
                ),
                geometry_calibration_loss=(
                    torch.stack(calibration_losses).mean()
                    if calibration_losses
                    else logits.new_zeros(())
                ),
            )
        return logits

    def _probe_rate(self, training_step: int) -> float:
        fixed = self.config.counterfactual_probe_fixed_rate
        if fixed is not None:
            return fixed
        if training_step < self.config.counterfactual_probe_early_steps:
            return self.config.counterfactual_probe_early_rate
        if training_step < self.config.counterfactual_probe_stable_steps:
            return self.config.counterfactual_probe_stable_rate
        return self.config.counterfactual_probe_mature_rate

    def _sample_counterfactual_probes(
        self,
        routing: RoutingDecision,
        already_probed: Tensor,
        budget: int,
        *,
        training_step: int,
        trajectory_step: int,
    ) -> tuple[Tensor, tuple[str, ...]]:
        if budget <= 0:
            return already_probed.new_empty((0,), dtype=torch.long), ()
        generator = torch.Generator(device="cpu").manual_seed(
            self.config.counterfactual_probe_seed
            + training_step * 1_000_003
            + trajectory_step * 9_973
        )
        draws = torch.rand(already_probed.numel(), generator=generator).to(
            already_probed.device
        )
        scheduled = draws < self._probe_rate(training_step)
        uncertain = torch.zeros_like(scheduled)
        if (
            self.config.counterfactual_uncertainty_enabled
            and routing.action_margin is not None
        ):
            uncertain = (
                routing.action_margin.reshape(-1)
                <= self.config.counterfactual_uncertainty_margin
            )
        candidates = (~already_probed) & (scheduled | uncertain)
        indices = candidates.nonzero(as_tuple=False).reshape(-1)[:budget]
        triggers = tuple(
            "scheduled+uncertainty"
            if bool(scheduled[index]) and bool(uncertain[index])
            else "uncertainty"
            if bool(uncertain[index])
            else "scheduled"
            for index in indices.tolist()
        )
        return indices, triggers

    def _counterfactual_probe(
        self,
        pre_routing_latent: Tensor,
        targets: Tensor,
        routing: RoutingDecision,
        sample_indices: Tensor,
        triggers: tuple[str, ...],
    ) -> tuple[Tensor, CounterfactualProbeTrace]:
        if not isinstance(self.router, GeometricNexusRouter):
            raise RuntimeError("counterfactual calibration requires geometric Nexus")
        if not isinstance(self.integrator, SequentialAcceptanceIntegrator):
            raise RuntimeError(
                "counterfactual calibration requires the sequential acceptance gate"
            )
        shared_state = pre_routing_latent.index_select(0, sample_indices)
        probe_targets = targets.index_select(0, sample_indices)
        candidate_losses: list[Tensor] = []
        # Alternatives are evidence only: no task-loss gradient reaches any expert
        # or the Integrator through these counterfactual branches.
        with torch.no_grad():
            for expert_index, expert in enumerate(self.emc_modules):
                proposal = expert(shared_state).unsqueeze(2)
                expert_ids = torch.full(
                    (shared_state.size(0), 1, 1),
                    expert_index,
                    dtype=torch.long,
                    device=shared_state.device,
                )
                candidate = self.integrator(
                    shared_state,
                    proposal,
                    torch.ones_like(expert_ids, dtype=shared_state.dtype).expand(
                        -1, shared_state.size(1), -1
                    ),
                    selected_indices=expert_ids,
                )
                if not isinstance(candidate, Tensor):
                    raise RuntimeError("counterfactual Integrator returned diagnostics")
                logits = self.output_projection(self.output_norm(candidate))
                losses = torch.nn.functional.cross_entropy(
                    logits.transpose(1, 2), probe_targets, reduction="none"
                ).mean(dim=1)
                candidate_losses.append(losses)
        losses = torch.stack(candidate_losses, dim=-1)
        target_preference = torch.softmax(
            -losses / self.config.counterfactual_probe_temperature, dim=-1
        )
        if routing.base_actions is None:
            raise RuntimeError("geometric route did not expose base actions")
        base_actions = routing.base_actions.reshape(
            pre_routing_latent.size(0), -1
        ).index_select(0, sample_indices)
        predicted_log_preference = torch.log_softmax(
            -base_actions / self.config.geometry_temperature, dim=-1
        )
        calibration_loss = -(
            target_preference.detach() * predicted_log_preference
        ).sum(dim=-1).mean()

        best = losses.argmin(dim=-1)
        chosen = routing.selected_indices.reshape(-1).index_select(0, sample_indices)
        geometric = routing.geometric_winner.reshape(-1).index_select(
            0, sample_indices
        )
        chosen_losses = losses.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        regret = chosen_losses - losses.min(dim=-1).values
        ordered = losses.sort(dim=-1).values
        best_second_margin = (
            ordered[:, 1] - ordered[:, 0]
            if losses.size(-1) > 1
            else torch.zeros_like(ordered[:, 0])
        )
        top2 = base_actions.topk(
            k=min(2, base_actions.size(-1)), largest=False, dim=-1
        ).indices
        trace = CounterfactualProbeTrace(
            sample_indices=sample_indices.detach().cpu(),
            trigger=triggers,
            candidate_losses=losses.detach().cpu(),
            counterfactual_best=best.detach().cpu(),
            chosen_expert=chosen.detach().cpu(),
            geometric_winner=geometric.detach().cpu(),
            top1_correct=(geometric == best).detach().cpu(),
            top2_correct=(top2 == best.unsqueeze(-1)).any(dim=-1).detach().cpu(),
            routing_regret=regret.detach().cpu(),
            best_second_margin=best_second_margin.detach().cpu(),
            effectively_tied=(
                best_second_margin <= self.config.counterfactual_tie_epsilon
            ).detach().cpu(),
        )
        return calibration_loss, trace

    @torch.no_grad()
    def _observe_geometric_routing(
        self, routing: RoutingDecision, *, evaluation: bool
    ) -> None:
        selected = routing.selected_indices.reshape(-1).detach().cpu().tolist()
        counts = self._geometry_statistics[
            "evaluation_routing_counts" if evaluation else "routing_counts"
        ]
        occupancy = self._geometry_statistics["basin_occupancy"]
        assert isinstance(counts, list) and isinstance(occupancy, list)
        for index in selected:
            counts[index] += 1
            if not evaluation:
                occupancy[index] += 1
        if evaluation:
            return
        if routing.action_margin is not None:
            action_margins = self._geometry_statistics["action_margins"]
            assert isinstance(action_margins, list)
            action_margins.extend(
                float(value) for value in routing.action_margin.reshape(-1).detach().cpu().tolist()
            )
        if routing.need_norm is not None:
            need_norms = self._geometry_statistics["need_norms"]
            assert isinstance(need_norms, list)
            need_norms.extend(
                float(value) for value in routing.need_norm.reshape(-1).detach().cpu().tolist()
            )
        if routing.base_actions is not None and routing.effective_actions is not None:
            base_sums = self._geometry_statistics["base_action_sums"]
            effective_sums = self._geometry_statistics["effective_action_sums"]
            assert isinstance(base_sums, list) and isinstance(effective_sums, list)
            base_rows = routing.base_actions.reshape(-1, self.config.num_modules).detach().cpu()
            effective_rows = routing.effective_actions.reshape(-1, self.config.num_modules).detach().cpu()
            for index in range(self.config.num_modules):
                base_sums[index] += float(base_rows[:, index].sum().item())
                effective_sums[index] += float(effective_rows[:, index].sum().item())
            self._geometry_statistics["action_observations"] = int(
                self._geometry_statistics["action_observations"]
            ) + base_rows.size(0)
        if routing.geometric_winner is not None:
            self._geometry_statistics["refractory_winner_changes"] = int(
                self._geometry_statistics["refractory_winner_changes"]
            ) + int(
                (
                    routing.geometric_winner.reshape(-1)
                    != routing.selected_indices.reshape(-1)
                ).sum().item()
            )

    @torch.no_grad()
    def _observe_counterfactual(
        self, trace: CounterfactualProbeTrace, trajectory_step: int
    ) -> None:
        stats = self._geometry_statistics
        regrets = stats["regrets"]
        margins = stats["best_second_margins"]
        wins = stats["counterfactual_wins"]
        confusion = stats["confusion"]
        per_step = stats["per_step"]
        assert isinstance(regrets, list) and isinstance(margins, list)
        assert isinstance(wins, list) and isinstance(confusion, list)
        assert isinstance(per_step, dict)
        values = [float(value) for value in trace.routing_regret.tolist()]
        loss_margins = [float(value) for value in trace.best_second_margin.tolist()]
        stats["probe_count"] = int(stats["probe_count"]) + len(values)
        stats["top1_correct"] = int(stats["top1_correct"]) + int(
            trace.top1_correct.sum().item()
        )
        stats["top2_correct"] = int(stats["top2_correct"]) + int(
            trace.top2_correct.sum().item()
        )
        stats["tie_count"] = int(stats["tie_count"]) + int(
            trace.effectively_tied.sum().item()
        )
        regrets.extend(values)
        margins.extend(loss_margins)
        for trigger in trace.trigger:
            if "scheduled" in trigger:
                stats["scheduled_probes"] = int(stats["scheduled_probes"]) + 1
            if "uncertainty" in trigger:
                stats["uncertainty_probes"] = int(stats["uncertainty_probes"]) + 1
        for selected, best in zip(
            trace.chosen_expert.tolist(), trace.counterfactual_best.tolist()
        ):
            wins[best] += 1
            confusion[selected][best] += 1
        step = per_step.setdefault(
            str(trajectory_step + 1),
            {"probe_count": 0, "regrets": [], "top1_correct": 0},
        )
        step["probe_count"] += len(values)
        step["regrets"].extend(values)
        step["top1_correct"] += int(trace.top1_correct.sum().item())

    def geometric_routing_report(self) -> dict[str, object] | None:
        if not isinstance(self.router, GeometricNexusRouter):
            return None
        stats = self._geometry_statistics
        probe_count = int(stats["probe_count"])
        regrets = [float(value) for value in stats["regrets"]]
        margins = [float(value) for value in stats["best_second_margins"]]
        routing_counts = [int(value) for value in stats["routing_counts"]]
        evaluation_counts = [
            int(value) for value in stats["evaluation_routing_counts"]
        ]
        counterfactual_wins = [int(value) for value in stats["counterfactual_wins"]]
        action_margins = [float(value) for value in stats["action_margins"]]
        need_norms = [float(value) for value in stats["need_norms"]]
        total_routes = max(sum(routing_counts), 1)
        probabilities = [value / total_routes for value in routing_counts]
        evaluation_total = max(sum(evaluation_counts), 1)
        evaluation_probabilities = [
            value / evaluation_total for value in evaluation_counts
        ]
        routing_distribution_l1 = sum(
            abs(train - evaluation)
            for train, evaluation in zip(probabilities, evaluation_probabilities)
        )
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probabilities)
        prototypes = self.router.normalized_prototypes().detach()
        initial = self.router.initial_competence_prototypes.to(prototypes.device)
        drift = (prototypes - initial).norm(dim=-1)
        centers = prototypes.mean(dim=1)
        pairwise = torch.cdist(centers, centers)
        non_diagonal = pairwise[~torch.eye(
            pairwise.size(0), dtype=torch.bool, device=pairwise.device
        )]
        minimum_basin_separation = (
            float(non_diagonal.min().item()) if non_diagonal.numel() else None
        )

        def percentile(values: list[float], fraction: float) -> float | None:
            if not values:
                return None
            ordered = sorted(values)
            return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]

        per_step: dict[str, object] = {}
        for step_name, values in dict(stats["per_step"]).items():
            count = int(values["probe_count"])
            step_regrets = [float(value) for value in values["regrets"]]
            per_step[step_name] = {
                "probe_count": count,
                "mean_routing_regret": (
                    sum(step_regrets) / count if count else None
                ),
                "top1_accuracy": (
                    int(values["top1_correct"]) / count if count else None
                ),
            }
        return {
            "schema_version": 1,
            "router_type": "geometric_competence_basin",
            "geometry_dimension": self.config.resolved_routing_geometry_dim,
            "prototypes_per_expert": self.config.competence_prototypes_per_expert,
            "distance": "squared_euclidean_on_unit_sphere",
            "probe_config": {
                "enabled": self.config.counterfactual_calibration_enabled,
                "fixed_rate": self.config.counterfactual_probe_fixed_rate,
                "early_rate": self.config.counterfactual_probe_early_rate,
                "stable_rate": self.config.counterfactual_probe_stable_rate,
                "mature_rate": self.config.counterfactual_probe_mature_rate,
                "early_steps": self.config.counterfactual_probe_early_steps,
                "stable_steps": self.config.counterfactual_probe_stable_steps,
                "uncertainty_enabled": self.config.counterfactual_uncertainty_enabled,
                "uncertainty_margin": self.config.counterfactual_uncertainty_margin,
                "maximum_probes_per_forward": self.config.counterfactual_max_probes_per_forward,
                "probe_temperature": self.config.counterfactual_probe_temperature,
                "geometry_temperature": self.config.geometry_temperature,
            },
            "total_probes": probe_count,
            "counterfactual_top1_accuracy": (
                int(stats["top1_correct"]) / probe_count if probe_count else None
            ),
            "counterfactual_top2_accuracy": (
                int(stats["top2_correct"]) / probe_count if probe_count else None
            ),
            "mean_routing_regret": (
                sum(regrets) / len(regrets) if regrets else None
            ),
            "median_routing_regret": percentile(regrets, 0.5),
            "p90_routing_regret": percentile(regrets, 0.9),
            "effectively_tied_fraction": (
                int(stats["tie_count"]) / probe_count if probe_count else None
            ),
            "average_best_second_loss_margin": (
                sum(margins) / len(margins) if margins else None
            ),
            "counterfactual_win_rates": [
                value / max(probe_count, 1) for value in counterfactual_wins
            ],
            "actual_routing_rates": probabilities,
            "total_routing_events": sum(routing_counts),
            "evaluation_routing_rates": evaluation_probabilities,
            "training_evaluation_routing_consistency": {
                "l1_distance": routing_distribution_l1,
                "mismatch": (
                    sum(evaluation_counts) > 0 and routing_distribution_l1 > 0.25
                ),
                "threshold": 0.25,
            },
            "basin_occupancy": [
                value / total_routes for value in stats["basin_occupancy"]
            ],
            "effective_active_basins": math.exp(entropy),
            "geometry_entropy": entropy,
            "mean_geometric_margin": (
                sum(action_margins) / len(action_margins) if action_margins else None
            ),
            "mean_need_norm_before_normalization": (
                sum(need_norms) / len(need_norms) if need_norms else None
            ),
            "mean_base_actions": [
                value / max(int(stats["action_observations"]), 1)
                for value in stats["base_action_sums"]
            ],
            "mean_effective_actions": [
                value / max(int(stats["action_observations"]), 1)
                for value in stats["effective_action_sums"]
            ],
            "refractory_winner_change_rate": (
                int(stats["refractory_winner_changes"]) / total_routes
            ),
            "minimum_basin_separation": minimum_basin_separation,
            "counterfactual_matrix": stats["confusion"],
            "per_step": per_step,
            "per_capability": {},
            "scheduled_probe_count": stats["scheduled_probes"],
            "uncertainty_probe_count": stats["uncertainty_probes"],
            "prototype_drift": {
                "mean": float(drift.mean().item()),
                "maximum": float(drift.max().item()),
                "per_expert": drift.mean(dim=-1).cpu().tolist(),
            },
            "loss_free_balancing_enabled": self.config.loss_free_balance_enabled,
        }


class SequentialEMCModel(EMCModel):
    """Primary EMC: one route, one expert transform, one gated integration."""

    def __init__(self, config: EMCConfig) -> None:
        if config.architecture_stage != "n1_sequential":
            raise ValueError("SequentialEMCModel requires architecture_stage=n1_sequential")
        if config.modules_per_cycle != 1 or config.active_top_k not in {None, 1}:
            raise ValueError("sequential EMC executes exactly one expert per routing step")
        super().__init__(config)


def _force_token_routing(
    routing: RoutingDecision,
    forced_modules: Tensor,
    *,
    batch: int,
    sequence: int,
    modules_per_cycle: int,
    num_modules: int,
) -> RoutingDecision:
    forced = forced_modules.to(device=routing.scores.device, dtype=torch.long)
    if forced.ndim == 1:
        forced = forced.reshape(1, 1, -1).expand(batch, sequence, -1)
    elif forced.ndim == 2 and tuple(forced.shape) == (batch, modules_per_cycle):
        forced = forced.unsqueeze(1).expand(batch, sequence, -1)
    if tuple(forced.shape) != (batch, sequence, modules_per_cycle):
        raise ValueError(
            "diagnostic_forced_modules must provide exactly modules_per_cycle "
            "indices globally, per request, or per token"
        )
    if bool(((forced < 0) | (forced >= num_modules)).any().item()):
        raise ValueError("diagnostic_forced_modules contains an invalid module index")
    forced_scores = torch.gather(routing.scores, dim=-1, index=forced)
    if bool(torch.isneginf(forced_scores).any().item()):
        raise ValueError("a forced module is unavailable")
    return RoutingDecision(
        scores=routing.scores,
        selected_indices=forced,
        selected_weights=torch.softmax(forced_scores, dim=-1),
        raw_scores=routing.raw_scores,
        pre_inhibition_scores=routing.pre_inhibition_scores,
        refractory_penalty=routing.refractory_penalty,
    )
