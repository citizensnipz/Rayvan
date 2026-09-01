from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .balancing import router_balance_loss
from .integrator import Integrator, IntegratorTrace, WeightedAverageIntegrator
from .modules import (
    EMCModule,
    EMCModuleBase,
    RecurrentEMCModule,
    StateSpaceEMCModule,
    create_emc_module,
)
from .nexus import ModuleAwareNexusRouter, NexusRouter, RoutingDecision


@dataclass(frozen=True)
class EMCConfig:
    latent_dim: int = 32
    num_modules: int = 6
    modules_per_cycle: int = 2
    num_cycles: int = 3
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
        if self.architecture_stage not in {"token", "n1_chunked", "n2"}:
            raise ValueError("architecture_stage must be token, n1_chunked, or n2")
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
        if self.router_type not in {"fixed_index", "module_aware"}:
            raise ValueError("router_type must be fixed_index or module_aware")
        if self.integrator_type not in {"weighted_average", "proposal_attention"}:
            raise ValueError(
                "integrator_type must be weighted_average or proposal_attention"
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
    def resolved_request_pool_size(self) -> int:
        return self.request_pool_size or self.num_modules

    @property
    def resolved_active_top_k(self) -> int:
        return self.active_top_k or self.modules_per_cycle

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


@dataclass(frozen=True)
class EMCOutput:
    logits: Tensor
    trace: tuple[EMCCycleTrace, ...]
    router_balance_loss: Tensor | None = None
    cycle_logits: tuple[Tensor, ...] | None = None
    chunk_trace: object | None = None
    n2_state: object | None = None


class EMCModel(nn.Module):
    def __init__(self, config: EMCConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.latent_dim)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.latent_dim
        )
        if config.router_type == "module_aware":
            self.router: NexusRouter | ModuleAwareNexusRouter = (
                ModuleAwareNexusRouter(config)
            )
        else:
            self.router = NexusRouter(config)
        self.emc_modules = nn.ModuleList(
            create_emc_module(config, family)
            for family in config.resolved_module_families
        )
        if config.integrator_type == "proposal_attention":
            self.integrator: WeightedAverageIntegrator | Integrator = Integrator(
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

    @property
    def module_families(self) -> tuple[str, ...]:
        return tuple(module.family for module in self.emc_modules)

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
        cycles_to_run = evaluation_cycle_limit or self.config.num_cycles
        positions = torch.arange(sequence_length, device=token_ids.device)
        latent = self.token_embedding(token_ids) + self.position_embedding(positions)
        cycle_traces: list[EMCCycleTrace] = []
        cycle_balance_losses: list[Tensor] = []
        per_cycle_logits: list[Tensor] = []

        for cycle in range(cycles_to_run):
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
                    sequence=sequence_length,
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
            module_updates = self.execute_selected_modules(
                latent, routing.selected_indices
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
            integrated = self.integrator(
                latent,
                module_updates,
                routing.selected_weights,
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
                    )
                )

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
            )
        return logits


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
    )
