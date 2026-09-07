"""Continuation-value routing. See docs/counterfactual-value-routing.md.

Only observed-prefix endpoint logits are computed in one trajectory. The public
all-position forward explicitly evaluates separate prefixes to preserve causality.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import EMCConfig, EMCOutput, EMCCycleTrace, SequentialEMCModel


def center(values: Tensor) -> Tensor:
    return values - values.mean(dim=-1, keepdim=True)


class ValueNexusRouter(nn.Module):
    """A(s, e) = [P(W f(s, remaining) + b)]_e; smaller is better."""

    def __init__(self, config: EMCConfig) -> None:
        super().__init__()
        width = config.latent_dim
        dim = config.resolved_routing_geometry_dim
        self.max_steps = config.resolved_trajectory_steps
        self.positions = nn.Embedding(config.max_sequence_length, width)
        self.key = nn.Linear(width, dim)
        self.query = nn.Parameter(torch.randn(dim) / dim**0.5)
        self.encoder = nn.Sequential(nn.Linear(2 * width + 1, width), nn.GELU(), nn.Linear(width, dim))
        if config.value_head_type == "mlp":
            self.value = nn.Sequential(nn.Linear(dim, config.value_head_hidden_dim), nn.GELU(),
                                       nn.Linear(config.value_head_hidden_dim, config.num_modules))
        else:
            self.value = nn.Linear(dim, config.num_modules)
        output = self.value[-1] if isinstance(self.value, nn.Sequential) else self.value
        # No arbitrary expert preference in an uncalibrated value head.
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def need(self, latent: Tensor, remaining: int) -> Tensor:
        if not 1 <= remaining <= self.max_steps:
            raise ValueError("remaining steps outside configured horizon")
        positions = self.positions(torch.arange(latent.size(1), device=latent.device))
        keys = torch.tanh(self.key(latent + positions))
        attention = torch.softmax(keys @ self.query, dim=1)
        pooled = (attention.unsqueeze(-1) * latent).sum(dim=1)
        horizon = latent.new_full((latent.size(0), 1), remaining / self.max_steps)
        return self.encoder(torch.cat((pooled, latent[:, -1], horizon), dim=-1))

    def forward(self, latent: Tensor, remaining: int) -> Tensor:
        return center(self.value(self.need(latent, remaining)))

    def choose(self, latent: Tensor, remaining: int, availability: Tensor | None = None) -> Tensor:
        costs = self(latent, remaining)
        if availability is not None:
            mask = availability.to(device=latent.device, dtype=torch.bool)
            mask = torch.broadcast_to(mask, costs.shape)
            if not mask.any(dim=-1).all():
                raise ValueError("each request must have an available expert")
            costs = costs.masked_fill(~mask, torch.inf)
        return costs.argmin(dim=-1)

    def calibration_loss(self, state: Tensor, remaining: int, losses: Tensor) -> Tensor:
        # Detach INPUT, not encoder output: router learns, upstream experts do not.
        prediction = self(state.detach(), remaining).float()
        return F.mse_loss(prediction, center(losses.detach().float()))

    def metric(self) -> Tensor:
        if isinstance(self.value, nn.Sequential):
            raise ValueError("The nonlinear head has no single global Mahalanobis metric")
        coefficients = self.value.weight - self.value.weight.mean(dim=0, keepdim=True)
        return coefficients.T @ coefficients / coefficients.size(0)


class CompetenceEnergy(nn.Module):
    """Scaled soft minimum over unit-sphere prototypes; smaller is better."""
    temperature = 0.25
    scale = 0.05

    def __init__(self, experts: int, prototypes: int, dim: int):
        super().__init__()
        # Identical initial basins across experts avoid random initial winners.
        # Supervision separates experts; prototypes within each expert differ.
        initial = F.normalize(torch.randn(prototypes, dim), dim=-1)
        self.prototypes = nn.Parameter(initial.unsqueeze(0).repeat(experts, 1, 1))
        self.bias = nn.Parameter(torch.zeros(experts))

    def forward(self, z: Tensor) -> Tensor:
        z = F.normalize(z, dim=-1)
        mu = F.normalize(self.prototypes, dim=-1)
        views = z[:, None, None, :] if z.ndim == 2 else z[:, :, None, :]
        distance = (views - mu[None]).square().sum(-1)
        softmin = -self.temperature * (torch.logsumexp(-distance / self.temperature, dim=-1)
                                       - math.log(mu.size(1)))
        return self.bias + self.scale * softmin


class RelationalGeometricRouter(ValueNexusRouter):
    """Position-aware slot attention followed by counterfactual competence basins."""
    def __init__(self, config: EMCConfig):
        nn.Module.__init__(self)
        self.max_steps = config.resolved_trajectory_steps
        self.regret_weight = config.value_geometry_regret_weight
        self.policy_temperature = 0.05
        width, slot_width = config.latent_dim, 32
        self.positions = nn.Embedding(config.max_sequence_length, width)
        self.input_norm = nn.LayerNorm(width)
        self.input_projection = nn.Linear(width, slot_width)
        self.queries = nn.Parameter(torch.randn(config.value_geometry_slots, slot_width) / slot_width**0.5)
        self.condition = nn.Linear(width + 1, slot_width)
        self.cross_attention = nn.MultiheadAttention(slot_width, 4, batch_first=True, dropout=0.)
        self.cross_norm = nn.LayerNorm(slot_width)
        self.slot_attention = nn.MultiheadAttention(slot_width, 4, batch_first=True, dropout=0.)
        self.slot_norm = nn.LayerNorm(slot_width)
        self.slot_ff = nn.Sequential(nn.Linear(slot_width, 2*slot_width), nn.GELU(), nn.Linear(2*slot_width, slot_width))
        self.output_norm = nn.LayerNorm(slot_width)
        self.encoder = nn.Sequential(nn.Linear(config.value_geometry_slots * slot_width + width + 1, slot_width),
                                     nn.GELU(), nn.Linear(slot_width, config.resolved_routing_geometry_dim))
        self.value = CompetenceEnergy(config.num_modules, config.value_geometry_prototypes,
                                      config.resolved_routing_geometry_dim)

    def need(self, latent: Tensor, remaining: int) -> Tensor:
        if not 1 <= remaining <= self.max_steps:
            raise ValueError("remaining steps outside configured horizon")
        position = self.positions(torch.arange(latent.size(1), device=latent.device))
        sequence = self.input_projection(self.input_norm(latent + position))
        horizon = latent.new_full((latent.size(0), 1), remaining / self.max_steps)
        context = torch.cat((latent[:, -1], horizon), -1)
        queries = self.queries[None] + self.condition(context)[:, None]
        slots = self.cross_norm(queries + self.cross_attention(queries, sequence, sequence, need_weights=False)[0])
        slots = self.slot_norm(slots + self.slot_attention(slots, slots, slots, need_weights=False)[0])
        slots = self.output_norm(slots + self.slot_ff(slots))
        return F.normalize(self.encoder(torch.cat((slots.flatten(1), context), -1)), dim=-1)

    def calibration_loss(self, state: Tensor, remaining: int, losses: Tensor) -> Tensor:
        prediction = self(state.detach(), remaining).float()
        truth = losses.detach().float()
        mse = F.mse_loss(prediction, center(truth))
        regret = truth - truth.min(-1, keepdim=True).values
        probabilities = torch.softmax(-prediction / self.policy_temperature, dim=-1)
        return mse + self.regret_weight * (probabilities * regret).sum(-1).mean()

    def metric(self) -> Tensor:
        raise ValueError("Multiple competence basins have no single global Mahalanobis metric")


class ExpertConditionedGeometricRouter(RelationalGeometricRouter):
    """One state view per candidate; common metric and calibrated cost units.

    Only the router runs at inference. Effect decoder/targets are training-only.
    Queries differ initially, but collection uses a frozen reference, so this
    cannot change expert training opportunity or the evidence sampled.
    """
    def __init__(self, config: EMCConfig):
        nn.Module.__init__(self)
        self.max_steps = config.resolved_trajectory_steps
        self.regret_weight = config.value_geometry_regret_weight
        self.effect_weight = config.value_effect_weight
        self.policy_temperature = 0.05
        width, attention_width = config.latent_dim, 32
        dim = config.resolved_routing_geometry_dim
        self.positions = nn.Embedding(config.max_sequence_length, width)
        self.input_norm = nn.LayerNorm(width)
        self.input_projection = nn.Linear(width, attention_width)
        self.queries = nn.Parameter(torch.randn(config.num_modules, attention_width) / attention_width**0.5)
        self.condition = nn.Linear(width + 1, attention_width)
        self.cross_attention = nn.MultiheadAttention(attention_width, 4, batch_first=True, dropout=0.)
        self.cross_norm = nn.LayerNorm(attention_width)
        self.encoder = nn.Sequential(nn.Linear(attention_width, attention_width), nn.GELU(), nn.Linear(attention_width, dim))
        self.value = CompetenceEnergy(config.num_modules, config.value_geometry_prototypes, dim)
        self.effect_decoder = nn.Sequential(nn.Linear(dim + attention_width, 32), nn.GELU(), nn.Linear(32, config.value_effect_dim))
        # A fixed structured linear projection of the *integrated* update:
        # R(delta) = concat(mean_token(delta), last_token(delta)) @ projection.
        # Separate RNG makes effect targets identical across router seeds.
        rng = torch.Generator().manual_seed(1729)
        projection = torch.randn(2 * width, config.value_effect_dim, generator=rng) / (2 * width)**0.5
        self.register_buffer("effect_projection", projection)

    def need(self, latent: Tensor, remaining: int) -> Tensor:
        if not 1 <= remaining <= self.max_steps:
            raise ValueError("remaining steps outside configured horizon")
        position = self.positions(torch.arange(latent.size(1), device=latent.device))
        sequence = self.input_projection(self.input_norm(latent + position))
        horizon = latent.new_full((latent.size(0), 1), remaining / self.max_steps)
        q = self.queries[None] + self.condition(torch.cat((latent[:, -1], horizon), -1))[:, None]
        views = self.cross_norm(q + self.cross_attention(q, sequence, sequence, need_weights=False)[0])
        return F.normalize(self.encoder(views), dim=-1)

    @torch.no_grad()
    def project_effect(self, before: Tensor, after: Tensor) -> Tensor:
        delta = after.float() - before.float()
        return torch.cat((delta.mean(1), delta[:, -1]), -1) @ self.effect_projection.float()

    def objective(self, state: Tensor, remaining: int, losses: Tensor, effects: Tensor):
        z = self.need(state.detach(), remaining)
        prediction = center(self.value(z)).float()
        truth = losses.detach().float()
        mse = F.mse_loss(prediction, center(truth))
        regret = (torch.softmax(-prediction / self.policy_temperature, -1) *
                  (truth - truth.min(-1, keepdim=True).values)).sum(-1).mean()
        identities = self.queries[None].expand(z.size(0), -1, -1)
        effect_prediction = self.effect_decoder(torch.cat((z, identities), -1)).float()
        effect_mse = F.mse_loss(effect_prediction, effects.detach().float())
        total = mse + self.regret_weight * regret + self.effect_weight * effect_mse
        return total, dict(value_mse=mse, effect_mse=effect_mse, soft_regret=regret)


def make_value_router(config: EMCConfig) -> ValueNexusRouter:
    router_type = {"geometric": RelationalGeometricRouter,
                   "expert_geometric": ExpertConditionedGeometricRouter}.get(config.value_head_type, ValueNexusRouter)
    return router_type(config)


@dataclass(frozen=True)
class StateBatch:
    latent: Tensor
    targets: Tensor
    depth: int


class CounterfactualValueEMC(SequentialEMCModel):
    """Reusable experts with greedy sequential cost-to-go decisions."""

    prefix_endpoint_objective = True

    def __init__(self, config: EMCConfig) -> None:
        if config.router_type != "counterfactual_value" or config.integrator_type != "identity_free_gate":
            raise ValueError("value EMC requires counterfactual_value and identity_free_gate")
        if config.refractory_enabled or config.loss_free_balance_enabled or config.switch_cost or config.persistence_bonus:
            raise ValueError("value EMC requires unbiased scores: disable refractory, balance and persistence")
        from .research_config import validate_value_settings
        validate_value_settings(config)
        super().__init__(config)
        self.reset_router(config.value_router_seed)

    def reset_router(self, seed: int) -> None:
        # Isolate router randomness from expert initialization and data sampling.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            router = make_value_router(self.config)
        self.router = router.to(device=self.token_embedding.weight.device,
                                dtype=self.token_embedding.weight.dtype)

    def embed(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or not 0 < tokens.size(1) <= self.config.max_sequence_length:
            raise ValueError("expected a nonempty observed prefix within context length")
        return self.token_embedding(tokens) + self.position_embedding(torch.arange(tokens.size(1), device=tokens.device))

    def read_endpoint(self, state: Tensor) -> Tensor:
        return self.output_projection(self.output_norm(state[:, -1]))

    def apply_expert(self, state: Tensor, selected: Tensor, *, diagnostics: bool = False,
                     zero_mask: Tensor | None = None):
        proposal = self.execute_selected_requests(state, selected)
        if zero_mask is not None:
            mask = zero_mask.to(state.device, dtype=torch.bool)[selected]
            proposal = proposal.masked_fill(mask[:, None, None, None], 0)
        return self.integrator(state, proposal, state.new_ones(state.size(0), state.size(1), 1),
                               selected_indices=selected, return_diagnostics=diagnostics)

    def continue_state(self, state: Tensor, depth: int, *, first: int | Tensor | None = None,
                       stop: int | None = None) -> Tensor:
        end = self.config.resolved_trajectory_steps if stop is None else stop
        for t in range(depth, end):
            if t == depth and first is not None:
                selected = (torch.full((state.size(0),), first, device=state.device, dtype=torch.long)
                            if isinstance(first, int) else first)
            else:
                selected = self.router.choose(state, self.config.resolved_trajectory_steps - t)
            state = self.apply_expert(state, selected)
        return state

    @torch.no_grad()
    def counterfactual_losses(self, batch: StateBatch, *, target: str = "suffix") -> Tensor:
        if target not in {"suffix", "immediate"}:
            raise ValueError("unknown counterfactual target")
        # Caller supplies a fixed eval-mode snapshot; each branch reroutes its suffix.
        end = batch.depth + 1 if target == "immediate" else self.config.resolved_trajectory_steps
        return torch.stack([
            F.cross_entropy(self.read_endpoint(self.continue_state(batch.latent, batch.depth, first=e, stop=end)).float(),
                            batch.targets, reduction="none")
            for e in range(self.config.num_modules)
        ], dim=-1)

    @torch.no_grad()
    def counterfactual_effects_and_losses(self, batch: StateBatch, observer: ExpertConditionedGeometricRouter):
        """Reuse each first insertion for both targets; no duplicate expert work."""
        losses, effects = [], []
        for e in range(self.config.num_modules):
            selected = torch.full((batch.latent.size(0),), e, device=batch.latent.device, dtype=torch.long)
            after = self.apply_expert(batch.latent, selected)
            effects.append(observer.project_effect(batch.latent, after))
            terminal = self.continue_state(after, batch.depth + 1)
            losses.append(F.cross_entropy(self.read_endpoint(terminal).float(), batch.targets, reduction="none"))
        return torch.stack(losses, 1), torch.stack(effects, 1)

    @torch.no_grad()
    def collect_states(self, tokens: Tensor, targets: Tensor, *, generator: torch.Generator,
                       exploration: float) -> list[StateBatch]:
        state = self.embed(tokens)
        batches = []
        for depth in range(self.config.resolved_trajectory_steps):
            batches.append(StateBatch(state.detach(), targets.detach(), depth))
            if depth + 1 < self.config.resolved_trajectory_steps:
                selected = self.router.choose(state, self.config.resolved_trajectory_steps - depth)
                random_ids = torch.randint(self.config.num_modules, (state.size(0),), generator=generator).to(state.device)
                explore = (torch.rand(state.size(0), generator=generator) < exploration).to(state.device)
                selected = torch.where(explore, random_ids, selected)
                state = self.apply_expert(state, selected)
        return batches

    def endpoint(self, tokens: Tensor, *, return_trace: bool = False, return_cycle_logits: bool = False,
                 exploration: float = 0.0, generator: torch.Generator | None = None,
                 availability_mask: Tensor | None = None, evaluation_cycle_limit: int | None = None,
                 diagnostic_forced_modules: Tensor | None = None,
                 diagnostic_zero_proposal_mask: Tensor | None = None) -> Tensor | EMCOutput:
        state = self.embed(tokens)
        horizon = self.config.resolved_trajectory_steps
        steps = horizon if evaluation_cycle_limit is None else evaluation_cycle_limit
        if not 1 <= steps <= horizon:
            raise ValueError("invalid evaluation cycle limit")
        traces, cycle_logits = [], []
        for depth in range(steps):
            costs = self.router(state, horizon - depth)
            selected = self.router.choose(state, horizon - depth, availability_mask)
            if diagnostic_forced_modules is not None:
                forced = diagnostic_forced_modules.to(state.device, dtype=torch.long).reshape(-1)
                if forced.numel() != 1:
                    raise ValueError("value routing forces exactly one expert per step")
                selected = forced.expand(state.size(0))
            elif exploration:
                random_ids = torch.randint(self.config.num_modules, (state.size(0),), generator=generator).to(state.device)
                mask = (torch.rand(state.size(0), generator=generator) < exploration).to(state.device)
                selected = torch.where(mask, random_ids, selected)
            integrated = self.apply_expert(state, selected, diagnostics=return_trace, zero_mask=diagnostic_zero_proposal_mask)
            if return_trace:
                state, gate_trace = integrated
                traces.append(EMCCycleTrace(
                    cycle=depth + 1, selected_modules=tuple(torch.unique(selected).tolist()),
                    router_scores=-costs.detach().unsqueeze(1), router_weights=state.new_ones(state.size(0), 1, 1),
                    latent_shape=tuple(state.shape), selected_indices=selected.detach().view(-1, 1, 1),
                    integrator_trace=gate_trace, module_families=self.module_families, expert_names=self.expert_names,
                    base_actions=costs.detach().unsqueeze(1), effective_actions=costs.detach().unsqueeze(1),
                ))
            else:
                state = integrated
            if return_cycle_logits:
                cycle_logits.append(self.read_endpoint(state).unsqueeze(1))
        logits = self.read_endpoint(state).unsqueeze(1)
        if return_trace or return_cycle_logits:
            return EMCOutput(logits, tuple(traces), logits.new_zeros(()), tuple(cycle_logits) if return_cycle_logits else None)
        return logits

    def forward(self, token_ids: Tensor, *, return_trace: bool = False, return_cycle_logits: bool = False,
                balance_entropy_floor: float = 0.75, counterfactual_targets: Tensor | None = None,
                training_step: int = 0, force_counterfactual_probe: bool = False, **kwargs) -> Tensor | EMCOutput:
        del balance_entropy_floor, counterfactual_targets, training_step
        if force_counterfactual_probe:
            raise ValueError("use the value trainer's snapshot audit for value counterfactuals")
        # Compatibility path is genuinely causal, including forced Delta routes.
        rows, cycles = [], []
        final = None
        for length in range(1, token_ids.size(1) + 1):
            result = self.endpoint(token_ids[:, :length], return_trace=return_trace and length == token_ids.size(1),
                                   return_cycle_logits=return_cycle_logits, **kwargs)
            rows.append(result.logits if isinstance(result, EMCOutput) else result)
            if isinstance(result, EMCOutput):
                final = result
                if return_cycle_logits:
                    cycles.append(result.cycle_logits)
        logits = torch.cat(rows, dim=1)
        if return_trace or return_cycle_logits:
            assert final is not None
            per_cycle = tuple(torch.cat([row[t] for row in cycles], dim=1) for t in range(len(cycles[0]))) if cycles else None
            return EMCOutput(logits, final.trace, logits.new_zeros(()), per_cycle)
        return logits
