"""Multiple diagonal competence basins with positive weighted spectral cosine."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .nexus import RoutingDecision, _mask_unavailable_action, _update_balance_bias
from .spectral_geometry import Descriptor, extract_descriptor, feature_names
from .spectral_config import spectral_config


class RunningStandardizer(nn.Module):
    """Population Welford statistics. Explicit training-only updates; persistent freeze."""
    def __init__(self, dim, floor=1e-3, frozen=False):
        super().__init__()
        self.floor = floor
        self.register_buffer('count', torch.zeros((), dtype=torch.float64))
        self.register_buffer('mean', torch.zeros(dim, dtype=torch.float64))
        self.register_buffer('m2', torch.zeros(dim, dtype=torch.float64))
        self.register_buffer('frozen', torch.tensor(frozen))

    @torch.no_grad()
    def update(self, x):
        if not self.training or bool(self.frozen):
            return
        x = x.detach().reshape(-1, self.mean.numel()).double()
        if not torch.isfinite(x).all():
            raise ValueError('nonfinite statistics input')
        n = x.size(0)
        if not n:
            return
        delta = x.mean(0) - self.mean
        total = self.count + n
        self.m2.add_(((x - x.mean(0)).square()).sum(0) + delta.square() * self.count * n / total)
        self.mean.add_(delta * n / total)
        self.count.copy_(total)

    def forward(self, x):
        if self.count == 0:
            return x.float()
        std = (self.m2 / self.count.clamp_min(1)).sqrt().clamp_min(self.floor)
        return ((x.double() - self.mean) / std).float()


def weighted_resonance(q, template, weight, eps=1e-8):
    # Inputs broadcast to [..., Q].
    numerator = (weight * q * template).sum(-1)
    denominator = (weight * q.square()).sum(-1).sqrt() * (weight * template.square()).sum(-1).sqrt()
    return (numerator / (denominator + eps)).clamp(0, 1)


def routing_objective(scores, losses, c):
    y = F.softmax(-losses.detach().float() / c.spectral_target_temperature, -1)
    logp = F.log_softmax(scores.float() / c.spectral_router_temperature, -1)
    kl = F.kl_div(logp, y, reduction='batchmean')
    regret = (logp.exp() * (losses.detach() - losses.detach().amin(-1, keepdim=True))).sum(-1).mean()
    return c.spectral_route_weight * kl + c.spectral_regret_weight * regret


class SpectralGeometricRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c = spectral_config(config)
        self.q_names, self.g_names = feature_names(self.c)
        qdim, gdim = len(self.q_names), len(self.g_names)
        if self.c.geometry_temporal_delta_enabled:
            self.g_names += tuple('delta.' + n for n in self.g_names) + tuple('delta.' + n for n in self.q_names) + ('previous_descriptor_valid',)
        e, m, g = config.num_modules, self.c.basins_per_expert, len(self.g_names)
        self.centres = nn.Parameter(torch.zeros(e, m, g))
        self.raw_variance = nn.Parameter(torch.full((e, m, g), math.log(math.expm1(1.0))))
        self.raw_templates = nn.Parameter(torch.zeros(e, m, qdim))
        self.raw_weights = nn.Parameter(torch.zeros(e, m, qdim))
        self.bias = nn.Parameter(torch.zeros(e, m))
        # Same deterministic basin set for every expert: no random expert winner.
        # Tiny distinct basin offsets avoid an exactly symmetric collapse before a bank exists.
        with torch.no_grad():
            phase = torch.arange(m, dtype=torch.float32)[:, None] + torch.arange(g, dtype=torch.float32)[None] * 1.618
            self.centres.copy_((.01 * phase.sin())[None].expand(e, -1, -1))
        # Equal expert scores until training-only evidence initializes basins.
        self.standardizer = RunningStandardizer(g, self.c.geometry_std_floor, self.c.geometry_statistics_frozen)
        self.register_buffer('balance_bias', torch.zeros(e))
        self.register_buffer('basin_occupancy', torch.zeros(e, m, dtype=torch.long))
        self.register_buffer('initialized', torch.tensor(False))

    def describe(self, latent, *, profile=False):
        return extract_descriptor(latent, self.c, profile=profile)

    def geometry_vector(self, descriptor, previous=None):
        g = descriptor.g
        if self.c.geometry_temporal_delta_enabled:
            dg = g - previous.g if previous is not None else torch.zeros_like(g)
            dq = descriptor.q - previous.q if previous is not None else torch.zeros_like(descriptor.q)
            mask = g.new_full((g.size(0), 1), float(previous is not None))
            g = torch.cat((g, dg, dq, mask), -1)
        return g

    def score_descriptor(self, descriptor, previous=None, *, update_stats=False):
        g = self.geometry_vector(descriptor, previous)
        if update_stats:
            self.standardizer.update(g)
        standardized = self.standardizer(g)
        variance = (F.softplus(self.raw_variance.float()) + self.c.basin_variance_floor).clamp_max(self.c.basin_variance_ceiling)
        distance = ((standardized[:, None, None] - self.centres.float()).square() / variance).mean(-1)
        templates = F.normalize(F.softplus(self.raw_templates.float()), dim=-1)
        weight = F.softplus(self.raw_weights.float()) + self.c.spectral_eps
        resonance = weighted_resonance(descriptor.q[:, None, None], templates, weight, self.c.spectral_eps)
        dissonance = 1 - resonance
        basin = self.bias.float() - self.c.geometry_score_weight * distance - self.c.resonance_score_weight * dissonance
        temp = self.c.basin_temperature
        scores = temp * (torch.logsumexp(basin / temp, -1) - math.log(basin.size(-1)))
        return scores, dict(geometry_vector=g, standardized_geometry=standardized,
                            D_geo=distance, D_res=dissonance, resonance=resonance, basin_scores=basin,
                            winning_basin=basin.argmax(-1), expert_scores=scores,
                            expert_probabilities=(scores / self.c.spectral_router_temperature).softmax(-1))

    def route_one(self, latent, *, previous_descriptor=None, availability_mask=None, module_descriptors=None,
                  score_adjustment=None, refractory_penalty=None):
        if module_descriptors is not None or (score_adjustment is not None and torch.count_nonzero(score_adjustment)):
            raise ValueError('spectral basins do not accept external descriptors or legacy score adjustments')
        descriptor = self.describe(latent)
        scores, details = self.score_descriptor(descriptor, previous_descriptor, update_stats=self.training)
        base = -scores[:, None]
        balance = -self.balance_bias[None, None] if self.config.loss_free_balance_enabled else torch.zeros_like(base)
        effective = base + balance
        pre = effective
        if refractory_penalty is not None:
            effective = effective + refractory_penalty
        effective = _mask_unavailable_action(effective, availability_mask)
        selected = effective.argmin(-1, keepdim=True)
        ordered = base.sort(-1).values
        margin = ordered[..., 1] - ordered[..., 0] if base.size(-1) > 1 else torch.zeros_like(base[..., 0])
        details.update(descriptor.diagnostics)
        details['uninhibited_winner'] = pre.argmin(-1)
        details['inhibited_winner'] = selected.squeeze(-1)
        details['effective_expert_probabilities'] = (-effective[:, 0] / self.c.spectral_router_temperature).softmax(-1)
        details['refractory_winner_changed'] = pre.argmin(-1) != selected.squeeze(-1)
        if self.training:
            with torch.no_grad():
                ids = selected.reshape(-1)
                wins = details['winning_basin'].gather(1, ids[:, None]).squeeze(1)
                self.basin_occupancy.view(-1).add_(torch.bincount(ids * self.c.basins_per_expert + wins,
                                                               minlength=self.basin_occupancy.numel()).view(-1))
        return RoutingDecision(scores=-effective, selected_indices=selected, selected_weights=torch.ones_like(selected, dtype=scores.dtype),
                               raw_scores=-base, pre_inhibition_scores=-pre, refractory_penalty=refractory_penalty,
                               base_actions=base, effective_actions=effective, balance_action=balance,
                               winning_prototypes=details['winning_basin'][:, None], geometric_winner=base.argmin(-1, keepdim=True),
                               action_margin=margin, spectral_descriptor=descriptor, spectral_diagnostics=details)

    def update_balance_bias(self, selected):
        _update_balance_bias(self, selected)

    def forward(self, latent, remaining=None):
        return -self.score_descriptor(self.describe(latent))[0]

    def calibration_loss(self, latent, remaining, losses):
        scores, _ = self.score_descriptor(self.describe(latent))
        return routing_objective(scores, losses, self.c) + self.c.basin_redundancy_weight * self.redundancy_loss()

    def redundancy_loss(self):
        if self.c.basins_per_expert < 2:
            return self.centres.sum() * 0
        a, b = torch.triu_indices(self.c.basins_per_expert, self.c.basins_per_expert, 1, device=self.centres.device)
        distance = (self.centres[:, a] - self.centres[:, b]).square().mean(-1)
        template = F.normalize(F.softplus(self.raw_templates), dim=-1)
        similarity = (template[:, a] * template[:, b]).sum(-1)
        return (F.relu(1 - distance / self.c.basin_redundancy_distance**2) *
                F.relu(similarity - self.c.basin_redundancy_similarity).square()).mean()

    @torch.no_grad()
    def initialize(self, descriptors, losses, previous=None, *, geometry_vectors=None):
        """Deterministic usefulness-weighted farthest seeds and weighted Lloyd fits.

        Caller must supply training observations only. Freezes fitted statistics.
        """
        if not self.training:
            raise ValueError('basin initialization requires training mode and training observations')
        g = self.geometry_vector(descriptors, previous) if geometry_vectors is None else geometry_vectors
        self.standardizer.update(g)
        self.standardizer.frozen.fill_(True)
        g = self.standardizer(g)
        q = descriptors.q
        y = (-losses.detach() / self.c.spectral_target_temperature).softmax(-1)
        for e in range(self.config.num_modules):
            # Competitive means at least uniform target mass; safe weighted fallback.
            useful = y[:, e] >= 1 / self.config.num_modules
            indices = useful.nonzero().flatten()
            if not indices.numel():
                indices = y[:, e].argmax().view(1)
            ge, qe, we = g[indices], q[indices], y[indices, e]
            seeds = [int(we.argmax())]
            for _ in range(1, self.c.basins_per_expert):
                distances = torch.cdist(ge, ge[seeds]).square().amin(-1)
                seeds.append(int((distances * we).argmax()))
            centres = ge[seeds].clone()
            for _ in range(8):
                assignment = torch.cdist(ge, centres).argmin(-1)
                for m in range(self.c.basins_per_expert):
                    member = assignment == m
                    if member.any():
                        w = we[member] / we[member].sum()
                        centres[m] = (ge[member] * w[:, None]).sum(0)
            for m in range(self.c.basins_per_expert):
                member = assignment == m
                if not member.any():
                    member[seeds[m]] = True
                w = we[member] / we[member].sum()
                variance = ((ge[member] - centres[m]).square() * w[:, None]).sum(0).clamp(self.c.basin_variance_floor * 2, self.c.basin_variance_ceiling)
                template = (qe[member] * w[:, None]).sum(0).clamp_min(1e-5)
                self.centres[e, m].copy_(centres[m])
                self.raw_variance[e, m].copy_(torch.log(torch.expm1((variance-self.c.basin_variance_floor).clamp_max(50))))
                self.raw_templates[e, m].copy_(torch.log(torch.expm1(template)))
        self.initialized.fill_(True)
