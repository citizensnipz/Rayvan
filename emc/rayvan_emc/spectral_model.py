"""Causal sequential spectral EMC, isolated from every historical router path."""
from dataclasses import replace
import torch
from torch.nn import functional as F
from .model import SequentialEMCModel, EMCOutput, EMCCycleTrace, CounterfactualProbeTrace
from .balancing import router_balance_loss
from .value_routing import CounterfactualValueEMC
from .spectral_routing import RunningStandardizer, routing_objective
from .transformation_geometry import transformation_signature, transformation_names, complementarity


def cpu_trace(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_trace(v) for k, v in value.items()}
    return value


class SpectralGeometricEMC(SequentialEMCModel):
    prefix_endpoint_objective = True
    # Reuse existing expert execution, endpoint readout and counterfactual losses.
    embed = CounterfactualValueEMC.embed
    read_endpoint = CounterfactualValueEMC.read_endpoint
    apply_expert = CounterfactualValueEMC.apply_expert
    counterfactual_losses = CounterfactualValueEMC.counterfactual_losses

    def __init__(self, config):
        if config.router_type != 'spectral_geometric':
            raise ValueError('SpectralGeometricEMC requires router_type=spectral_geometric')
        if config.integrator_type not in {'acceptance_gate', 'identity_free_gate'}:
            raise ValueError('spectral EMC requires a sequential acceptance integrator')
        super().__init__(config)
        c = self.router.c
        self._spectral_sample = None
        self.transformation_standardizer = RunningStandardizer(len(transformation_names(c)), c.geometry_std_floor,
                                                              c.geometry_statistics_frozen)

    def continue_state(self, state, depth, *, first=None, stop=None):
        """Fixed suffix policy with request-local refractory/descriptor state."""
        end = self.config.resolved_trajectory_steps if stop is None else stop
        previous = None
        refractory = state.new_zeros(state.size(0), 1, self.config.num_modules)
        for t in range(depth, end):
            decision = self.router.route_one(state, previous_descriptor=previous,
                                            refractory_penalty=refractory if self.config.refractory_enabled else None)
            selected = decision.selected_indices.reshape(-1)
            if t == depth and first is not None:
                selected = torch.full_like(selected, first) if isinstance(first, int) else first
            previous = decision.spectral_descriptor
            state = self.apply_expert(state, selected)
            refractory = refractory * self.config.refractory_decay
            refractory.scatter_add_(-1, selected[:, None, None], torch.full_like(selected[:, None, None], self.config.refractory_strength, dtype=state.dtype))
        return state

    def _probe(self, state, targets, decision, indices, depth, previous):
        c = self.router.c
        observed = state.detach().index_select(0, indices)
        target = targets.index_select(0, indices)
        losses, signatures, ranks = [], [], []
        active = c.transformation_complementarity_weight > 0 and self.training
        for e in range(self.config.num_modules):
            selected = torch.full((observed.size(0),), e, device=state.device, dtype=torch.long)
            with torch.set_grad_enabled(active):
                if active:
                    # Keep Integrator parameters fixed: complementarity updates experts only.
                    proposal = self.execute_selected_requests(observed, selected)
                    fixed = {name: parameter.detach() for name, parameter in self.integrator.named_parameters()}
                    after = torch.func.functional_call(self.integrator, fixed,
                        (observed, proposal, observed.new_ones(observed.size(0), observed.size(1), 1)),
                        dict(selected_indices=selected))
                else:
                    after = self.apply_expert(observed, selected)
                if c.transformation_diagnostics_enabled or active:
                    transform = transformation_signature(observed, after, c, active=active)
                    signatures.append(transform.signature); ranks.append(transform.rank)
            with torch.no_grad():
                # Immediate endpoint usefulness matches the legacy geometric probe.
                losses.append(F.cross_entropy(self.read_endpoint(after.detach()).float(), target, reduction='none'))
        losses = torch.stack(losses, -1)
        scores = decision.raw_scores[:, 0].index_select(0, indices)
        y = (-losses / c.spectral_target_temperature).softmax(-1)
        objective = routing_objective(scores, losses, c) + c.basin_redundancy_weight * self.router.redundancy_loss()
        details = dict(candidate_losses=losses, target_usefulness=y, target='immediate_endpoint')
        if signatures:
            signature = torch.stack(signatures, 1)
            self.transformation_standardizer.update(signature.detach())
            comp, similarity, overlap = complementarity(signature, y, c.transformation_complementarity_threshold,
                                                       standardizer=self.transformation_standardizer)
            objective = objective + c.transformation_complementarity_weight * comp
            details.update(transformation_signature=signature, transformation_names=transformation_names(c),
                           transformation_rank=torch.stack(ranks, 1), transformation_similarity=similarity,
                           usefulness_overlap=overlap, complementarity_loss=comp, complementarity_active=active)
        best = losses.argmin(-1)
        selected = decision.selected_indices.reshape(-1).index_select(0, indices)
        winner = scores.argmax(-1)
        ordered = losses.sort(-1).values
        margin = ordered[:, 1] - ordered[:, 0] if losses.size(-1) > 1 else losses[:, 0]*0
        regret = losses.gather(1, selected[:, None]).squeeze(1) - losses.amin(-1)
        trace = CounterfactualProbeTrace(indices.cpu(), tuple('spectral_probe' for _ in indices), losses.cpu(),
                    best.cpu(), selected.cpu(), winner.cpu(), (winner == best).cpu(),
                    (scores.topk(min(2, scores.size(-1))).indices == best[:, None]).any(-1).cpu(),
                    regret.cpu(), margin.cpu(), (margin <= self.config.counterfactual_tie_epsilon).cpu())
        return objective, trace, cpu_trace(details)

    def endpoint(self, tokens, *, return_trace=False, return_cycle_logits=False, counterfactual_targets=None,
                 training_step=0, force_counterfactual_probe=False, balance_entropy_floor=.75,
                 availability_mask=None, evaluation_cycle_limit=None, diagnostic_forced_modules=None,
                 diagnostic_zero_proposal_mask=None, module_descriptors=None):
        if module_descriptors is not None:
            raise ValueError('spectral routing owns its competence basins')
        state = self.embed(tokens)
        steps = evaluation_cycle_limit or self.config.resolved_trajectory_steps
        if not 1 <= steps <= self.config.resolved_trajectory_steps:
            raise ValueError('invalid evaluation cycle limit')
        previous = None
        refractory = state.new_zeros(state.size(0), 1, self.config.num_modules)
        traces, cycles, balances, objectives, selections = [], [], [], [], []
        probed = torch.zeros(state.size(0), dtype=torch.bool, device=state.device)
        budget = self.config.counterfactual_max_probes_per_forward
        for depth in range(steps):
            decision = self.router.route_one(state, previous_descriptor=previous, availability_mask=availability_mask,
                       refractory_penalty=refractory if self.config.refractory_enabled else None)
            selected = decision.selected_indices.reshape(-1)
            if diagnostic_forced_modules is not None:
                forced = diagnostic_forced_modules.to(state.device).long().reshape(-1)
                if forced.numel() != 1 or not 0 <= int(forced[0]) < self.config.num_modules:
                    raise ValueError('force exactly one valid expert')
                selected = forced.expand(state.size(0))
                decision = replace(decision, selected_indices=selected[:, None, None])
            probe_trace = None
            diagnostic = decision.spectral_diagnostics
            if counterfactual_targets is not None and budget > 0 and self.config.counterfactual_calibration_enabled and (self.training or force_counterfactual_probe):
                if force_counterfactual_probe:
                    ids = (~probed).nonzero().flatten()[:budget] if depth == training_step % steps else torch.empty(0, device=state.device, dtype=torch.long)
                else:
                    ids, _ = self._sample_counterfactual_probes(decision, probed, budget, training_step=training_step, trajectory_step=depth)
                if ids.numel():
                    target = counterfactual_targets[:, -1] if counterfactual_targets.ndim == 2 else counterfactual_targets
                    objective, probe_trace, measurement = self._probe(state, target, decision, ids, depth, previous)
                    objectives.append(objective)
                    diagnostic = {**diagnostic, 'counterfactual': measurement}
                    probed[ids] = True; budget -= ids.numel()
                    self._observe_counterfactual(probe_trace, depth)
            if return_trace:
                # One recent sampled request, bounded independently of run length.
                def sample(value):
                    if isinstance(value, torch.Tensor):
                        return value[:1].detach().cpu() if value.ndim else value.detach().cpu()
                    if isinstance(value, dict):
                        return {key: sample(v) for key, v in value.items()}
                    return value
                self._spectral_sample = sample(diagnostic)
            previous = decision.spectral_descriptor
            integrated = self.apply_expert(state, selected, diagnostics=return_trace, zero_mask=diagnostic_zero_proposal_mask)
            state, gate = integrated if return_trace else (integrated, None)
            balances.append(router_balance_loss(decision.scores, decision.selected_indices, entropy_floor=balance_entropy_floor))
            selections.append(decision.selected_indices.detach())
            if return_cycle_logits:
                cycles.append(self.read_endpoint(state).unsqueeze(1))
            if return_trace:
                traces.append(EMCCycleTrace(cycle=depth+1, selected_modules=tuple(torch.unique(selected).tolist()),
                  router_scores=decision.scores.detach().cpu(), router_weights=decision.selected_weights.detach().cpu(),
                  latent_shape=tuple(state.shape), selected_indices=decision.selected_indices.detach().cpu(), integrator_trace=gate,
                  module_families=self.module_families, expert_names=self.expert_names,
                  raw_router_scores=decision.raw_scores.detach().cpu(), pre_inhibition_router_scores=decision.pre_inhibition_scores.detach().cpu(),
                  effective_router_scores=decision.scores.detach().cpu(), refractory_penalty=refractory.detach().cpu(),
                  base_actions=decision.base_actions.detach().cpu(), effective_actions=decision.effective_actions.detach().cpu(),
                  geometric_winner=decision.geometric_winner.detach().cpu(), winning_prototypes=decision.winning_prototypes.detach().cpu(),
                  spectral_diagnostics=cpu_trace(diagnostic), counterfactual=probe_trace))
            self._observe_geometric_routing(decision, evaluation=not self.training)
            refractory = refractory * self.config.refractory_decay
            refractory.scatter_add_(-1, selected[:, None, None], torch.full_like(selected[:, None, None], self.config.refractory_strength, dtype=state.dtype))
        if self.training:
            self.router.update_balance_bias(torch.cat(selections))
        logits = self.read_endpoint(state).unsqueeze(1)
        if return_trace or return_cycle_logits:
            return EMCOutput(logits, tuple(traces), torch.stack(balances).mean(), tuple(cycles) if cycles else None,
                             geometry_calibration_loss=torch.stack(objectives).mean() if objectives else logits.new_zeros(()))
        return logits

    def forward(self, token_ids, *, return_trace=False, return_cycle_logits=False, **kwargs):
        # Every output position starts from its own observed prefix; later token
        # routing decisions cannot influence earlier logits, even with Delta experts.
        rows, outputs = [], []
        for length in range(1, token_ids.size(1)+1):
            options = dict(kwargs)
            target = options.get('counterfactual_targets')
            if target is not None and target.ndim == 2:
                options['counterfactual_targets'] = target[:, :length]
            output = self.endpoint(token_ids[:, :length], return_trace=return_trace,
                                   return_cycle_logits=return_cycle_logits, **options)
            rows.append(output.logits if isinstance(output, EMCOutput) else output)
            if isinstance(output, EMCOutput):
                outputs.append(output)
        logits = torch.cat(rows, 1)
        if not outputs:
            return logits
        cycles = tuple(torch.cat([o.cycle_logits[i] for o in outputs], 1) for i in range(len(outputs[-1].cycle_logits))) if return_cycle_logits else None
        return EMCOutput(logits, outputs[-1].trace, torch.stack([o.router_balance_loss for o in outputs]).mean(), cycles,
                         geometry_calibration_loss=torch.stack([o.geometry_calibration_loss for o in outputs]).mean())

    def geometric_routing_report(self):
        stats = self._geometry_statistics
        regrets = stats['regrets']
        count = len(regrets)
        def json_value(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, dict):
                return {k: json_value(v) for k, v in value.items()}
            return value
        return dict(router_type='spectral_geometric', total_probes=count,
                    mean_routing_regret=sum(regrets)/count if count else None,
                    counterfactual_top1_accuracy=stats['top1_correct']/count if count else None,
                    counterfactual_top2_accuracy=stats['top2_correct']/count if count else None,
                    routing_counts=stats['routing_counts'], evaluation_routing_counts=stats['evaluation_routing_counts'],
                    basin_occupancy=self.router.basin_occupancy.cpu().tolist(),
                    basin_redundancy=float(self.router.redundancy_loss().detach()),
                    harmonic_feature_names=self.router.q_names, geometry_feature_names=self.router.g_names,
                    per_step=stats['per_step'], sampled_geometry=json_value(self._spectral_sample))
