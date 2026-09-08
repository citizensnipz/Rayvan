"""Frozen-expert policy improvement with genuine branch-specific sequential routing.

Bank warmup reproduces the selected spectral candidate. Online targets are fresh
suffix losses under the current (unchanged within a block) spectral policy.
No expert/shared parameter gradients, cached moving-policy labels or held-out
early stopping. Evaluation prefixes are excluded from all training samples.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .spectral_geometry import Descriptor
from .spectral_routing import SpectralGeometricRouter, routing_objective
from .spectral_experiment import cache, cached_scores, metrics
from .value_fit import unique_prefixes, load_bank_payload
from .training import TrainingCancelledError


def check_cancel(cancelled):
    if cancelled and cancelled():
        raise TrainingCancelledError('Sequential spectral test stopped at a safe boundary')


def expert_batch_limit(model, length, requested):
    limit = max(1, min(4, requested))
    for module in model.modules():
        if hasattr(module, 'max_transition_bytes'):
            per_item = length * module.heads * module.head_dim**2 * 4
            limit = min(limit, max(1, module.max_transition_bytes // per_item))
    return limit


@torch.no_grad()
def rollout(model, router, state, *, start=0, first=None, constant=None, original=False, cancelled=None):
    """Re-read the transformed latent at every depth; no preselected Top-K path."""
    choices = []
    horizon = model.config.resolved_trajectory_steps
    for depth in range(start, horizon):
        check_cancel(cancelled)
        if first is not None and depth == start:
            selected = torch.full((state.size(0),), first, device=state.device, dtype=torch.long)
        elif constant is not None:
            selected = torch.full((state.size(0),), constant[depth], device=state.device, dtype=torch.long)
        elif original:
            selected = model.router.choose(state, horizon-depth)
        else:
            descriptor = router.describe(state)
            scores, _ = router.score_descriptor(descriptor)
            selected = scores.argmax(-1)
        choices.append(selected)
        state = model.apply_expert(state, selected)
    return state, torch.stack(choices, 1)


@torch.no_grad()
def suffix_targets(model, router, state, targets, depth, cancelled=None):
    """All insertions share exactly one prestate and immutable current policy."""
    losses = []
    for expert in range(model.config.num_modules):
        terminal, _ = rollout(model, router, state, start=depth, first=expert, cancelled=cancelled)
        losses.append(F.cross_entropy(model.read_endpoint(terminal).float(), targets, reduction='none'))
    return torch.stack(losses, -1)


@torch.no_grad()
def evaluate_trajectory(model, router, prefixes, targets, *, constant=None, original=False, cancelled=None):
    if prefixes.is_cuda: torch.cuda.synchronize(prefixes.device)
    started = time.perf_counter()
    losses, routes = [], []
    batch = expert_batch_limit(model, prefixes.size(1), 4)
    for offset in range(0, prefixes.size(0), batch):
        check_cancel(cancelled)
        terminal, chosen = rollout(model, router, model.embed(prefixes[offset:offset+batch]),
            constant=constant, original=original, cancelled=cancelled)
        losses.append(F.cross_entropy(model.read_endpoint(terminal).float(), targets[offset:offset+batch], reduction='none').cpu())
        routes.append(chosen.cpu())
    losses, routes = torch.cat(losses), torch.cat(routes)
    counts = [torch.bincount(routes[:,t], minlength=model.config.num_modules).tolist() for t in range(routes.size(1))]
    if prefixes.is_cuda: torch.cuda.synchronize(prefixes.device)
    elapsed = time.perf_counter()-started
    return dict(mean_loss=float(losses.mean()), per_prefix_loss=losses.tolist(), route_counts=counts,
                paths=routes.tolist(), endpoints=int(losses.numel()), elapsed_seconds=elapsed,
                endpoints_per_second=losses.numel()/max(elapsed,1e-12),
                context_tokens_per_second=losses.numel()*prefixes.size(1)/max(elapsed,1e-12))


def paired_delta(candidate, baseline):
    delta = torch.tensor(candidate['per_prefix_loss'], dtype=torch.float64) - torch.tensor(baseline['per_prefix_loss'], dtype=torch.float64)
    se = float(delta.std(unbiased=True)/delta.numel()**.5) if delta.numel()>1 else None
    return dict(mean=float(delta.mean()), standard_error=se,
                approximate_95_interval=[float(delta.mean())-1.96*se,float(delta.mean())+1.96*se] if se is not None else None)


def run_spectral_live(model, corpus, config, train, held, bank, output, emit, cancelled=None):
    """Only `router` is optimized; `model.router` remains the original baseline."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    c = config.routing
    device = torch.device(config.training.device)
    model.to(device).float().eval().requires_grad_(False)
    model.zero_grad(set_to_none=True)
    model.router = deepcopy(getattr(model, '_value_reference_router', model.router)).to(device).eval().requires_grad_(False)
    reference_hash = hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        reference_hash.update(name.encode()); reference_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    train = load_bank_payload(train.payload(),model,c.value_fit_prefixes,config.model.context_length,'cpu')
    held = load_bank_payload(held.payload(),model,c.value_fit_prefixes,config.model.context_length,'cpu')
    spectral = replace(model.config, router_type='spectral_geometric',
        spectral_neighbourhood_size=c.value_spectral_window, basins_per_expert=c.value_spectral_basins,
        geometry_temporal_delta_enabled=False, geometry_score_weight=1., resonance_score_weight=1.,
        spectral_route_weight=1., spectral_regret_weight=0., transformation_complementarity_weight=0.,
        refractory_enabled=False, loss_free_balance_enabled=False)
    router = SpectralGeometricRouter(spectral).cpu().train()
    train_desc, held_desc = cache(router,train), cache(router,held)
    joined = Descriptor(torch.cat([d.q for d in train_desc]),torch.cat([d.g for d in train_desc]),{})
    router.initialize(joined,torch.cat(train.losses))
    observer = deepcopy(router).to(device).eval().requires_grad_(False)
    means = [loss.mean(0) for loss in train.losses]
    constant = [int(mean.argmin()) for mean in means]
    exclude = set(map(tuple,torch.cat((train.prefixes,held.prefixes)).tolist()))
    emit('spectral_live_progress',phase='Preparing fresh trajectory evaluation panel',bank=bank)
    eval_x, eval_y, _ = unique_prefixes(corpus,'validation',c.value_spectral_eval_prefixes,
        config.model.context_length,torch.Generator().manual_seed(config.training.seed+8101),device,exclude,cancelled)
    eval_y = eval_y[:,-1]
    exclude.update(map(tuple,eval_x.cpu().tolist()))
    # Persist exact evidence and split identity, separate from the optimization bank.
    torch.save(dict(prefixes=eval_x.cpu(),targets=eval_y.cpu()),root/'spectral-live-evaluation.pt')
    panel_hash = hashlib.sha256(eval_x.cpu().contiguous().numpy().tobytes()).hexdigest()
    original = evaluate_trajectory(model,observer,eval_x,eval_y,original=True,cancelled=cancelled)
    fixed = evaluate_trajectory(model,observer,eval_x,eval_y,constant=constant,cancelled=cancelled)
    optimizer = torch.optim.AdamW(router.parameters(),lr=config.training.learning_rate,weight_decay=0)
    history = []
    started = time.perf_counter()
    online_items = 0
    total = c.value_fit_updates+c.value_spectral_live_updates

    def audit(step, phase, training_objective=None):
        check_cancel(cancelled)
        observer.load_state_dict(router.state_dict())
        observer.eval()
        with torch.no_grad():
            train_scores, held_scores = cached_scores(router,train_desc),cached_scores(router,held_desc)
            tr, he = metrics(train_scores,train.losses,means),metrics(held_scores,held.losses,means)
            kl = lambda scores,bank: float(torch.stack([routing_objective(s,y,router.c) for s,y in zip(scores,bank.losses)]).mean())
            live = evaluate_trajectory(model,observer,eval_x,eval_y,cancelled=cancelled)
        row = dict(step=step,phase=phase,total=total,warmup_updates=c.value_fit_updates,elapsed_seconds=time.perf_counter()-started,
            train_kl=kl(train_scores,train),held_kl=kl(held_scores,held),
            train_regret=tr['routing_regret'],held_regret=he['routing_regret'],
            uniform_regret=he['uniform_random_regret'],constant_regret=he['constant_per_step_regret'],
            trajectory_loss=live['mean_loss'],original_loss=original['mean_loss'],constant_loss=fixed['mean_loss'],
            delta_original=paired_delta(live,original),delta_constant=paired_delta(live,fixed),
            route_counts=live['route_counts'],training_objective=training_objective,online_endpoints=online_items)
        row.update(evaluation_endpoints_per_second=live['endpoints_per_second'],
                   evaluation_context_tokens_per_second=live['context_tokens_per_second'])
        history.append(row)
        emit('spectral_live_audit',**row)
        torch.save(dict(kind='spectral_live_router_v1',router=router.state_dict(),config=asdict(spectral),
            source_checkpoint=c.value_checkpoint_path,reference_weights_sha256=reference_hash.hexdigest(),
            bank_sha256=bank['bank_sha256'],step=step,optimizer=optimizer.state_dict()),root/'spectral-live-router-latest.pt')
        return live

    last_live = audit(0,'bank warmup')
    for step in range(1,c.value_fit_updates+1):
        check_cancel(cancelled)
        optimizer.zero_grad(set_to_none=True)
        scores = cached_scores(router,train_desc)
        loss = torch.stack([routing_objective(s,y,router.c) for s,y in zip(scores,train.losses)]).mean()
        loss = loss+router.c.basin_redundancy_weight*router.redundancy_loss()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite spectral warmup loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(router.parameters(),1.,error_if_nonfinite=True); optimizer.step()
        emit('spectral_live_progress',phase='bank warmup',step=step,total=total,training_objective=float(loss.detach()))
        if step % config.training.evaluation_interval == 0 or step==c.value_fit_updates:
            last_live = audit(step,'bank warmup',float(loss.detach()))

    # Fresh optimizer clock for policy-dependent online evidence; bank initialization
    # and normalization remain fixed. Never replay labels from an older policy.
    optimizer = torch.optim.AdamW(router.parameters(),lr=c.value_spectral_online_lr,weight_decay=0)
    generator = torch.Generator().manual_seed(config.training.seed+9101)
    probe_rng = torch.Generator().manual_seed(c.value_router_seed+9102)
    batch = expert_batch_limit(model,config.model.context_length,config.training.batch_size)
    for update in range(1,c.value_spectral_live_updates+1):
        check_cancel(cancelled)
        observer.load_state_dict(router.state_dict()); observer.eval()
        x,y,_ = unique_prefixes(corpus,'train',batch,config.model.context_length,generator,device,exclude,cancelled)
        depth = int(torch.randint(model.config.resolved_trajectory_steps,(),generator=probe_rng))
        request = int(torch.randint(batch,(),generator=probe_rng))
        with torch.no_grad():
            state = model.embed(x)
            # Reach the sampled position using this iteration's spectral policy.
            for t in range(depth):
                check_cancel(cancelled)
                d = observer.describe(state)
                selected = observer.score_descriptor(d)[0].argmax(-1)
                state = model.apply_expert(state,selected)
            state = state[request:request+1]
            labels = suffix_targets(model,observer,state,y[request:request+1,-1],depth,cancelled)
            d = observer.describe(state)
            descriptor = Descriptor(d.q.cpu(),d.g.cpu(),{})
        optimizer.zero_grad(set_to_none=True)
        scores = router.score_descriptor(descriptor)[0]
        loss = routing_objective(scores,labels.cpu(),router.c)+router.c.basin_redundancy_weight*router.redundancy_loss()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite spectral online loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(router.parameters(),1.,error_if_nonfinite=True); optimizer.step()
        online_items += batch
        step = c.value_fit_updates+update
        emit('spectral_live_progress',phase='sequential policy learning',step=step,total=total,
             training_objective=float(loss.detach()),online_endpoints=online_items)
        if update % config.training.evaluation_interval==0 or update==c.value_spectral_live_updates:
            last_live = audit(step,'sequential policy learning',float(loss.detach()))
    return dict(bank=bank,history=history,final=last_live,original=original,constant=fixed,
        constant_experts=constant,expert_names=list(model.expert_names),panel_sha256=panel_hash,
        evaluation_prefixes=c.value_spectral_eval_prefixes,config=asdict(spectral),
        source_checkpoint=c.value_checkpoint_path,reference_weights_sha256=reference_hash.hexdigest(),
        online_updates=c.value_spectral_live_updates,online_probes=c.value_spectral_live_updates,
        online_learning_rate=c.value_spectral_online_lr,warmup_learning_rate=config.training.learning_rate,
        online_endpoints=online_items,expert_updates=0,normalization='training bank only; frozen',
        objective='Bank KL warmup then fresh current-spectral-policy suffix KL; experts and shared layers frozen',
        caveat='Evaluation panel is repeatedly monitored, not used for gradients or best-checkpoint selection. Prefix equality is excluded; nearby packed-text contexts may still overlap. Paired intervals are descriptive, not proof of significance.')
