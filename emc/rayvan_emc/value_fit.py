"""Fixed-measurement router fitting diagnostic; no expert execution while fitting."""
from dataclasses import asdict, dataclass
from copy import deepcopy
from pathlib import Path
import hashlib
import math
import sys
import time

import torch

from .checkpoint import save_training_checkpoint
from .value_routing import StateBatch, center
from .value_training import frozen_snapshot, trainable_only


@dataclass(frozen=True)
class FitBank:
    states: tuple[StateBatch, ...]
    losses: tuple[torch.Tensor, ...]
    prefixes: torch.Tensor

    def payload(self):
        return dict(prefixes=self.prefixes.cpu(), states=[s.latent.cpu() for s in self.states],
                    targets=[s.targets.cpu() for s in self.states], losses=[y.cpu() for y in self.losses])


def unique_prefixes(corpus, split, count, length, generator, device, excluded=(), cancelled=None):
    seen = set(excluded)
    inputs, targets = [], []
    for _ in range(max(64, count * 8)):
        if cancelled and cancelled():
            from .training import TrainingCancelledError
            raise TrainingCancelledError('Cancelled while preparing router fit bank')
        x, y = corpus.sample_batch(split, min(count, 16), length, generator=generator, device=device)
        for i, row in enumerate(x.cpu().tolist()):
            key = tuple(row)
            if key not in seen:
                seen.add(key)
                inputs.append(x[i]); targets.append(y[i])
                if len(inputs) == count:
                    return torch.stack(inputs), torch.stack(targets), seen
    raise ValueError(f'Could not obtain {count} distinct {split} prefixes; reduce fit bank size or use a larger corpus')


@torch.no_grad()
def measure_bank(reference, inputs, targets, generator, cancelled=None):
    # Bank size is statistical coverage, not an expert execution batch size.
    # Bound both state collection and counterfactual suffix execution, including
    # branches where every item selects the same expert.
    batch_size = min(4, inputs.size(0))
    for module in reference.modules():
        if hasattr(module, 'max_transition_bytes'):
            per_item = inputs.size(1) * module.heads * module.head_dim**2 * 4
            batch_size = min(batch_size, max(1, module.max_transition_bytes // per_item))
    pieces = []
    for start in range(0, inputs.size(0), batch_size):
        if cancelled and cancelled():
            from .training import TrainingCancelledError
            raise TrainingCancelledError('Cancelled while measuring router fit bank')
        states = reference.collect_states(inputs[start:start+batch_size], targets[start:start+batch_size, -1],
                                          generator=generator, exploration=1.)
        labels = []
        for state in states:
            if cancelled and cancelled():
                from .training import TrainingCancelledError
                raise TrainingCancelledError('Cancelled while measuring router fit bank')
            labels.append(reference.counterfactual_losses(state, target='suffix').detach())
        pieces.append((states, labels))
    states = tuple(StateBatch(torch.cat([s[d].latent for s, _ in pieces]),
                              torch.cat([s[d].targets for s, _ in pieces]), d)
                   for d in range(reference.config.resolved_trajectory_steps))
    labels = tuple(torch.cat([y[d] for _, y in pieces]) for d in range(len(states)))
    return FitBank(states, labels, inputs.detach())


def bank_loss(router, bank):
    horizon = len(bank.states)
    return torch.stack([router.calibration_loss(s.latent, horizon - s.depth, y)
                        for s, y in zip(bank.states, bank.losses)]).mean()


@torch.no_grad()
def bank_metrics(router, bank, train_means):
    rows = []
    for state, labels, mean in zip(bank.states, bank.losses, train_means):
        predicted = router(state.latent, len(bank.states) - state.depth).float()
        truth = center(labels.float())
        mse = float((predicted - truth).square().mean())
        zero_mse = float(truth.square().mean())
        mean_mse = float((mean - truth).square().mean())
        chosen = predicted.argmin(-1)
        oracle = labels.min(-1).values
        picked = labels.gather(1, chosen[:, None]).squeeze(1)
        constant = int(mean.argmin())
        rows.append(dict(depth=state.depth, probe_count=labels.size(0), mse=mse, zero_mse=zero_mse,
                         mean_predictor_mse=mean_mse, regret=float((picked-oracle).mean()),
                         uniform_regret=float((labels.mean(-1)-oracle).mean()),
                         constant_regret=float((labels[:, constant]-oracle).mean()), constant_expert=constant,
                         route_counts=torch.bincount(chosen, minlength=labels.size(1)).tolist()))
    average = lambda field: sum(r[field] for r in rows)/len(rows)
    mse, zero = average('mse'), average('zero_mse')
    return dict(by_depth=rows, mse=mse, zero_mse=zero, mean_predictor_mse=average('mean_predictor_mse'),
                normalized_mse=mse/zero if zero > 1e-20 else None,
                mean_regret=average('regret'), uniform_regret=average('uniform_regret'),
                constant_regret=average('constant_regret'), gap_rmse=math.sqrt(2*mse),
                probe_count=sum(r['probe_count'] for r in rows), target='fixed_bank_suffix')


def source_identity(reference, tokenizer, sequence_length):
    digest = hashlib.sha256()
    for name, tensor in sorted(reference.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return dict(weights_sha256=digest.hexdigest(), tokenizer=tokenizer.to_config(),
                sequence_length=sequence_length, horizon=reference.config.resolved_trajectory_steps,
                expert_names=list(reference.expert_names))


def load_bank_payload(data, model, count, length, device):
    states, losses, targets = data['states'], data['losses'], data['targets']
    horizon = model.config.resolved_trajectory_steps
    if not len(states) == len(losses) == len(targets) == horizon:
        raise ValueError('Saved bank trajectory length differs')
    prefixes = data['prefixes']
    if tuple(prefixes.shape) != (count, length):
        raise ValueError('Saved bank size/context differs; match Fixed prefixes per split and context length')
    for x, y, target in zip(states, losses, targets):
        if (tuple(x.shape) != (count, length, model.config.latent_dim)
                or tuple(y.shape) != (count, model.config.num_modules) or tuple(target.shape) != (count,)):
            raise ValueError('Saved bank tensor shapes differ from the model')
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('Saved bank contains non-finite values')
    return FitBank(tuple(StateBatch(x.to(device), y.to(device), d) for d, (x, y) in enumerate(zip(states, targets))),
                   tuple(y.to(device) for y in losses), prefixes.to(device))


def train_fixed_bank(model, corpus, config, *, print_progress=True, evaluation_callback=None,
                     progress_callback=None, progress_callback_interval=1, cancellation_callback=None, prepare_only=False):
    from .training import TrainingMetrics, TrainingResult, TrainingCancelledError
    c = model.config
    if c.value_expert_training != 'frozen' or not c.value_fixed_reference or c.value_target != 'suffix':
        raise ValueError('Fixed-bank fitting requires frozen experts, fixed continuation and suffix targets')
    if config.precision != 'fp32':
        raise ValueError('Use FP32 for the fixed-bank fitting diagnostic')
    if config.resume_from:
        raise ValueError('Fixed-bank tests start fresh from their source checkpoint; optimizer resume is not supported')
    if progress_callback_interval <= 0:
        raise ValueError('progress_callback_interval must be positive')
    device = torch.device(config.device)
    model.to(device).float().eval()
    model.zero_grad(set_to_none=True)
    reference = frozen_snapshot(model)
    reference.router = deepcopy(getattr(model, '_value_reference_router', model.router))
    reference.router.to(device).eval().requires_grad_(False)
    setup_start = time.perf_counter()
    identity = source_identity(reference, corpus.tokenizer, config.sequence_length)
    reused = bool(c.value_fit_bank_path)
    if reused:
        print('Router fit: loading saved measurement banks; no expert measurements.', file=sys.stderr, flush=True)
        payload = torch.load(Path(c.value_fit_bank_path).expanduser(), map_location='cpu', weights_only=True)
        if 'source_identity' not in payload:
            raise ValueError('This bank predates source validation. Run one new baseline with Saved bank path empty, then reuse its bank.')
        if payload['source_identity'] != identity or payload['data_seed'] != config.seed:
            raise ValueError('Saved bank source checkpoint, tokenizer, context, trajectory or data seed differs')
        train = load_bank_payload(payload['train'], model, c.value_fit_prefixes, config.sequence_length, device)
        held = load_bank_payload(payload['held_out'], model, c.value_fit_prefixes, config.sequence_length, device)
        train_keys = set(map(tuple, train.prefixes.cpu().tolist()))
        held_keys = set(map(tuple, held.prefixes.cpu().tolist()))
        if len(train_keys) != c.value_fit_prefixes or len(held_keys) != c.value_fit_prefixes or train_keys & held_keys:
            raise ValueError('Saved bank prefixes must be unique and disjoint')
    else:
        print('Router fit: measuring fixed train/held-out banks; experts run only during this setup.', file=sys.stderr, flush=True)
        x, y, keys = unique_prefixes(corpus, 'train', c.value_fit_prefixes, config.sequence_length,
                                   torch.Generator().manual_seed(config.seed+3101), device, cancelled=cancellation_callback)
        vx, vy, _ = unique_prefixes(corpus, 'validation', c.value_fit_prefixes, config.sequence_length,
                                   torch.Generator().manual_seed(config.seed+3102), device, keys, cancellation_callback)
        train = measure_bank(reference, x, y, torch.Generator().manual_seed(config.seed+3103), cancellation_callback)
        held = measure_bank(reference, vx, vy, torch.Generator().manual_seed(config.seed+3104), cancellation_callback)
    means = [center(y).mean(0) for y in train.losses]
    bank_file = None
    if config.checkpoint_directory:
        bank_file = Path(config.checkpoint_directory)/'router-fit-bank.pt'
        bank_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(train=train.payload(), held_out=held.payload(), reference_router=reference.router.state_dict(),
                        data_seed=config.seed, source_identity=identity), bank_file)
    fingerprint = hashlib.sha256()
    for bank in (train, held):
        for tensor in (bank.prefixes, *(s.latent for s in bank.states), *(s.targets for s in bank.states), *bank.losses):
            fingerprint.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    metadata = dict(bank_sha256=fingerprint.hexdigest(), prefixes_per_split=c.value_fit_prefixes,
                    states_per_split=c.value_fit_prefixes*c.resolved_trajectory_steps,
                    unique_disjoint_prefixes=True, full_batch=True, precision='fp32',
                    setup_seconds=time.perf_counter()-setup_start,
                    bank_file=str(bank_file) if bank_file else None,
                    bank_reused=reused, need_dimension=c.resolved_routing_geometry_dim, head_type=c.value_head_type,
                    head_hidden_dim=c.value_head_hidden_dim, geometry_slots=c.value_geometry_slots,
                    geometry_prototypes=c.value_geometry_prototypes, geometry_regret_weight=c.value_geometry_regret_weight,
                    bank_expert_items=0 if reused else 2*c.value_fit_prefixes*(c.resolved_trajectory_steps-1 +
                        c.num_modules*c.resolved_trajectory_steps*(c.resolved_trajectory_steps+1)//2),
                    fitting_expert_items=0)
    del reference
    if prepare_only:
        return train, held, metadata
    parameters = list(model.router.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    initial_train = bank_metrics(model.router, train, means)
    initial_held = bank_metrics(model.router, held, means)
    history = []
    total = c.value_fit_updates
    latest_path = best_path = None
    if config.checkpoint_directory:
        root = Path(config.checkpoint_directory)
        latest_path = root/f'{config.checkpoint_prefix}-latest.pt'
        best_path = root/f'{config.checkpoint_prefix}-best.pt' if config.retain_best_checkpoint else None
    started = time.perf_counter()
    best = math.inf
    last = {}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with trainable_only(model, parameters):
        for step in range(total+1):
            if cancellation_callback and cancellation_callback():
                raise TrainingCancelledError('Fixed-bank fitting cancelled between router updates')
            step_started = time.perf_counter()
            gradient = encoder_gradient = head_gradient = parameter_change = 0.
            if step:
                optimizer.zero_grad(set_to_none=True)
                loss = bank_loss(model.router, train)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite fixed-bank calibration loss')
                loss.backward()
                norm = lambda ps: math.sqrt(sum(float(p.grad.detach().square().sum()) for p in ps if p.grad is not None))
                head_gradient = norm(model.router.value.parameters())
                encoder_gradient = norm(p for name,p in model.router.named_parameters() if not name.startswith('value.'))
                gradient = float(torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm, error_if_nonfinite=True))
                before = [p.detach().clone() for p in parameters]
                optimizer.step()
                parameter_change = math.sqrt(sum(float((p.detach()-old).square().sum()) for p,old in zip(parameters,before)))
            due = step == 0 or step == total or step % config.evaluation_interval == 0
            if not due:
                continue
            tm, vm = bank_metrics(model.router, train, means), bank_metrics(model.router, held, means)
            elapsed = time.perf_counter()-started
            rate = step/max(elapsed,1e-9)
            memory = torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
            peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
            vm['snapshot_version'] = step
            last = dict(router_phase='fixed_bank_fit', fixed_reference=True,
                        label_snapshot_version=step-1, total_training_probes=metadata['states_per_split'],
                        expert_update_batches=[0]*c.num_modules, expert_training_items=[0]*c.num_modules,
                        held_out=vm, initial_held_out=initial_held,
                        fixed_bank=dict(**metadata, train=tm, held_out=vm,
                                        initial_train=initial_train, initial_held_out=initial_held,
                                        head_gradient_norm=head_gradient, encoder_gradient_norm=encoder_gradient,
                                        router_parameter_change_norm=parameter_change))
            metric = TrainingMetrics(step, step, tm['mse'], vm['mse'], None, None, None,
                                     0.,0.,rate,elapsed,memory,peak)
            history.append(metric)
            values = dict(step=step,tokens_processed=step,target_tokens=total,objective='router_fixed_bank',
                          throughput_unit='updates/s',tokens_per_second=rate,context_tokens_per_second=None,
                          training_loss=tm['mse'],validation_loss=vm['mse'],gradient_norm=gradient,
                          learning_rate=config.learning_rate,elapsed_seconds=elapsed,
                          step_time_seconds=time.perf_counter()-step_started,gpu_memory_used_bytes=memory,
                          gpu_peak_memory_bytes=peak,evaluation_completed=True,
                          routing=dict(expert_names=list(model.expert_names),value_routing=last))
            if progress_callback:
                progress_callback(step,model,values)
            if evaluation_callback:
                evaluation_callback(step,model,metric)
            improved = vm['mse'] < best
            best = min(best,vm['mse'])
            if latest_path:
                kwargs = dict(model=model,optimizer=optimizer,tokenizer=corpus.tokenizer,step=step,tokens_processed=step,
                              validation_loss=vm['mse'],best_validation_loss=best,
                              training_config={**asdict(config),'objective':'router_fixed_bank'},
                              training_diagnostics={'value_routing':last})
                save_training_checkpoint(latest_path,**kwargs)
                if best_path and improved:
                    save_training_checkpoint(best_path,**kwargs)
            if print_progress:
                print(f'fit update {step}/{total} | train NMSE {tm["normalized_mse"]} | held-out NMSE {vm["normalized_mse"]}',flush=True)
    model.zero_grad(set_to_none=True)
    m = history[-1]
    return TrainingResult(tuple(history),total,total,m.training_loss,m.validation_loss,None,best,None,None,
                          0.,0.,m.tokens_per_second,m.elapsed_seconds,m.gpu_memory_used_bytes,m.gpu_peak_memory_bytes,
                          None,str(latest_path) if latest_path else None,str(best_path) if best_path else None,
                          {'value_routing':last,'objective':'router_fixed_bank'},(),None)
