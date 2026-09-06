from copy import deepcopy
from dataclasses import replace
import json
from unittest.mock import patch

import pytest
import torch

from rayvan_emc.value_fit import FitBank, bank_loss, bank_metrics, measure_bank
from rayvan_emc.value_routing import CounterfactualValueEMC, StateBatch, center
from rayvan_emc.research_runner import _build_model, run_experiment
from rayvan_emc.checkpoint import load_model_checkpoint
from test_value_experiment import experiment


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_full_bank_loss_and_metrics_have_exact_targets():
    model = _build_model(experiment(),16)
    states=tuple(StateBatch(torch.randn(4,3,8),torch.arange(4),depth) for depth in range(3))
    labels=tuple(torch.randn(4,2) for _ in states)
    bank=FitBank(states,labels,torch.ones(4,3,dtype=torch.long))
    means=[center(y).mean(0) for y in labels]
    expected=sum(float(center(y).square().mean()) for y in labels)/3
    assert float(bank_loss(model.router,bank).detach()) == pytest.approx(expected)
    m=bank_metrics(model.router,bank,means)
    assert m['normalized_mse'] == pytest.approx(1)
    assert m['gap_rmse']**2 == pytest.approx(2*expected)
    assert all(y.grad is None for y in labels)


def test_fixed_bank_fits_synthetic_signal_without_expert_calls():
    model=_build_model(experiment(),16)
    rng=torch.Generator().manual_seed(32)
    x=torch.randn(24,3,8,generator=rng)
    labels=torch.stack((x[:,-1,0],-x[:,-1,0]),-1)
    bank=FitBank(tuple(StateBatch(x,torch.zeros(24,dtype=torch.long),d) for d in range(3)),
                 tuple(labels for _ in range(3)),torch.zeros(24,3,dtype=torch.long))
    before=deepcopy(model.state_dict())
    initial=float(bank_loss(model.router,bank).detach())
    opt=torch.optim.AdamW(model.router.parameters(),lr=.01,weight_decay=0)
    with patch.object(model,'apply_expert',side_effect=AssertionError('Experts must not run during fitting')):
        for _ in range(200):
            opt.zero_grad(set_to_none=True);bank_loss(model.router,bank).backward();opt.step()
    assert float(bank_loss(model.router,bank).detach()) < initial*.05
    for name,p in model.state_dict().items():
        if not name.startswith('router.'):
            torch.testing.assert_close(p,before[name],rtol=0,atol=0)


def test_fixed_bank_console_run_freezes_model_and_saves_disjoint_banks(tmp_path):
    c=experiment()
    source=run_experiment(c,runs_directory=tmp_path,run_id='source')
    source_path=source['training_result']['best_checkpoint']
    c=replace(c,routing=replace(c.routing,value_expert_training='frozen',value_fixed_reference=True,
              value_checkpoint_path=source_path,value_fit_enabled=True,value_fit_prefixes=2,value_fit_updates=4))
    with patch.object(CounterfactualValueEMC, 'apply_expert', autospec=True, side_effect=CounterfactualValueEMC.apply_expert) as calls:
        result=run_experiment(c,runs_directory=tmp_path,run_id='fit')
        assert calls.call_count == 2*(3-1+2*3*4//2)  # Setup banks only, independent of fit updates.
    assert result['status']=='completed'
    assert result['model']['objective']=='router_fixed_bank'
    assert result['headline']['throughput_unit']=='updates/s'
    assert result['headline']['perplexity'] is None
    assert result['headline']['context_tokens_per_second'] is None
    fit=result['value_routing']['fixed_bank']
    assert fit['states_per_split']==6 and fit['fitting_expert_items']==0
    assert fit['initial_train']['normalized_mse']==pytest.approx(1)
    assert fit['router_parameter_change_norm'] > 0
    data=torch.load(fit['bank_file'],weights_only=True)
    assert not set(map(tuple,data['train']['prefixes'].tolist())) & set(map(tuple,data['held_out']['prefixes'].tolist()))
    original=load_model_checkpoint(source_path).model.state_dict()
    final=load_model_checkpoint(result['training_result']['latest_checkpoint']).model.state_dict()
    assert all(torch.equal(p,final[n]) for n,p in original.items() if not n.startswith('router.'))
    events=[json.loads(l) for l in (tmp_path/'fit'/'metrics.jsonl').read_text().splitlines()]
    assert not any(e['type']=='projection_update' for e in events)
    assert len([e for e in events if e['type']=='validation'])==5
    # Different router seed must measure exactly the same bank, not different answers.
    again=run_experiment(replace(c,routing=replace(c.routing,value_router_seed=7)),runs_directory=tmp_path,run_id='fit-other-seed')
    assert again['value_routing']['fixed_bank']['bank_sha256']==fit['bank_sha256']


def test_fit_rejects_moving_targets_and_reduced_precision():
    c=experiment()
    with pytest.raises(ValueError,match='frozen'):
        replace(c,routing=replace(c.routing,value_fit_enabled=True))
    with pytest.raises(ValueError,match='FP32'):
        replace(c,routing=replace(c.routing,value_fit_enabled=True,value_expert_training='frozen'),
                training=replace(c.training,precision='auto'))


def test_measure_bank_batches_collection_and_suffixes_under_delta_limit():
    model = _build_model(experiment(), 16).eval()
    inputs = torch.randint(16, (7, 8))
    targets = torch.randint(16, (7, 8))
    guarded = [m for m in model.modules() if hasattr(m, 'max_transition_bytes')]
    assert guarded
    for module in guarded:
        module.max_transition_bytes = 2 * 8 * module.heads * module.head_dim**2 * 4
    # Reproduce the failure when measuring the entire bank at once.
    state = StateBatch(model.embed(inputs).detach(), targets[:, -1], 0)
    with pytest.raises(RuntimeError, match='transition tensor exceeds'):
        model.counterfactual_losses(state)
    with patch.object(model, 'apply_expert', wraps=model.apply_expert) as calls:
        bank = measure_bank(model, inputs, targets, torch.Generator().manual_seed(42))
    assert max(call.args[0].size(0) for call in calls.call_args_list) == 2
    assert any(call.args[0].size(0) == 1 for call in calls.call_args_list)
    assert torch.equal(bank.prefixes, inputs)
    for state, labels in zip(bank.states, bank.losses):
        assert state.latent.size(0) == 7 and labels.shape == (7, 2)
        assert torch.equal(state.targets, targets[:, -1])
    # Concatenation preserves order and the original suffix-loss mathematics.
    for module in guarded:
        module.max_transition_bytes *= 100
    for state, labels in zip(bank.states, bank.losses):
        torch.testing.assert_close(labels, model.counterfactual_losses(state), rtol=1e-5, atol=1e-6)
