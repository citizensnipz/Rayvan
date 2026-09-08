from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from rayvan_emc.spectral_live import rollout, suffix_targets
from rayvan_emc.spectral_geometry import Descriptor
from rayvan_emc.training import TrainingCancelledError


class ToyModel:
    config = SimpleNamespace(resolved_trajectory_steps=3, num_modules=2)

    def apply_expert(self, state, selected):
        return state + torch.where(selected==0,1.,10.)[:,None,None]

    def read_endpoint(self, state):
        value = state[:,-1,0]
        return torch.stack((value,-value),-1)


class ToyRouter:
    def __init__(self): self.seen=[]
    def describe(self,state):
        self.seen.append(state.clone())
        return Descriptor(state[:,-1],state[:,-1],{})
    def score_descriptor(self,d):
        prefer_second = d.g[:,0] > .5
        return torch.stack((~prefer_second,prefer_second),-1).float(),{}


def test_live_reroutes_transformed_state_at_every_step():
    model,router=ToyModel(),ToyRouter()
    state,paths=rollout(model,router,torch.zeros(1,1,1))
    assert paths.tolist()==[[0,1,1]]
    assert [float(h.item()) for h in router.seen]==[0.,1.,11.]
    assert state.item()==21.


def test_counterfactual_suffix_uses_each_branch_transformed_state():
    model,router=ToyModel(),ToyRouter()
    initial=torch.zeros(1,1,1)
    labels=suffix_targets(model,router,initial,torch.tensor([1]),0)
    expected=F.cross_entropy(torch.tensor([[21.,-21.],[30.,-30.]]),torch.ones(2,dtype=torch.long),reduction='none')
    torch.testing.assert_close(labels[0],expected)
    assert initial.item()==0.
    assert not labels.requires_grad


def test_live_cancel_at_expert_boundary():
    with pytest.raises(TrainingCancelledError):
        rollout(ToyModel(),ToyRouter(),torch.zeros(1,1,1),cancelled=lambda:True)


@pytest.mark.parametrize('snapshot',[False,True])
def test_console_live_roundtrip_curves_frozen_weights_and_fresh_panel(tmp_path,monkeypatch,snapshot):
    from test_value_experiment import experiment
    from rayvan_emc.research_runner import run_experiment
    from rayvan_emc.checkpoint import load_model_checkpoint
    import rayvan_emc.spectral_live as live
    import json
    torch.set_num_threads(1)
    updates=3 if snapshot else 2
    probes=4 if snapshot else 2
    c=experiment(experts={'gpt':2,'ssm':2,'recurrent':2,'delta':2})
    source=run_experiment(c,runs_directory=tmp_path,run_id='source')
    path=source['training_result']['latest_checkpoint']
    original=deepcopy(load_model_checkpoint(path).model.state_dict())
    c=replace(c,routing=replace(c.routing,value_checkpoint_path=path,value_expert_training='frozen',
        value_fit_enabled=True,value_spectral_live=True,value_fit_prefixes=3,value_fit_updates=2,
        value_spectral_live_updates=updates,value_spectral_eval_prefixes=3,
        value_spectral_snapshot=snapshot,value_spectral_snapshot_states=2,value_spectral_snapshot_updates=2),
        training=replace(c.training,weight_decay=0,learning_rate=.01))
    assert type(c).from_dict(c.to_dict()).routing.value_spectral_live
    actual=live.run_spectral_live
    actual_targets=live.suffix_targets
    snapshots=[]
    def checked_targets(model,router,*args,**kw):
        assert not any(p.requires_grad for p in router.parameters())
        snapshots.append({k:v.clone() for k,v in router.state_dict().items()})
        return actual_targets(model,router,*args,**kw)
    monkeypatch.setattr(live,'suffix_targets',checked_targets)
    def checked(model,*args,**kw):
        before={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('router.')}
        report=actual(model,*args,**kw)
        for k,v in before.items(): torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
        assert all(p.grad is None for p in model.parameters())
        return report
    monkeypatch.setattr(live,'run_spectral_live',checked)
    summary=run_experiment(c,runs_directory=tmp_path,run_id='live')
    report=summary['spectral_live']
    assert len(report['history'])==3+updates
    assert report['history'][2]['phase']=='bank warmup'
    assert report['history'][-1]['phase']=='sequential policy learning'
    assert report['expert_updates']==0 and report['online_probes']==probes
    assert report['snapshot_batch_enabled']==snapshot
    assert report['snapshot_rounds']==2
    if snapshot:
        assert all(torch.equal(v,snapshots[1][k]) for k,v in snapshots[0].items())
        assert all(torch.equal(v,snapshots[3][k]) for k,v in snapshots[2].items())
        assert any(not torch.equal(v,snapshots[2][k]) for k,v in snapshots[0].items())
    assert report['config']['spectral_neighbourhood_size']==8
    assert report['config']['basins_per_expert']==1
    assert report['config']['spectral_regret_weight']==0
    assert report['history'][-1]['delta_original']['mean']==pytest.approx(
        report['final']['mean_loss']-report['original']['mean_loss'],abs=1e-6)
    panel=torch.load(tmp_path/'live/checkpoints/spectral-live-evaluation.pt',weights_only=True)
    bank=torch.load(tmp_path/'live/checkpoints/router-fit-bank.pt',weights_only=True)
    reserved=set(map(tuple,panel['prefixes'].tolist()))
    assert not reserved & set(map(tuple,bank['train']['prefixes'].tolist()))
    assert not reserved & set(map(tuple,bank['held_out']['prefixes'].tolist()))
    saved=torch.load(tmp_path/'live/checkpoints/spectral-live-router-latest.pt',weights_only=True)
    assert saved['step']==2+updates and saved['bank_sha256']==report['bank']['bank_sha256']
    after=load_model_checkpoint(path).model.state_dict()
    assert all(torch.equal(v,after[k]) for k,v in original.items())
    events=[json.loads(line) for line in (tmp_path/'live/metrics.jsonl').read_text().splitlines()]
    assert len([e for e in events if e['type']=='spectral_live_audit'])==3+updates
    assert not any(e['type']=='projection_update' for e in events)
    online=[e for e in events if e['type']=='spectral_live_progress' and e.get('phase') in ('collecting snapshot evidence','sequential policy learning')]
    assert [e['phase'] for e in online]==(
        ['collecting snapshot evidence']*2+['sequential policy learning']*2+['collecting snapshot evidence']*2+['sequential policy learning'] if snapshot else
        ['collecting snapshot evidence','sequential policy learning']*2)
    import rayvan_emc.value_fit as vf
    def no_remeasurement(*args,**kwargs):
        raise AssertionError('Saved bank must not be remeasured')
    monkeypatch.setattr(vf,'measure_bank',no_remeasurement)
    reused = replace(c,routing=replace(c.routing,value_fit_bank_path=report['bank']['bank_file']))
    repeat = run_experiment(reused,runs_directory=tmp_path,run_id='reuse')['spectral_live']
    assert repeat['bank']['bank_reused']
    assert repeat['panel_sha256']==report['panel_sha256']
    assert repeat['final']['paths']==report['final']['paths']
    assert repeat['final']['per_prefix_loss']==report['final']['per_prefix_loss']
    assert repeat['online_probes']==probes


def test_live_config_rejects_combined_modes():
    from test_value_experiment import experiment
    from rayvan_emc.research_config import RoutingConfig
    with pytest.raises(ValueError,match='Live spectral test'):
        experiment(routing=RoutingConfig(value_spectral_live=True,value_spectral_comparison=True))
