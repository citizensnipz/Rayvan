"""Controlled development: math, assignment, matching, frozen state and real pipeline."""
import copy
import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch
from rayvan_emc import token_lab_emergence as e
from rayvan_emc import token_lab_emergence_analysis as a
from rayvan_emc.token_lab import LabConfig,Dataset,ProbeModel
from rayvan_emc.token_lab_calibration import digest
from rayvan_emc.token_lab_routing import Policy
from test_token_lab_routing import make_source


def small(root):
    return e.DEFAULTS|dict(source_sweep=str(root),seeds=[42],expert_count=2,asymmetry_strengths=[0,.01],feedback_strengths=[0,.5],train_steps=10,batch_size=2,probe_interval=10,probe_samples=20,validation_samples=20,evaluation_samples=20,diagnostic_samples=200,reference_size=20,fit_epochs=2,checkpoints=[0,1])


def test_condition_grid_and_application_estimate():
    c=copy.deepcopy(e.DEFAULTS)
    assert len(e.condition_plan(c))==14
    assert e.cost(c)['jobs']==42
    assert e.cost(c)['training_applications']==168000
    assert len(e.condition_plan(c|{'design':'factorial'}))==18
    c=small('unused');assert [v['condition'] for v in e.condition_plan(c)]==list('ABCDE')
    assert e.cost(c)['training_applications']==200


def test_zero_identity_and_exact_relative_perturbation():
    torch.manual_seed(42);torch.set_num_threads(1)
    c=LabConfig(dataset='capability_10',latent_dim=8,hidden_dim=16,heads=2,families=['gpt']*4)
    m=ProbeModel(c,Dataset(c).tokenizer.vocab_size)
    for x in m.experts:x.load_state_dict(m.experts[0].state_dict())
    before=digest(m.state_dict());e.perturb(m,0,42);assert before==digest(m.state_dict())
    other=copy.deepcopy(m);result=e.perturb(m,.01,42);e.perturb(other,.01,42)
    assert digest(m.state_dict())==digest(other.state_dict())
    assert len({digest(v.state_dict()) for v in m.experts})==4
    assert all(v['relative_delta_norm']==pytest.approx(.01,rel=1e-5) for v in result['experts'])
    with pytest.raises(ValueError,match='identical'):e.perturb(m,.01,42)


def test_exposure_formula_and_random_control_counts():
    p=e.PopulationRouter(4,fixed=0);rows=[{}]*10000
    tickets,counts=e.assignment_tickets(rows,p,.6,.2,random.Random(42),4)
    expected=np.array([.1+.6*(.8+.2/4)]+[.1+.6*.2/4]*3)
    assert np.bincount(tickets)/len(tickets)==pytest.approx(expected,abs=.01)
    shuffled=e.randomize_tickets(tickets,random.Random(23))
    assert np.bincount(tickets).tolist()==np.bincount(shuffled).tolist()
    assert not np.array_equal(tickets,shuffled)
    assert counts['explored']/counts['redirected']==pytest.approx(.2,abs=.01)
    zero,_=e.assignment_tickets(rows,p,0,.2,random.Random(42),4)
    assert np.bincount(zero).tolist()==[10000]*4
    assert e.assignment_tickets(rows,p,.6,.2,random.Random(42),4)[0].tolist()==tickets.tolist()


@pytest.mark.parametrize('family',['gpt','ssm','recurrent','delta'])
def test_general_loading_and_shared_freeze(tmp_path,family):
    root,source=make_source(tmp_path,family)
    c=small(root)|dict(family=family,expert_count=4)
    cfg,ds,m,excluded,proof=e.initialize(c,42)
    assert e.base_path(c,42)==source/'common-base.pt'
    assert len(set(proof['clone_hashes']))==1
    shared=digest(e.shared_state(m));before=[digest(x.state_dict()) for x in m.experts]
    x=torch.ones(1,48,dtype=torch.long)
    with torch.no_grad():h=m.embed(x)
    opt=[torch.optim.AdamW(x.parameters(),lr=.001) for x in m.experts]
    counts,updates=e.train_batch(m,[dict(h=h,y=1)],np.array([0]*4),opt,42,lambda:None)
    assert counts.tolist()==[4,0,0,0];assert updates.tolist()==[1,0,0,0]
    assert shared==digest(e.shared_state(m))
    assert digest(m.experts[0].state_dict())!=before[0]
    assert [digest(x.state_dict()) for x in m.experts[1:]]==before[1:]


def test_pairwise_router_math_roundtrip_and_neutral_pairs(tmp_path):
    from torch import nn
    net=nn.Sequential(nn.Linear(1,32),nn.GELU(),nn.Linear(32,1))
    with torch.no_grad():
        for p in net.parameters():p.zero_()
        net[-1].bias.fill_(2.)
    p=Policy([1],['hidden_norm'],np.array([0.]),np.array([1.]),net)
    router=e.PopulationRouter(2,{(0,1):p})
    scores,_=router.scores({'hidden_norm':3})
    assert scores[0]==pytest.approx(float(torch.sigmoid(torch.tensor(2.))))
    assert router.choose({'hidden_norm':3})==0
    router.save(tmp_path/'router');restored=e.PopulationRouter.load(tmp_path/'router')
    assert restored.scores({'hidden_norm':3})[0].tolist()==scores.tolist()
    with torch.no_grad():net[-1].bias.fill_(1e-8)
    assert router.choose({'hidden_norm':3})==0
    neutral=e.PopulationRouter(2,{(0,1):Policy([0])},neutral=[(0,1)])
    assert neutral.scores({})[0].tolist()==[.5,.5]
    assert neutral.choose({})==1


def test_signature_removes_common_difficulty_and_global_strength():
    base=np.arange(100)/10+1
    constant=base[:,None]-np.array([.1,.4,.2])
    s,_,_=a.performance_signature(constant,base,.001)
    assert s['interaction_rms']<1e-12
    assert s['meaningful_winning_experts']==1
    assert s['oracle_gain_over_best_constant']==pytest.approx(0)
    alternating=constant[:,:2].copy();alternating[:50,0]-=1
    s,_,_=a.performance_signature(alternating,base,.001)
    assert s['interaction_rms']>.1;assert s['meaningful_winning_experts']==2


def test_actual_small_study_and_disjoint_reproducible_panels(tmp_path):
    root,_=make_source(tmp_path)
    c=e.validate(small(root))
    result=e.run_study(c,tmp_path/'runs','smoke')
    assert result['status']=='completed',[(v['id'],v.get('error')) for v in result['entries']]
    out=tmp_path/'runs'/'smoke'
    assert len(result['entries'])==5
    pools=torch.load(out/'seed-42'/'matched-data.pt',weights_only=True)
    groups=[v['meta']['analysis_group'] for pool in pools.values() for v in pool]
    hashes=[v['meta']['prefix_sha256'] for pool in pools.values() for v in pool]
    assert len(groups)==len(set(groups));assert len(hashes)==len(set(hashes))
    excluded=e.initialize(c,42)[3];assert not excluded.intersection(hashes)
    for key in ['train','validation','evaluation','diagnostic']:
        tasks=[v['meta']['task_id'] for v in pools[key]]
        assert len(set(tasks.count(t) for t in e.CAPABILITIES))==1
    forbidden={'baseline_loss','target_probability','target_margin','task_id','improvement','relative_advantage','delta_norm','success_familiarity'}
    assert all(not forbidden.intersection(v['features']) for k,pool in pools.items() if k!='reference' for v in pool)
    by={v['condition']:v for v in result['entries']}
    assert by['A']['final']['exposure']==[20,20]
    assert by['A']['final']['specialization']['interaction_rms']<1e-8
    assert by['D']['final']['exposure']==by['E']['final']['exposure']
    assert by['D']['final']['optimizer_updates']==by['E']['final']['optimizer_updates']
    assert all(v['cost']['training_applications']==40 and v['shared_unchanged'] for v in result['entries'])
    assert all(v['hypothesis_1']==v['hypothesis_2']==v['hypothesis_3']=='INCONCLUSIVE' for v in result['summary']['conditions'].values())
    cfg,ds,m,ex,_=e.initialize(c,42);again,_=e.make_data(c,42,cfg,ds,m,ex,lambda:None)
    assert [v['meta'] for p in pools.values() for v in p]==[v['meta'] for p in again.values() for v in p]
    assert (out/'report.md').exists();assert json.loads((out/'analysis.json').read_text())['emergence']['status']=='completed'


def test_missing_seed_and_constant_selection_cannot_be_success():
    def entry(seed,selected=2):
        spec=dict(interaction_rms=.2,meaningful_winning_experts=2)
        util=dict(policy_loss=1.,uniform=1.2,gain_over_fixed=.1,gain_over_each_fixed=[{'ci':[.01,.2]}]*2,meaningful_selected_experts=selected)
        final=dict(specialization=spec,utility=util,stability={'mean_advantage_correlation':.9})
        return dict(id='C',seed=seed,status='completed',condition='C',asymmetry=0,feedback=.5,final=final,checkpoints=[final])
    c=e.DEFAULTS|dict(seeds=[42,43,44])
    assert a.summarize([entry(s) for s in c['seeds']],c)['conditions']['C']['hypothesis_2']=='WEAKLY SUPPORTED'
    assert a.summarize([entry(s,selected=1) for s in c['seeds']],c)['conditions']['C']['hypothesis_2']=='NOT SUPPORTED'
    assert a.summarize([entry(42),entry(43)],c)['conditions']['C']['hypothesis_2']=='INCONCLUSIVE'


def test_saved_common_base_hash_enforced(tmp_path):
    root,path=make_source(tmp_path)
    p=path/'common-base.pt';saved=torch.load(p,weights_only=True)
    next(iter(saved['model'].values())).add_(.1);torch.save(saved,p)
    with pytest.raises(ValueError,match='hash mismatch'):e.initialize(small(root),42)


def test_population_preserves_consistent_pairwise_winner_despite_margins():
    from torch import nn
    def constant(prob):
        net=nn.Sequential(nn.Linear(1,32),nn.GELU(),nn.Linear(32,1))
        with torch.no_grad():
            for p in net.parameters():p.zero_()
            net[-1].bias.fill_(float(np.log(prob/(1-prob))))
        return Policy([1],['hidden_norm'],np.zeros(1),np.ones(1),net)
    pairs={(0,1):constant(.51),(0,2):constant(.51),(0,3):constant(.51),(1,2):constant(.99),(1,3):constant(.99),(2,3):constant(.99)}
    router=e.PopulationRouter(4,pairs)
    assert router.choose({'hidden_norm':0})==0  # averaging confidences alone would incorrectly choose 1
