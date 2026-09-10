"""Scientific controls and a real small frozen-bank execution test."""
import json
from dataclasses import asdict
import numpy as np
import pytest
import torch
from torch import nn
from rayvan_emc import token_lab_routing as r
from rayvan_emc.token_lab import LabConfig,Dataset,ProbeModel,write
from rayvan_emc.token_lab_calibration import CalibrationDataset,train_specialists,digest
from rayvan_emc import token_lab_features as mf


def test_regret_weighted_preference_math():
    score=torch.tensor([0.,0.,0.],requires_grad=True);a=torch.tensor([2.,-1.,0.])
    loss=r.preference_loss(score,a)
    assert float(loss.detach())==pytest.approx(float(np.log(2)))
    loss.backward()
    assert score.grad.tolist()==pytest.approx([-1/3,1/6,0.])


class Add(nn.Module):
    def __init__(self,n):super().__init__();self.n=n;self.calls=0
    def forward(self,h):self.calls+=1;return torch.ones_like(h)*self.n


def test_sequential_reobserves_transformed_state_and_remaining_steps():
    model=nn.Module();model.experts=nn.ModuleList([Add(2),Add(5)])
    seen=[]
    class StatePolicy:
        def choose(self,observer,h,x,pos,remaining,kind):
            seen.append((float(h[0,0,0]),remaining));return 0 if h[0,0,0]<1 else 1
    h,path=r.rollout(model,torch.zeros(1,2,1),None,0,3,StatePolicy(),None,'controls',lambda:None)
    assert path==[0,1,1];assert seen==[(0,3),(2,2),(7,1)];assert float(h[0,0,0])==12
    assert [e.calls for e in model.experts]==[1,2]
    seen.clear()
    r.rollout(model,torch.zeros(1,2,1),None,0,3,StatePolicy(),None,'controls',lambda:None,first=1)
    assert seen==[(5,2),(10,1)]  # Intervene first, preserve teacher on suffix.


def test_input_api_has_no_target_and_never_runs_candidate_experts():
    c=LabConfig(sequence_length=8,latent_dim=8,hidden_dim=16,heads=2,dataset='capability_10')
    ds=Dataset(c);m=ProbeModel(c,ds.tokenizer.vocab_size).eval();x=torch.ones(1,8,dtype=torch.long)
    bank=dict(hidden=torch.empty(0,8),descriptor=torch.empty(0,8),mean=torch.zeros(8),std=torch.ones(8))
    obs=r.Observer(m,c,ds,bank)
    m.experts=nn.ModuleList([Add(2),Add(3)])
    with torch.no_grad():h=m.embed(x)
    row=obs.features(h,x,20,2,3,'full')
    forbidden={'baseline_loss','target_probability','target_margin','task_id','success_familiarity','improvement','delta_norm'}
    assert not forbidden.intersection(row)
    assert set(row)<=set(mf.blind_keys(True))|{'remaining_steps','trajectory_step'}
    assert [e.calls for e in m.experts]==[0,0]


def test_data_pools_are_balanced_disjoint_and_reproducible():
    cfg=LabConfig(dataset='capability_10',sequence_length=48)
    c=r.DEFAULTS|dict(training_samples=20,validation_samples=20,test_samples=20)
    ds=Dataset(cfg);a=r.make_pools(c,cfg,ds,set(),set(),lambda:None)
    b=r.make_pools(c,cfg,ds,set(),set(),lambda:None)
    allgroups=[];allprefix=[]
    for split in a:
        assert [m for _,_,m in a[split]]==[m for _,_,m in b[split]]
        tasks=[m['task_id'] for _,_,m in a[split]]
        assert all(tasks.count(t)==2 for t in r.CAPABILITIES)
        allgroups.extend(m['analysis_group'] for _,_,m in a[split]);allprefix.extend(m['prefix_sha256'] for _,_,m in a[split])
    assert len(set(allgroups))==60;assert len(set(allprefix))==60
    blocked=r.make_pools(c,cfg,ds,set(allprefix),set(allgroups),lambda:None)
    assert not set(allprefix)&{m['prefix_sha256'] for pool in blocked.values() for _,_,m in pool}


def make_source(tmp_path,family='gpt',strength=1):
    root=tmp_path/'sweep';source=root/'jobs'/f'{family}-seed-42-s{strength*100}';source.mkdir(parents=True)
    c=LabConfig(experiment_mode='forced_specialization',dataset='capability_10',families=(family,family),
        sequence_length=48,latent_dim=8,hidden_dim=16,heads=2,common_pretrain_steps=2,train_steps=2,batch_size=5,
        evaluation_samples=100,analysis_cap=100,calibration_samples_per_task=1,specialization_strength=strength)
    torch.manual_seed(c.seed);torch.set_num_threads(1);ds=Dataset(c);model=ProbeModel(c,ds.tokenizer.vocab_size)
    cds=CalibrationDataset(c,ds);excluded=set()
    info,_,_=train_specialists(c,model,cds,source,excluded,lambda:None,lambda *a,**k:None)
    model.eval().requires_grad_(False)
    torch.save(dict(config=asdict(c),model=model.state_dict(),tokenizer=ds.tokenizer.to_config(),excluded_prefixes=sorted(excluded)),source/'model.pt')
    write(source/'config.json',asdict(c))
    # Small real training-only reference cloud.
    with torch.no_grad():h=model.embed(torch.ones(1,16,dtype=torch.long))[0]
    feats=mf.core(h[-c.window:]);values=torch.tensor([[feats[k] for k in mf.NOVEL_DESCRIPTOR]])
    bank=dict(hidden=torch.nn.functional.normalize(h,dim=-1),descriptor=torch.zeros_like(values),mean=values[0],std=torch.ones(8))
    torch.save(dict(banks=[bank,bank],excluded_training_prefix_hashes=sorted(excluded)),source/'reference-bank.pt')
    (source/'observations.jsonl').write_text('')
    entry=dict(key=source.name,family=family,seed=42,strength=strength,status='completed',source_directory=str(source),controls=dict(common_hash=info['common_hash']))
    previous=json.loads((root/'sweep-analysis.json').read_text())['entries'] if (root/'sweep-analysis.json').exists() else []
    write(root/'sweep-analysis.json',dict(schema_version=1,entries=previous+[entry]))
    return root,source


@pytest.mark.parametrize('family',['gpt','ssm','recurrent','delta'])
def test_loads_all_sweep_families_without_training(tmp_path,family):
    root,path=make_source(tmp_path,family)
    cfg,ds,model,bank,ex,groups,info=r.load_model(path,'cpu')
    assert all(not p.requires_grad for p in model.parameters());assert not model.training
    assert [digest(e.state_dict()) for e in model.experts]==info['final_hashes']
    c=r.validate(r.DEFAULTS|dict(source_sweep=str(root),family=family,include_control=False))
    assert len(r.sources(c))==1


def test_real_frozen_routing_smoke(tmp_path):
    root,path=make_source(tmp_path)
    c=r.DEFAULTS|dict(source_sweep=str(root),include_control=False,training_samples=100,validation_samples=100,
        test_samples=100,rounds=1,fit_epochs=2,sequential_steps=2)
    before=(path/'model.pt').read_bytes()
    result=r.run_test(c,tmp_path,'routing-test')
    assert result['status']=='completed',result
    e=result['entries'][0];assert e['frozen_verified'];assert len(e['results'])==4
    assert (path/'model.pt').read_bytes()==before
    for v in e['results']:
        assert np.isfinite(v['mean_loss']);assert v['suffix_regret']>=0
        assert sum(v['expert_counts'])==100*v['horizon']
        assert v['comparisons']['oracle_sequence']['gain']<=1e-6
        if v['selected_policy']=='fixed baseline retained':assert abs(v['comparisons']['fixed_sequence']['gain'])<1e-6
    saved=json.loads((tmp_path/'routing-test'/'analysis.json').read_text())
    assert saved['routing_test']==result
    with pytest.raises(ValueError,match='missing model.pt'):
        (path/'model.pt').unlink();r.validate(c)


def test_fitted_policy_learns_state_dependent_choices_and_reloads(tmp_path):
    c=r.DEFAULTS|dict(fit_epochs=80)
    rows=[{'predictive_entropy':float(v)} for v in np.linspace(-2,2,200)]
    advantages=np.linspace(-2,2,200)
    policy=r.fit_policy(rows,advantages,[0],c,42,lambda:None)
    class Observer:
        def features(self,h,x,pos,remaining,horizon,kind):return {'predictive_entropy':h}
    assert policy.choose(Observer(),-1,None,0,1,'controls')==1
    assert policy.choose(Observer(),1,None,0,1,'controls')==0
    policy.save(tmp_path/'router.pt');loaded=r.Policy.load(tmp_path/'router.pt')
    for v in [-1,1]:assert loaded.choose(Observer(),v,None,0,1,'controls')==policy.choose(Observer(),v,None,0,1,'controls')
    # Zero advantages must not manufacture a preference based on network initialization.
    fixed=r.fit_policy(rows,np.zeros(200),[1],c,42,lambda:None)
    assert fixed.net is None and fixed.sequence==(1,)


def test_audit_cannot_override_selected_choice():
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.anchor=nn.Parameter(torch.zeros(1));self.experts=nn.ModuleList([Add(1),Add(-1)])
        def embed(self,x):return torch.zeros(1,2,1)
        def logits(self,h):return torch.cat([h[:,-1],-h[:,-1]],1)
    model=Model();pool=[(torch.zeros(1,2,dtype=torch.long),0,{'source_position':1,'sample_id':'a','analysis_group':'a'})]
    observer=type('Observer',(),{'seconds':0.})()
    # Pick the worse expert; audit must report regret without replacing it.
    policy=r.Policy([1]);plain,_=r.execute(model,pool,1,policy,observer,'controls',lambda:None)
    audited,stats=r.execute(model,pool,1,policy,observer,'controls',lambda:None,audit=True)
    assert np.array_equal(plain,audited);assert stats['routes']==[[1]];assert stats['mean_suffix_regret']>1


def test_matched_control_and_cli_estimate(tmp_path,monkeypatch,capsys):
    from rayvan_emc.token_lab import main
    import sys
    root,_=make_source(tmp_path,strength=1)
    _,control=make_source(tmp_path,strength=0)
    cfg=r.DEFAULTS|dict(source_sweep=str(root))
    assert len(r.sources(r.validate(cfg)))==2
    info=json.loads((control/'specialization.json').read_text())
    assert info['exact_control_final_identity']
    config_path=tmp_path/'request.json';write(config_path,cfg)
    monkeypatch.setattr(sys,'argv',['token_lab','estimate',str(config_path)])
    main();out=json.loads(capsys.readouterr().out)
    assert out['valid'] and '2 saved expert banks' in out['warning']
    sweep=json.loads((root/'sweep-analysis.json').read_text())
    sweep['entries'][1]['controls']['common_hash']='incorrect'
    write(root/'sweep-analysis.json',sweep)
    with pytest.raises(ValueError,match='same common base'):r.validate(cfg)
