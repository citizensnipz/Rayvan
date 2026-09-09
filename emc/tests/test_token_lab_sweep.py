"""Scientific controls and orchestration without training in unit tests."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pytest
from rayvan_emc import token_lab_sweep as sweep
from rayvan_emc import token_lab_discovery as discovery


def config(**kw):return sweep.DEFAULTS | dict(sweep_seed_count=1,device='cpu',cross_task=False) | kw


def test_validation_clears_stale_manual_settings():
    c=sweep.validate(config(common_checkpoint='wrong-gpt.pt',families=['gpt'],specialization_strength=.75))
    assert 'common_checkpoint' not in c
    for f in sweep.FAMILIES:
        a=sweep.child_config(c,f,43,1)
        b=sweep.child_config(c,f,43,0,'correct/common-base.pt')
        assert a.families==b.families==(f,f)
        assert a.common_base_source=='pretrain' and a.common_checkpoint==''
        assert b.common_base_source=='checkpoint' and b.specialist_checkpoint==''
        assert b.specialization_strength==0 and b.identical_stream
        assert a.train_steps==b.train_steps and a.batch_size==b.batch_size
        assert a.seed==b.seed==43
    with pytest.raises(ValueError):sweep.validate(config(sweep_seed_count=0))
    with pytest.raises(ValueError):sweep.validate(config(evaluation_samples=99))


def test_cli_estimate_dispatches_sweep(tmp_path,monkeypatch,capsys):
    import sys
    from rayvan_emc.token_lab import main
    path=tmp_path/'sweep-config.json';sweep.write(path,config())
    monkeypatch.setattr(sys,'argv',['token_lab','estimate',str(path)])
    main();result=json.loads(capsys.readouterr().out)
    assert result['valid'] and '8 calibration runs' in result['warning']


def fake_infrastructure(monkeypatch,fail=None):
    calls=[]
    def run(c,root,run_id,cancel_path=None):
        calls.append((c,run_id));p=Path(root)/run_id;p.mkdir()
        if c.families[0]==fail:raise RuntimeError('synthetic backend unavailable')
        common=f'{c.families[0]}-{c.seed}'
        info=dict(clone_identity_verified=True,initial_hashes=[common]*2,shared_unchanged=True,
            common_hash=common,shared_hash=common,expert_parameter_counts=[10,10],optimizer={'lr':.001},
            exact_control_final_identity=c.specialization_strength==0,strength=c.specialization_strength,
            budgets=[dict(steps=c.train_steps,samples=4000,target_tokens=4000,context_tokens=192000)]*2)
        for n,v in [('config.json',asdict(c)),('status.json',{'status':'completed'}),('specialization.json',info),('calibration-analysis.json',{'reveal':{}})]:sweep.write(p/n,v)
        (p/'common-base.pt').touch()
    def analyze(request,root,run_id,cancel_path=None):
        p=Path(root)/run_id;p.mkdir();sweep.write(p/'discovery-analysis.json',{'targets':{},'raw_sha256':'raw'})
    monkeypatch.setattr(sweep,'run',run)
    monkeypatch.setattr(sweep.discovery,'run_saved',analyze)
    monkeypatch.setattr(sweep,'extra_analysis',lambda *args:{'status':'test'})
    return calls


def test_sweep_all_families_seeds_matching_and_resume(tmp_path,monkeypatch):
    calls=fake_infrastructure(monkeypatch)
    c=config(sweep_seed_count=2)
    r=sweep.run_sweep(c,tmp_path,'sweep')
    assert r['status']=='completed' and len(calls)==16
    for i in range(0,16,2):
        a,aid=calls[i];b,bid=calls[i+1]
        assert a.specialization_strength==1 and b.specialization_strength==0
        assert b.common_checkpoint==str(tmp_path/'sweep'/'jobs'/aid/'common-base.pt')
        assert a.seed==b.seed and a.families==b.families
    assert (tmp_path/'sweep'/'report.md').is_file()
    calls.clear()
    resumed=sweep.run_sweep(c|dict(resume_from=str(tmp_path/'sweep')),tmp_path,'resumed')
    assert resumed['status']=='completed' and not calls
    with pytest.raises(ValueError):sweep.validate(c|dict(resume_from=str(tmp_path/'sweep'),train_steps=2000))


def test_failure_continues_and_is_reported(tmp_path,monkeypatch):
    fake_infrastructure(monkeypatch,fail='ssm')
    r=sweep.run_sweep(config(),tmp_path,'sweep')
    assert r['status']=='partial'
    assert sum(e['status']=='completed' for e in r['entries'])==6
    assert 'synthetic backend unavailable' in (tmp_path/'sweep'/'report.md').read_text()


def test_parent_cancel_preserves_manifest(tmp_path,monkeypatch):
    fake_infrastructure(monkeypatch)
    real=sweep.run
    def stop(c,root,run_id,cancel_path=None):
        real(c,root,run_id,cancel_path);Path(cancel_path).touch()
    monkeypatch.setattr(sweep,'run',stop)
    with pytest.raises(InterruptedError):sweep.run_sweep(config(),tmp_path,'sweep')
    assert sweep.load(tmp_path/'sweep'/'status.json')['status']=='cancelled'
    assert (tmp_path/'sweep'/'sweep-manifest.json').is_file()


def test_control_hash_mismatch_fails(tmp_path):
    common=dict(clone_identity_verified=True,initial_hashes=['x','x'],shared_unchanged=True,
        budgets=[dict(steps=1,samples=2,target_tokens=2,context_tokens=4)]*2,common_hash='x',shared_hash='x',
        expert_parameter_counts=[2,2],optimizer={},strength=0,exact_control_final_identity=True)
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    sweep.write(a/'specialization.json',common);sweep.write(b/'specialization.json',common|{'common_hash':'different'})
    with pytest.raises(ValueError,match='common_hash'):sweep.verify_controls(a,b)


def test_decision_sign_and_training_constant():
    y=np.array([1.,2.,1.,2.,-2.,3.]);masks=(np.array([1,1,0,0,0,0],bool),np.array([0,0,1,1,0,0],bool),np.array([0,0,0,0,1,1],bool))
    rows=[dict(sample_id=str(i)) for i in range(6)]
    d=sweep.decision_scores(rows,y,y,masks,42)
    assert d['mean_regret']==0 and d['constant_regret']==1 and d['gain_over_constant']==1
    assert d['non_tie_accuracy']==1


def test_group_features_and_cross_task_preprocessing_are_blind():
    rows=[dict(sample_id=str(i),value=float(i%5),hidden_mean=float(i),task_id='SECRET',change_baseline_loss=-999,expert_success_familiarity=999) for i in range(90)]
    split=np.array([0]*30+[6]*30+[8]*30)
    X,y,keys,masks=discovery.prepare(rows,split)
    assert keys==['hidden_mean']
    assert abs(X[:30].mean())<1e-9 and X[60:].mean()>1
    g=sweep.group_indices(['context_start','hidden_mean','hidden_novelty_knn','spectral_hks.std_1'])
    assert g['controls']==[0] and g['without_geometry']==[0,2]
    assert g['without_spectral']==[0,1,2] and g['without_novelty']==[0,1,3]
