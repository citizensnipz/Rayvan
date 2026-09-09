import hashlib
import json
import numpy as np
import pytest
import torch
from rayvan_emc.token_lab_discovery import read_locations,targets,prepare,select,analyze_target,run_saved,validate


def fixture_rows(n=180):
    rng=np.random.default_rng(25);groups=[]
    for i in range(n):
        x=float(rng.normal()*.3);noise=float(rng.normal());rs=[]
        for e,improvement in [('1:gpt',.4+2*x),('2:gpt',.4-x)]:
            rs.append(dict(sample_id=f'location-{i}',analysis_group=f'group-{i}',expert_id=e,split='validation',topology='independent',
                prefix_sha256=hashlib.sha256(str(i).encode()).hexdigest(),baseline_loss=5.,expert_loss=5-improvement,improvement=improvement,
                hidden_mean=x,hidden_skewness=noise,delta_norm=improvement,task_id=improvement,success_familiarity=improvement))
        groups.append(rs)
    return groups


def test_pair_sign_and_identical_control():
    groups=fixture_rows();sets=targets(groups);rs=sets['1:gpt minus 2:gpt advantage']
    assert rs[0]['value']==pytest.approx(3*rs[0]['hidden_mean'])
    for g in groups:g[1]['expert_loss']=g[0]['expert_loss'];g[1]['improvement']=g[0]['improvement']
    rows=targets(groups)['1:gpt minus 2:gpt advantage']
    assert analyze_target(rows,{},lambda:None)['status'].startswith('constant outcome')
    groups[0][1]['prefix_sha256']='different'
    with pytest.raises(ValueError,match='identical input'):targets(groups)


def test_selection_does_not_use_test_labels_and_excludes_leaks():
    torch.set_num_threads(1)
    rows=targets(fixture_rows())['1:gpt improvement'];X,y,keys,masks=prepare(rows)
    assert 'task_id' not in keys and 'delta_norm' not in keys and 'success_familiarity' not in keys and 'baseline_loss' not in keys
    chosen,meta=select(X,y,keys,masks,3,4,42,lambda:None)
    corrupted=y.copy();corrupted[masks[2]]=np.arange(masks[2].sum())*10000
    again,_=select(X,corrupted,keys,masks,3,4,42,lambda:None)
    assert chosen==again and 'hidden_mean' in [keys[j] for j in chosen]


def test_compact_signal_and_removal():
    torch.set_num_threads(1)
    d=analyze_target(targets(fixture_rows())['1:gpt improvement'],dict(max_features=3,candidate_count=4),lambda:None)
    assert d['scores']['compact_ridge']['r2']>.9
    assert 'hidden_mean' in d['selected_features']
    v=next(v for v in d['ablations'] if v['feature']=='hidden_mean')
    assert v['mae_increase']>0 and v['mae_increase_ci'][0]>0


def source(tmp_path,n=180):
    p=tmp_path/'source';p.mkdir();(p/'config.json').write_text(json.dumps(dict(name='fixture',families=['gpt','gpt'])))
    (p/'status.json').write_text('{"status":"completed"}')
    (p/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for g in fixture_rows(n) for r in g))
    return p


def test_streaming_reservoir_keeps_pairs_and_finite_policy(tmp_path):
    p=source(tmp_path);a,n,h=read_locations(p/'observations.jsonl',20,42,lambda:None)
    b,_,_=read_locations(p/'observations.jsonl',20,42,lambda:None)
    assert a==b and len(a)==20 and n==180 and all(len(g)==2 for g in a)
    assert h==hashlib.sha256((p/'observations.jsonl').read_bytes()).hexdigest()
    assert all('task_id' not in r and 'delta_norm' not in r for g in a for r in g)


def test_saved_run_no_model_execution_or_source_mutation(tmp_path,monkeypatch):
    p=source(tmp_path);before={v.name:v.read_bytes() for v in p.iterdir()}
    def forbidden(*a,**kw):raise AssertionError('must not load checkpoints')
    monkeypatch.setattr(torch,'load',forbidden)
    s=run_saved(dict(source_run_directory=str(p),max_features=2,candidate_count=3,location_cap=200),tmp_path,'analysis')
    assert s['training_targets']==0 and s['measured_locations']==180
    assert before=={v.name:v.read_bytes() for v in p.iterdir()}
    result=json.loads((tmp_path/'analysis/analysis.json').read_text())['discovery']
    assert len(result['targets'])==3 and result['targets']['1:gpt improvement']['status']=='ok'
    assert (tmp_path/'analysis/report.md').is_file() and (tmp_path/'analysis/discovery-analysis.json').is_file()


def test_cancel_and_missing_raw(tmp_path):
    with pytest.raises(ValueError,match='observations'):validate(dict(source_run_directory=str(tmp_path)))
    p=source(tmp_path)
    def cancel():raise InterruptedError('cancel')
    with pytest.raises(InterruptedError):read_locations(p/'observations.jsonl',20,42,cancel)
