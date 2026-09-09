import copy
import json
import random
from dataclasses import replace,asdict
import numpy as np
import pytest
import torch
from rayvan_emc.token_lab import LabConfig,Dataset,ProbeModel,run
from rayvan_emc.token_lab_calibration import CalibrationDataset,distributions,train_specialists,digest,profiles
from rayvan_emc.token_lab_calibration_analysis import probe,classification_scores,split_rows
from rayvan_emc.token_lab_features import blind_keys,SCHEMA
from rayvan_emc.capability_tasks import CAPABILITIES


def config(**kw):
    return replace(LabConfig(dataset='capability_10',experiment_mode='forced_specialization',families=('gpt','gpt'),
        latent_dim=8,hidden_dim=16,heads=2,sequence_length=8,window=4,common_pretrain_steps=5,
        train_steps=2,batch_size=2,evaluation_samples=20,analysis_cap=20,reference_size=3,
        calibration_samples_per_task=2,spectral_enabled=False),**kw)


def train(c,root):
    torch.manual_seed(c.seed);torch.set_num_threads(1)
    ds=CalibrationDataset(c,Dataset(c));m=ProbeModel(c,ds.tokenizer.vocab_size)
    info,_,_=train_specialists(c,m,ds,root,set(),lambda:None,lambda *a,**kw:None)
    return info,m,ds


def test_profiles_strength_and_sampling():
    c=config();p=distributions(c)
    assert p[0]['probabilities']['arithmetic']==pytest.approx(1/3)
    assert p[0]['probabilities']['language']==0
    z=distributions(replace(c,specialization_strength=0))
    assert z[0]['probabilities']==z[1]['probabilities']==dict.fromkeys(CAPABILITIES,.1)
    mid=distributions(replace(c,specialization_strength=.5))
    assert mid[0]['probabilities']['language']==pytest.approx(.05)
    rng=random.Random(42);draws=rng.choices(CAPABILITIES,weights=list(mid[0]['probabilities'].values()),k=20000)
    assert max(abs(draws.count(t)/len(draws)-mid[0]['probabilities'][t]) for t in CAPABILITIES)<.02
    assert len(profiles('one_task',8))==8


def test_clone_freeze_budget_and_reproducibility(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    c=config(specialization_strength=0)
    info,m,ds=train(c,a);again,m2,_=train(c,b)
    assert len(set(info['initial_hashes']))==1
    assert info['clone_identity_verified'] and info['shared_unchanged']
    assert info['exact_control_final_identity']
    for i,names in enumerate(info['changed_names']):
        assert names and all(n.startswith(f'experts.{i}.') for n in names)
    assert {v['target_tokens'] for v in info['budgets']}=={4}
    assert {v['context_tokens'] for v in info['budgets']}=={32}
    assert {v['steps'] for v in info['budgets']}=={2}
    assert info['budgets'][0]['task_counts']==info['budgets'][1]['task_counts']
    assert digest(m.state_dict())==digest(m2.state_dict())
    assert info['initial_hashes']==again['initial_hashes']


def test_task_dataset_answer_semantics_and_split(tmp_path):
    c=config();ds=CalibrationDataset(c,Dataset(c));train_keys=set()
    for i,t in enumerate(CAPABILITIES):
        x,y,m=ds.sample_task('train',i,t);assert x.shape==(1,c.sequence_length)
        assert m['target_region']=='answer' and m['task_id']==t
        train_keys.add(m['prefix_sha256'])
    ds.excluded=train_keys
    for i,t in enumerate(CAPABILITIES):
        x,y,m=ds.sample_task('validation',i,t)
        assert m['prefix_sha256'] not in train_keys


def test_blind_allowlist_and_target_free():
    keys=blind_keys()
    for k in ['task_id','specialization_strength','task_weights','assigned_weight','success_familiarity','delta_norm','change_baseline_loss','improvement','relative_advantage','attention_entropy_mean','ffn_pre_mean','gradient_norm']:
        assert k not in keys
    for k in ['baseline_loss','target_probability','target_margin','effective_rank','predictive_entropy']:assert k in keys
    assert 'baseline_loss' not in blind_keys(True)
    assert all(SCHEMA[k]['origin']=='pre_expert' for k in keys)


def test_probe_exclusion_even_if_caller_requests_forbidden_features():
    torch.set_num_threads(1)
    rng=np.random.default_rng(19)
    rows=[]
    for i in range(180):
        x=float(rng.normal());y=2*x+.05*float(rng.normal())
        rows.append(dict(sample_id=f'probe-{i}',effective_rank=x,baseline_loss=4.,improvement=y,task_id=y,delta_norm=y,success_familiarity=y))
    p=probe(rows,['effective_rank','task_id','delta_norm','success_familiarity'],42,.2,lambda:None)
    assert p['features']==['effective_rank']
    assert p['ridge']['test']['r2']>.9
    assert p['mlp']['test']['r2']>.7
    assert p['success_classifier']['test']['roc_auc']>.8
    assert 'baseline_surprisal_only' in p
    assert split_rows([dict(sample_id='a',analysis_group='g'),dict(sample_id='b',analysis_group='g')])[0]==split_rows([dict(sample_id='a',analysis_group='g')])[0]


def test_auc_ties_and_missing_classes():
    a=classification_scores(np.array([0,1,0,1]),np.ones(4)*.5)
    assert a['roc_auc']==pytest.approx(.5) and a['pr_auc']==pytest.approx(.5)
    assert classification_scores(np.array([0,1]),np.array([.1,.9]))['roc_auc']==1
    assert classification_scores(np.ones(3),np.ones(3))['roc_auc'] is None


def test_calibration_end_to_end_and_reopen(tmp_path):
    c=config(specialization_strength=0,sequence_length=32);s=run(c,tmp_path,'calibration');root=tmp_path/'calibration'
    rows=[json.loads(v) for v in (root/'observations.jsonl').read_text().splitlines()]
    assert s['measured_locations']==20
    assert all(r['target_region']=='answer' for r in rows)
    for j in range(0,len(rows),2):
        assert rows[j]['prefix_sha256']==rows[j+1]['prefix_sha256']
        assert rows[j]['baseline_loss']==rows[j+1]['baseline_loss']
        assert rows[j]['hidden_norm']==rows[j+1]['hidden_norm']
        assert rows[j]['expert_loss']==pytest.approx(rows[j+1]['expert_loss'],abs=1e-6)
    a=json.loads((root/'calibration-analysis.json').read_text())
    assert a['experts'] and a['reveal']
    for e in a['reveal'].values():assert not e['specialization_formed']
    assert (root/'common-base.pt').exists() and (root/'specialist-1.pt').exists()
    saved=json.loads((root/'specialization.json').read_text());assert saved['shared_unchanged']
    s2=run(replace(c,specialist_checkpoint=str(root)),tmp_path,'reopened')
    assert s2['training_targets']==0
    assert s2['overall_full_stream']==s['overall_full_stream']
    assert s2['calibration']['base_hash']==s['calibration']['base_hash']


def test_checkpoint_base_and_validation(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    c=config();info,_,_=train(c,a)
    loaded,_,_=train(replace(c,common_base_source='checkpoint',common_checkpoint=str(a/'common-base.pt')),b)
    assert loaded['initial_hashes']==info['initial_hashes']
    assert loaded['common_hash']==info['common_hash']
    assert LabConfig(**{'name':'old run'}).experiment_mode=='standard'
    with pytest.raises(ValueError):replace(c,families=('gpt','ssm')).validate()
    with pytest.raises(ValueError):replace(c,measurement_rate=.5).validate()
    with pytest.raises(ValueError):replace(c,evaluation_samples=21).validate()
    with pytest.raises(ValueError):replace(c,task_weights={},specialization_profile='custom').validate()
