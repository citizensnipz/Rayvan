"""Mixed architecture banks share one trunk, preserve all pairs, and feed emergence."""
from dataclasses import asdict
import json
import numpy as np
import pytest
import torch
from rayvan_emc import token_lab_sweep as s
from rayvan_emc import token_lab_emergence as e
from rayvan_emc.token_lab import Dataset,ProbeModel
from rayvan_emc.token_lab_calibration import digest,train_specialists,CalibrationDataset,load_compatible


def small():
    return s.DEFAULTS|dict(population='heterogeneous',sweep_seed_count=1,common_pretrain_steps=2,train_steps=2,batch_size=5,latent_dim=8,hidden_dim=16,heads=2,sequence_length=48,evaluation_samples=100,calibration_samples_per_task=2,reference_size=10,spectral_enabled=False,cross_task=False,max_features=1,candidate_count=2)


def test_mixed_control_semantics_and_rotating_task_profiles():
    c=s.validate(small());first=s.child_config(c,'mixed',42,1)
    assert first.families==s.FAMILIES and first.architecture_confounded
    assert s.child_config(c,'mixed',42,0).families==s.FAMILIES
    assert s.child_config(c,'mixed',43,1).task_weights!=first.task_weights
    for task in first.task_weights['0']:
        assert sum(s.child_config(c,'mixed',seed,1).task_weights['0'][task] for seed in range(42,46))==1
    with pytest.raises(ValueError):e.validate(e.DEFAULTS|dict(family='mixed',expert_count=2))


def test_mixed_common_training_budget_freeze_and_loading(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(42)
    c=s.child_config(small(),'mixed',42,1);ds=Dataset(c);m=ProbeModel(c,ds.tokenizer.vocab_size)
    before=[digest(x.state_dict()) for x in m.experts]
    info,_,_=train_specialists(c,m,CalibrationDataset(c,ds),tmp_path,set(),lambda:None,lambda *a,**k:None)
    saved=torch.load(tmp_path/'common-base.pt',weights_only=True)
    assert info['architecture_confounded'] and not info['clone_identity_verified']
    assert len(set(info['initial_hashes']))==4
    assert all(a!=b for a,b in zip(before,info['initial_hashes']))  # all four general experts trained
    assert info['common_pretraining_expert_applications']==40
    assert [b['target_tokens'] for b in info['budgets']]==[10]*4
    assert info['shared_unchanged']
    for i,names in enumerate(info['changed_names']):assert all(n.startswith(f'experts.{i}.') for n in names)
    restore=ProbeModel(c,ds.tokenizer.vocab_size);load_compatible(tmp_path/'common-base.pt',c,restore,ds.tokenizer)
    assert digest(restore.state_dict())==digest(saved['model'])
    h=digest(restore.state_dict());e.perturb(restore,0,42,allow_nonidentical=True);assert digest(restore.state_dict())==h
    assert all(p['l2'] is None for p in e.parameter_distances(restore))
    bad=s.child_config(small(),'gpt',42,1)
    with pytest.raises(ValueError,match='exact ordered'):load_compatible(tmp_path/'common-base.pt',bad,ProbeModel(bad,ds.tokenizer.vocab_size),ds.tokenizer)


def test_real_mixed_sweep_and_emergence(tmp_path):
    torch.set_num_threads(1)
    result=s.run_sweep(small(),tmp_path,'mixed-sweep')
    assert result['status']=='completed',[(v['key'],v.get('error')) for v in result['entries']]
    assert len(result['entries'])==2
    a,b=result['entries'];assert a['controls']['common_hash']==b['controls']['common_hash']
    assert a['controls']['architecture_confounded'] and not b['controls']['exact_control_final_identity']
    assert len(a['pair_extras'])==len(b['pair_extras'])==6
    assert len(result['families']['mixed']['pair_replication'])==6
    c=e.DEFAULTS|dict(source_sweep=str(tmp_path/'mixed-sweep'),family='mixed',expert_count=4,seeds=[42],asymmetry_strengths=[0,.01],feedback_strengths=[0,.5],train_steps=5,batch_size=2,probe_interval=5,probe_samples=20,validation_samples=20,evaluation_samples=20,diagnostic_samples=100,reference_size=10,fit_epochs=1,checkpoints=[0,1])
    cfg,ds,m,excluded,proof=e.initialize(c,42)
    assert cfg.families==list(s.FAMILIES)
    assert proof['architecture_confounded'] and not proof['clones_identical']
    # Mixed source is general, not the later task-biased specialist checkpoint.
    expected=torch.load(tmp_path/'mixed-sweep'/'jobs'/'mixed-seed-42-s100'/'common-base.pt',weights_only=True)
    assert digest(m.state_dict())==digest(expected['model'])
    study=e.run_study(c,tmp_path,'mixed-emergence')
    assert study['status']=='completed',[(v['id'],v.get('error')) for v in study['entries']]
    by={v['condition']:v for v in study['entries']}
    assert by['D']['final']['exposure']==by['E']['final']['exposure']
    assert all(v['shared_unchanged'] for v in study['entries'])
    assert by['A']['final']['specialization']['interaction_rms']>0
    path=tmp_path/'mixed-emergence'/'jobs'/'seed-42-A-a0-b0'/'checkpoint-0'/'observations.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert {r['expert_family'] for r in rows}==set(s.FAMILIES)
    assert 'NOT identical experts' in (tmp_path/'mixed-emergence'/'report.md').read_text()
