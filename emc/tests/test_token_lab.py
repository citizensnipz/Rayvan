import json
from dataclasses import replace
import pytest
import torch
from rayvan_emc.token_lab_features import core,prediction,cosine,novelty,spectral,outcomes,instrumentation,NOVEL_DESCRIPTOR
from rayvan_emc.token_lab import LabConfig,ProbeModel,run


def test_identical_dispersion_and_missing_shape():
    v=core(torch.ones(8,4));assert v['radius']==0 and v['pair_distance_mean']==0
    assert v['effective_rank'] is None


def test_scale_and_pca_energy():
    torch.manual_seed(1);h=torch.randn(16,4);a,b=core(h),core(3*h)
    assert b['radius']==pytest.approx(3*a['radius'],rel=1e-5)
    assert b['variance']==pytest.approx(9*a['variance'],rel=1e-5)
    assert sum(a[f'pca_p{i}'] for i in range(1,5))==pytest.approx(1)


def test_line_plane_volume_rank():
    values=[]
    for rank in [1,2,3]:
        h=torch.zeros(6,4)
        for j in range(rank):h[j*2,j]=1;h[j*2+1,j]=-1
        values.append(core(h)['effective_rank'])
    assert values==pytest.approx([1,2,3])


def test_entropy_cosine():
    assert prediction(torch.zeros(3),0)['predictive_entropy']>prediction(torch.tensor([10.,0.,0.]),0)['predictive_entropy']
    assert cosine(torch.tensor([1.,0.]),torch.tensor([1.,0.]))==pytest.approx(1)
    assert cosine(torch.tensor([1.,0.]),torch.tensor([0.,1.]))==pytest.approx(0)
    assert cosine(torch.zeros(2),torch.ones(2)) is None


def test_novelty_reference_only():
    bank=dict(hidden=torch.tensor([[1.,0.]]),descriptor=torch.zeros(1,len(NOVEL_DESCRIPTOR)),mean=torch.zeros(len(NOVEL_DESCRIPTOR)),std=torch.ones(len(NOVEL_DESCRIPTOR)))
    features=dict.fromkeys(NOVEL_DESCRIPTOR,0.)
    a=novelty(torch.tensor([1.,0.]),features,bank);b=novelty(torch.tensor([0.,1.]),features,bank)
    assert a['hidden_novelty_nearest']<b['hidden_novelty_nearest']
    assert a['descriptor_novelty_nearest']==0


def test_spectral_relabeling_finite():
    torch.manual_seed(3);h=torch.randn(8,4);a=spectral(h);b=spectral(h[torch.randperm(8)])
    assert a.keys()==b.keys()
    for k in a:
        if a[k] is not None:assert a[k]==pytest.approx(b[k],abs=2e-4)


def test_advantage_serial_attribution():
    a=outcomes([3.,3.],[1.,2.],True,.001)
    assert a[0]['improvement']==2 and a[0]['relative_advantage']==1 and a[0]['expert_rank']==1
    b=outcomes([3.,1.],[1.,2.],False,.001)
    assert b[1]['improvement']==-1 and b[1]['relative_advantage'] is None and b[1]['expert_rank'] is None


def test_missing_attention_and_instrumentation_preserves_output():
    c=LabConfig(families=('gpt','recurrent'),latent_dim=8,hidden_dim=16,heads=2)
    model=ProbeModel(c,20).eval();h=torch.randn(1,8,8)
    for i in range(2):
        with torch.no_grad():expected=model.experts[i](h)
        actual,features=instrumentation(model.experts[i],h)
        torch.testing.assert_close(actual,expected)
        assert (features['attention_entropy_mean'] is None)==(i==1)


@pytest.mark.parametrize('topology',['independent','serial'])
def test_standalone_run_storage_splits_and_no_router(tmp_path,topology):
    c=LabConfig(dataset='capability_10',families=('gpt','ssm','recurrent','delta'),latent_dim=8,hidden_dim=16,heads=2,sequence_length=8,window=4,
        topology=topology,train_steps=1,batch_size=1,evaluation_samples=5,reference_size=3,deep_enabled=True,deep_rate=1.)
    summary=run(c,tmp_path,topology);root=tmp_path/topology
    rows=[json.loads(l) for l in (root/'observations.jsonl').read_text().splitlines()]
    assert len(rows)==20 and summary['observations']==20
    assert all(r['split']=='validation' for r in rows)
    assert all(r['gradient_norm'] is not None for r in rows)
    bank=torch.load(root/'reference-bank.pt',weights_only=True)
    assert not set(bank['reference_ids'])&{r['sample_id'] for r in rows}
    assert not set(bank['excluded_training_prefix_hashes'])&{r['prefix_sha256'] for r in rows}
    if topology=='independent':assert len({r['baseline_loss'] for r in rows[:4]})==1
    else:
        assert rows[1]['baseline_loss']==pytest.approx(rows[0]['expert_loss'])
        assert all(r['relative_advantage'] is None for r in rows)
    assert (root/'analysis.json').exists() and (root/'report.md').exists()
    assert not any('router' in k for k in ProbeModel(c,20).state_dict())


def test_ridge_excludes_outcomes_and_group_split():
    from rayvan_emc.token_lab_analysis import multivariate,ranks
    import numpy as np
    rows=[dict(sample_id=str(i),x=float(i),improvement=float(i)*2) for i in range(60)]
    r=multivariate(rows,['x'],'improvement');assert r['held_out']['r2']>.9
    assert ranks(np.array([1,1,3])).tolist()==[.5,.5,2]


def test_config_validation():
    with pytest.raises(ValueError):LabConfig(window=3).validate()
    with pytest.raises(ValueError):LabConfig(families=('bogus',)).validate()


def test_completed_analysis_reservoir_and_cancellation(tmp_path):
    c=LabConfig(dataset='capability_10',families=('gpt','recurrent'),latent_dim=8,hidden_dim=16,heads=2,
        sequence_length=8,window=4,train_steps=0,evaluation_samples=40,reference_size=3,analysis_cap=32,spectral_enabled=False)
    summary=run(c,tmp_path,'analysis')
    assert summary['measured_locations']+summary['duplicate_prefixes_skipped']==40 and summary['analysis_locations']==32
    a=json.loads((tmp_path/'analysis/analysis.json').read_text());assert a['pairs']
    assert len(a['experts']['1:gpt']['low_high'])>0
    assert len((tmp_path/'analysis/observations.jsonl').read_text().splitlines())==summary['measured_locations']*2


def test_analysis_cancel():
    from rayvan_emc.token_lab_analysis import analyze
    with pytest.raises(InterruptedError):analyze([dict(expert_id='a')],cancelled=lambda:True)


def test_tinystories_uses_existing_loader_and_disjoint_blocks(monkeypatch):
    import rayvan_emc.token_lab as lab
    from rayvan_emc.data import LanguageCorpus
    corpus=LanguageCorpus.from_texts(['train story abc '*30],['held out story xyz '*30])
    calls=[]
    def loader(**kw):calls.append(kw);return corpus
    monkeypatch.setattr(lab,'load_tinystories',loader)
    ds=lab.Dataset(LabConfig(sequence_length=8))
    x,y,meta=ds.sample('validation',0);x2,y2,meta2=ds.sample('validation',1)
    assert calls and meta['sample_id']!=meta2['sample_id'] and x.shape[1]<=8
    assert torch.equal(ds.counts,torch.bincount(corpus.train_tokens,minlength=corpus.tokenizer.vocab_size))


def test_run_cancel_preserves_status(tmp_path,monkeypatch):
    import rayvan_emc.token_lab as lab
    original=lab.Dataset
    def dataset(c):
        (tmp_path/'cancel/cancel.requested').touch()
        return original(c)
    monkeypatch.setattr(lab,'Dataset',dataset)
    c=LabConfig(dataset='capability_10',latent_dim=8,hidden_dim=16,heads=2,train_steps=1)
    with pytest.raises(InterruptedError):run(c,tmp_path,'cancel')
    assert json.loads((tmp_path/'cancel/status.json').read_text())['status']=='cancelled'
