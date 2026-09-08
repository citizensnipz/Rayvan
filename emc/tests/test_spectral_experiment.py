from copy import deepcopy
from dataclasses import replace
import torch
from rayvan_emc.spectral_experiment import descriptor_oracle_disagreement, fit_comparison, metrics
from rayvan_emc.spectral_geometry import Descriptor
from rayvan_emc.spectral_routing import SpectralGeometricRouter
from rayvan_emc.value_fit import FitBank
from rayvan_emc.value_routing import StateBatch
from test_spectral_routing import config


def test_collision_detects_missing_information():
    descriptors = torch.tensor([[0.,0.],[0.,0.],[5.,5.],[5.,5.]])
    losses = torch.tensor([[0.,1.],[1.,0.],[0.,1.],[1.,0.]])
    report = descriptor_oracle_disagreement(descriptors,losses)
    assert report['mean_nearest_distance'] == 0
    assert report['near_identical_disagreement'] == 1
    assert report['top_k_oracle_overlap'] == 1


def test_initialization_reproducible_and_held_out_does_not_change_stats():
    torch.manual_seed(9);torch.set_num_threads(1)
    router = SpectralGeometricRouter(config())
    x = torch.randn(12,16,8)
    descriptor = router.describe(x)
    losses = torch.rand(12,3)
    router.initialize(descriptor,losses)
    other = SpectralGeometricRouter(config())
    other.initialize(descriptor,losses)
    assert all(torch.equal(v,other.state_dict()[n]) for n,v in router.state_dict().items())
    state = deepcopy(router.standardizer.state_dict())
    router.eval();router.route_one(x+500)
    assert all(torch.equal(v,router.standardizer.state_dict()[n]) for n,v in state.items())


def test_controlled_ablation_smoke_uses_shared_bank():
    torch.manual_seed(31);torch.set_num_threads(1)
    def bank(offset):
        x=torch.randn(6,32,8)
        y=torch.stack((x.square().mean((1,2)),x[:,:,0].std(1),x[:,:,1].std(1)),1)
        return FitBank((StateBatch(x,torch.zeros(6,dtype=torch.long),0),),(y,),torch.arange(192).reshape(6,32)+offset)
    train,held=bank(0),bank(1000)
    before=[s.latent.clone() for s in train.states]
    rows=fit_comparison(train,held,config(),updates=2)
    assert len(rows)==13
    assert len({r['held_out']['uniform_random_regret'] for r in rows})==1
    assert all(torch.equal(x,s.latent) for x,s in zip(before,train.states))
    assert all(r['parameters']>0 for r in rows)


def test_active_complementarity_updates_experts_only():
    from rayvan_emc.spectral_model import SpectralGeometricEMC
    torch.manual_seed(73);torch.set_num_threads(1)
    model=SpectralGeometricEMC(config(transformation_complementarity_weight=.1,
         transformation_complementarity_threshold=-1.,spectral_route_weight=0.,basin_redundancy_weight=0.))
    out=model.endpoint(torch.randint(0,16,(2,16)),counterfactual_targets=torch.randint(0,16,(2,)),return_trace=True)
    out.geometry_calibration_loss.backward()
    gradients=[p.grad for e in model.emc_modules for p in e.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum()>0 for g in gradients)
    assert all(p.grad is None for p in model.integrator.parameters())
    assert model.token_embedding.weight.grad is None


def test_research_console_training_and_json_report(tmp_path):
    from test_value_experiment import experiment
    from rayvan_emc.research_runner import run_experiment
    from rayvan_emc.research_config import RoutingConfig
    from rayvan_emc.checkpoint import load_model_checkpoint
    from rayvan_emc.spectral_model import SpectralGeometricEMC
    torch.set_num_threads(1)
    c=experiment(architecture='spectral_geometric_emc',routing=RoutingConfig(counterfactual_probe_preset='fixed',
        counterfactual_probe_fixed_rate=1.,counterfactual_max_probes_per_forward=1))
    result=run_experiment(c,runs_directory=tmp_path,run_id='spectral')
    assert result['status']=='completed'
    model=load_model_checkpoint(result['training_result']['latest_checkpoint']).model
    assert isinstance(model,SpectralGeometricEMC)
    assert model.router.standardizer.count>0
    import json
    report=json.loads((tmp_path/'spectral'/'geometric-routing.json').read_text())
    assert report['router_type']=='spectral_geometric'
    assert report['sampled_geometry']['window_size']>1


def test_existing_bank_cli_and_reference_identity(tmp_path,monkeypatch):
    import sys,json
    from rayvan_emc.spectral_experiment import main
    torch.manual_seed(19);torch.set_num_threads(1)
    def bank(offset):
        latent=torch.randn(3,8,8)
        return FitBank((StateBatch(latent,torch.zeros(3,dtype=torch.long),0),),
                       (torch.rand(3,3),),torch.arange(24).reshape(3,8)+offset)
    path=tmp_path/'bank.pt'
    torch.save(dict(train=bank(0).payload(),held_out=bank(100).payload(),source_identity={'test':'fixed'}),path)
    monkeypatch.setattr(sys,'argv',['spectral_experiment','--bank',str(path),'--output',str(tmp_path/'fit'),'--updates','1'])
    main()
    report=json.loads((tmp_path/'fit'/'report.json').read_text())
    assert report['source']=={'test':'fixed'}
    assert len(report['results'])==13
    assert report['results'][-1]['window_truncated']
    assert report['results'][-1]['actual_window_sizes']==[8]
