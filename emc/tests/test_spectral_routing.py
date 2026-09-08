from dataclasses import replace
from copy import deepcopy
import pytest
import torch
from rayvan_emc.model import EMCConfig
from rayvan_emc.spectral_model import SpectralGeometricEMC
from rayvan_emc.spectral_routing import SpectralGeometricRouter, RunningStandardizer, weighted_resonance, routing_objective
from rayvan_emc.spectral_geometry import Descriptor


def config(**kw):
    base = EMCConfig(latent_dim=8, num_modules=3, modules_per_cycle=1, num_cycles=2, trajectory_steps=2,
        vocab_size=16, max_sequence_length=32, module_hidden_dim=16, attention_heads=2, integrator_heads=2,
        module_families=('gpt', 'recurrent', 'delta'), router_type='spectral_geometric',
        integrator_type='acceptance_gate', architecture_stage='n1_sequential', loss_free_balance_enabled=False,
        counterfactual_probe_fixed_rate=1., counterfactual_max_probes_per_forward=2)
    return replace(base, **kw)


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(11)
    torch.set_num_threads(1)


def test_resonance_and_basin_two():
    q = torch.tensor([1., 0., 0.])
    assert weighted_resonance(q, q, torch.ones(3)).item() == pytest.approx(1)
    assert weighted_resonance(q, q.roll(1), torch.ones(3)).item() == pytest.approx(0)
    router = SpectralGeometricRouter(config()).eval()
    d = router.describe(torch.randn(1, 16, 8))
    with torch.no_grad():
        router.centres.fill_(100)
        router.centres[1, 1].copy_(d.g[0])
    scores, detail = router.score_descriptor(d)
    assert scores.argmax(-1).item() == 1
    assert detail['winning_basin'][0, 1].item() == 1


def test_stats_train_only_checkpoint_and_freeze():
    stats = RunningStandardizer(3)
    x = torch.randn(10, 3)
    stats.update(x[:4]); stats.update(x[4:])
    torch.testing.assert_close(stats.mean.float(), x.mean(0))
    torch.testing.assert_close(stats(x), (x-x.mean(0))/x.std(0, unbiased=False))
    saved = deepcopy(stats.state_dict())
    stats.eval(); stats.update(x*100)
    assert all(torch.equal(saved[k], v) for k, v in stats.state_dict().items())
    loaded = RunningStandardizer(3); loaded.load_state_dict(saved)
    torch.testing.assert_close(loaded(x), stats(x))
    loaded.frozen.fill_(True); loaded.update(x)
    assert loaded.count == 10


def test_router_learning_detached_geometry_and_targets():
    router = SpectralGeometricRouter(config())
    x = torch.randn(8, 16, 8, requires_grad=True)
    losses = torch.randn(8, 3, requires_grad=True)
    d = router.describe(x)
    router.initialize(d, losses)
    before = router.centres.detach().clone()
    opt = torch.optim.Adam(router.parameters(), lr=.01)
    router.calibration_loss(x, 1, losses).backward(); opt.step()
    assert x.grad is None and losses.grad is None
    assert not torch.equal(before, router.centres)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in router.parameters())


def test_trajectory_full_sequence_causal_and_temporal(monkeypatch):
    model = SpectralGeometricEMC(config(geometry_temporal_delta_enabled=True)).eval()
    seen, previous = [], []
    original = model.router.route_one
    def observe(latent, **kw):
        seen.append(latent.detach().clone()); previous.append(kw.get('previous_descriptor'))
        return original(latent, **kw)
    monkeypatch.setattr(model.router, 'route_one', observe)
    tokens = torch.randint(0, 16, (2, 5))
    output = model.endpoint(tokens, return_trace=True)
    assert all(x.shape == (2, 5, 8) for x in seen)
    assert not torch.equal(seen[0], seen[1])
    assert previous[0] is None and previous[1] is not None
    assert output.trace[0].spectral_diagnostics['window_size'] == 5
    a = model(tokens)
    altered = tokens.clone(); altered[:, 3:] = (altered[:, 3:] + 1) % 16
    b = model(altered)
    torch.testing.assert_close(a[:, :3], b[:, :3])
    torch.testing.assert_close(model.endpoint(tokens), model.endpoint(tokens))


def test_probe_gradients_and_refractory():
    model = SpectralGeometricEMC(config())
    tokens, targets = torch.randint(0, 16, (2, 8)), torch.randint(0, 16, (2, 8))
    out = model.endpoint(tokens, counterfactual_targets=targets, return_trace=True)
    out.geometry_calibration_loss.backward()
    assert any(p.grad is not None for p in model.router.parameters())
    assert all(p.grad is None for e in model.emc_modules for p in e.parameters())
    assert model.token_embedding.weight.grad is None
    detail = out.trace[0].spectral_diagnostics['counterfactual']
    assert detail['candidate_losses'].shape == (2, 3)
    assert not detail['complementarity_active']
    model.eval()
    x = torch.randn(2, 8, 8)
    base = model.router.route_one(x)
    penalty = torch.zeros(2, 1, 3).scatter_(-1, base.selected_indices, 100)
    inhibited = model.router.route_one(x, refractory_penalty=penalty)
    torch.testing.assert_close(base.base_actions, inhibited.base_actions)
    assert (base.selected_indices != inhibited.selected_indices).all()


def test_registration_checkpoint():
    from rayvan_emc.research_config import ExperimentConfig, RoutingConfig, ModelConfig, research_schema
    from rayvan_emc.research_runner import _build_model
    from rayvan_emc.checkpoint import _model_type, _create_model
    cfg = ExperimentConfig(architecture='spectral_geometric_emc', experts={'gpt': 2},
                           model=ModelConfig(preset='custom', latent_dim=8, attention_heads=2, integrator_heads=2, module_hidden_dim=16),
                           routing=RoutingConfig(spectral_neighbourhood_size=8))
    model = _build_model(cfg, 16)
    assert isinstance(model, SpectralGeometricEMC)
    assert model.config.spectral_neighbourhood_size == 8
    restored = _create_model(_model_type(model), __import__('dataclasses').asdict(model.config))
    restored.load_state_dict(model.state_dict())
    assert any(x['id']=='spectral_geometric_emc' for x in research_schema()['architectures'])
    assert ExperimentConfig.from_dict(cfg.to_dict()).architecture == cfg.architecture
