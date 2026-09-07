"""Behavioral checks for expert-conditioned geometry and measured-effect replay."""
from copy import deepcopy
from dataclasses import replace
import math

import pytest
import torch
from torch.nn import functional as F

from rayvan_emc.checkpoint import load_model_checkpoint
from rayvan_emc.data import LanguageCorpus
from rayvan_emc.research_runner import _build_model, run_experiment
from rayvan_emc.training import TrainingConfig, train_model
from rayvan_emc.value_routing import CounterfactualValueEMC, ExpertConditionedGeometricRouter, StateBatch, center
from rayvan_emc.value_training import ValueOptimizers, frozen_snapshot, train_block
from test_value_routing import config, deterministic_cpu
from test_value_experiment import experiment


def settings(**kw):
    return config(value_head_type='expert_geometric', value_expert_training='frozen',
                  value_fixed_reference=True, value_replay_capacity=3, value_replay_batch_size=2, **kw)


def test_candidate_views_energy_and_first_step_gradients():
    model = CounterfactualValueEMC(settings())
    r = model.router
    h = torch.randn(3, 4, 8, requires_grad=True)
    z = r.need(h, 3)
    assert z.shape == (3, 2, 8)
    assert not torch.allclose(z[:, 0], z[:, 1])
    mu = F.normalize(r.value.prototypes, dim=-1)
    squared = ((z[:, :, None] - mu[None])**2).sum(-1)
    raw = r.value.bias - .05 * .25 * (torch.logsumexp(-squared/.25, -1) - math.log(4))
    torch.testing.assert_close(r(h, 3), center(raw))
    labels = torch.tensor([[3., 3.1], [3.1, 3.], [3., 3.05]], requires_grad=True)
    effects = torch.randn(3, 2, 16, requires_grad=True)
    loss, parts = r.objective(h, 3, labels, effects)
    torch.testing.assert_close(loss, parts['value_mse'] + .01*parts['soft_regret'] + .01*parts['effect_mse'])
    loss.backward()
    assert h.grad is None and labels.grad is None and effects.grad is None
    assert r.queries.grad.abs().sum() > 0
    assert r.input_projection.weight.grad.abs().sum() > 0
    assert r.encoder[0].weight.grad.abs().sum() > 0
    assert r.effect_projection.grad is None


def test_effect_target_is_integrated_update_and_suffix_is_unchanged():
    live = CounterfactualValueEMC(settings())
    snapshot = frozen_snapshot(live)
    h = torch.randn(2, 4, 8)
    batch = StateBatch(h, torch.tensor([1, 2]), 0)
    counts = [0, 0]
    hooks = [e.register_forward_hook(lambda m, a, o, i=i: counts.__setitem__(i, counts[i] + a[0].size(0)))
             for i, e in enumerate(snapshot.emc_modules)]
    labels, effects = snapshot.counterfactual_effects_and_losses(batch, live.router)
    for hook in hooks:
        hook.remove()
    assert sum(counts) == 2 * 2 * 3  # B * E * remaining; no extra first insertion.
    torch.testing.assert_close(labels, snapshot.counterfactual_losses(batch))
    for e in range(2):
        after = snapshot.apply_expert(h, torch.full((2,), e, dtype=torch.long))
        delta = after - h
        expected = torch.cat((delta.mean(1), delta[:, -1]), -1) @ live.router.effect_projection
        torch.testing.assert_close(effects[:, e], expected)
    other = CounterfactualValueEMC(settings(value_router_seed=19))
    torch.testing.assert_close(live.router.effect_projection, other.router.effect_projection)


def test_inference_runs_only_selected_experts_and_never_effect_decoder():
    model = CounterfactualValueEMC(settings()).eval()
    def forbidden(*_):
        raise AssertionError('effect decoder is training-only')
    hook = model.router.effect_decoder.register_forward_hook(forbidden)
    counts = []
    expert_hooks = [e.register_forward_hook(lambda m, args, o: counts.append(args[0].size(0))) for e in model.emc_modules]
    seen = []
    original = model.router.choose
    def choose(h, remaining, availability=None):
        seen.append((h.detach().clone(), remaining))
        return original(h, remaining, availability)
    model.router.choose = choose
    model.endpoint(torch.randint(16, (2, 4)))
    assert sum(counts) == 6
    assert [t for _, t in seen] == [3, 2, 1]
    assert not torch.equal(seen[0][0], seen[1][0])
    hook.remove()
    for hook in expert_hooks:
        hook.remove()


def test_replay_is_bounded_updates_only_router_and_roundtrips():
    model = CounterfactualValueEMC(settings())
    snapshot = frozen_snapshot(model)
    before = deepcopy(model.state_dict())
    opts = ValueOptimizers(model, .001, 0., 42)
    for step in range(1, 4):
        block = train_block(model, opts, torch.randint(16, (2, 4)), torch.randint(16, (2, 4)), step, 1., snapshot)
    assert len(opts.replay) == 3
    assert block['expert_conditioned']['replay_samples'] == 2
    assert block['expert_conditioned']['fresh_samples'] == 2
    assert block['expert_conditioned']['router_gradient_norm'] > 0
    assert block['expert_update_batches'] == [0, 0]
    assert all(not tensor.requires_grad and tensor.device.type == 'cpu'
               for item in opts.replay for tensor in (item[0], item[2], item[3]))
    for name, p in model.state_dict().items():
        if not name.startswith('router.'):
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
    restored = ValueOptimizers(model, .001, 0., 42)
    restored.load_state_dict(opts.state_dict())
    assert restored.replay_cursor == opts.replay_cursor
    torch.testing.assert_close(restored.replay_generator.get_state(), opts.replay_generator.get_state())
    for a, b in zip(opts.replay, restored.replay):
        assert a[1] == b[1]
        for i in (0, 2, 3):
            torch.testing.assert_close(a[i], b[i])


@pytest.mark.parametrize('change', [dict(value_fixed_reference=False), dict(value_expert_training='ordinary'),
                                  dict(value_fit_enabled=True), dict(value_target='immediate')])
def test_stale_or_missing_effect_label_modes_rejected(change):
    with pytest.raises(ValueError, match='Expert-conditioned'):
        CounterfactualValueEMC(replace(settings(), **change))


def test_checkpoint_resume_preserves_original_different_reference_router(tmp_path):
    corpus = LanguageCorpus.from_texts(['ababacabbc\n'*10], ['bacabbcaba\n'*10])
    c = experiment()
    source = _build_model(c, corpus.tokenizer.vocab_size)
    target_config = replace(source.config, value_head_type='expert_geometric', value_expert_training='frozen',
                            value_fixed_reference=True, value_replay_capacity=8, value_replay_batch_size=4)
    initial = CounterfactualValueEMC(target_config)
    object.__setattr__(initial, '_value_reference_router', deepcopy(source.router))
    from dataclasses import asdict
    object.__setattr__(initial, '_value_reference_config', asdict(source.config))
    full, part = deepcopy(initial), deepcopy(initial)
    train = TrainingConfig(steps=3, batch_size=2, sequence_length=4, learning_rate=.001,
                           evaluation_interval=1, evaluation_batches=1, seed=42, router_balance_coefficient=0.,
                           checkpoint_directory=str(tmp_path/'full'), retain_milestone_checkpoints=False)
    train_model(full, corpus, train, print_progress=False)
    first = train_model(part, corpus, replace(train, steps=1, checkpoint_directory=str(tmp_path/'part')), print_progress=False)
    loaded = load_model_checkpoint(first.latest_checkpoint).model
    train_model(loaded, corpus, replace(train, resume_from=first.latest_checkpoint, checkpoint_directory=str(tmp_path/'resume')), print_progress=False)
    for name, p in full.state_dict().items():
        torch.testing.assert_close(p, loaded.state_dict()[name], rtol=0, atol=0)


def test_console_configuration_warmstart_and_metrics(tmp_path):
    c = experiment()
    trained = run_experiment(c, runs_directory=tmp_path, run_id='source')
    path = trained['training_result']['best_checkpoint']
    variant = replace(c, routing=replace(c.routing, value_head_type='expert_geometric', routing_geometry_dim=32,
                      value_checkpoint_path=path, value_expert_training='frozen', value_reset_router=True,
                      value_fixed_reference=True, value_probe_budget=2))
    result = run_experiment(variant, runs_directory=tmp_path, run_id='expert-query')
    assert result['status'] == 'completed'
    v = result['value_routing']
    assert v['expert_conditioned']['replay_size'] == 4
    assert v['held_out']['effect_mse'] >= 0
    assert v['held_out']['shuffled_state_regret'] >= 0
    assert v['expert_update_batches'] == [0, 0]
    saved = load_model_checkpoint(result['training_result']['latest_checkpoint']).model
    assert isinstance(saved.router, ExpertConditionedGeometricRouter)
    old = load_model_checkpoint(path).model.state_dict()
    for name, tensor in saved.state_dict().items():
        if not name.startswith('router.'):
            torch.testing.assert_close(tensor, old[name], rtol=0, atol=0)


def test_learns_state_dependent_competence_on_controlled_signal():
    # Known answer: opposite experts help on opposite signs. Generalizes to new
    # nuisance features; no claim this signal exists in the real checkpoint.
    r = CounterfactualValueEMC(settings()).router
    rng = torch.Generator().manual_seed(29)
    h = torch.randn(96, 4, 8, generator=rng) * .05
    sign = torch.where(torch.arange(96) % 2 == 0, 1., -1.)
    h[:, :, 0] = sign[:, None]
    labels = torch.stack((3. - .04*sign, 3. + .04*sign), -1)
    effects = torch.zeros(96, 2, 16)
    effects[:, 0, 0] = sign*.1
    effects[:, 1, 0] = -sign*.1
    optimizer = torch.optim.Adam(r.parameters(), lr=.003)
    for _ in range(160):
        optimizer.zero_grad(set_to_none=True)
        objective, _ = r.objective(h[:64], 3, labels[:64], effects[:64])
        objective.backward()
        optimizer.step()
    chosen = r.choose(h[64:], 3)
    loss = labels[64:].gather(1, chosen[:, None]).mean()
    assert float(loss - labels[64:].min(-1).values.mean()) < .005


def test_decision_only_gradient_matches_expected_regret_and_ignores_effect_targets():
    r = CounterfactualValueEMC(settings(value_cost_mse_weight=0., value_effect_weight=0.,
                                       value_geometry_regret_weight=1.)).router
    h = torch.randn(3, 4, 8, requires_grad=True)
    labels = torch.tensor([[3., 3.1], [3.1, 3.], [3., 3.05]], requires_grad=True)
    effects = torch.randn(3, 2, 16, requires_grad=True)
    actual, parts = r.objective(h, 3, labels, effects)
    prediction = r(h.detach(), 3).float()
    truth = labels.detach()
    expected = (torch.softmax(-prediction/.05, -1) * (truth-truth.min(-1, keepdim=True).values)).sum(-1).mean()
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, r.queries, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, r.queries)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    other, _ = r.objective(h, 3, labels, effects * 100)
    torch.testing.assert_close(other, actual)
    actual.backward()
    assert all(p.grad is None for p in r.effect_decoder.parameters())
    assert h.grad is None and labels.grad is None and effects.grad is None
    assert parts['value_mse'] > 0  # Diagnostics still computed, not optimized.


def test_decision_control_config_validation_and_roundtrip():
    from rayvan_emc.research_config import ExperimentConfig
    c = experiment()
    c = replace(c, routing=replace(c.routing, value_head_type='expert_geometric', value_expert_training='frozen',
                value_cost_mse_weight=0., value_effect_weight=0., value_geometry_regret_weight=1.))
    assert ExperimentConfig.from_dict(c.to_dict()).routing.value_cost_mse_weight == 0.
    for weight in (-1., float('nan'), 1.1):
        with pytest.raises(ValueError, match='Cost MSE weight'):
            CounterfactualValueEMC(settings(value_cost_mse_weight=weight))
    with pytest.raises(ValueError, match='At least one'):
        CounterfactualValueEMC(settings(value_cost_mse_weight=0., value_effect_weight=0., value_geometry_regret_weight=0.))


def test_audit_reports_known_top_two_margin():
    from rayvan_emc.value_training import audit_values
    model = CounterfactualValueEMC(config(value_expert_training='frozen'))
    with torch.no_grad():
        model.router.value.bias.copy_(torch.tensor([0., .0005]))
    audit = audit_values(model, torch.randint(16, (2, 4)), torch.randint(16, (2, 4)))
    assert audit['predicted_margin_mean'] == pytest.approx(.0005)
    assert audit['predicted_margin_min'] == pytest.approx(.0005)
    assert audit['predicted_margin_below_1e_3_fraction'] == 1.
