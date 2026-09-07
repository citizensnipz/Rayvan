from dataclasses import replace
import math

import pytest
import torch
from torch.nn import functional as F

from rayvan_emc.value_routing import CounterfactualValueEMC, pairwise_advantage_loss
from rayvan_emc.checkpoint import load_model_checkpoint
from rayvan_emc.research_config import ExperimentConfig
from rayvan_emc.research_runner import run_experiment
from test_value_routing import config, deterministic_cpu
from test_value_experiment import experiment


def pair_config(**kw):
    return config(value_head_type='expert_geometric', value_expert_training='frozen',
                  value_fixed_reference=True, value_cost_mse_weight=0., value_effect_weight=0.,
                  value_geometry_regret_weight=0., value_pairwise_weight=1., **kw)


def test_gap_weighted_pair_equation_sign_and_detached_labels():
    q = torch.tensor([[.3, -.2, .1]], requires_grad=True)
    y = torch.tensor([[1., 2., 4.]], requires_grad=True)
    actual = pairwise_advantage_loss(q, y, .05, .001)
    expected = (1*F.softplus((q[0,0]-q[0,1])/.05) +
                3*F.softplus((q[0,0]-q[0,2])/.05) +
                2*F.softplus((q[0,1]-q[0,2])/.05))/3
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert y.grad is None
    assert q.grad[0,0] > 0  # Gradient descent lowers the best expert's cost.
    assert q.grad[0,2] < 0  # Raises the worst expert's cost.
    torch.testing.assert_close(pairwise_advantage_loss(q+5, y+10, .05, .001), actual)
    perm = [2,0,1]
    torch.testing.assert_close(pairwise_advantage_loss(q[:,perm], y[:,perm], .05, .001), actual)


def test_confidently_wrong_comparison_keeps_corrective_gradient():
    q = torch.tensor([[10., -10.]], requires_grad=True)
    y = torch.tensor([[1., 2.]])  # Expert zero is better, despite negligible policy probability.
    pairwise_advantage_loss(q, y, .05, .001).backward()
    torch.testing.assert_close(q.grad, torch.tensor([[20., -20.]]))
    assert torch.isfinite(q.grad).all()


@pytest.mark.parametrize('q,y', [([[1.,-1.]], [[2.,2.]]), ([[1.,-1.]], [[2.,2.0001]]), ([[1.]], [[2.]])])
def test_ties_and_single_candidate_have_zero_loss_and_gradient(q,y):
    q = torch.tensor(q, requires_grad=True)
    loss = pairwise_advantage_loss(q, torch.tensor(y), .05, .001)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(q.grad) == 0


def test_pairwise_only_trains_encoder_without_auxiliary_gradients():
    router = CounterfactualValueEMC(pair_config()).router
    h = torch.randn(3,4,8,requires_grad=True)
    y = torch.tensor([[1.,2.],[2.,1.],[1.,2.]],requires_grad=True)
    effects = torch.randn(3,2,16,requires_grad=True)
    objective, parts = router.objective(h,3,y,effects)
    torch.testing.assert_close(objective, parts['pairwise_loss'])
    objective.backward()
    assert h.grad is None and y.grad is None and effects.grad is None
    assert all(p.grad is None for p in router.effect_decoder.parameters())
    assert router.queries.grad.abs().sum() > 0
    assert router.input_projection.weight.grad.abs().sum() > 0


def test_pairwise_learns_known_conditional_signal():
    r = CounterfactualValueEMC(pair_config()).router
    g = torch.Generator().manual_seed(29)
    h = torch.randn(96,4,8,generator=g)*.05
    sign = torch.where(torch.arange(96)%2 == 0, 1., -1.)
    h[:,:,0] = sign[:,None]
    y = torch.stack((3.-.04*sign,3.+.04*sign),-1)
    optimizer = torch.optim.Adam(r.parameters(),lr=.003)
    for _ in range(160):
        optimizer.zero_grad(set_to_none=True)
        loss,_ = r.objective(h[:64],3,y[:64],torch.zeros(64,2,16))
        loss.backward()
        optimizer.step()
    selected = r.choose(h[64:],3)
    assert (selected == y[64:].argmin(-1)).float().mean() > .95


@pytest.mark.parametrize('field,value', [('value_pairwise_weight',-1),('value_pairwise_temperature',0),
                                       ('value_pairwise_temperature',math.inf),('value_pairwise_tie_tolerance',-.1)])
def test_invalid_pairwise_settings_rejected(field,value):
    with pytest.raises(ValueError,match='Pairwise'):
        CounterfactualValueEMC(replace(pair_config(),**{field:value}))


def test_mixed_checkpoint_console_roundtrip_and_pairwise_telemetry(tmp_path):
    source = replace(experiment(),experts={'gpt':1,'ssm':1,'recurrent':1,'delta':1})
    trained = run_experiment(source,runs_directory=tmp_path,run_id='mixed-source')
    path = trained['training_result']['latest_checkpoint']
    c = replace(source,routing=replace(source.routing,value_checkpoint_path=path,
        value_head_type='expert_geometric',value_expert_training='frozen',value_fixed_reference=True,
        value_reset_router=True,value_pairwise_weight=1.,value_cost_mse_weight=0.,
        value_effect_weight=0.,value_geometry_regret_weight=0.,value_probe_budget=2))
    assert ExperimentConfig.from_dict(c.to_dict()).routing.value_pairwise_weight == 1.
    result = run_experiment(c,runs_directory=tmp_path,run_id='mixed-pairwise')
    v = result['value_routing']
    assert v['router_objective'] == pytest.approx(v['expert_conditioned']['pairwise_loss'])
    assert v['held_out']['pairwise_loss'] >= 0
    assert v['expert_update_batches'] == [0]*4
    old = load_model_checkpoint(path).model.state_dict()
    model = load_model_checkpoint(result['training_result']['latest_checkpoint']).model
    assert model.router.pairwise_weight == 1.
    for name,tensor in model.state_dict().items():
        if not name.startswith('router.'):
            torch.testing.assert_close(tensor,old[name],rtol=0,atol=0)
