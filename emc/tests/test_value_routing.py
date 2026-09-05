from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from types import MethodType

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from rayvan_emc.model import EMCConfig, EMCOutput
from rayvan_emc.value_routing import CounterfactualValueEMC, StateBatch, ValueNexusRouter, center
from rayvan_emc.value_training import (
    ValueOptimizers, audit_values, curriculum_probabilities, frozen_snapshot,
    insertion_loss, sample_probe_pairs, train_block,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(previous)


def config(**updates):
    base = EMCConfig(latent_dim=8, num_modules=2, modules_per_cycle=1, num_cycles=3,
                     trajectory_steps=3, vocab_size=16, max_sequence_length=8, module_hidden_dim=16,
                     attention_heads=2, integrator_heads=2, delta_heads=2, routing_geometry_dim=8,
                     module_families=('delta', 'delta'), architecture_stage='n1_sequential',
                     router_type='counterfactual_value', integrator_type='identity_free_gate',
                     refractory_enabled=False, loss_free_balance_enabled=False, switch_cost=0., persistence_bonus=0.,
                     value_development_batch_size=4, value_probe_rate=1., value_probe_budget=2)
    return replace(base, **updates)


def test_centered_values_and_induced_metric_match_equations():
    router = ValueNexusRouter(config())
    x = torch.randn(3, 4, 8)
    z = router.need(x, 3)
    raw = z @ router.value.weight.T + router.value.bias
    expected = raw - raw.mean(-1, keepdim=True)
    torch.testing.assert_close(router(x, 3), expected)
    W = router.value.weight
    P = torch.eye(2) - torch.ones(2, 2) / 2
    torch.testing.assert_close(router.metric(), W.T @ P @ W / 2)
    delta = z[0] - z[1]
    torch.testing.assert_close(delta @ router.metric() @ delta, ((expected[0] - expected[1])**2).mean())


def test_calibration_preserves_magnitudes_and_detaches_state_and_labels():
    router = ValueNexusRouter(config())
    state = torch.randn(2, 4, 8, requires_grad=True)
    labels = torch.tensor([[1., 3.], [4., 4.]], requires_grad=True)
    actual = router.calibration_loss(state, 2, labels)
    torch.testing.assert_close(actual, ((router(state, 2) - center(labels))**2).mean())
    actual.backward()
    assert state.grad is None and labels.grad is None
    assert router.encoder[0].weight.grad.abs().sum() > 0
    assert router.value.weight.grad.abs().sum() > 0
    torch.testing.assert_close(center(labels)[1], torch.zeros(2))


def test_need_is_order_sensitive_and_horizon_sensitive():
    router = ValueNexusRouter(config())
    x = torch.randn(2, 4, 8)
    # Keep endpoint fixed and permute only earlier positions, preserving the mean.
    permuted = x[:, [2, 0, 1, 3]]
    torch.testing.assert_close(x.mean(1), permuted.mean(1))
    assert not torch.allclose(router.need(x, 3), router.need(permuted, 3))
    assert not torch.allclose(router.need(x, 3), router.need(x, 1))


def test_gate_has_no_identity_and_starts_at_half():
    model = CounterfactualValueEMC(config())
    x = torch.randn(2, 4, 8)
    proposal = torch.randn(2, 4, 1, 8)
    assert not hasattr(model.integrator, 'expert_identity')
    y = model.integrator(x, proposal, torch.ones(2, 4, 1))
    torch.testing.assert_close(y, x + .5 * proposal.squeeze(2))


def test_sequential_execution_recomputes_and_allows_repeated_expert():
    model = CounterfactualValueEMC(config()).eval()
    with torch.no_grad():
        model.router.value.weight.zero_()
        model.router.value.bias.copy_(torch.tensor([-2., 2.]))
    calls, seen = [0, 0], []
    hooks = [e.register_forward_hook(lambda _m, args, _o, i=i: calls.__setitem__(i, calls[i] + args[0].size(0)))
             for i, e in enumerate(model.emc_modules)]
    original = model.router.choose
    def choose(x, remaining, availability=None):
        seen.append((x.detach().clone(), remaining))
        return original(x, remaining, availability)
    model.router.choose = choose
    result = model.endpoint(torch.randint(16, (3, 4)), return_trace=True)
    for hook in hooks:
        hook.remove()
    assert calls == [9, 0]  # Exactly B*T, not E*B*T.
    assert [r for _, r in seen] == [3, 2, 1]
    assert not torch.equal(seen[0][0], seen[1][0])
    assert all((t.selected_indices == 0).all() for t in result.trace)


@pytest.mark.parametrize('family', ['delta', 'gpt', 'ssm', 'recurrent'])
def test_public_logits_are_causal_even_for_forced_experts(family):
    model = CounterfactualValueEMC(config(module_families=(family, family))).eval()
    a = torch.tensor([[1, 2, 3, 4]])
    b = torch.tensor([[1, 2, 9, 8]])
    with torch.no_grad():
        out_a = model(a, diagnostic_forced_modules=torch.tensor([0]))
        out_b = model(b, diagnostic_forced_modules=torch.tensor([0]))
        torch.testing.assert_close(out_a[:, :2], out_b[:, :2])
        torch.testing.assert_close(out_a[:, 1:2], model.endpoint(a[:, :2], diagnostic_forced_modules=torch.tensor([0])))
        torch.testing.assert_close(model(a)[:, :2], model(b)[:, :2])


class AffineProposal(nn.Module):
    family = 'test'
    def __init__(self, scale, offset):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.offset = nn.Parameter(torch.tensor(float(offset)))
    def forward(self, state):
        return self.scale * state + self.offset


def scalar_readout(self, state):
    x = state[:, -1, 0]
    return torch.stack((-x, x), dim=-1)


def test_delayed_benefit_requires_branch_specific_rerouted_suffix():
    model = CounterfactualValueEMC(config(trajectory_steps=2, num_cycles=2))
    model.emc_modules = nn.ModuleList([AffineProposal(0, 2), AffineProposal(-6, -.2)])
    model.read_endpoint = MethodType(scalar_readout, model)
    choices = []
    def choose(state, remaining, availability=None):
        result = (state[:, -1, 0] > .5).long()
        choices.extend(result.tolist())
        return result
    model.router.choose = choose
    model.requires_grad_(False).eval()
    batch = StateBatch(torch.zeros(1, 2, 8), torch.tensor([0]), 0)
    immediate = model.counterfactual_losses(batch, target='immediate')
    downstream = model.counterfactual_losses(batch, target='suffix')
    assert immediate.argmin(-1).item() == 1
    assert downstream.argmin(-1).item() == 0
    assert choices == [1, 0]  # Different continuation after each candidate.
    expected = F.cross_entropy(torch.tensor([[2.1, -2.1]]), torch.tensor([0]))
    torch.testing.assert_close(downstream[0, 0], expected)


def test_live_insertion_gradient_crosses_frozen_suffix_including_same_expert():
    model = CounterfactualValueEMC(config(trajectory_steps=2, num_cycles=2))
    model.emc_modules = nn.ModuleList([AffineProposal(2, 0), AffineProposal(1, 0)])
    model.read_endpoint = MethodType(scalar_readout, model)
    with torch.no_grad():
        model.router.value.weight.zero_()
        model.router.value.bias.copy_(torch.tensor([-2., 2.]))
    snapshot = frozen_snapshot(model)
    state = torch.ones(1, 2, 8, requires_grad=True)
    loss = insertion_loss(model.emc_modules[0], snapshot, StateBatch(state, torch.tensor([1]), 0))
    loss.backward()
    # x1=1+0.5*p, x2=(1+0.5*2)*x1. Only first p is live.
    p = torch.tensor(2., requires_grad=True)
    final_x = (1 + .5 * p) * 2
    reference = F.cross_entropy(torch.stack((-final_x, final_x)).view(1,2), torch.tensor([1]))
    reference.backward()
    torch.testing.assert_close(model.emc_modules[0].scale.grad, p.grad)
    assert state.grad is None
    assert all(p.grad is None for p in snapshot.parameters())
    assert all(p.grad is None for p in model.emc_modules[1].parameters())


def test_curriculum_is_normalized_common_specialist_mixture():
    model = frozen_snapshot(CounterfactualValueEMC(config()))
    states = model.collect_states(torch.randint(16,(3,4)), torch.tensor([1,2,3]),
                                  generator=torch.Generator().manual_seed(3), exploration=.5)
    rho, temperature = .5, .25
    q = curriculum_probabilities(model, states, rho, temperature)
    r = torch.cat([(-model.router(s.latent, 3-s.depth)/temperature).softmax(-1) for s in states])
    torch.testing.assert_close(q, rho / 9 + (1-rho) * r / r.sum(0))
    torch.testing.assert_close(q.sum(0), torch.ones(2))
    assert q.min() >= rho / 9


def test_probe_sampling_is_reproducible_and_not_front_or_first_depth_biased():
    def collect():
        generator = torch.Generator().manual_seed(42)
        return [sample_probe_pairs(8, 3, 1., 1, generator)[0] for _ in range(1200)]
    samples = collect()
    assert samples == collect()
    requests = torch.bincount(torch.tensor([r for r,_ in samples]), minlength=8)
    depths = torch.bincount(torch.tensor([d for _,d in samples]), minlength=3)
    assert requests.min() > 100 and depths.min() > 320
    assert not sample_probe_pairs(8, 3, 0., 1, torch.Generator())


def test_controlled_updates_are_equal_despite_complete_traffic_monopoly():
    model = CounterfactualValueEMC(config(value_exploration_rate=0., value_probe_rate=0.))
    with torch.no_grad():
        model.router.value.weight.zero_()
        model.router.value.bias.copy_(torch.tensor([-100., 100.]))
    before = [[p.detach().clone() for p in expert.parameters()] for expert in model.emc_modules]
    opts = ValueOptimizers(model, .001, 0., 42)
    output = train_block(model, opts, torch.randint(16,(3,4)), torch.randint(16,(3,4)), 1, 1.)
    assert output['expert_update_batches'] == [1,1]
    assert output['expert_training_items'] == [4,4]
    assert all((t.selected_indices == 0).all() for t in output['output'].trace)
    for expert, old in zip(model.emc_modules, before):
        assert any(not torch.equal(p, q) for p,q in zip(expert.parameters(), old))
    assert all(p.grad is None for p in model.parameters())
    assert opts.costs['supervised_targets'] == 3
    assert opts.costs['input_tokens'] == 12


def test_controlled_shared_updates_do_not_update_experts_between_practice_blocks():
    model = CounterfactualValueEMC(config(value_development_interval=2))
    before = [p.detach().clone() for p in model.emc_modules.parameters()]
    opts = ValueOptimizers(model, .001, .01, 42)
    train_block(model, opts, torch.randint(16,(2,4)), torch.randint(16,(2,4)), 1, 1.)
    for p,q in zip(model.emc_modules.parameters(), before):
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert opts.updates == [0,0]


def test_frozen_mode_changes_only_router_and_audit_is_read_only():
    model = CounterfactualValueEMC(config(value_expert_training='frozen'))
    before = deepcopy(model.state_dict())
    opts = ValueOptimizers(model,.001,.01,42)
    x,y = torch.randint(16,(2,4)),torch.randint(16,(2,4))
    train_block(model,opts,x,y,1,1.)
    for name,p in model.state_dict().items():
        if not name.startswith('router.'):
            torch.testing.assert_close(p,before[name],rtol=0,atol=0)
    before = deepcopy(model.state_dict())
    audit = audit_values(model,x,y)
    assert audit['mean_regret'] >= 0
    assert len(audit['by_depth']) == 3
    for name,p in model.state_dict().items():
        torch.testing.assert_close(p,before[name],rtol=0,atol=0)


def test_availability_and_empty_prefix_fail_explicitly():
    model = CounterfactualValueEMC(config())
    with pytest.raises(ValueError, match='available'):
        model.endpoint(torch.ones(2,4,dtype=torch.long), availability_mask=torch.tensor([False,False]))
    with pytest.raises(ValueError, match='prefix'):
        model.endpoint(torch.empty(2,0,dtype=torch.long))


def test_permuting_experts_and_value_rows_preserves_computation():
    model = CounterfactualValueEMC(config()).eval()
    permuted = deepcopy(model)
    permuted.emc_modules = nn.ModuleList([permuted.emc_modules[1], permuted.emc_modules[0]])
    with torch.no_grad():
        permuted.router.value.weight.copy_(model.router.value.weight[[1, 0]])
        permuted.router.value.bias.copy_(model.router.value.bias[[1, 0]])
        x = torch.randint(16, (4, 5))
        original = model.endpoint(x, return_trace=True)
        changed = permuted.endpoint(x, return_trace=True)
    torch.testing.assert_close(original.logits, changed.logits)
    for a, b in zip(original.trace, changed.trace):
        torch.testing.assert_close(a.selected_indices, 1-b.selected_indices)


@pytest.mark.parametrize('target', ['suffix', 'immediate'])
@pytest.mark.parametrize('mode', ['controlled', 'ordinary'])
def test_mixed_population_factorial_updates_are_finite(target, mode):
    model = CounterfactualValueEMC(config(num_modules=4, module_families=('gpt','ssm','recurrent','delta'),
                                        value_target=target, value_expert_training=mode))
    opts = ValueOptimizers(model,.001,0.,42)
    block = train_block(model,opts,torch.randint(16,(2,4)),torch.randint(16,(2,4)),1,1.)
    assert math.isfinite(block['loss']) and math.isfinite(block['calibration_mse'])
    if mode == 'controlled':
        assert block['expert_update_batches'] == [1]*4
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('precision', ['fp32', 'bf16'])
def test_cuda_mixed_family_development_and_router_gradients(precision):
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        pytest.skip('CUDA device has no bf16 support')
    model = CounterfactualValueEMC(config(num_modules=4, module_families=('gpt','ssm','recurrent','delta'))).cuda()
    opts = ValueOptimizers(model,.001,0.,42)
    before = model.router.value.weight.detach().clone()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=precision=='bf16'):
        block = train_block(model,opts,torch.randint(16,(2,4),device='cuda'),
                            torch.randint(16,(2,4),device='cuda'),1,1.)
    assert block['expert_update_batches']==[1]*4
    assert all(value > 0 and math.isfinite(value) for value in block['expert_parameter_change_norms'])
    assert not torch.equal(before, model.router.value.weight)
    assert math.isfinite(block['calibration_mse'])


def test_specialist_distribution_survives_extreme_incumbent_advantage():
    model = CounterfactualValueEMC(config())
    with torch.no_grad():
        model.router.value.weight.zero_()
        model.router.value.bias.copy_(torch.tensor([-1000.,1000.]))
    snapshot = frozen_snapshot(model)
    states = [StateBatch(torch.randn(4,2,8),torch.ones(4,dtype=torch.long),0)]
    q = curriculum_probabilities(snapshot,states,0.,.01)
    torch.testing.assert_close(q.sum(0), torch.ones(2))
    torch.testing.assert_close(q,torch.full((4,2),.25))
