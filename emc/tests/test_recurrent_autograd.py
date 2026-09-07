from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from rayvan_emc.modules import RecurrentEMCModule
from rayvan_emc.value_routing import CounterfactualValueEMC, StateBatch
from rayvan_emc.value_training import frozen_snapshot, insertion_loss
from test_value_routing import config, deterministic_cpu


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_frozen_eval_gru_preserves_input_gradient_and_backend_flags(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    cfg = SimpleNamespace(latent_dim=8, resolved_recurrent_dim=8)
    module = RecurrentEMCModule(cfg).to(device).eval().requires_grad_(False)
    before = deepcopy(module.state_dict())
    x = torch.randn(2, 7, 8, device=device, requires_grad=True)
    observed = []
    hook = module.recurrent.register_forward_pre_hook(lambda m, args: observed.append(torch.backends.cudnn.enabled))
    with torch.backends.cudnn.flags(enabled=True):
        output = module(x)
        assert observed[-1] is False
        assert torch.backends.cudnn.enabled
        output.square().sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        with torch.no_grad():
            reference = module(x)
        assert observed[-1] is True
        torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-5)
        module.train()
        module(x)
        assert observed[-1] is True
    hook.remove()
    assert all(p.grad is None for p in module.parameters())
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_expert_insertion_backpropagates_through_forced_frozen_gru_suffix(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    model = CounterfactualValueEMC(config(module_families=('gpt', 'recurrent'))).to(device)
    with torch.no_grad():
        model.router.value.bias.copy_(torch.tensor([10., -10.], device=device))
    snapshot = frozen_snapshot(model)
    batch = StateBatch(torch.randn(2, 5, 8, device=device), torch.tensor([1, 2], device=device), 0)
    with torch.backends.cudnn.flags(enabled=True):
        loss = insertion_loss(model.emc_modules[0], snapshot, batch)
        loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.emc_modules[0].parameters())
    assert all(p.grad is None and not p.requires_grad for p in snapshot.parameters())
    assert not snapshot.training and not snapshot.emc_modules[1].recurrent.training
