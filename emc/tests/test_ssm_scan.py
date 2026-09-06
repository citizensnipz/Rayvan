from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from torch.nn import functional as F
from rayvan_emc.modules import StateSpaceEMCModule
from rayvan_emc.ssm_scan import affine_scan


def original_forward(module, latent):
    internal = module.input_adapter(module.input_norm(latent))
    convolved = module.causal_convolution(F.pad(internal.transpose(1, 2), (module.kernel_size - 1, 0))).transpose(1, 2)
    state = torch.zeros_like(convolved[:, 0])
    outputs = []
    for token in convolved.unbind(1):
        decay = torch.exp(-F.softplus(module.log_decay) * F.softplus(module.delta_projection(token)))
        state = decay * state + (1 - decay) * torch.tanh(module.input_projection(token))
        outputs.append(torch.sigmoid(module.gate_projection(token)) * state)
    return module.output_adapter(torch.stack(outputs, 1))


@pytest.mark.parametrize('length', [1, 7, 256, 257])
def test_scan_outputs_and_gradients_match_recurrence(length):
    torch.manual_seed(4)
    a = torch.rand(2, length, 5, dtype=torch.float64, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    weight = torch.randn_like(a)
    actual = affine_scan(a, b)
    reference = affine_scan(a, b, 'reference')
    torch.testing.assert_close(actual, reference)
    actual_grads = torch.autograd.grad((actual * weight).sum(), (a, b), retain_graph=True)
    expected_grads = torch.autograd.grad((reference * weight).sum(), (a, b))
    for x, y in zip(actual_grads, expected_grads):
        torch.testing.assert_close(x, y)


def test_scan_gradcheck_and_decay_extremes():
    a = torch.tensor([0., 1., 1e-8, .999999], dtype=torch.float64).reshape(1, 4, 1).requires_grad_()
    b = torch.randn_like(a, requires_grad=True)
    assert torch.autograd.gradcheck(affine_scan, (a, b))


def test_module_matches_original_parameters_input_gradients_and_causality():
    torch.set_num_threads(1)
    config = SimpleNamespace(latent_dim=8, resolved_state_space_dim=8, state_space_kernel_size=4, ssm_backend='parallel_scan')
    module = StateSpaceEMCModule(config).double()
    reference = deepcopy(module)
    x = torch.randn(2, 17, 8, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    output, expected = module(x), original_forward(reference, y)
    torch.testing.assert_close(output, expected)
    weight = torch.randn_like(output)
    (output * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    for p, q in zip(module.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    torch.testing.assert_close(output[:, :9], module(x[:, :9]))
    module.requires_grad_(False)
    assert torch.autograd.grad(module(x).square().sum(), x)[0].abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('length', [1, 7, 256, 257])
def test_fused_cuda_forward_backward_nondefault_stream(length):
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        pytest.skip('CUDA toolkit required to compile fused extension')
    with torch.cuda.stream(torch.cuda.Stream()):
        a = torch.rand(3, length, 65, device='cuda', requires_grad=True)
        b = torch.randn_like(a, requires_grad=True)
        g = torch.randn_like(a)
        output = affine_scan(a, b, 'cuda')
        reference = affine_scan(a, b, 'reference')
        torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-5)
        actual = torch.autograd.grad((output * g).sum(), (a, b), retain_graph=True)
        expected = torch.autograd.grad((reference * g).sum(), (a, b))
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, atol=3e-6, rtol=3e-5)
