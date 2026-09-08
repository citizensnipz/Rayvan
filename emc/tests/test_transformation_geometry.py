import math
import pytest
import torch
from rayvan_emc.spectral_config import SpectralConfig
from rayvan_emc.transformation_geometry import transformation_signature, complementarity


@pytest.fixture(autouse=True)
def setup():
    torch.manual_seed(10)
    torch.set_num_threads(1)


def values(x, y, c=None, active=False):
    result = transformation_signature(x, y, c or SpectralConfig(transformation_rank=3), active=active)
    return {n: result.signature[:, i] for i, n in enumerate(result.names)}


def test_identity_scale_rotation_and_shear():
    x = torch.randn(2, 16, 3) * torch.tensor([1., 2., 4.])
    identity = values(x, x)
    for name in ('log_stretch_0', 'log_volume', 'anisotropic_deformation', 'tangent_escape', 'spectral_delta_norm'):
        assert identity[name].abs().max() < 1e-4
    scale = values(x, x * 2)
    torch.testing.assert_close(scale['log_volume'], torch.full((2,), 3*math.log(2)), atol=1e-4, rtol=1e-4)
    rotation = torch.tensor([[0., -1, 0], [1., 0, 0], [0., 0, 1]])
    rotated = values(x, x @ rotation)
    assert rotated['log_stretch_0'].abs().max() < 1e-4
    assert rotated['rotation_identity_distance'].min() > .5
    shear = torch.tensor([[1., 1., 0], [0., 1., 0], [0., 0., 1.]])
    assert values(x, x @ shear)['shear_like_off_axis_stretch'].min() > .01


def test_rank_deficiency_and_escape():
    for x in (torch.ones(2, 16, 3), torch.randn(2, 1, 3), torch.randn(2, 2, 3)):
        result = values(x, x)
        assert all(torch.isfinite(v).all() for v in result.values())
    x = torch.randn(1, 16, 3)
    x[..., 2] = 0
    y = x.clone(); y[..., 2] = x[..., 0].square()
    assert values(x, y)['tangent_escape'].item() > .1


def test_observational_and_active_gradients():
    x = torch.randn(2, 16, 3)
    y = (x @ torch.diag(torch.tensor([.5, 1.5, 3.]))).requires_grad_()
    c = SpectralConfig(transformation_rank=3)
    assert not transformation_signature(x, y, c).signature.requires_grad
    active = transformation_signature(x, y, c, active=True)
    active.signature.square().mean().backward()
    assert torch.isfinite(y.grad).all() and y.grad.abs().sum() > 0
    labels = torch.tensor([[.8, .2]], requires_grad=True)
    signatures = torch.ones(1, 2, 5, requires_grad=True)
    loss, _, _ = complementarity(signatures, labels, .5)
    loss.backward()
    assert labels.grad is None
    assert loss.item() == pytest.approx(.8*.2*.25)
