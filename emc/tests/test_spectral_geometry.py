from dataclasses import replace
import math
import pytest
import torch
from rayvan_emc.spectral_config import SpectralConfig
from rayvan_emc.spectral_geometry import extract_descriptor, feature_names, decompose_graph


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(81)
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_invariances():
    c = SpectralConfig()
    x = torch.randn(2, 16, 9)
    base = extract_descriptor(x, c)
    rotation = torch.linalg.qr(torch.randn(9, 9)).Q
    for y in (x + torch.randn(1, 1, 9), x @ rotation, x[:, torch.randperm(16)]):
        result = extract_descriptor(y, c)
        torch.testing.assert_close(result.q, base.q, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(result.g, base.g, atol=2e-4, rtol=2e-4)
    result = extract_descriptor(-3 * x, c)
    torch.testing.assert_close(result.q, base.q, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(result.g[:, 1:], base.g[:, 1:], atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(result.g[:, 0] - base.g[:, 0], torch.full((2,), math.log(3)))


@pytest.mark.parametrize('length', [1, 2, 8, 16, 32])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_degenerate(length, dtype):
    c = SpectralConfig(spectral_neighbourhood_size=length)
    for x in (torch.ones(2, length, 4), torch.randn(2, length, 4) * 1e-12):
        d = extract_descriptor(x.to(dtype), c)
        assert torch.isfinite(d.q).all() and torch.isfinite(d.g).all()
        assert (d.q >= 0).all()
        assert d.q.shape[-1] == len(feature_names(c)[0])
        assert d.g.shape[-1] == len(feature_names(c)[1])
        for name in ('hks', 'wks'):
            assert (d.diagnostics[name] >= 0).all()


def test_dimension_proxy():
    c = SpectralConfig(spectral_neighbourhood_size=32)
    values = []
    # Orthonormal centred columns have equal energy in each occupied dimension.
    x = torch.randn(32, 6)
    x = torch.linalg.qr(x - x.mean(0)).Q
    for rank in (1, 2, 4):
        cloud = torch.zeros(1, 32, 6)
        cloud[0, :, :rank] = x[:, :rank]
        d = extract_descriptor(cloud, c)
        values.append(d.g[0, 1+c.geometry_pca_components].item())
    assert values == pytest.approx([1, 2, 4], abs=0.01)


def test_spectrum_disconnected_and_repeated():
    c = SpectralConfig(spectral_modes=1)
    a = torch.zeros(1, 6, 6)
    a[0, :3, :3] = 1 - torch.eye(3)
    a[0, 3:, 3:] = 1 - torch.eye(3)
    values, _, selected = decompose_graph(a, c)
    assert (values.diff() >= 0).all() and values.min() >= 0 and values.max() <= 2
    assert (values <= c.spectral_zero_threshold).sum() == 2
    assert selected.sum() == 4  # the full tied eigenspace
    x = torch.eye(8)[None]
    base = extract_descriptor(x, c)
    moved = extract_descriptor(x[:, torch.randperm(8)], c)
    torch.testing.assert_close(base.q, moved.q, atol=2e-4, rtol=2e-4)


def test_detachment_and_config_validation():
    x = torch.randn(1, 8, 4, requires_grad=True)
    d = extract_descriptor(x, SpectralConfig())
    assert not d.q.requires_grad and not d.g.requires_grad
    for kw in ({'spectral_modes': 0}, {'spectral_eps': float('nan')},
               {'spectral_router_temperature': 0}, {'spectral_graph_mode': 'semantic'}):
        with pytest.raises(ValueError):
            SpectralConfig(**kw)
