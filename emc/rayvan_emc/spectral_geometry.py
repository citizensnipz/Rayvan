"""Small graph spectral signatures without a learned coordinate projection.

Boundary eigenspaces and kth-distance ties are retained whole for relabeling
invariance. Linear algebra runs outside autocast in float32, never fp16.
"""
from dataclasses import dataclass
import math
import time
import torch
from torch import Tensor
from .spectral_config import SpectralConfig


@dataclass(frozen=True)
class Descriptor:
    q: Tensor
    g: Tensor
    diagnostics: dict


def feature_names(c):
    q = [f'spectrum.band_{i}' for i in range(c.spectral_bands)]
    q += [f'{s}.band_{i}' for s in ('radial', 'mean_knn_distance', 'local_nonflatness') for i in range(c.spectral_bands)]
    q += [f'{kind}.{stat}_{i}' for kind, n in [('hks', c.hks_scales), ('wks', c.wks_bands)]
          for stat in ('mean', 'std') for i in range(n)]
    g = ['log_rms_scale'] + [f'pca_energy_{i}' for i in range(c.geometry_pca_components)]
    g += ['effective_dimension_proxy', 'anisotropy', 'nearest_distance_mean', 'nearest_distance_std',
          'pairwise_cosine_mean', 'pairwise_cosine_std', 'local_nonflatness_mean', 'local_nonflatness_std']
    return tuple(q), tuple(g)


def neighbourhood(latent, c):
    if latent.ndim != 3 or min(latent.shape) <= 0:
        raise ValueError('expected nonempty [batch, observed_prefix, latent_dim]')
    return latent[:, -c.spectral_neighbourhood_size:]


def centre_normalize(h, eps):
    centroid = h.mean(1, keepdim=True)
    x = h - centroid
    scale = x.square().sum(-1).mean(-1).sqrt()
    return x / (scale[:, None, None] + eps), scale, centroid


def build_graph(x, c):
    b, m, _ = x.shape
    distances = torch.cdist(x, x, compute_mode='donot_use_mm_for_euclid_dist')
    eye = torch.eye(m, dtype=torch.bool, device=x.device)[None]
    k = min(c.spectral_knn_k, m - 1)
    if k == 0:
        return distances.new_zeros(b, m, m), distances.new_zeros(b, m, 0), distances.new_full((b,), c.spectral_eps), distances.new_zeros(b, m, m)
    knn = distances.masked_fill(eye, float('inf')).topk(k, largest=False).values
    selected = (distances <= knn[..., -1:] + c.spectral_eps) & ~eye
    sigmas = []
    for row, mask in zip(distances, selected):
        positive = row[mask & (row > c.spectral_eps)]
        sigmas.append(positive.median() if positive.numel() else row.new_tensor(c.spectral_eps))
    sigma = torch.stack(sigmas).clamp_min(c.spectral_eps)
    directed = torch.exp(-0.5 * (distances / sigma[:, None, None]).square()) * selected
    adjacency = torch.maximum(directed, directed.transpose(-1, -2))
    if c.spectral_graph_mode == 'latent_knn_plus_sequence':
        seq = torch.diag(x.new_full((m - 1,), c.spectral_sequence_edge_weight), diagonal=1)
        adjacency = torch.maximum(adjacency, seq + seq.T)
    return adjacency, knn, sigma, selected


def decompose_graph(adjacency, c):
    degree = adjacency.sum(-1)
    inv = degree.clamp_min(c.spectral_eps).rsqrt()
    # Isolated vertices have zero rows and a trivial mode each.
    laplacian = torch.diag_embed((degree > c.spectral_eps).float()) - inv[..., :, None] * adjacency * inv[..., None, :]
    values, vectors = torch.linalg.eigh(laplacian.float())
    values = values.clamp(0, 2)
    nontrivial = values > c.spectral_zero_threshold
    selected = nontrivial & (nontrivial.cumsum(-1) <= c.spectral_modes)
    boundary = values.masked_fill(~selected, 0).amax(-1, keepdim=True)
    selected = nontrivial & (values <= boundary + c.spectral_zero_threshold)
    return values, vectors, selected


def filters(values, bands, c):
    centers = torch.linspace(math.log(c.spectral_log_frequency_min), math.log(c.spectral_log_frequency_max), bands, device=values.device)
    return torch.exp(-0.5 * ((values.clamp_min(c.spectral_eps).log()[..., None] - centers) / c.spectral_filter_width).square())


def kernel_summaries(values, vectors, selected, c):
    times = torch.logspace(math.log10(c.hks_time_min), math.log10(c.hks_time_max), c.hks_scales, device=values.device)
    heat_mask = selected | (values <= c.spectral_zero_threshold)
    heat = torch.exp(-values[..., None] * times) * heat_mask[..., None]
    hks = torch.einsum('bjk,bkt->bjt', vectors.square(), heat)
    wave = filters(values, c.wks_bands, c) * selected[..., None]
    wave = wave / wave.sum(1, keepdim=True).clamp_min(c.spectral_eps)
    wks = torch.einsum('bjk,bkt->bjt', vectors.square(), wave)
    return tuple(torch.cat((v.mean(1), v.std(1, unbiased=False)), -1) for v in (hks, wks))


def nonflatness(x, adjacency, c):
    displacement = x[:, None, :, :] - x[:, :, None, :]
    displacement = displacement * (adjacency > 0)[..., None]
    energies = torch.linalg.svdvals(displacement).square()
    return energies[..., c.geometry_tangent_rank:].sum(-1) / energies.sum(-1).clamp_min(c.spectral_eps)


def explicit_geometry(x, scale, knn, curvature, c):
    singular = torch.linalg.svdvals(x)
    energy = singular.square()
    p = energy / energy.sum(-1, keepdim=True).clamp_min(c.spectral_eps)
    kept = torch.nn.functional.pad(p[:, :c.geometry_pca_components], (0, max(0, c.geometry_pca_components - p.size(-1))))
    eff = torch.where(energy.sum(-1) > c.spectral_eps, 1 / p.square().sum(-1).clamp_min(c.spectral_eps), 0)
    anisotropy = singular[:, 0] / singular.mean(-1).clamp_min(c.spectral_eps)
    nearest = knn[..., 0] if knn.size(-1) else x.new_zeros(x.shape[:2])
    unit = torch.nn.functional.normalize(x, dim=-1, eps=c.spectral_eps)
    cosine = unit @ unit.transpose(-1, -2)
    valid = x.norm(dim=-1) > c.spectral_eps
    mask = valid[:, :, None] & valid[:, None, :] & ~torch.eye(x.size(1), dtype=torch.bool, device=x.device)[None]
    count = mask.sum((1, 2)).clamp_min(1)
    angle_mean = (cosine * mask).sum((1, 2)) / count
    angle_std = (((cosine - angle_mean[:, None, None]).square() * mask).sum((1, 2)) / count).sqrt()
    scalars = torch.stack((eff, anisotropy, nearest.mean(1), nearest.std(1, unbiased=False), angle_mean, angle_std,
                           curvature.mean(1), curvature.std(1, unbiased=False)), -1)
    return torch.cat(((scale + c.spectral_eps).log()[:, None], kept, scalars), -1)


def extract_descriptor(latent, c: SpectralConfig, *, detach=True, profile=False):
    def stamp():
        if profile and latent.is_cuda:
            torch.cuda.synchronize(latent.device)
        return time.perf_counter()
    start = stamp()
    with torch.autocast(device_type=latent.device.type, enabled=False):
        h = neighbourhood(latent.detach() if detach else latent, c).float()
        x, scale, centroid = centre_normalize(h, c.spectral_eps)
        a, knn, sigma, local_neighbours = build_graph(x, c)
        graph_end = stamp()
        values, vectors, selected = decompose_graph(a, c)
        eigen_end = stamp()
        weights = filters(values, c.spectral_bands, c) * selected[..., None]
        occupancy = weights.sum(1)
        occupancy = occupancy / occupancy.sum(-1, keepdim=True).clamp_min(c.spectral_eps)
        curvature = nonflatness(x, local_neighbours, c)
        mean_distance = knn.mean(-1) if knn.size(-1) else x.new_zeros(x.shape[:2])
        signals = torch.stack((x.norm(dim=-1), mean_distance, curvature), -1)
        signals = signals - signals.mean(1, keepdim=True)
        energy = (vectors.transpose(-1, -2) @ signals).square() * selected[..., None]
        band_energy = torch.einsum('bks,bkf->bsf', energy, weights) / energy.sum(1)[..., None].clamp_min(c.spectral_eps)
        hks, wks = kernel_summaries(values, vectors, selected, c)
        raw = torch.cat((occupancy, band_energy.flatten(1), hks, wks), -1).clamp_min(0)
        q = raw / (raw.norm(dim=-1, keepdim=True) + c.spectral_eps)
        g = explicit_geometry(x, scale, knn, curvature, c)
        if not torch.isfinite(q).all() or not torch.isfinite(g).all():
            raise FloatingPointError('nonfinite spectral-geometric descriptor')
    end = stamp()
    diagnostic = dict(window_size=h.size(1), k=min(c.spectral_knn_k, h.size(1)-1), rms_scale=scale,
                      centroid_norm=centroid.squeeze(1).norm(dim=-1), bandwidth=sigma,
                      eigenvalues=values, retained_modes=selected,
                      trivial_modes=(values <= c.spectral_zero_threshold).sum(-1),
                      spectral_occupancy=occupancy, graph_signal_energy=band_energy, hks=hks, wks=wks,
                      geometry=g, harmonic=q)
    if profile:
        diagnostic['timing_seconds'] = dict(graph=graph_end-start, eigendecomposition=eigen_end-graph_end,
                                           descriptor=end-start)
    return Descriptor(q, g, diagnostic)
