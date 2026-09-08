"""Approximate local linear maps in the input PCA basis, not expert weight space."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .spectral_geometry import neighbourhood, extract_descriptor, feature_names


def transformation_names(c):
    return (tuple(f'log_stretch_{i}' for i in range(c.transformation_rank)) +
            ('mean_log_stretch', 'anisotropic_deformation', 'maximum_stretch', 'minimum_stretch',
             'log_volume', 'rotation_normalized_trace', 'rotation_identity_distance',
             'shear_like_off_axis_stretch', 'tangent_escape') +
            tuple('delta.'+n for n in feature_names(c)[0]) + ('spectral_delta_norm',))


@dataclass(frozen=True)
class Transformation:
    signature: torch.Tensor
    rank: torch.Tensor
    names: tuple[str, ...]


def transformation_signature(before, after, c, *, active=False):
    """Active gradients use the regularized fit and distinct-stretch SVD only.

    Input basis and graph spectral delta are fixed measurements. Repeated or
    near-zero stretch spectra reject active mode instead of producing bad grads.
    """
    with torch.autocast(device_type=before.device.type, enabled=False):
        h = neighbourhood(before.detach(), c).float()
        out = neighbourhood(after if active else after.detach(), c).float()
        if h.shape != out.shape:
            raise ValueError('transformation requires aligned input/output neighbourhoods')
        x = h - h.mean(1, keepdim=True)
        xp = out - out.mean(1, keepdim=True)
        q_before = extract_descriptor(before, c).q
        q_after = extract_descriptor(after, c).q
        _, singular, vh = torch.linalg.svd(x, full_matrices=False)
        rows, ranks = [], []
        for b in range(x.size(0)):
            threshold = max(c.spectral_eps, float(singular[b, 0]) * 1e-5)
            r = min(c.transformation_rank, int((singular[b] > threshold).sum()))
            ranks.append(r)
            delta = q_after[b] - q_before[b]
            if not r:
                rows.append(torch.cat((xp[b].sum().reshape(1).expand(c.transformation_rank+9) * 0, delta, delta.norm().view(1))))
                continue
            basis = vh[b, :r].T.detach()
            coords, transformed = x[b] @ basis, xp[b] @ basis
            identity = torch.eye(r, device=x.device)
            local_map = torch.linalg.solve(coords.T @ coords + c.transformation_ridge * identity, coords.T @ transformed)
            u, stretch, v = torch.linalg.svd(local_map, full_matrices=False)
            if active and (stretch.min() < 1e-5 or (r > 1 and stretch.diff().abs().min() < 1e-5)):
                raise ValueError('active complementarity requires nonzero, distinct local stretch values')
            logstretch = (stretch + c.spectral_eps).log()
            rotation = u @ v
            polar_stretch = v.T @ torch.diag(stretch) @ v
            offaxis = polar_stretch - torch.diag(torch.diag(polar_stretch))
            escape = (xp[b] - transformed @ basis.T).norm() / (xp[b].norm() + c.spectral_eps)
            scalars = torch.stack((logstretch.mean(), logstretch.std(unbiased=False), stretch.max(), stretch.min(),
                                   logstretch.sum(), rotation.trace()/r, (rotation-identity).norm()/r**0.5,
                                   offaxis.norm()/(polar_stretch.norm()+c.spectral_eps), escape))
            rows.append(torch.cat((F.pad(logstretch, (0, c.transformation_rank-r)), scalars, delta, delta.norm().view(1))))
        signature = torch.stack(rows)
        if not torch.isfinite(signature).all():
            raise FloatingPointError('nonfinite transformation signature')
        return Transformation(signature, torch.tensor(ranks, device=x.device), transformation_names(c))


def complementarity(signatures, usefulness, threshold, *, standardizer=None):
    """[B,E,P] signatures; detached measured usefulness, never router probabilities."""
    values = standardizer(signatures) if standardizer is not None else signatures
    normalized = F.normalize(values, dim=-1, eps=1e-8)
    similarity = normalized @ normalized.transpose(-1, -2)
    y = usefulness.detach()
    overlap = y[:, :, None] * y[:, None, :]
    a, b = torch.triu_indices(y.size(-1), y.size(-1), 1, device=y.device)
    if not a.numel():
        loss = signatures.sum() * 0
    else:
        loss = (overlap[:, a, b] * F.relu(similarity[:, a, b] - threshold).square()).mean()
    return loss, similarity, overlap
