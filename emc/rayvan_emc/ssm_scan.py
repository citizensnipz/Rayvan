"""Diagonal affine recurrence with fused CUDA autograd and portable scan fallback.

h[t] = a[t] * h[t-1] + b[t], h[-1] = 0. CUDA accumulates in FP32;
the portable path preserves FP64 for gradcheck. No persistent request state.
"""
from functools import lru_cache
from pathlib import Path
import warnings
import sys

import torch
from torch.autograd.function import once_differentiable


@lru_cache(maxsize=1)
def _cuda_extension():
    from torch.utils.cpp_extension import load
    print("SSM: loading/building fused CUDA extension (cached between runs).", file=sys.stderr, flush=True)
    root = Path(__file__).parent / "csrc"
    return load(name="rayvan_ssm_scan_v1", sources=[str(root / "ssm_scan.cpp"), str(root / "ssm_scan.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


@lru_cache(maxsize=1)
def _automatic_extension():
    try:
        return _cuda_extension()
    except (RuntimeError, OSError, ImportError) as error:
        warnings.warn(f"SSM CUDA extension unavailable; using parallel_scan: {error}", RuntimeWarning)
        return None


def resolve_backend(requested: str, device: torch.device) -> str:
    if requested not in {"auto", "cuda", "parallel_scan", "reference"}:
        raise ValueError(f"unknown SSM backend: {requested}")
    if requested == "cuda":
        if device.type != "cuda":
            raise ValueError("SSM cuda backend requires a CUDA device")
        _cuda_extension()  # Explicit selection fails loudly; auto may fall back.
    if requested == "auto":
        return "cuda" if device.type == "cuda" and _automatic_extension() is not None else "parallel_scan"
    return requested


def parallel_scan(a, b):
    # (a2,b2) o (a1,b1) = (a2*a1, b2+a2*b1). Out-of-place
    # doubling preserves autograd and handles non-power-of-two sequence lengths.
    if a.size(1) == 1:
        return b + a * 0  # Preserve the zero derivative w.r.t. a at h[-1]=0.
    stride = 1
    while stride < a.size(1):
        b = torch.cat((b[:, :stride], b[:, stride:] + a[:, stride:] * b[:, :-stride]), 1)
        a = torch.cat((a[:, :stride], a[:, stride:] * a[:, :-stride]), 1)
        stride *= 2
    return b


class _CudaScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        h = _cuda_extension().forward(a, b)
        ctx.save_for_backward(a, h)
        return h

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        a, h = ctx.saved_tensors
        # q[t] = dL/dh[t] + a[t+1]*q[t+1]; dL/da[t] = q[t]*h[t-1].
        return tuple(_cuda_extension().backward(a, h, gradient.contiguous()))


def affine_scan(a, b, backend="parallel_scan"):
    if a.shape != b.shape or a.ndim != 3 or not a.size(1):
        raise ValueError("SSM scan expects matching nonempty [batch, time, width] tensors")
    backend = resolve_backend(backend, a.device)
    if backend == "cuda":
        return _CudaScan.apply(a.float().contiguous(), b.float().contiguous())
    if backend == "reference":
        state = torch.zeros_like(b[:, 0])
        rows = []
        for t in range(a.size(1)):
            state = a[:, t] * state + b[:, t]
            rows.append(state)
        return torch.stack(rows, 1)
    return parallel_scan(a, b)
