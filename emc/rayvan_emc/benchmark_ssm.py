"""Whole-module forward/backward timing. Run with --device cuda --backend auto.

Compilation/warmup excluded. Reports milliseconds, not language-model tok/s.
"""
import argparse
from copy import deepcopy
import time
import torch
from .model import EMCConfig
from .modules import StateSpaceEMCModule


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--backend', default='auto', choices=['auto', 'cuda', 'parallel_scan'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--length', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--iterations', type=int, default=50)
    args = parser.parse_args()
    if min(args.length, args.batch_size, args.width, args.iterations) < 1:
        parser.error('dimensions and iterations must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    c = EMCConfig(latent_dim=args.width, state_space_dim=args.width, ssm_backend=args.backend)
    model = StateSpaceEMCModule(c).to(args.device)
    reference = deepcopy(model)
    reference.ssm_backend = 'reference'
    x = torch.randn(args.batch_size, args.length, args.width, device=args.device, requires_grad=True)
    def sync():
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
    def measure(module):
        def step():
            module.zero_grad(set_to_none=True)
            x.grad = None
            module(x).square().mean().backward()
        for _ in range(3):
            step()
        sync()
        start = time.perf_counter()
        for _ in range(args.iterations):
            step()
        sync()
        return 1000 * (time.perf_counter() - start) / args.iterations
    # Both paths batch projections; this isolates the recurrence speedup.
    expected, actual = measure(reference), measure(model)
    print(f'backend={model.last_backend} | reference={expected:.3f} ms | optimized={actual:.3f} ms | speedup={expected/actual:.2f}x')


if __name__ == '__main__':
    main()
