# Validation and measured CUDA baseline

Measurements were made on 2026-09-02. This page records the pre-Delta native-runtime baseline. The subsequent Delta implementation, GPU stress measurements, and 1M-token experiment are reported in [`delta-native.md`](delta-native.md).

The later routing-free heterogeneous N1 implementation, 37-case verification,
50k stability gate, legacy control, causal ablations, and CUDA phase profile are
reported separately in [`routing-free-n1.md`](routing-free-n1.md).

## Environment alignment

Before the upgrade, the host had an RTX 5070 (compute capability 12.0), NVIDIA driver 591.86 (`nvidia-smi` CUDA 13.1), and one local CUDA toolkit, 13.0.48. The `pytorch_env` environment used Python 3.12.11, PyTorch 2.8.0+cu129, CUDA runtime 12.9, and cuDNN 9.1.0.2. CMake resolved `Torch_DIR` from that Python wheel, so C++ also consumed the 2.8/cu129 wheel libraries rather than a standalone LibTorch archive. The mismatch was the 12.9 PyTorch/LibTorch runtime versus the 13.0 local development toolkit. An unrelated base-Python CPU Torch install was not part of either runtime.

The selected stable target is PyTorch and LibTorch 2.13.0 with CUDA 13.0. PyTorch 2.13 was the newest release with an official GA announcement; 2.14 was still published only as a final release candidate. CUDA 13.0 is the 2.13 default binary target and matches the installed toolkit. NVIDIA documents driver 580 or newer for CUDA 13.x minor-version compatibility, so driver 591.86 required no change. CUDA 13.2 was not selected because it was not a stable Windows binary target for the chosen GA release and exceeded the driver's advertised CUDA 13.1 level.

After the upgrade:

| component | resolved value |
|---|---|
| GPU / driver | NVIDIA GeForce RTX 5070 / 591.86 |
| driver-advertised CUDA | 13.1 |
| local toolkit / compiler | CUDA 13.0 / nvcc 13.0.48 |
| Python | 3.12.11 |
| Python PyTorch | 2.13.0+cu130 |
| Python CUDA / cuDNN | 13.0 / 9.2.0 |
| C++ LibTorch | 2.13.0+cu130 files from the same CUDA Python wheel |
| CMake `Torch_DIR` | `pytorch_env/Lib/site-packages/torch/share/cmake/Torch` |
| C++ compiler | MSVC 19.44.35225, toolset 14.44.35207 |

The wheel and native build now use the exact same Torch generation and CUDA runtime target. The release build was configured from a completely removed build directory with CUDA 13.0 and detected SM 12.0. The Torch CMake package reports `USE_CUDNN=0` during its local configure probe; the prebuilt wheel still reports cuDNN 9.2.0 and supplies its CUDA operator libraries.

Package changes were limited to `torch` 2.8.0+cu129 to 2.13.0+cu130, `torchvision` 0.23.0+cu129 to 0.28.0+cu130, and pytest plus its small direct dependencies because the environment did not contain a test runner.

## Correctness

The current clean CUDA build passes 31/31 native cases. In addition to the original model, routing, optimizer, checkpoint, and trainer coverage, these include the independent Delta recurrence, allocation refusal, FP32 and BF16 CUDA forward/backward parity, Delta checkpoint round-trip, causal interventions, and full mixed-population Python/C++ parity. The Python N2 regression suite remains 19/19.

The Python-to-C++ fixture compares embeddings, shared state, router scores/probabilities, exact selected IDs and weights, dispatch ordering, every supported N1 proposal, Integrator state/acceptance, logits, loss, and representative gradients. A three-step real-model trajectory compares loss, routes, gradient norm, and final parameters.

The optimized AdamW additionally has a deterministic Python-generated reference at steps 1, 10, and 100. Parameters, first moments, second moments, per-parameter step counters, and quadratic loss trajectory all match both Python AdamW and stock C++ AdamW at `atol=2e-7, rtol=2e-6` (loss `atol=2e-6, rtol=2e-6`) on CPU and CUDA. The implementation retains stock parameter-group options and skips parameters whose gradients are undefined.

No parity tolerance was widened for the upgrade.

## AdamW diagnosis and implementation

The old 2.8 audit measured Python AdamW at 1.010 ms versus stock C++ at 6.277 ms in FP32, about 6.2 times slower. PyTorch 2.13 did not fix the C++ frontend: `torch::optim::AdamWOptions` still exposes no foreach or fused selection and its implementation remains a per-tensor loop.

On the aligned 2.13 stack, stock C++ measured 8.151 ms FP32 and 9.044 ms BF16, versus Python's automatically selected foreach path at 1.940 ms and 1.802 ms. The replacement was therefore required.

`ForeachAdamW` subclasses stock LibTorch `AdamW`, preserving its `AdamWParamState`, options, parameter groups, and save/load format. It builds lists per optimizer group and compatible device/dtype bucket, lazily creates one standard FP32 moment pair per parameter, and calls generated ATen `_foreach_*` operators for decoupled decay, first/second moments, optional AMSGrad maximum, square root, bias correction, epsilon, and the final update. It does not flatten parameters, duplicate persistent moments, add a custom CUDA kernel, or run a per-parameter CUDA operation chain.

The generated ATen foreach headers are shipped and linkable in stable LibTorch 2.13, but their leading-underscore names are not covered by the high-level C++ optimizer API's compatibility contract. Upgrades must rebuild and rerun the parity/performance tests.

### Optimizer launch profile

One warmed FP32 optimizer step on the 142-parameter tiny EMC model:

| path | CUDA launches | CUDA execution | wall time | peak temporary CUDA allocation |
|---|---:|---:|---:|---:|
| Python 2.13 foreach | 8 | 0.885 ms | benchmarked separately | 111,616 B |
| stock C++ per-tensor | 882 | 8.048 ms | 8.057 ms | 10,240 B |
| optimized C++ foreach | 8 | 0.575 ms | 0.578 ms | 111,616 B |

Python values come from `torch.profiler`; C++ launch counts use CUPTI driver callbacks and CUDA execution uses events around exactly one step. The 110-fold C++ launch-count reduction is the intended result. Foreach's small temporary tensor-list output allocation is transient and matches Python; persistent optimizer memory remains one moment pair per active parameter.

## Exact tiny-fixture performance

The timings below are the original three-family fixture: batch 2, sequence 8, latent width 16, vocabulary 67, population `[GPT, SSM, recurrent]`, top-K 2. The current correctness fixture is four-family and includes Delta; these historical timings were not silently relabeled. Both timed processes used identical weights and inputs, one CPU thread, 10 warmups, 100 timed iterations, and explicit CUDA synchronization at timing boundaries. It is deliberately host-overhead dominated.

### FP32

| metric | Python 2.13 | optimized C++ | C++ change |
|---|---:|---:|---:|
| forward | 4.907 ms | 3.998 ms | -18.5% |
| forward + backward | 27.100 ms | 13.528 ms | -50.1% |
| AdamW | 1.940 ms | 0.677 ms | -65.1% |
| complete train step | 30.049 ms | 13.753 ms | -54.2% |
| tokens/sec | 532.5 | 1,163.3 | +118.5% |

### BF16 autocast

| metric | Python 2.13 | optimized C++ | C++ change |
|---|---:|---:|---:|
| forward | 7.071 ms | 5.740 ms | -18.8% |
| forward + backward | 35.915 ms | 17.432 ms | -51.5% |
| AdamW | 1.802 ms | 0.658 ms | -63.5% |
| complete train step | 39.028 ms | 17.467 ms | -55.3% |
| tokens/sec | 410.0 | 916.0 | +123.4% |

Relative to stock C++ 2.13, foreach makes AdamW 12.0 times faster in FP32 and 13.8 times faster in BF16.

## Short realistic-scale sanity check

The non-Delta scale check uses latent width 64, vocabulary 8,192, sequence length 128, batch 4, GPT/SSM/recurrent population, top-K 2, two N1 blocks, and the current Nexus/Integrator. It has 3,808,528 parameter bytes. Each process used 3 warmups and 10 timed iterations; this is a short throughput check, not training to convergence.

| precision / metric | Python | C++ | C++ change |
|---|---:|---:|---:|
| FP32 forward | 14.840 ms | 12.537 ms | -15.5% |
| FP32 forward + backward | 49.323 ms | 43.402 ms | -12.0% |
| FP32 AdamW | 5.094 ms | 4.453 ms | -12.6% |
| FP32 train step | 66.224 ms | 59.793 ms | -9.7% |
| FP32 tokens/sec | 7,731 | 8,563 | +10.8% |
| BF16 forward | 20.180 ms | 18.669 ms | -7.5% |
| BF16 forward + backward | 59.420 ms | 56.663 ms | -4.6% |
| BF16 AdamW | 5.067 ms | 4.420 ms | -12.8% |
| BF16 train step | 82.005 ms | 73.134 ms | -10.8% |
| BF16 tokens/sec | 6,244 | 7,001 | +12.1% |

Optimizer time is about 7.7% of Python and 7.4% of C++ FP32 train-step time; in BF16 it is about 6.2% and 6.0%. The native advantage narrows as GPU work dominates, but remains an end-to-end throughput win.

## Memory recheck

Tiny FP32 after optimizer-state creation:

| metric | Python | optimized C++ |
|---|---:|---:|
| process RSS | 1,870,725,120 B | 1,546,870,784 B |
| parameter storage | 116,124 B | 116,124 B |
| optimizer state | 164,800 B | 164,408 B |
| CUDA allocated | 67,622,400 B | 67,622,400 B |
| CUDA reserved | 90,177,536 B | 90,177,536 B |
| peak CUDA allocated | 76,087,296 B | 76,096,512 B |

The tiny BF16 run used 2,360,750,080 B Python RSS versus 1,883,836,416 B C++ RSS. CUDA reserved memory was identical and peak allocated differed by less than 0.02%.

Realistic FP32 peak/current RSS was 1,948,794,880 B in Python and 1,626,468,352 B in C++ (-16.5%). Peak CUDA allocated was 146,346,496 B and 163,154,944 B respectively; allocator reservation history differed by one 16 MiB block. Realistic BF16 RSS was 2,296,127,488 B versus 1,968,799,744 B (-14.3%), while peak CUDA allocated differed by 3.7%. Parameter and optimizer state storage agree within 568 bytes in both precisions.

Fresh-process model-only checkpoint loading used 683,257,856 B Python RSS versus 435,077,120 B native RSS (-36.3%). Both allocated 158,208 B CUDA and reserved 2,097,152 B. The native evaluation loader never opens optimizer state.

The optimizer change does not reduce persistent VRAM: Python and C++ use the same ATen allocator and mathematical state. Its benefit is launch amortization, training throughput, lower host-runtime RSS, and explicit checkpoint lifetime.

## Decision

Yes. With PyTorch/LibTorch 2.13 aligned on CUDA 13.0 and equivalent multi-tensor AdamW semantics, native C++ preserves its faster forward/backward path as an actual end-to-end training advantage: about 2.2 times the tiny-fixture throughput and 10.8–12.1% higher throughput in the short realistic configuration.

Remaining cautions are the underscored ATen foreach API surface, Windows timing variability, and the fact that a larger production configuration may become entirely GPU-bound. CUDA graph capture and distributed execution remain out of scope. Delta and its custom CUDA path are now implemented and audited separately in [`delta-native.md`](delta-native.md).
