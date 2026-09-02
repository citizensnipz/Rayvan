# Validation and measured baseline

## Test configuration

Reference fixture: `D=16`, `V=67`, sequence length 8, batch 2, two blocks per N1, population `[GPT, SSM, recurrent]`, top-K 2, tied embeddings, FP32 reference weights, deterministic seed 20260902. This is deliberately small enough for exhaustive intermediate and gradient checks; it is not a throughput-sizing model.

Test host: NVIDIA GeForce RTX 5070 (12,227 MiB), driver 591.86, PyTorch/LibTorch 2.8.0 CUDA 12.9 wheel, CUDA 13.0 local toolkit, MSVC 19.44, release build. The CMake package warned that its local configure probe did not discover cuDNN, while the prebuilt Torch CUDA libraries remained the operator backend. No custom CUDA code was compiled.

## Parity results

`rayvan-emc-tests` passed 23/23 on the CUDA build, including the CUDA BF16 case and native trainer resume/telemetry. The CPU build previously passed the 22 parity/architecture cases; its BF16 CUDA case was explicitly skipped because the CPU LibTorch build has no CUDA device.

Compared FP32 intermediates:

- embeddings and shared state;
- Nexus scores and probabilities;
- exact top-K indices and selected weights;
- dispatch permutation and inverse ordering;
- direct proposal from every supported N1;
- sparsely restored proposal stack;
- Integrator state and acceptance;
- logits and language-model loss.

Compared gradients:

- tied token embedding/output weight;
- Nexus score projection;
- active GPT attention projection;
- active SSM log-decay;
- active recurrent GRU input weight;
- Integrator query projection;
- final output normalization.

A forced two-request route `[[GPT,SSM],[SSM,recurrent]]` activates every supported family for gradient parity. Three AdamW steps compare loss trajectory, exact routing decisions, global gradient norm, and representative final parameters. All checks passed.

Tolerances:

| check | absolute | relative |
|---|---:|---:|
| ordinary FP32 intermediates | `2e-5` | `2e-4` |
| proposal/integrated/logit accumulation | `4e-5` | `4e-4` |
| selected FP32 gradients | `8e-5` | `8e-4` |
| three-step training trajectory | `3e-4` | `3e-3` |
| BF16 sanity envelope | `3e-2` | `5e-2` |

Top-K IDs must match exactly. The test fails with router-score context on mismatch; no tie is silently ignored.

Python regression after adding the non-Delta `supported` N2 preset: `19/19` tests passed in `tests/test_n2.py`.

## CUDA performance

Both processes used the same fixture weights, inputs, batch, architecture, one CPU thread, 10 warmups, and 100 timed iterations. Explicit CUDA synchronization brackets each timed region. Numbers below are one stable warmed run; the tiny workload is host-overhead dominated and showed transient Windows scheduling outliers in discarded earlier runs.

### FP32

| metric | Python | C++ | change |
|---|---:|---:|---:|
| forward latency | 4.728 ms | 3.761 ms | -20.4% |
| forward + backward | 16.953 ms | 12.697 ms | -25.1% |
| AdamW `step()` | 1.010 ms | 6.277 ms | +521.8% |
| complete train step | 16.983 ms | 21.170 ms | +24.7% |
| tokens/sec | 942.1 | 755.8 | -19.8% |
| normalized process CPU utilization | 4.92% | 4.30% | -0.62 pp |

### BF16 autocast

| metric | Python | C++ | change |
|---|---:|---:|---:|
| forward latency | 6.776 ms | 5.448 ms | -19.6% |
| forward + backward | 24.519 ms | 16.629 ms | -32.2% |
| AdamW `step()` | 0.934 ms | 5.529 ms | +491.8% |
| complete train step | 24.864 ms | 25.016 ms | +0.6% |
| tokens/sec | 643.5 | 639.6 | -0.6% |

The native forward path reduced host/framework overhead for this fixture. Native LibTorch AdamW was much slower than Python's current CUDA optimizer path and erased the training gain. This is a library/frontend limitation, not evidence that EMC math is slower in C++. The port does not replace AdamW or add fused optimizer kernels, per scope.

GPU utilization and implicit CUDA synchronization counts were not measured reliably in this run. The benchmark scripts report those fields as unavailable rather than inventing values. Explicit synchronization is used only around timing boundaries.

## Memory comparison

FP32 training benchmark after optimizer state creation:

| metric | Python | C++ | change |
|---|---:|---:|---:|
| model CPU tensor storage after CUDA move | 0 B | 0 B | 0 |
| optimizer CPU tensor storage | approximately 0 B | approximately 0 B | not material |
| parameter storage | 116,124 B | 116,124 B | 0% |
| optimizer state observed for active parameters | 164,800 B | 164,408 B | -0.2% |
| peak/current process RSS | 1,709,137,920 B | 1,491,173,376 B | -12.8% |
| CUDA allocated | 17,552,896 B | 17,552,896 B | 0% |
| CUDA reserved | 44,040,192 B | 44,040,192 B | 0% |
| peak CUDA allocated | 26,017,792 B | 26,027,008 B | +0.04% |
| peak CUDA reserved | 44,040,192 B | 44,040,192 B | 0% |

BF16 peak RSS was 1,877,757,952 B in Python and 1,532,772,352 B in C++ (-18.4%). BF16 CUDA reserved memory was identical at 23,068,672 B; peak allocated differed by less than 1%.

Fresh-process model-only checkpoint load:

| metric | Python fixture load | native checkpoint load | change |
|---|---:|---:|---:|
| peak RSS | 596,979,712 B | 436,609,024 B | -26.9% |
| CUDA allocated | 158,208 B | 158,208 B | 0% |
| CUDA reserved | 2,097,152 B | 2,097,152 B | 0% |
| peak CUDA allocated | 158,208 B | 158,208 B | 0% |

The checkpoint probe is model-only on both sides. The native loader never opens `optimizer.pt`, so evaluation cannot accidentally materialize optimizer state on CUDA.

Interpretation: the meaningful immediate saving is process/runtime RSS and explicit checkpoint lifetime control. Model, activation, and allocator-backed CUDA memory are effectively identical because both runtimes use ATen's CUDA kernels and caching allocator. There is no evidence that C++ intrinsically reduces VRAM.

## Decision

The current EMC N1/N2 architecture can be reproduced faithfully in native C++ without a Python runtime. The parity evidence covers every major intermediate, supported N1, routing, integration, backward gradients, and a short optimizer trajectory.

Measured results justify C++ as a native inference/runtime-control candidate: lower forward latency in this overhead-dominated fixture, 13–18% lower warmed-process RSS, 27% lower model-load RSS, deterministic ownership, and evaluation loading that excludes optimizer state. They do **not** establish a blanket training-performance win. FP32 end-to-end training was about 25% slower because the LibTorch AdamW frontend was much slower; BF16 training was effectively tied. VRAM did not improve.

Therefore: use the port as the production EMC engine when eliminating Python, controlling lifecycle/checkpoints, and preparing a stable FFI host are primary requirements. Do not justify the cutover with claims of faster training or lower GPU memory. Before production training adoption, benchmark the intended model size and investigate a mature supported fused/foreach optimizer path without changing EMC mathematics.
