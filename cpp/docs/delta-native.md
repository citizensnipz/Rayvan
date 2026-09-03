# CUDA-native Delta N1 audit

Measurements in this report were taken on 2026-09-02 on an NVIDIA GeForce RTX 5070 with CUDA 13.0 and PyTorch/LibTorch 2.13.0+cu130. All timed CUDA regions were warmed up and explicitly synchronized. CUPTI driver callbacks counted launches and CUDA events measured elapsed device work. Unless stated otherwise, the research shape is batch 4, sequence 256, latent width 256, depth 3, top-K 2, and BF16 autocast with FP32 recurrent state.

## Exact operator implemented

For each head, the unchanged Python node first conditions each token on the shared-state mean and applies LayerNorm. It then produces L2-normalized `q` and `k`, `tanh`-bounded `v`, and independent sigmoid gates `alpha` and `beta`. The initial memory is the outer product of two projections of the sequence summary, scaled by `1/sqrt(head_dim)`.

The native CUDA core implements, in FP32 state arithmetic,

```text
S_t = S_(t-1) [alpha_t (I - beta_t k_t k_t^T)] + beta_t v_t k_t^T
    = alpha_t S_(t-1) + beta_t [v_t - alpha_t S_(t-1) k_t] k_t^T
y_t = S_t q_t
```

The output gate, output projection, residual post-FFN, outer N1 residual scaling, and state proposal are unchanged. This is deliberately the project-specific operator. Compared with the commonly published Gated DeltaNet form, this implementation has separate `alpha` and `beta`, applies `alpha` to the erase term, normalizes `q/k`, bounds `v`, and retains the existing conditioning, output gate, learned initial memory, and post-FFN.

## FLA study and implementation boundary

Flash Linear Attention is MIT-licensed. Its current chunked gated-delta implementation uses chunk-local `K K^T` structure, a lower-triangular solve/WY representation, and recomputation of intermediates in backward. Those ideas informed the review, particularly avoiding sequence-length collections of full transition matrices and trading recomputation for saved activation memory. No FLA source code was copied.

The production kernel chosen here is a bounded fused recurrent implementation, not a WY kernel. A CUDA block owns one batch/head pair, keeps its `D x D` state in shared FP32 memory, walks all chunks in order inside one kernel, and parallelizes the state rows. The current kernel therefore still has serial token dependence within each batch/head. Backward saves only vector inputs, the initial state, and chunk-boundary states, then recomputes states inside each chunk. This is an honest remaining optimization gap relative to the requested ideal of parallel work within a chunk and serial dependence only between chunks.

Supported head dimensions are at most 64 and supported chunk sizes are 16, 32, and 64. Unsupported shapes fail validation; CUDA execution never silently falls back to a host or generic recurrence.

## Memory model and safety

Backward scratch and saved chunk-boundary storage are checked with overflow-safe integer arithmetic before allocation:

```text
scratch_bytes = B * H * (chunk_size + 1) * head_dim * head_dim * sizeof(float)
boundary_bytes = B * H * (ceil(S / chunk_size) + 1) * head_dim * head_dim * sizeof(float)
```

At `B=4`, `H=8`, `S=256`, `chunk=64`, `head_dim=64`, scratch is 34,078,720 bytes (32.5 MiB) and boundaries are 2,621,440 bytes (2.5 MiB). The implementation rejects the call when either exceeds the configured `delta_max_scratch_bytes`. State, dot products, and gate reductions use FP32; projected `q/k/v` may be BF16. It retains no per-token `D x D` tensor: the explicit sequence-dependent term is the capped `ceil(S/chunk_size)+1` boundary collection.

For comparison, the old affine scan's single full `[B,H,S,D,D]` transition tensor at the same batch/sequence/head shape is 134,217,728 bytes. Transition plus write is already 256 MiB; including the corresponding prefix objects and per-token states gives a conservative roughly 640 MiB lower bound before concatenations, copies, and autograd saves. The old path's exact allocator peak was not captured, so this is a tensor-shape lower bound rather than a claimed measured peak.

The research Delta N1 has 11,511,344 parameters (46,045,376 bytes in FP32). Across depth 3 and four outer 64-token chunks, its core saves approximately 28,508,160 bytes of inputs, initial states, and boundaries. The full mixed model has 236,808,472 parameters.

## Correctness

The clean GPU suite passes 31/31 cases. Delta-specific coverage includes:

- an independent direct recurrence across seeds 7, 42, and 901, sequence lengths 1, 17, and 37, and chunk sizes 16, 32, and 64;
- FP32 CUDA forward/final-state parity at `atol=3e-5, rtol=3e-4` and all six input gradients at `atol=8e-5, rtol=8e-4`;
- BF16 vector/FP32-state parity at `atol=4e-4, rtol=4e-3` for outputs and `atol=1.5e-2, rtol=2e-2` for gradients, with finite-gradient checks;
- overflow and scratch-budget refusal, config rejection, checkpoint round-trip, and causal-intervention compatibility;
- a deterministic four-family Python/C++ fixture comparing routing, every N1 proposal, Integrator output, logits, loss, representative gradients, and a three-step training trajectory.

## Kernel and N1 benchmark

Each entry uses 20 timed iterations. “F+B” includes the custom backward. Peak temporary allocation is allocator-observed and does not include persistent model parameters.

| chunk | core forward | core F+B | core launches F / F+B | scratch | core peak temp F+B | full Delta N1 forward | full Delta N1 F+B | full N1 launches F / F+B | full N1 peak temp F / F+B |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1.508 ms | 7.928 ms | 1 / 2 | 8,912,896 B | 20,447,232 B | 47.734 ms | 246.006 ms | 3,503 / 10,906 | 8,475,136 / 500,424,192 B |
| 32 | 1.474 ms | 7.958 ms | 1 / 2 | 17,301,504 B | 24,641,536 B | 30.430 ms | 121.447 ms | 1,775 / 5,482 | 10,641,920 / 303,001,088 B |
| 64 | 1.469 ms | 8.170 ms | 1 / 2 | 34,078,720 B | 39,321,600 B | 15.778 ms | 63.563 ms | 911 / 2,770 | 12,616,192 / 213,559,808 B |

The core uses one custom launch in forward and one in backward. A full depth-3 Delta N1 called over four outer chunks therefore contributes 12 custom launches in forward or 24 in forward+backward; the larger CUPTI counts above include its unchanged projections, gates, normalizations, post-FFN, and autograd operations. Chunk 64 remains the default. Against the historical 69 ms Delta forward measurement, 15.778 ms is 4.37x faster (77.1% lower latency), and is in the same broad latency class as the unchanged SSM (20.211 ms) and recurrent (18.769 ms) nodes. GPT measured 6.647 ms.

## Mixed-model benchmark

At the same shape, natural top-2 routing selected two node families per token rather than evaluating the entire population.

| region | time | CUDA launches | peak temporary allocation |
|---|---:|---:|---:|
| Nexus | 0.175 ms | 13 | — |
| Integrator | 0.858 ms | 46 | — |
| natural mixed forward | 30.772 ms | 1,628 | 110,578,176 B |
| natural mixed forward+backward | 110.454 ms | 4,695 | 903,181,312 B |
| foreach optimizer | 5.288 ms | 8 | 145,566,208 B |
| complete train micro-step | 123.341 ms | 7,001 | 813,348,864 B |

The micro-step rate was 8,302 tokens/s. Persistent parameter bytes were 236,808,472 and optimizer-state bytes were 382,458,928; current allocated/reserved CUDA memory was 880,950,784 / 1,958,739,968 bytes and observed peak allocated was 1,694,299,648 bytes. System-wide utilization samples ranged from 29% to 64% (44.3% mean), but another process was using the GPU, so those samples cannot be attributed to EMC and are not used as evidence of saturation.

Forced-pair forward latencies were 37.218 ms for GPT+SSM, 36.556 ms for SSM+GPT, 40.256 ms for recurrent+GPT, and 31.531 ms for Delta+GPT. These are workload observations, not a claim that family pairs are semantically interchangeable.

## Deterministic 1M-token training run

TinyStories was prepared outside the runtime with the GPT-2 tokenizer into fingerprinted `.rvtok` files: the first 10,000 training stories produced 2,162,078 tokens with fingerprint `5227b790975bc8b1`, and the first 1,000 validation stories produced 194,559 tokens with fingerprint `ad4ccf06c89282df`. The exact run used the natural mixed population `[GPT, SSM, recurrent, Delta]`, seed 42, batch 4, sequence 256, latent width 256, GPT hidden width 6,144, SSM width 960, recurrent width 704, Delta width 512 with 8 heads and FFN width 5,120, depth 3, BF16 autocast, chunk 64, top-K 2, learning rate `3e-4`, and the unchanged Nexus policy. It processed 1,000,448 tokens in 977 steps. No all-Delta stress training was run.

| result | native mixed run | prior Python context | comparison |
|---|---:|---:|---:|
| wall time | 132.61 s | 364 s | 63.6% lower (2.75x) |
| cumulative throughput | 7,544 tokens/s | 2,866 tokens/s | 2.63x (+163.2%) |
| validation loss | 9.8825 | 5.553 | +4.3295 |
| perplexity | 19,585 | 258 | 75.9x |
| validation accuracy | 9.81% | 10.47% | -0.66 percentage points |

The performance gate passed; the quality comparison did not. The prior run is contextual rather than controlled because dataset/checkpoint lineage was not identical. The new run reached 2,935,899,136 bytes peak allocated CUDA memory, 3,225,419,776 bytes reserved, and about 2,374,660,096 bytes process RSS. Checkpoints were written at 100k, 250k, 500k, 750k, and 1M tokens, plus `best` and `latest`.

## Causal audit and decision

The 16-window audit of the best checkpoint naturally selected GPT+recurrent; Delta received no natural selections. Disabling or zeroing Delta was therefore numerically identical to the natural result. Forcing recurrent+Delta changed audit loss from 28.2256 to 28.3784 and forward latency from 29.636 ms for forced GPT+recurrent to 34.998 ms (+18.1%). Accuracy moved from 9.918% to 9.937%, too small and internally mixed with worse loss to establish value. This audit samples different fixed validation windows from trainer validation, so its absolute loss is not directly comparable to 9.8825.

The implementation decision is therefore split:

- **Engineering:** keep the CUDA-native Delta path. It is mathematically faithful, bounded, checkpoint-compatible, 4.37x faster than the historical Delta forward, and passes the full parity suite.
- **Model evidence:** do not claim that learned Delta value was preserved. This checkpoint did not route to Delta naturally, forcing it worsened loss and latency, and the broader quality result missed the prior Python context.
- **Next optimization:** if Delta becomes important in a controlled-quality run, replace the token-serial core with a true chunk-parallel WY/triangular formulation while retaining this recurrence as the small-shape oracle. Kernel fusion around projections and the post-FFN is a larger launch-count opportunity than the already one-launch core.

Reproducibility entry points are `rayvan-emc-tests`, `rayvan-emc-delta-benchmark`, `rayvan-emc-train`, and `rayvan-emc-causal`. Dataset preparation is `cpp/python/prepare_tinystories.py`; the native runtime consumes only the resulting `.rvtok` files.

The corresponding repository-root commands are:

```powershell
$env:PYTHONPATH = "emc"
python cpp/python/prepare_tinystories.py build/delta-data `
  --train-stories 10000 --validation-stories 1000 --tokenizer gpt2

build/emc-delta10/rayvan-emc-tests.exe
build/emc-delta10/rayvan-emc-delta-benchmark.exe --iterations 20
build/emc-delta10/rayvan-emc-train.exe `
  build/delta-data/train.rvtok build/delta-data/validation.rvtok `
  build/delta-1m --tokens 1000000
build/emc-delta10/rayvan-emc-causal.exe `
  build/delta-1m/best build/delta-data/validation.rvtok 16
```

The retained run artifacts are `build/delta-1m/telemetry.tsv`, milestone directories at 100k/250k/500k/750k/1M tokens, and `best`/`latest` checkpoints. Build artifacts are intentionally ignored by source control; the tables above preserve the synchronized profiler summary.

## References

- Gated DeltaNet paper: <https://arxiv.org/abs/2412.06464>
- Flash Linear Attention license: <https://github.com/fla-org/flash-linear-attention/blob/main/LICENSE>
- FLA gated-delta chunk forward: <https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk_fwd.py>
- FLA WY utilities: <https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/delta_rule/wy_fast.py>
