# Rayvan EMC research prototype

EMC (Emergent Modular Cognition) is an experimental language-model computation graph. Token embeddings form a shared latent state; a learned Nexus selects a sparse module set; those independent modules produce latent updates; a learned Integrator updates the shared state. The route-integrate cycle repeats a fixed number of times before normalization and vocabulary projection.

## N1 components

- **Module-aware Nexus:** projects each causal latent token into a query and scores it against learned per-module descriptor keys. Descriptors contain no semantic labels. Top-K and the existing balancing objective remain unchanged. An availability mask can temporarily remove local modules without changing the scoring head.
- **Heterogeneous modules:** every family implements `EMCModuleBase.forward([B,S,D]) → proposal [B,S,D]`. The existing GPT-style family remains. A pure-PyTorch selective diagonal state-space family and a GRU recurrent family add different established computation. Input/output adapters permit different internal widths.
- **Proposal-aware Integrator:** preserves all selected proposals. Multi-head cross-attention uses the current latent as query, proposals as keys/values, and Nexus weights as learned-strength priors. Proposal mean/variance provide set context, followed by a learned token-dimensional gated residual update.
- **Fixed cycles and output:** the integrated latent becomes the next cycle's input. No adaptive halting or persistent private memory is introduced.

State-space and GRU state exists only while scanning one sequence in one module forward. It resets between EMC cycles, batches, and inference calls. Modules never call each other and communicate only through proposals and the Integrator.

The pre-N1 `NexusRouter` and `WeightedAverageIntegrator` remain available for isolated baselines. No new dependency is required: the state-space implementation uses ordinary PyTorch rather than the CUDA/custom-kernel-oriented `mamba-ssm` package.

This research area remains independent from Rayvan's Rust networking code. It contains no networking, distributed execution, pretrained model weights, instruction tuning, semantic module roles, or custom CUDA path.

## N2 CUDA execution

`--model n2_emc` builds one request-routed N2 event over four depth-3 N1 nodes. The mixed population is GPT, SSM, recurrent, and DeltaNet. The default `--n2-execution-mode sparse` keeps top-K semantics: Nexus scores the whole request batch with one projection, GPU tensors sort the selected request/slot assignments by expert, each heterogeneous N1 receives one batched request tensor, and one inverse permutation restores proposals before matrix-native integration. Unselected requests do not execute an N1's local blocks. The only host loop is the fixed four-family launch boundary required by the four different computational graphs; no request or token is dispatched through Python.

Routing, dispatch metadata, proposal tensors, and streaming diagnostic reductions remain on the accelerator. CPU materialization occurs when a diagnostic report or milestone snapshot is requested. `--n2-execution-mode dense` is an experimental training control that executes all four N1s with continuous Nexus probabilities. `--n2-cuda-streams` enables experimental concurrent family streams; both controls are off by default.

The CUDA path uses PyTorch operations only. It does not add custom CUDA/Triton kernels or an external MoE framework. PyTorch grouped GEMM is useful only for compatible homogeneous projections; the heterogeneous N1 graphs remain separate.

## Chunk-routed N1

`--n1-stage n1_chunked` selects the execution architecture. A request is embedded by a deliberately small shared core, initializes canonical state `[B, shared_state_slots, D]`, selects a request-level descriptor pool once, then processes contiguous `chunk_size` blocks. Chunk routing uses the first causal token plus the previous canonical state; it never summarizes future tokens inside the current chunk.

For every chunk, only the selected top-K modules execute. Chunks assigned to the same module are gathered into one module batch and proposals are scattered back by request and routing slot. Computed and retained chunk-module pairs are therefore identical. The old union-of-token-selections executor remains only in token-routed stages.

Private state is lease-scoped. A continuously selected module reuses its state and increments lease age. When inactive, `end_lease` runs and state is discarded. Selection after a gap calls `begin_lease(shared_state)`; stale state is never resumed. Canonical shared state is the only cross-module history.

The chunk Integrator applies proposal-aware attention independently to `[B,K,C,D]` token proposals and `[B,K,M,D]` state proposals, producing both updated chunk latent and updated canonical state. Nexus confidence remains a prior, not the final acceptance decision.

### Chunk module boundary

```text
ModuleInput:
  chunk_latent [B,C,D]
  shared_state [B,M,D]
  lease_state
  structural metadata

ModuleOutput:
  token_proposal [B,C,D]
  state_proposal [B,M,D]
  new_lease_state
```

GPT prepends canonical state slots as causal context. The SSM uses vectorized chunk projections and an exact diagonal parallel scan with chunk-boundary state/convolution history. The recurrent family calls whole-chunk `nn.GRU` and can use an internal FP16 CUDA autocast while returning EMC dtype. None contains a Python token loop.

Gated DeltaNet follows Yang, Kautz, and Hatamizadeh, [“Gated Delta Networks: Improving Mamba2 with Delta Rule”](https://arxiv.org/abs/2412.06464):

```text
S_t = S_{t-1}[alpha_t(I - beta_t k_t k_t^T)] + beta_t v_t k_t^T
o_t = S_t q_t
```

Each token defines an affine memory transform `(A_t, B_t)`. The implementation composes prefix transforms with a logarithmic-depth associative scan, so there is no token-by-token Python recurrence. Lease memory is `[B, heads, value_dim, key_dim]` and is initialized from canonical shared state. The current fallback implements the paper’s gated associative-memory core; it omits the paper’s optional local convolution/attention hybrids and replaces the optimized WY/Triton implementation with mathematically equivalent PyTorch affine-prefix composition. Scan math uses FP32 for stability and returns EMC dtype through adapters.

The chunk Nexus keeps learned module descriptors at both timescales. Chunk scores add configurable persistence, switching, availability, and bounded loss-free global balancing bias. The balancing bias is a checkpointed buffer, not a learned parameter or auxiliary model loss.

Architecture metrics expose request pools, family composition, chunk selections, routing entropy, lease ages/lengths, switch/retention rates, persistence/switch contributions, executed modules, population touched, exact sparse compute pairs, balancing bias/totals, and separate token/state Integrator acceptance.

Grouped configuration is carried by `EMCConfig`: execution (`architecture_stage`, `chunk_size`, `shared_state_slots`, `request_pool_size`, `active_top_k`), leases (`switch_cost`, `persistence_bonus`, `minimum_lease_chunks`), loss-free balance (`loss_free_balance_enabled`, `balance_target_utilization`, `balance_bias_lr`, `balance_bias_limit`, `balance_warmup_chunks`), shared core (`shared_core_enabled`, `shared_core_hidden_dim`), and family backends/widths (`state_space_dim`, `ssm_backend`, `recurrent_dim`, `recurrent_backend`, `recurrent_precision`, `delta_internal_dim`, `delta_heads`, `delta_ffn_dim`, `delta_backend`). Defaults keep the pool equal to the current four-module population and use four canonical state slots.

## TinyStories language training

The main TinyStories path uses the standard fast GPT-2 BPE tokenizer (`gpt2`, 50,257 tokens), inserts EOS between packed stories, and samples fixed-length causal windows without padding or DataLoader workers. Long documents are encoded through the tokenizer's raw fast backend, which has no language-model context limit; only the resulting 128/256-token chunks enter EMC. This fixes the former oversized-sequence warning at its source rather than suppressing it. The repository-local overfit test keeps its character tokenizer.

The final mixed N1 research preset keeps a compact 256-wide shared latent, tied GPT-2 vocabulary weights, 4 modules, top-2 routing, and 2 cycles. Its population is `GPT, state-space, recurrent, GPT`; family sizes are approximately comparable (3.27M–3.42M parameters each). The proposal-aware Integrator and descriptor Nexus bring the full model to about 27.24M parameters.

Install from this directory:

```sh
python -m pip install -e ".[test,data]"
```

## Staged N1 experiments

The flags keep H1/H2/H3 separately testable:

```text
--n1-stage baseline       fixed router + weighted-average Integrator + GPT-only
--n1-stage integrator     fixed router + proposal Integrator + GPT-only
--n1-stage heterogeneous  fixed router + proposal Integrator + chosen families
--n1-stage n1             descriptor router + proposal Integrator + token routing
--n1-stage n1_chunked     request pool + chunk routing + leases + canonical state
```

Module populations:

```text
gpt-only, ssm-only, recurrent-only, delta-only,
gpt-ssm, gpt-recurrent, gpt-delta,
ssm-recurrent, ssm-delta, recurrent-delta, mixed
```

Stage A:

```sh
python -m rayvan_emc.experiments.train --n1-stage baseline --module-population gpt-only ...
python -m rayvan_emc.experiments.train --n1-stage integrator --module-population gpt-only ...
```

Stage B:

```sh
python -m rayvan_emc.experiments.train --n1-stage heterogeneous --module-population gpt-only ...
python -m rayvan_emc.experiments.train --n1-stage heterogeneous --module-population ssm-only ...
python -m rayvan_emc.experiments.train --n1-stage heterogeneous --module-population recurrent-only ...
python -m rayvan_emc.experiments.train --n1-stage heterogeneous --module-population mixed ...
```

Stage C:

```sh
python -m rayvan_emc.experiments.train --n1-stage heterogeneous --module-population mixed ...
python -m rayvan_emc.experiments.train --n1-stage n1 --module-population mixed ...
```

Chunk-routed N1:

```sh
python -m rayvan_emc.experiments.train \
  --n1-stage n1_chunked --module-population mixed ...
```

Diagnostics now include module/family traffic, average router probability, proposal norm, Integrator acceptance/contribution, proposal similarity, gate/update magnitude, parameter and gradient counts, and quality/perplexity/runtime after each fixed cycle.

### Larger-model smoke test — approximately 1M tokens

```sh
python -m rayvan_emc.experiments.train \
  --model emc --dataset tinystories --preset research --budget quick \
  --sequence-length 256 --batch-size 1 --gradient-accumulation 4 \
  --precision auto --checkpoint-dir checkpoints
```

### Meaningful language run — approximately 10M tokens

```sh
python -m rayvan_emc.experiments.train \
  --model emc --dataset tinystories --preset research --budget medium \
  --sequence-length 256 --batch-size 1 --gradient-accumulation 4 \
  --train-stories 50000 --precision auto --checkpoint-dir checkpoints
```

### Longer run — approximately 25M tokens

```sh
python -m rayvan_emc.experiments.train \
  --model emc --dataset tinystories --preset research --budget research \
  --sequence-length 256 --batch-size 1 --gradient-accumulation 4 \
  --train-stories 100000 --precision auto --checkpoint-dir checkpoints
```

Use `--train-tokens 50000000` for a 50M-token run. Token budgets become whole optimizer steps using `ceil(train_tokens / (batch_size × context_length × gradient_accumulation))`; the final count can exceed the request by less than one accumulated optimizer batch.

Training evaluates periodically and reports LM loss, validation loss/perplexity, processed tokens, throughput, elapsed time, routing distributions, concentration, entropy/effective modules, balance contribution, and CUDA current/peak allocated memory. It samples fixed TinyStories prompts every few evaluations. `--device`, `--precision`, `--batch-size`, `--sequence-length`, and `--gradient-accumulation` are configurable. `auto` uses BF16 when supported, otherwise FP16 on CUDA and FP32 on CPU. A CUDA OOM directly recommends reducing physical batch size or context length and resuming from the latest checkpoint.

### Baseline and comparison

Train only the conventional baseline with the exact same tokenizer/data/context/budget:

```sh
python -m rayvan_emc.experiments.train \
  --model baseline --dataset tinystories --preset research --budget medium \
  --sequence-length 256 --batch-size 1 --gradient-accumulation 4
```

Train both only when a direct comparison is wanted:

```sh
python -m rayvan_emc.experiments.compare \
  --dataset tinystories --preset research --budget medium \
  --sequence-length 256 --batch-size 1 --gradient-accumulation 4
```

### Checkpoints and resume

Each evaluation atomically writes:

```text
checkpoints/emc-latest.pt
checkpoints/emc-best.pt
```

The files include model and optimizer state, model configuration, tokenizer identifier/config, step and token counts, best validation loss, RNG state, precision, and routing-balance configuration. Resume with the same model/context settings and a larger total budget:

```sh
python -m rayvan_emc.experiments.train \
  --model emc --dataset tinystories --preset research --budget research \
  --sequence-length 256 --batch-size 1 --gradient-accumulation 4 \
  --resume checkpoints/emc-latest.pt
```

At training completion, the command loads `emc-best.pt`, prints its token count, validation loss/perplexity and routing report, then generates all fixed prompt samples. It does not blindly sample the final step.

### Generate from the best checkpoint

```sh
python -m rayvan_emc.generate \
  --checkpoint checkpoints/emc-best.pt \
  --prompt "Once upon a time there was a little boy named Sam" \
  --max-new-tokens 180 --temperature 0.8 --top-k 50
```

Use `--greedy` for deterministic decoding or `--top-p 0.95` for nucleus sampling. This is one-shot continuation, not a chatbot.

### Tiny overfit and tests

```sh
python -m rayvan_emc.experiments.overfit
python -m pytest
python example.py
```

## Router balancing and interpretation

EMC retains the weak thresholded balance objective:

```text
total_loss = language_model_loss + 0.01 * router_balance_loss
router_balance_loss = max(0, 0.75 - normalized_utilization_entropy)²
```

The dead zone allows uneven specialization; severe concentration receives smooth pressure. Diagnostics report top-1/top-2/minimum traffic, normalized entropy, effective active modules, route variation, router gradients, and module update divergence. All modules merely receiving some traffic is not sufficient evidence against collapse.

TinyStories should eventually yield coherent simple-English story continuations after enough tokens. That would not make EMC a general-purpose assistant, prove cognitive specialization, or establish an advantage over transformers. The experiment only asks whether the existing sparse circulating architecture can learn useful small-language-model behavior.

## Deliberately deferred

The current N2 stage does not include clusters beyond one four-node N2 event, adaptive halting, automatic module creation/destruction, semantic routing labels, persistent cross-request state, compact global workspaces, networking, distributed gradients, or online inference-time weight updates.

## Performance diagnostics

Run a warmed, synthetic-token benchmark before long experiments:

```sh
python -m rayvan_emc.benchmark \
  --preset research --n1-stage n1 --module-population mixed \
  --batch-size 1 --sequence-length 256 --gradient-accumulation 4 \
  --precision bf16 --warmup-steps 3 --benchmark-steps 5
```

Add `--profile --output-dir benchmark-results/research --json benchmark-results/research/report.json` for a PyTorch Chrome trace and structured report. The command uses the real forward, balance loss, backward, gradient diagnostics, clipping, and AdamW step over synthetic token batches. It does not save weights, alter model code, or run a corpus-scale training job. Reports cover phase timing, module-family isolation, Integrator/Nexus timing, one-versus-two-cycle cost, routed-compute retention, CUDA allocation, `nvidia-smi` utilization samples, and optional operator/kernel tables.
