# Rayvan EMC research prototype

EMC (Emergent Modular Cognition) is an experimental computation graph. Token embeddings form a shared latent state; a learned Nexus selects one module for the current computational need; that module proposes a transformation; and a learned Integrator gates the update into shared state. Nexus then reevaluates the changed state before the next fixed trajectory step. Normal EMC never chooses a parallel expert set in advance.

## N1 components

- **Geometric Nexus (primary):** mean-pools the current state, applies a LayerNorm–MLP NeedEncoder, and L2-normalizes the result `z`. Every expert owns `P` trainable, L2-normalized competence prototypes. Its base action is the nearest squared Euclidean distance on the unit sphere, `a_i = min_p ||z - μ_ip||² = min_p (2 - 2 cos(z, μ_ip))`; routing is direct `argmin` over effective action, with no classifier after the distance. Prototypes are randomly initialized and carry no task or role labels.
- **Legacy module-aware Nexus:** projects state to an unnormalized query, takes scaled dot products with learned module descriptor keys, and selects by adjusted score. It remains available as `sequential_module_aware_emc`; Top-K remains only in explicitly labeled parallel/N2 comparisons.
- **Heterogeneous modules:** every family implements `EMCModuleBase.forward([B,S,D]) → proposal [B,S,D]`. The existing GPT-style family remains. A pure-PyTorch selective diagonal state-space family and a GRU recurrent family add different established computation. Input/output adapters permit different internal widths.
- **Sequential acceptance Integrator:** receives current state, the selected expert's raw proposal, a learned expert identity, and `log(1 + proposal norm)`. A bounded scalar gate per token produces `S_next = S + sigmoid(gate_features) * proposal`; proposal magnitude is measured and is not normalized away. Multi-proposal attention remains only in legacy paths.
- **Fixed trajectory and output:** the integrated latent becomes the next trajectory step's input. No adaptive halting or persistent private memory is introduced.

State-space and GRU state exists only while scanning one sequence in one module forward. It resets between EMC cycles, batches, and inference calls. Modules never call each other and communicate only through proposals and the Integrator.

The pre-N1 `NexusRouter` and `WeightedAverageIntegrator` remain available for isolated baselines. No new dependency is required: the state-space implementation uses ordinary PyTorch rather than the CUDA/custom-kernel-oriented `mamba-ssm` package.

This research area remains independent from Rayvan's Rust networking code. It contains no networking, distributed execution, pretrained model weights, instruction tuning, semantic module roles, or custom CUDA path.

## N2 CUDA execution

`--model n2_emc` builds one request-routed N2 event over four depth-3 N1 nodes. The mixed population is GPT, SSM, recurrent, and DeltaNet. The default `--n2-execution-mode sparse` keeps top-K semantics: Nexus scores the whole request batch with one projection, GPU tensors sort the selected request/slot assignments by expert, each heterogeneous N1 receives one batched request tensor, and one inverse permutation restores proposals before matrix-native integration. Unselected requests do not execute an N1's local blocks. The only host loop is the fixed four-family launch boundary required by the four different computational graphs; no request or token is dispatched through Python.

Routing, dispatch metadata, proposal tensors, and streaming diagnostic reductions remain on the accelerator. CPU materialization occurs when a diagnostic report or milestone snapshot is requested. `--n2-execution-mode dense` is an experimental training control that executes all four N1s with continuous Nexus probabilities. `--n2-cuda-streams` enables experimental concurrent family streams; both controls are off by default.

The CUDA path uses PyTorch operations only. It does not add custom CUDA/Triton kernels or an external MoE framework. PyTorch grouped GEMM is useful only for compatible homogeneous projections; the heterogeneous N1 graphs remain separate.

## Legacy parallel chunk-routed N1

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

Grouped configuration is carried by `EMCConfig`. The primary `n1_sequential` stage performs `trajectory_steps` route/execute/integrate transitions with exactly one expert per step. Nexus recomputes `z` and all basin distances from the updated latent after every accepted transformation. Refractory pressure adds to effective action after base geometry, decays geometrically, and resets for each independent forward; it discourages but never prohibits repetition and never moves a basin. Optional loss-free balance is a separately reported additive action bias and defaults off for geometric EMC.

Counterfactual calibration samples at most one trajectory position per example in a forward, subject to a bounded per-forward budget. The default deterministic schedule uses 8% before step 1,000, 2% through step 9,999, and 1% afterward; a fixed rate and optional small-margin trigger are available. A probe runs every expert and the same acceptance Integrator from the identical detached pre-routing state. Candidate task losses are converted to `softmax(-loss / τ_probe)` and train the NeedEncoder/prototypes by cross-entropy against `softmax(-base_action / τ_geometry)`. All candidate task losses are evaluated under `no_grad`, so unselected experts and the Integrator are not trained by probe branches; the real selected path still receives ordinary task gradients.

Geometric warnings use explicit thresholds: basin separation below 0.05, mean action margin below 0.01, routing regret above 0.25 nats/token after 10 probes, accuracy no better than uniform chance after 50 probes, prototype drift below 1e-5 after 100 routes, monopoly at 98%, never-selected after 100 routes, and train/evaluation routing-frequency L1 distance above 0.25. Unequal utilization by itself is not a failure.

`n1_chunked` now names the legacy parallel comparison path: it selects Top-K modules for each chunk, executes simultaneous proposals, integrates the proposal set, and may retain bounded module leases across chunks. It is exposed by the Research Console as `legacy_parallel_emc`; the normal `emc` identifier is geometric and never reads or applies Top-K. The previous sequential scorer is `sequential_module_aware_emc`, and the older token-routed parallel model remains `old_emc`. Schema-v2 `emc` configs migrate to the explicit sequential legacy identifier so historical semantics are preserved; ambiguous schema-v1 configs remain rejected.

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
