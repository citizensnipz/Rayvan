# Native EMC architecture audit

## Frozen Python reference

The native port follows `emc/rayvan_emc/n2.py`, `chunk_modules.py`, and `integrator.py`. The target is the request-routed `N2EMCModel`, not the older token-routed `EMCModel` or the chunk-router experiment.

One N2 event performs:

1. Token and absolute-position embedding addition.
2. Request-level Nexus routing from the sequence mean after LayerNorm.
3. Sparse request assignment to independent depth-`L` homogeneous N1 nodes.
4. Restoration of N1 proposals to request and selected-slot order.
5. Proposal-attention integration with Nexus weights as a log prior.
6. Final LayerNorm and vocabulary projection.

The native population is a variable-length ordered list of GPT, SSM, recurrent, and Delta N1 nodes. Repeated families construct independent modules and parameters. Delta uses the custom CUDA path on CUDA tensors and a straightforward ATen recurrence as its CPU test/reference path; there is no silent CUDA fallback.

## Routing-free experimental path

`N1Mode::routing_free_collective` is a separate construction path and does not
register the legacy position embedding, Nexus, Integrator, or legacy N1 module
list. Token embeddings first pass through pre-norm causal grouped-query
attention with RoPE (eight query heads and two KV heads by default). For every
sequence chunk, each expert independently computes a rank-16 response and a
thresholded activation. Active request rows alone execute that expert's two
native blocks; persistent block state is gathered/scattered by row.

Strength-weighted token proposals form a residual update. Expert-local compact
latent proposals update four persistent shared slots through masked multi-head
attention and a gated residual. No Top-K, Nexus score, cross-expert activation
softmax, or CPU routing decision exists in this path. See
[`routing-free-n1.md`](routing-free-n1.md) for equations, tensor behavior,
telemetry, measurements, and the 50k stability decision.

## Frozen mathematical behavior

### Shared path

For token IDs `x`, positions `p`, token embedding `E`, and position embedding `P`:

```text
latent = E[x] + P[p]
```

The current N2 reference has no additional shared-core MLP. The pre-N2 latent is the shared state for routing and N1 input. Output is:

```text
logits = Linear_output(LayerNorm_output(integrated))
```

When tying is enabled, `Linear_output.weight` is the same tensor object as `E.weight`; the native model does not allocate a second matrix.

### Nexus

```text
pooled = mean_sequence(LayerNorm(latent))
scores = Linear(pooled)
probabilities = softmax(scores)
selected_scores, selected_ids = topk(scores, K)
selected_weights = softmax(selected_scores)
```

An availability mask replaces unavailable scores with negative infinity before top-K. Forced causal routing gathers the original scores at forced IDs and recomputes selected weights with softmax. Disabling a node through availability therefore selects the natural next-best node. Routing remains differentiable through selected scores and weights; indices are discrete as in Python.

### Sparse dispatch

Assignments are flattened in request-major, slot-minor order. A deterministic expert-major permutation groups them. Each heterogeneous N1 is launched once with its selected request batch; an empty batch returns before local blocks. Concatenated expert proposals are indexed by the inverse permutation and reshaped back to `[B,S,K,D]`. There is no request or token loop and no CPU reconstruction of route metadata. The fixed host loop is over the configured N1 population.

### N1 node wrapper

Each N1 contains `L = n1_depth` independent blocks and a local shared-state initializer. For each block and sequence chunk:

```text
chunk_next = chunk + block_proposal / sqrt(L)
local_shared_next = local_shared + state_proposal / sqrt(L)
```

After all blocks, the node returns `current - input_latent`. Stateful nodes return block lease state and local shared state. GPT returns no persistent local state. N2 state is used only when explicitly passed into the next call; ordinary training batches and independent inference calls reset it.

Every block conditions its chunk with:

```text
context = Linear_shared(mean_slots(shared_state))
conditioned = chunk + context
state_proposal = Linear_state(concat(shared_state,
    expand_slots(mean_sequence(token_proposal))))
```

### GPT N1

The GPT block prepends normalized shared-state slots to the conditioned chunk. It applies pre-norm causal multi-head self-attention, an attention residual, pre-norm `Linear -> GELU -> Linear`, and an FFN residual. The token proposal removes the conditioned input after dropping shared slots. The boolean upper-triangular mask is registered once at maximum chunk-plus-state length and sliced per call. LibTorch `MultiheadAttention` uses the same projections and semantics as Python `nn.MultiheadAttention`; no custom or fused attention kernel is introduced.

### SSM N1

The SSM block preserves the Python selective diagonal scan:

```text
internal = Linear_in(LayerNorm(conditioned))
convolved = depthwise_causal_conv(concat(conv_history, internal))
delta = softplus(Linear_delta(convolved))             # FP32 scan
log_a = -softplus(log_decay) * delta
candidate = tanh(Linear_input(convolved))              # FP32 scan
a = exp(log_a)
write = (1 - a) * candidate
state[t] = a[t] * state[t-1] + write[t]
gate = sigmoid(Linear_gate(convolved))
proposal = Linear_out(gate * state)
```

The recurrence is evaluated with the same lower-triangular parallel coefficient matrix as Python. Lease state contains final diagonal state and the `kernel_size - 1` convolution history. Scan calculations are FP32 and adapters return the model dtype. No replacement SSM algorithm is used.

### Recurrent N1

The recurrent block applies input LayerNorm and projection, then one batch-first single-layer GRU initialized from the mean shared state. Lease hidden state is `[1,A,Hr]` at the LibTorch call boundary and stored as `[A,Hr]`. The whole chunk is passed to LibTorch GRU, allowing cuDNN execution on CUDA. Output projection returns the model dtype. Global BF16 autocast controls compatible CUDA operations while parameters and AdamW state remain FP32.

### Delta N1

The Delta block preserves the Python project-specific gated recurrence exactly. It projects normalized conditioned tokens to L2-normalized Q/K, tanh V, sigmoid alpha decay, sigmoid beta update, and a sigmoid output gate. Its learned initial associative memory is the outer product of projected shared-state value and key summaries divided by the square root of head width. The recurrent state and gate scalars are FP32; Q/K/V may be BF16. A custom CUDA autograd operation keeps one `[D_h,D_h]` state per batch/head in shared memory, emits token outputs, stores chunk-boundary states, and recomputes one bounded window in reverse during backward. See [`delta-native.md`](delta-native.md) for equations, memory formulas, and measured limitations.

### Integrator

For normalized latent and proposals, the Integrator constructs query `[B,S,H,Dh]` and keys/values `[B,S,H,K,Dh]`. Attention logits are scaled dot products plus:

```text
routing_prior_scale * log(clamp_min(selected_weight, 1e-9))
```

Softmax is over selected proposals. The attended value is projected. Proposal mean and population variance are computed over `K`; normalized latent, attended value, mean, and variance are concatenated. Independent linear projections produce a candidate and sigmoid vector gate. The update is residual:

```text
integrated = latent + sigmoid(gate_projection(context)) * update_projection(context)
```

Diagnostics retain detached acceptance, proposal norms, pairwise cosine similarity, value contribution norms, candidate-update norms, and mean absolute gate magnitude.

## Tensor shapes

Symbols: `B` requests, `S` tokens, `D` latent width, `N` N1 nodes, `K` top-K, `A=B*K` assignments, `A_n` assignments for node `n`, `C<=chunk_size`, `M` shared-state slots, `L` N1 depth, `H` attention heads, `Dh=D/H`, `Hs` SSM width, `Hr` recurrent width, `V` vocabulary.

| Stage | Tensor | Shape |
|---|---|---|
| input | token IDs | `[B,S]` int64 |
| shared | token embedding | `[B,S,D]` |
| shared | position embedding lookup | `[S,D]` |
| shared | embedded/shared latent | `[B,S,D]` |
| Nexus | normalized latent | `[B,S,D]` |
| Nexus | pooled descriptor | `[B,D]` |
| Nexus | scores / pre-top-K probabilities | `[B,N]` |
| Nexus | selected IDs / selected weights / slots | `[B,K]` |
| dispatch | expert/source/slot IDs | `[A]` |
| dispatch | permutation / inverse permutation | `[A]` |
| dispatch | counts / offsets | `[N]` / `[N+1]` |
| N1 | selected node input | `[A_n,S,D]` |
| N1 | local shared state | `[A_n,M,D]` |
| N1 | chunk input/proposal | `[A_n,C,D]` |
| GPT | shared-plus-token attention input | `[A_n,M+C,D]` |
| SSM | internal/convolved/scan state | `[A_n,C,Hs]` |
| SSM | convolution history | `[A_n,kernel_size-1,Hs]` |
| SSM | scan coefficients | `[A_n,Hs,C,C]` |
| recurrent | GRU input/output | `[A_n,C,Hr]` |
| recurrent | GRU hidden at call | `[1,A_n,Hr]` |
| Delta | Q/K/V | `[A_n,C,Hd,Dd]` |
| Delta | alpha/beta | `[A_n,C,Hd]` FP32 |
| Delta | recurrent memory | `[A_n,Hd,Dd,Dd]` FP32 |
| N1 | grouped proposal | `[A,S,D]` |
| N2 | restored proposal stack | `[B,S,K,D]` |
| Integrator | Q | `[B,S,H,Dh]` |
| Integrator | K/V | `[B,S,H,K,Dh]` |
| Integrator | per-head acceptance | `[B,S,H,K]` |
| Integrator | acceptance | `[B,S,K]` |
| Integrator | integrated state | `[B,S,D]` |
| output | logits | `[B,S,V]` |

Runtime checks cover public inputs, router masks, top-K, dispatch rank, N1 latent/shared state, Integrator shapes, forced routes, and causal masks.

## Implementation-only memory choices

- Causal attention and SSM scan masks are registered buffers, allocated once per block and sliced as views. LibTorch's C++ registration API serializes them in native checkpoints.
- Tied embedding/output storage is one tensor.
- Dispatch tensors stay on the input device. Proposal restoration uses one inverse index-select and a view/permute.
- Empty expert batches return before N1 blocks, so unselected nodes do no expensive local work.
- Evaluation checkpoint loading opens only `model.pt`; `optimizer.pt` is a separate file.
- Diagnostics accumulate reductions on-device and materialize reports only when requested.
- Stateful tensors use RAII-owned `Tensor` handles. No unsafe in-place mutation is applied to autograd-visible model values.
- Delta never materializes `[B,H,T,D_h,D_h]` transitions. Its backward scratch is computed with checked integer arithmetic and refused before launch when it exceeds `delta_max_scratch_bytes`.

The exact diagonal SSM scan still constructs `[A_n,Hs,C,C]` coefficients because that is the Python reference mathematics. Replacing it is an architecture/algorithm change and is not part of this port.
