# Routing-free heterogeneous N1 experiment

## Outcome

The native runtime now contains a second, isolated N1 architecture:

```text
token embedding
  -> shared pre-norm causal GQA + RoPE
  -> independent chunk-local expert activations
  -> sparse depth-2 GPT / SSM / recurrent / Delta execution
  -> strength-weighted token residual + compact latent attention
  -> output norm and tied vocabulary projection
```

It has no Nexus, Top-K, routing softmax, or centralized router. The existing
`LegacyNexus` path remains the default `ModelConfig` mode and retains its
original parameter names and behavior. This experiment is selected with
`N1Mode::routing_free_collective`; the native training tool defaults to it and
accepts `--mode legacy` for a controlled comparison.

The 50k-token stability gate did **not** pass. One expert was starved at the
20k and 30k observations, and the recurrent expert was completely inactive at
the final 50,176-token observation. Per the experiment protocol, no 1M-token
run was started. The architecture is implemented and testable, but this gate
configuration should not replace the legacy path yet.

A controlled rerun with RFMoE-matching `lambda_0=1e-10` (all architecture,
seed, data, batches, and other hyperparameters unchanged) also failed. It began
near-all-on at 90.625% density, but ended at 43.75% density with recurrent
activation zero. A 64-batch fixed-window audit measured only 9 recurrent
activations in 1,024 opportunities.

## Routing-free mechanism

For routing item `z` and expert `i`, the implementation follows Liu et al.:

```text
G_i(z) = ReLU(||z A_i||_2 - b_i)
f_i(z) = 1{G_i(z) >= theta}
```

Each `A_i` is a bias-free rank-16 projection owned by its expert. A routing
item is one 64-token chunk of one request, rather than one token: recurrent,
SSM, and Delta state therefore advances over intact causal chunks. The routing
representation is the variance-preserving sum of RMS-normalized pooled shared
GQA context and pooled persistent latent state.

The official reference stores `b_i=-1e-6` while evaluating `norm-bias`, whereas
the paper describes a positive `1e-6` bias. This implementation deliberately
matches the official code's stored sign. The first calibration used ordinary
Kaiming initialization and made every gate active. The retained configuration
uses `A_i ~ Normal(0, 0.02)` and `theta=1.0`, matching the scale of the reference
implementation more closely.

### Exact shared-attention configuration

The experiment uses latent width 256, RMSNorm epsilon `1e-6`, eight query heads,
two KV heads, head dimension 32, RoPE base 10,000, causal masking, no attention
dropout, a learned output projection, and a residual connection. Q and K receive
RoPE; V does not. BF16 autocast is enabled on CUDA while RMS statistics and RoPE
angles are formed in FP32. KV heads are explicitly repeated in four groups for
the ATen SDPA call because this selected fused backend 2, while this LibTorch
build's experimental `enable_gqa=true` route selected the math backend.

### Expert definitions

Every N0 has exactly two sequential blocks and uses the existing validated
family implementation:

- GPT: eight-head chunk-local causal attention plus a 6,144-wide FFN;
- SSM: 960-wide causal depthwise convolution and selective diagonal scan,
  kernel size four;
- recurrent: a 704-wide GRU path;
- Delta: two CUDA-native gated-Delta blocks, width 512, eight 64-wide heads,
  and a 5,120-wide FFN.

All operate at model width 256. Each block reads the four-slot shared latent
through the existing projected residual conditioning mechanism. GPT therefore
retains two chunk-local attention operations in addition to the shared GQA;
this is intentional preservation of its native semantics and is the main known
sequence-operation duplication.

### Deviations from Liu et al.

- Routing items are chunks rather than tokens to preserve stateful expert
  mathematics.
- Experts are heterogeneous depth-2 circuits rather than homogeneous FFNs.
- The compact routing representation includes both shared GQA context and the
  persistent latent; the paper has no equivalent EMC latent channel.
- Target density is 0.50 for this four-expert study (the inspected official
  training configuration used 0.25), and token integration uses a
  population-size `1/sqrt(N)` residual scale.
- A symmetric all-available-expert recovery handles all-off items. Only this
  exceptional path substitutes a smooth softplus proxy so dead gates retain
  gradient; it performs no ranking.
- Active expert latent proposals are RMS-normalized and integrated by masked
  latent attention, an EMC-specific communication stage after activation.

Masks are formed and compacted on the GPU. Each expert receives only its active
request rows through `nonzero` and `index_select`; an empty expert returns before
its two local blocks. Persistent state is gathered and scattered by request row,
so an inactive row remains unchanged. The only host loop is the fixed expert
population. The ordinary path performs no scalar `.item()` routing decision.
Diagnostic interventions validate their explicitly supplied availability mask
on the host, outside the normal training/inference path.

An all-off item activates every available expert symmetrically. This is a
recovery rule, not a ranking operation. Its smooth positive proxy preserves a
gate gradient; recovery is separately reported so it cannot masquerade as
healthy learned activation.

Active token proposals are multiplied by their absolute activation strengths,
summed, scaled by `1/sqrt(N)`, and added residually. There is no cross-expert
normalization. Each expert also emits a compact latent proposal. RMS-normalized
proposals provide attention keys and values for the shared latent slots, with
inactive experts masked before softmax. Bias-free K/V/output projections make a
zero-proposal intervention causally exact; a sigmoid-gated residual updates the
latent state.

## Balancing objective and telemetry

For binary activity `f`, continuous strength `G`, expert count `N`, and routing
items `Z`:

```text
L_EB = mean_i(mean_Z(f_i) * mean_Z(G_i))
L_TB = mean_Z(mean_i(f_i) * mean_i(G_i))
L_LB = mu * L_EB + (1-mu) * L_TB
L_train = L_LM + lambda * L_LB
lambda <- min(lambda_max,
              lambda * (1+eta)^sign(mean(f)-rho_target))
```

The gate used `rho_target=0.5`, `mu=0.5`, `lambda_0=1e-5`, `eta=0.02`, and
`lambda_max=1`. Lambda is a CUDA buffer and updates without a host
synchronization. Validation reports language-model loss only.

Telemetry includes activation rate, response strength mean/std, learned bias,
compute share, raw token and latent proposal norms, normalized latent proposal
norm, latent attention, latent norm, density/target/lambda, EB/TB/combined
losses, mean active and effective experts, recovery rate, co-activation,
correlation, and starvation/monopoly/all-on/all-off/scale warnings. Milestones
also record expert parameter, gradient, and update norms.

## Correctness verification

The CUDA release build passes **38/38** native tests. New coverage includes:

- shared causal GQA/RoPE prefix invariance and gradients;
- routing-free balancing formulas and gate gradients;
- genuinely sparse non-execution and symmetric all-off recovery;
- independent expert parameters and recurrent state carry;
- CUDA BF16 forward/backward and fused-SDPA backend reporting;
- routing-free checkpoint round-trip and N1-mode mismatch rejection.

The earlier failure while beginning the experiment was a BF16 state scatter
type mismatch. State destinations are now promoted to the returned state dtype
before scatter, and the CUDA BF16 test covers the repaired path.

## 50k stability gate

Both retained runs used seed 42, BF16, batch 4, sequence 256, chunk 64, learning
rate `3e-4`, weight decay `0.01`, clipping at 1.0, and depth 2 for all four
families. The dataset streams and evaluation schedule were identical.
Legacy executes exactly two experts per request; routing-free averaged 2.160 per
routing item on the fixed-window audit, so active compute is close but not
identical by design.

| metric | routing-free | legacy Top-K=2 |
|---|---:|---:|
| tokens | 50,176 | 50,176 |
| reported validation loss | 10.147350 | 10.291084 |
| reported validation token accuracy | 9.7656% | 6.8604% |
| cumulative training tokens/s | 2,712.6 | 4,310.5 |
| parameters | 44,918,133 | 44,409,846 |
| parameter bytes | 179,672,532 | 177,639,384 |
| optimizer bytes | 350,923,688 | 233,876,400 |
| peak CUDA allocation | 2,436,248,064 B | 1,276,360,704 B |

The cumulative throughput includes periodic evaluation and checkpoint work; it
is useful for the controlled comparison, not as a kernel-only rate. Routing-free
is 37.1% slower by this measure and uses 1.91x the peak CUDA allocation. The
larger persistent optimizer state reflects more parameters receiving gradients under
multi-activation, while hard Top-K leaves consistently unselected experts
without optimizer state.

Before the retained calibration, a 50,176-token trial with Kaiming-initialized
activation projections ended all-on (every expert active), with validation loss
10.252686 and accuracy 7.5806%. It was rejected as an initialization-scale
failure; its 557 tokens/s result is not used as the architecture comparison.

Activation rates at the logged evaluation observations were:

| tokens | GPT | SSM | recurrent | Delta | density | warning |
|---:|---:|---:|---:|---:|---:|---|
| 1,024 | 1.000 | 0.688 | 1.000 | 0.938 | 0.906 | none |
| 10,240 | 0.188 | 1.000 | 1.000 | 1.000 | 0.797 | none |
| 20,480 | 0.000 | 1.000 | 1.000 | 1.000 | 0.750 | starvation |
| 30,720 | 0.000 | 1.000 | 1.000 | 1.000 | 0.750 | starvation |
| 40,960 | 0.000 | 0.750 | 0.750 | 1.000 | 0.625 | starvation |
| 50,176 | 0.875 | 0.250 | 0.000 | 1.000 | 0.531 | starvation |

At the final observation, compute-share entropy was 0.97185 nats (0.7010 when
normalized by `ln(4)`), giving 2.6428 effective experts. The co-activation
matrix below contains joint activation probabilities in GPT/SSM/recurrent/Delta
order:

| | GPT | SSM | recurrent | Delta |
|---|---:|---:|---:|---:|
| GPT | 0.875 | 0.250 | 0.000 | 0.875 |
| SSM | 0.250 | 0.250 | 0.000 | 0.250 |
| recurrent | 0.000 | 0.000 | 0.000 | 0.000 |
| Delta | 0.875 | 0.250 | 0.000 | 1.000 |

The only nontrivial final activation correlation was GPT–SSM at approximately
0.218; Delta was constant and recurrent absent, so their correlations carry no
specialization information.

Raw token/latent proposal norms evolved as follows (each cell is
`GPT/SSM/recurrent/Delta`; inactive proposals contribute zero):

| tokens | token proposal norms | raw latent proposal norms |
|---:|---|---|
| 1,024 | `8.96/1.36/4.38/7.28` | `5.49/0.97/3.08/4.78` |
| 10,240 | `4.16/8.02/13.13/27.81` | `2.21/6.87/10.96/20.38` |
| 20,480 | `0/11.26/18.24/36.94` | `0/11.26/18.92/33.27` |
| 30,720 | `0/13.39/20.44/46.61` | `0/14.19/22.92/45.21` |
| 40,960 | `0/10.09/15.71/40.93` | `0/8.55/13.64/36.52` |
| 50,176 | `28.75/2.88/0/24.37` | `25.32/1.83/0/13.30` |

Despite large raw differences, an active normalized latent proposal remains
approximately norm 16; the averaged final values were `13.98/4.00/0/15.99`
because SSM activated one quarter of items and recurrent none. No configured
scale-collapse threshold fired.

There were no all-off recoveries, all-on, monopoly, scale, NaN, or CUDA-failure
warnings after calibration. Density approached the 0.5 target and lambda rose
to `2.438e-5`, but the density objective did not prevent family starvation. The
legacy control also collapsed: recurrent and Delta occupied essentially every
Top-K slot, so the failure is not unique to independent gating.

Final routing-free expert gradient norms were GPT 0.13356, SSM 0.00583,
recurrent 0.00340, and Delta 0.28285. Update norms were 0.14123, 0.08551,
0.06009, and 0.29281 respectively. These values support the starvation warning
rather than indicating a global optimization or numerical failure.

### RFMoE-matching lambda rerun

The gate was rerun from scratch with `lambda_0=1e-10`. The checkpoint config
diff against the retained `lambda_0=1e-5` run contains exactly one line: the
initial lambda. Both runs otherwise use seed 42, the same token streams and
batches, BF16, batch 4, sequence 256, chunk 64, depth 2, `eta=0.02`, `mu=0.5`,
target density 0.5, `Normal(0,0.02)` activation projections, threshold 1.0, no
per-family quota, and no forced minimum expert use.

| metric | `lambda_0=1e-10` | prior `lambda_0=1e-5` |
|---|---:|---:|
| tokens | 50,176 | 50,176 |
| reported validation loss | 10.158691 | 10.147350 |
| reported validation token accuracy | 9.7107% | 9.7656% |
| cumulative tokens/s | 2,899.96 | 2,712.59 |
| initial logged density | 0.90625 | 0.90625 |
| final logged density | 0.43750 | 0.53125 |
| final adaptive lambda | `2.2522e-10` | `2.4379e-5` |

The validation-loss trajectory was finite and generally decreasing:
`10.40341, 10.27885, 10.25286, 10.25050, 10.22363, 10.15869` at the six logged
observations. Activation evolved as follows:

| tokens | GPT | SSM | recurrent | Delta | density | warning |
|---:|---:|---:|---:|---:|---:|---|
| 1,024 | 1.000 | 0.688 | 1.000 | 0.938 | 0.906 | none |
| 10,240 | 0.188 | 1.000 | 1.000 | 1.000 | 0.797 | none |
| 20,480 | 0.000 | 1.000 | 1.000 | 1.000 | 0.750 | starvation |
| 30,720 | 0.000 | 1.000 | 1.000 | 1.000 | 0.750 | starvation |
| 40,960 | 0.000 | 0.750 | 0.750 | 1.000 | 0.625 | starvation |
| 50,176 | 0.500 | 0.250 | 0.000 | 1.000 | 0.438 | starvation |

At the final observation, the global parameter, gradient, and update norms were
188.137, 2.071, and 0.893. Per-expert gradient norms were
`0.11382/0.00623/~0/0.35116` for GPT/SSM/recurrent/Delta. No monopoly, global
all-on/all-off, recovery, scale, NaN, or CUDA warning fired.

The 64-batch fixed-window natural audit measured loss 6.829059, accuracy 9.7549%,
and activation counts `[556,256,9,1024]` from 1,024 opportunities per expert.
That is 45.04% density or 1.802 active experts per item. Counts across the four
successive chunks were `[256,256,9,256]`, `[256,0,0,256]`, `[44,0,0,256]`, and
`[0,0,0,256]`: Delta stayed on, SSM appeared only in chunk one, recurrent was
nearly absent, and GPT disappeared by chunk four. Shared-latent norm rose
smoothly `5.08 -> 8.52 -> 12.26 -> 15.79`, so this is an activation-health
failure rather than norm explosion.

The rerun therefore **fails the 50k stability gate due to expert starvation**.
Reducing only `lambda_0` to the RFMoE value did not cure the collapse, and the
protocol still forbids proceeding to 1M tokens.

## Controlled causal evaluation

The causal tool evaluated 64 fixed validation batches for both checkpoints.
Counts are active chunk-items in `[GPT, SSM, recurrent, Delta]` order; each
routing-free expert had 1,024 opportunities.

| routing-free intervention | loss | accuracy | latency/batch | counts |
|---|---:|---:|---:|---|
| natural | 6.944231 | 9.8450% | 31.025 ms | `[916,256,16,1024]` |
| disable GPT | 7.270090 | 9.8328% | 24.568 ms | `[0,256,9,1024]` |
| disable SSM | 6.955658 | 9.8495% | 27.486 ms | `[921,0,40,1024]` |
| disable recurrent | 6.945713 | 9.8450% | 29.157 ms | `[912,256,0,1024]` |
| disable Delta | 6.839743 | 6.7657% | 28.735 ms | `[1023,256,514,0]` |
| force GPT active | 6.927563 | 9.8434% | 31.993 ms | `[1024,256,16,1024]` |
| force SSM active | 6.910194 | 9.8450% | 43.305 ms | `[991,1024,12,1024]` |
| force recurrent active | 6.888042 | 9.8465% | 41.650 ms | `[1023,256,1024,1024]` |
| force Delta active | 6.944231 | 9.8450% | 29.660 ms | `[916,256,16,1024]` |

For every expert, zeroing its proposal produced the same loss and accuracy as
disabling it. Zeroing retained execution and was correspondingly slower; for
Delta it took 35.572 ms instead of 28.735 ms. This validates the intended
causal-control distinction. GPT removal raises loss by 0.326, making GPT the
clearest cross-entropy contributor. Delta removal lowers loss by 0.104 but drops
accuracy by 3.08 percentage points. SSM and the already-starved recurrent expert
have negligible natural ablation effects. Forcing SSM or recurrent reduces loss
slightly without changing accuracy, but costs roughly 12 ms/batch.

All six exact two-expert subsets were also evaluated:

| active pair | loss | accuracy | latency/batch |
|---|---:|---:|---:|
| GPT + SSM | 6.819299 | 6.7657% | 32.054 ms |
| GPT + recurrent | 6.835708 | 6.7657% | 29.495 ms |
| GPT + Delta | 6.941877 | 9.8480% | 24.760 ms |
| SSM + recurrent | 7.103892 | 7.6126% | 40.349 ms |
| SSM + Delta | 7.232256 | 9.8358% | 36.051 ms |
| recurrent + Delta | 7.231839 | 9.8373% | 35.881 ms |

This is early evidence of functional differentiation—GPT primarily helps
cross-entropy and Delta determines the observed accuracy regime—but not of a
healthy four-way collective. SSM/recurrent effects look mostly redundant or
under-trained.

Temporal activation over the four consecutive 64-token chunks was:

| chunk | GPT | SSM | recurrent | Delta |
|---:|---:|---:|---:|---:|
| 1 | 100.0% | 100.0% | 3.5% | 100.0% |
| 2 | 100.0% | 0.0% | 0.0% | 100.0% |
| 3 | 97.7% | 0.0% | 1.6% | 100.0% |
| 4 | 60.2% | 0.0% | 1.2% | 100.0% |

SSM specializes entirely to the first chunk, GPT participation falls late, and
Delta remains always on. Mean latent-attention shares were
`0.350/0.073/0.004/0.573`; attention-weighted raw-latent influence proxies were
`10.77/0.54/0.05/8.29`. Shared-latent norm rose smoothly across chunks from
5.34 to 9.29, 13.44, and 17.19 rather than exploding. The TinyStories stream
has no capability labels, so semantic input/task specialization cannot be
claimed from this run; adding labeled capability probes is required before any
scaling decision.

The legacy natural control produced loss 6.806732, accuracy 6.7657%, and 24.803
ms/batch. Thus legacy has better cross-entropy and latency on the fixed windows,
while routing-free has better argmax accuracy. The fixed-window causal metrics
use a separate deterministic sample protocol from the training tool's reported
validation summary and should be compared only within this table/protocol.

Using the natural routing-free counts and measured module sizes, the average
request activates 2.160 experts and approximately 30,537,082 parameters,
including the always-active shared path: 68.0% of the 44,918,133 total.

## CUDA phase benchmark

Twenty warmed CUDA-event iterations on the final checkpoint produced:

| phase | time |
|---|---:|
| shared GQA | 0.637 ms |
| all four local gates | 0.298 ms |
| sparse mask compaction | 0.285 ms |
| one 64-token GPT chunk | 1.096 ms |
| one 64-token SSM chunk | 3.631 ms |
| one 64-token recurrent chunk | 3.185 ms |
| one 64-token Delta chunk | 2.030 ms |
| latent integration | 0.596 ms |
| complete sparse forward | 54.130 ms |
| forward + backward | 149.994 ms |
| foreach AdamW | 6.499 ms |
| kernel-only train throughput | 6,543.4 tokens/s |

The selected SDPA backend was 2 (the fused efficient-attention path in this
LibTorch build). Explicitly expanding the two KV heads to the eight query heads
retains grouped-query semantics and allows the mature fused dispatcher; the
experimental `enable_gqa` call had selected backend 0. Benchmark inputs are
synthetic random tokens and happened to activate every expert, so their density
is not representative of the fixed validation distribution.

CUPTI recorded 3,923 CUDA launches per complete forward, 10,161 per
forward/backward, and eight per foreach optimizer step. A concurrent 200 ms
`nvidia-smi` sample observed 39–58% device utilization during the benchmark;
this short device-wide sample is descriptive, not a kernel occupancy metric.
The isolated benchmark used 2,221,662,208 B process RSS, 912,730,112 B current
CUDA allocation, 2,604,662,784 B reserved/peak-reserved CUDA memory, and
2,308,286,464 B peak allocated CUDA memory. The training-loop peak remains the
larger 2,436,248,064 B value reported above.

Parameter counts were shared 13,739,217; GPT 7,626,497; SSM 7,819,649;
recurrent 7,834,369; and Delta 7,898,401.

## Decision and next experiment

Keep the implementation as an experimental mode and retain legacy hard Top-K.
Do not run 1M tokens or scale the expert population from this checkpoint. The
next useful experiment is a balancing change that directly penalizes per-family
under-utilization over a moving window (or imposes a temporary per-expert
minimum participation schedule), followed by another 50–100k gate with the same
seed, data, and causal protocol. A density-only target can be satisfied by two
or three always-active families and therefore is insufficient for this
heterogeneous population.

The answer to the experiment's main question is therefore **no, not yet**.
Shared attention and compact normalized latent integration are numerically
stable, and the experts make measurably different causal contributions, but
decentralized activation did not prevent routing collapse. The strongest
measured cause is the density-only routing-free optimization: it can meet a
global 50% target while starving a family. Heterogeneous proposal scales and
costs likely amplify that optimization imbalance. The stable attention/latent
norms, 38/38 tests, exact zero/disable controls, and absence of CUDA failures
make shared attention, latent instability, and an implementation defect less
likely primary causes. Two blocks may still be insufficient and 50k tokens are
too few for a capability conclusion, but the predefined gate appropriately
prevents using more training to wave away the observed collapse.

Known limitations are the absent labeled capability benchmark, no exact
per-training-step inactive-run counter (starvation is measured at every logged
evaluation observation), and device-level rather than kernel-level utilization.
These do not change the failed scaling decision.

Reference: [Liu et al., Routing-Free Mixture-of-Experts](https://arxiv.org/abs/2604.00801),
[official implementation at commit `051b556`](https://github.com/liuyilun2000/RoutingFreeMoE).
