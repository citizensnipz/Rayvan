# Competence-basin routing experiment

## Outcome

The native routing-free collective now uses learned computational-need geometry
and success-shaped competence memory instead of the independent norm/bias
election gate. The experiment completed 1,000,448 tokens on the unchanged
10-lane `CapabilityTaskSuite` configuration (seed 42, BF16, batch 4, sequence
256, chunk 64, latent 256, depth 2, GPT/SSM/recurrent/Delta, vocabulary 100).

The main hypothesis is **not supported by this run**. The mechanism became
sparse and mostly deterministic and produced more capability-conditioned
assignment than the prior routing-free gate, but GPT and Delta starved. SSM and
recurrent split essentially all deterministic work. No utilization quota,
load-balancing loss, centralized router, or Top-K operation was used.

## Implemented algorithm

For each chunk-cycle, the shared latent is pooled and passed through a learned
two-layer SiLU projection:

```text
z = normalize(W_out SiLU(W_in RMSNorm(mean_slots(h))))
```

Every expert owns four checkpointed competence basins. For each expert, the
runtime chooses only its nearest local basin and evaluates exactly:

```text
d[i,k] = ||z - mu[i,k]||^2 / (r[i,k]^2 + 1e-6)
k* = argmin_k d[i,k]
E_i = d[i,k*] - 64 q[i,k*]
g_i = sigmoid((1 - E_i) / 0.25)
```

The optional compute-cost coefficient is zero. Training samples independent
Bernoulli activations from `g_i`; evaluation uses `E_i < 1`. There is no
cross-expert ranking. `min_i E_i > 2` marks novelty. Novel or low-confidence
items sample one additional eligible expert from weights
`sigma_i/sqrt(n_i+1)`. An exploratory contribution uses participation weight
one so an uninitialized basin can receive a measurable backward signal.

The ordinary LM backward retains the integrated chunk gradient and computes,
for active experts only:

```text
u_i = -mean(dL/dh_out * (g_i p_i)) * 8192
u_i = clamp(u_i, -1, 1)
```

The mean is over proposal/gradient dimensions. Competence is an EMA of `u`;
evidence increments once per observation. Center and radius use only positive
utility, following the requested equations. Utility residual variance plus a
32-observation prior floor determines uncertainty. Novel positive observations
create an unused basin, or replace only an immature/weak basin. At most one
basin per expert is created in a cycle so a minibatch cannot fill all slots
with duplicate centers. CUDA RNG state is checkpointed because election is
stochastic during training.

## Final balanced evaluation

The evaluation contains 925 usable examples (75 are over the fixed context
limit) and 9,397 answer tokens. Activation order is
`[GPT, SSM, recurrent, Delta]`.

| lane | loss | token accuracy | exact accuracy | activation rate |
|---|---:|---:|---:|---|
| language | 0.2901 | 97.81% | 5.00% | `0/.593/.407/0` |
| associative recall | 17.4319 | 30.86% | 0.00% | `0/.583/.417/0` |
| fuzzy recall | 2.4962 | 69.94% | 1.25% | `0/.675/.325/0` |
| selective copying | 7.1452 | 30.70% | 0.00% | `0/.553/.447/0` |
| working memory | 18.1713 | 7.19% | 6.00% | `0/.451/.549/0` |
| compression | 4.1728 | 37.67% | 0.00% | `0/.574/.426/0` |
| arithmetic | 11.7788 | 40.79% | 4.00% | `0/0/.940/0` |
| symbolic | 14.5935 | 15.79% | 9.00% | `0/.130/.783/0` |
| program execution | 3.7788 | 38.24% | 10.00% | `0/0/1/0` |
| stateful action | 10.6082 | 52.44% | 34.00% | `0/.412/.488/0` |
| **overall** | **2.6123** | **80.71%** | **7.46%** | **`0/.465/.516/0`** |

Overall density is 0.2452 (0.9809 active experts/item). Assignment entropy is
0.4990 normalized by `ln(4)`, or approximately 1.997 effective experts.
Independent resonance entropy is 0.04385 nats (6.33% of the Bernoulli maximum).
Evaluation novelty is 0.868%. Capability/expert mutual information is 0.11130
nats and normalized mutual information is 0.08931. GPT and Delta are starved;
there is no single-expert monopoly, but the two-family collapse is decisive.

## Checkpoint trajectory

Balanced evaluation metrics and cumulative training-routing counters:

| tokens | eval loss | accuracy | exact | eval density | resonance H | NMI | train density | train novelty | train exploration |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100,352 | 2.7976 | 25.99% | 0.00% | .246 | .0480 | .0873 | .362 | 4.97% | 13.66% |
| 250,880 | 1.3087 | 74.47% | 2.16% | .243 | .0271 | .0813 | .294 | 1.99% | 5.47% |
| 500,736 | 1.5092 | 79.43% | 6.05% | .245 | .0281 | .0846 | .272 | 1.00% | 2.74% |
| 750,592 | 1.9337 | 80.21% | 7.46% | .246 | .0399 | .0905 | .275 | 0.72% | 2.86% |
| 1,000,448 | 2.6123 | 80.71% | 7.46% | .245 | .0439 | .0893 | .269 | 0.54% | 2.15% |

The 500k-750k interval briefly renewed exploration (3.12% of expert
opportunities) as the learned need geometry moved; otherwise post-100k interval
exploration was effectively zero. This is natural confidence/evidence decay,
not a scheduled utilization rule.

Training throughput (including scheduled validation/checkpoint overhead) rose
from 6,186 tokens/s at 100k to 7,834 tokens/s at 1M. Interval throughput was
6,186, 8,271, 8,226, 7,792, and 8,101 tokens/s across the five ranges. The
temporary throughput dip coincides with renewed exploration. Thus this run did
become sparser and faster than its early phase, although final throughput was
only 1.7% above the prior routing-free run because competence updates add many
small tensor operations.

## Basin evolution

All experts had filled four slots by 10k tokens (after the first 1,024 tokens,
counts were `3/1/3/2`). Evidence totals from 100k through 1M were:

| expert | 100k | 250k | 500k | 750k | 1M |
|---|---:|---:|---:|---:|---:|
| GPT | 299 | 299 | 299 | 458 | 458 |
| SSM | 1,008 | 2,750 | 5,639 | 8,543 | 11,431 |
| recurrent | 541 | 1,129 | 2,105 | 3,146 | 4,122 |
| Delta | 314 | 324 | 353 | 553 | 647 |

Mean uncertainty trajectories were GPT
`.572/.572/.572/.514/.514`, SSM `.468/.447/.438/.284/.269`, recurrent
`.583/.557/.544/.360/.340`, and Delta `.577/.573/.564/.510/.500`. Mean radii
were effectively fixed near `.502` for GPT and `.500` for Delta, while SSM
contracted `.469 -> .304` and recurrent `.500 -> .338`. The center RMS movement
between checkpoints was negligible except at 500k-750k, when SSM moved 0.388
and recurrent 0.459 in the normalized embedding. This matches the renewed
low-confidence exploration interval.

Final basin competence vectors were:

```text
GPT       [-.00659, -.00220, -.00614,  .00107]
SSM       [ .00154,  .00036,  .00555,  .00090]
recurrent [ .00203,  .00522, -.00491,  .00110]
Delta     [-.00001, -.00145, -.00137, -.00390]
```

Final expert marginal-utility EMAs were `-.00527/.00154/.00203/-.00001`.
Negative GPT/Delta utility lowered competence without moving their centers,
while SSM/recurrent accumulated evidence and lower uncertainty. This is the
specified success-shaping behavior, but in this run it reinforced a two-expert
collapse. Full 16-dimensional center trajectories, per-basin radii, evidence,
competence, and uncertainty are retained in each milestone evaluation JSON.

## Direct comparison with the prior routing-free 1M run

| metric | competence basins | prior norm gate |
|---|---:|---:|
| balanced loss | 2.6123 | 1.7468 |
| balanced token accuracy | 80.71% | 81.58% |
| balanced exact accuracy | 7.46% | 10.59% |
| activation density | 24.52% | 55.28% |
| mean active experts | 0.981 | 2.211 |
| capability/expert NMI | 0.08931 | 0.06154 |
| normalized assignment entropy | 0.4990 | 0.6961 |
| final training throughput | 7,834 tok/s | 7,704 tok/s |
| starvation | GPT + Delta | GPT |

The new geometry improved NMI by 45.1% and cut activation density by 55.6%, but
loss rose 49.6%, token accuracy fell 0.87 percentage point, exact accuracy fell
3.14 points, and starvation worsened from one to two experts. The result is
therefore useful evidence that success-shaped geometry can produce a sparse,
task-conditioned two-family partition, but not evidence for healthy emergent
specialization across the unchanged heterogeneous population.

## Verification and artifacts

The release CUDA build passes 39/39 native tests. Coverage includes the exact
distance/resistance equation, normalized need embedding, stochastic novelty
exploration, positive-only center/radius updates, negative competence updates,
novel creation, sparse/empty execution, ordinary-backward utility capture,
checkpoint round-trip, CUDA BF16, and explicit absence of routing auxiliary
loss.

Run artifacts are in `build/experiments/capability-competence-basin/standard-1m`:
`telemetry.tsv`, checkpoints at 100k/250k/500k/750k/1M, and a full capability
evaluation JSON for every milestone. The controlled predecessor is retained in
`build/experiments/capability-routing-free/standard-1m`.
