# Sequential counterfactual value routing: implementation and experiment protocol

This branch adds `counterfactual_value_emc` to the Python research console. It is
an experimental implementation of continuation-value geometry and controlled
expert development. Existing geometric, module-aware and parallel architectures
retain their routing behavior and checkpoint identities. The selective diagonal SSM now has batched projections and optional fused CUDA execution.

## What is implemented

The deployment decision is one expert per request at each trajectory step. The
router recomputes its decision after integration, and the same expert can be used
again. There are no utilization quotas, refractory penalties, persistence bonuses
or manual expert roles in this architecture.

For observed latent prefix $H_t$ and remaining budget $r=T-t$, the router computes
an order-sensitive attention summary, concatenates the last state and $r/T$, and
uses a small MLP to obtain $z=f_\phi(H_t,r)$. Attention keys include learned
positional embeddings; permuting earlier tokens while keeping the endpoint fixed
can change the summary. No need or expert vector is normalized to unit length.

$$
a=Wz+b,\qquad \hat A=a-\frac1E\mathbf1\mathbf1^\top a,\qquad
\pi(H_t,r)=\arg\min_{e\in\mathcal E_{available}}\hat A_e.
$$

`ValueNexusRouter` implements these equations. `metric()` returns the induced
pseudometric $G=W^\top P W/E$, where $P=I-\mathbf1\mathbf1^\top/E$.
Consequently $(z-z')^\top G(z-z')$ is exactly the mean squared difference between
the two predicted expert advantage profiles. This metric is diagnostic; routing
uses the value table directly. The representation is useful only if held-out
expert effects are predictable; the geometry itself is not proof of competence.

The identity-free Integrator implements

$$
H_{t+1}=H_t+\sigma(g_\eta[\mathrm{LN}(H_t),\Delta_e(H_t),
\log(1+\|\Delta_e(H_t)\|)])\odot\Delta_e(H_t).
$$

The scalar gate is per position. Its final affine layer starts at zero, hence
initial acceptance is exactly 0.5 for every proposal. Expert identity is not an
input. Proposal content and scale remain visible to the gate.

## Information boundary and budget units

Training scores **only the next token after the observed prefix**. If a sampled
window has `inputs[:, :L]` and shifted targets, the valid training target is
`targets[:, L-1]` and the scored logit is `endpoint(inputs)[:, -1]`. All expert
and router inputs are inference-available. Delta's full-prefix conditioning is
permissible under this objective.

The public `forward()` computes each all-position prediction from a separate
prefix. Therefore existing capability evaluations remain causal even when they
pass a complete teacher-forced answer. A returned trajectory trace describes the
last evaluated prefix, not all earlier prefixes. Generation calls `endpoint()`
directly and needs only one trajectory per generated token.

The console's **supervised endpoint budget** counts one target per request.
`tokens_processed` and learning-curve x coordinates use this unit for this
architecture. Reading 256 context tokens does not count as 256 supervised targets.
`costs.input_tokens` separately counts training context tokens. Do not compare an
old all-position run's token-axis curve directly with a new endpoint run.

The state-pool batch multiplier concatenates the nominal accumulation batches
into one in-memory block. It increases the candidate-state pool and memory use;
it is not microbatch memory-saving gradient accumulation. Start at 1.

## Counterfactual targets and router learning

At the beginning of each alternating block, `frozen_snapshot()` copies the whole
model, switches it to evaluation mode and freezes every parameter. State
collection, labels and developmental suffixes use this immutable version.
The current minibatch supplies fresh state batches at **all** trajectory depths;
no state or label replay survives a snapshot refresh.

For a sampled request/depth and each expert $e$, force $e$ once, apply the real
Integrator, then reroute every remaining step on that branch's changed state:

$$
Y_e=\ell_{endpoint}\left(F_{\bar\theta}^{\bar\pi}
       (H_t; e, T-t),y\right).
$$

This is a terminal-loss rollout under the frozen continuation policy, not an
estimate of globally optimal continuation. Cost price $\lambda=0$ in this first
implementation. There is no invented communication or compute-cost penalty.
`value_target=immediate` is an explicit control that stops the measurement after
one integrated update; developmental learning still uses the full suffix.

Each request is independently eligible with `value_probe_rate`, receives a
uniformly sampled depth, and eligible requests are shuffled before the block cap
is applied. Thus the cap cannot systematically favor early depths or request
indices. All experts are enumerated only for these sampled measurements.

$$
L_{router}=\frac1{|\mathcal P|E}\sum_{s\in\mathcal P}\sum_e
 [\hat A_e(\operatorname{stopgrad}H_s,T-t_s)
 -(Y_e-\overline Y)]^2.
$$

Both labels and router state inputs are detached. Router parameters learn;
upstream experts and shared embeddings receive no calibration gradient. Targets
keep their numerical scale and ties. No winner-classification softmax is used.
Each label carries `label_snapshot_version`. It is consumed within the block
that measured it; expert/shared changes make it at most one alternating block
old when the next inference policy is evaluated.

## Controlled expert development

The controlled arm blocks all main-trajectory parameter gradients into experts.
Every scheduled practice round gives **each** expert one optimizer batch with
exactly `value_development_batch_size` sampled valid items, independently of its
traffic. Experts have separate AdamW state and clocks. This equalizes training
opportunity counts, not inference usage or heterogeneous FLOPs.

Let the empirical reference bank contain $N=B T$ freshly collected states,
with equal coverage of trajectory depths. The pool includes multi-step deviations:
each collection decision independently explores uniformly with probability
`value_exploration_rate`. For each expert:

$$
r_e(s)=\mathrm{softmax}_e(-\hat A(s)/\tau),\qquad
q_e(s)=\rho/N+(1-\rho)\frac{r_e(s)}{\sum_{s'\in bank}r_e(s')}.
$$

`curriculum_probabilities()` implements this normalized mixture exactly on the
current empirical bank. During the first `value_warmup_steps` blocks, $\rho=1$.
Afterward it uses `value_common_fraction`, initially 0.5. Different experts sample
independent batches with replacement from their $q_e$. A small bank gives a noisy,
local specialist curriculum; no global curriculum-estimation claim is made.

The candidate expert is live at exactly one insertion:

$$
L_e=\mathbb E_{s\sim q_e}\ell_{endpoint}
\left(F_{\bar\theta}^{\bar\pi}
 [I_{\bar\eta}(\operatorname{stopgrad}H_s,
 \Delta_{\theta_e}(\operatorname{stopgrad}H_s))],y\right).
$$

Suffix parameters are frozen **but their operations retain gradients with
respect to the input state**. `insertion_loss()` must not run the suffix inside
`no_grad()`. If the same expert appears again in the suffix, it is the separate
snapshot copy, not the live insertion. A focused test checks the resulting
analytic gradient against a scalar reference.

The shared embedding, readout and gate train separately on exploratory full
trajectories with expert parameters fixed. Router fitting then uses measured
snapshot states and losses. This is a local insertion surrogate with hard,
piecewise-constant suffix choices, not the full gradient of a tied trajectory.

`ordinary` removes protected practice and learns experts on traffic-selected
main trajectories. `frozen` learns only the router, freezing both expert and
shared parameters. The console now requires a trained value-EMC checkpoint for
router-only runs. Open its saved report and use **Test router from checkpoint**;
this copies the suite, expert composition, horizon and model dimensions. A warm
start loads weights with fresh optimizers, not the source training clock. Incompatible
model settings or tokenizers are rejected rather than silently remapped.

This design removes the direct traffic-count feedback loop in the controlled
arm. It cannot guarantee useful specialization or prevent every monopoly:
shared coadaptation, curriculum feedback, insufficient practice and unseen
multi-expert combinations can still lock in an inferior solution.

## Telemetry and persistence

At validation checkpoints, the audit re-encodes the same held-out raw prefixes
under current parameters, measures every candidate at every depth with rerouted
suffixes, and reports:

- decision regret: mean $Y_{chosen}-\min_e Y_e$;
- pairwise cost-gap RMSE and centered-advantage RMSE;
- best evaluated current-action loss, chosen loss, and near-tie accuracy;
- greedy held-out route counts by depth;
- actual expert update batches, training items, sampled gradient norms and
  parameter-change norms.

Audit labels never enter router training or specialist sampling. Audit size is
`batch_size * evaluation_batches` prefixes, repeated at every depth. This is a
small fixed validation panel, not a statistically conclusive independent test set.
Use separate seeds/raw examples for final hypothesis evaluation. Charts do not
relabel old audit measurements as fresh observations on subsequent training steps.

`costs` separately counts collection, probe, development, shared, evaluation and
audit **expert-items** (one request passed through one expert). Development items
include their frozen continuation calls. Counts are forward-work proxies, not
FLOPs: backpropagation, family cost differences, snapshot copies and kernel
launches still require wall-time/profiler measurements. Per-block duration includes
snapshot/probe/practice overhead. Reported training elapsed time includes validation;
the final capability-report pass occurs afterward and has its own evaluator times.

Latest/best checkpoints store a distinct `emc_counterfactual_value` model type,
all expert/shared/router optimizer states, their clocks, sampler RNG, snapshot
version and cost counters. Exact CPU checkpoint continuation is tested. Legacy
milestone/causal/Top-K scheduling is not the value trainer's diagnostic path.

## Run locally

From the repository root:

```powershell
python -m pip install -e "./emc[test]"
cd emc
python -m pytest -q
python -m rayvan_emc.research run experiments/value-routing/smoke.json --runs-dir runs
```

The smoke preset uses CUDA/FP32, eight Delta copies, 16-wide states, context 256,
64 endpoints, two practice items per expert, 100% probe eligibility with cap one,
and an eight-block common warmup. It checks execution and recording; its loss is
not evidence for specialization. Change `training.device` to `cpu` for CPU use.

The original joint-training pilot used the following settings (historical; use the router-only protocol below for the next test):

| Control | Value |
|---|---|
| Suite | 10-task Mixed Diagnostic |
| Architecture | Sequential EMC — Counterfactual Value |
| Experts | 8 Delta; 0 GPT/SSM/recurrent |
| Trajectory steps / need dimension | 3 / 8 |
| Counterfactual target / expert learning | Final rerouted trajectory / Controlled |
| Common fraction / specialist temperature | 0.5 / 0.25 |
| Common-only warmup | 100 blocks |
| Practice interval / items per expert | Every block / 4 |
| Training exploration / probe probability / cap | 0.1 / 0.08 / 1 |
| Size | Custom; latent 64, hidden 128, heads 4 |
| Context / supervised endpoint budget | 256 / 4,096 |
| Batch / state-pool multiplier | 4 / 1 |
| Learning rate / weight decay | 0.0003 / 0.01 |
| Precision / device | FP32 / CUDA initially |
| Validation cadence / batches | 25 blocks / 4 batches |
| Seed | 42, then repeat 43–46 |

The corresponding JSON is `experiments/value-routing/delta8-suffix-controlled.json`.
The directory also contains the other three factorial arms:

| Counterfactual label | Expert opportunity | Config |
|---|---|---|
| Suffix | Controlled | `delta8-suffix-controlled.json` |
| Immediate | Controlled | `delta8-immediate-controlled.json` |
| Suffix | Traffic-driven | `delta8-suffix-ordinary.json` |
| Immediate | Traffic-driven | `delta8-immediate-ordinary.json` |

All four use the same causal endpoint objective and identity-free gate. Compare
paired seeds. First compare at equal endpoint budgets; then equalize actual total
training compute/time before crediting the scheduler. Practice has real cost.
Do not interpret a same-endpoint gain bought with substantially more expert work
as a demonstrated efficiency gain. Repeat with the 2/2/2/2 population using
`mixed8-suffix-controlled.json`, and make matching factorial copies.

Expected support: controlled expert counters remain equal even when route shares
do not; held-out cost-gap prediction and downstream regret improve; final task
quality gains survive compute matching and repeated seeds. Falsifiers include
low routing regret with a poor expert capability envelope, no downstream advantage
over immediate labels, or persistent inferior outcomes determined by early winners.
Longer 50K-endpoint runs should follow working diagnostics, not replace them.

## Validation and limits

Implementation validation in the development environment: **194 Python tests
passed, 3 CUDA-dependent tests skipped**; TypeScript and the Vite production build
passed. The supplied eight-Delta smoke preset also completed on CPU, evaluated
all ten capability examples without skips, and recorded 16 update batches for
every expert. These are execution/contract checks, not evidence that the routing
hypothesis improves learning. CUDA FP32/BF16 development tests are included for
the local GPU run.

Tests cover the centered score and metric identities, detached calibration,
order and remaining-step sensitivity, causal logits (all four expert families),
exact one-expert execution, repeated routing, an immediate-versus-delayed-benefit
counterexample, differentiable frozen suffixes, normalized curricula, unbiased
probe allocation, monopoly-resistant opportunity counts, shared/route-only freeze
boundaries, all four mixed-family factorial paths, expert permutation invariance,
configuration persistence, backend events and exact checkpoint resume.

The initial implementation is intentionally for small pools. Inference scoring is
$O(BEd)$ plus the need encoder at each step; it never executes all experts to
choose a route. Dispatch retains the repository's grouping loop over selected
expert IDs. Probes cost up to $E(T-t)$ expert-items per sampled state; controlled
practice and snapshot storage scale with the population. There is no ANN index,
uncertainty bandit, differentiable routing relaxation, native distributed executor,
nonzero cost price or optimal-trajectory guarantee in this version.


## Router isolation and calibration update

The value head starts with W=0 and b=0. The encoder has its own `value_router_seed`;
resetting it does not change expert/shared parameters or the data RNG. A zero head
has no expert preference, but greedy argmin would break ties by index. Therefore
**neutral initialization alone is not the mechanism**: during calibration,
collection and the shared training trajectory choose experts uniformly at each
step, and every request is eligible for probing (the configured cap still applies).
Greedy training traffic is enabled only when both conditions hold:

$$k>K_{calibration}\quad\text{and}\quad N_{probes}\geq N_{minimum}.$$

Defaults are 64 blocks and 64 probes. Afterwards joint training uses the configured
exploration/probe rates. `value_warmup_steps` remains the separate common-practice
schedule. Calibration counters are saved/restored in training checkpoints. With a
zero head the first gradient update trains W/b; encoder gradients begin once W
becomes nonzero. Setting both calibration thresholds to zero is an explicit ablation.

In frozen, fixed-reference mode, the source checkpoint's router is preserved **before**
resetting the trainable router. Collection stays uniformly random throughout this
test, while all suffix labels use the preserved checkpoint policy on each branch's
changed latent. Experts, shared parameters and that reference policy never learn.
Thus the target is stationary:

$$Y_e(s)=\ell(F_{\theta_*}^{\pi_*}(s;e),y),\qquad
L=\operatorname{MSE}(\hat A_\phi(s),Y(s)-\overline{Y(s)}).$$

The collection/probe RNG is independent of the router seed. No held-out labels are
used for fitting. Fixed validation prefixes and randomized collection routes are
recreated deterministically; initial and final audits measure the same targets.
`held_out.target=fixed_reference_suffix` distinguishes this test from the ordinary
live-policy suffix audit. The preserved reference router is saved with optimizer
state so CLI checkpoint resume keeps the same measuring policy.

This tests supervised competence prediction under one continuation policy. It does
**not** establish that repeatedly substituting the learned router improves full
trajectories: actual greedy endpoint validation runs separately. A subsequent test
can disable fixed continuation to study policy improvement with changing labels.
The console skips the expensive final capability-generation report for router-only
runs; it still performs endpoint validation and all-candidate routing audits.

### Next console test

1. Open **History**, open an existing trained value-EMC run, then click
   **Test router from checkpoint**. Prefer the completed checkpoint over the
   188-endpoint cancelled run. Keep its expert composition: eight Delta or 2/2/2/2
   are both supported; changing composition requires a different checkpoint.
2. The shortcut fills: router-only, reset router enabled, fixed checkpoint
   continuation enabled, suffix targets, calibration 64 blocks / minimum 64 probes,
   probe probability 1.0, cap 4, 1,024 endpoints, validation every 64 blocks with
   16 batches. Keep batch 4, multiplier 1, original context/dimensions and data seed.
3. Use learning rate 0.0003, weight decay 0.01, CUDA, precision Auto, SSM backend Auto.
4. Run router seeds 0, 1 and 2 **from the same original checkpoint**, keeping all
   other settings fixed. Do not use one router test's output as the next source.

Watch held-out decision regret, pairwise gap RMSE, initial regret, uniform regret,
and the constant-per-step baseline. That baseline chooses one expert at each depth
from accumulated **training** counterfactual losses, then evaluates it on held-out
states; it does not select the baseline on validation labels. Lower regret than
uniform shows useful selection; beating this constant-per-step baseline is stronger
evidence for state-dependent routing. Agreement across three seeds is preliminary
robustness evidence, not a guarantee of seed independence or useful specialization.

Expert update counts must remain zero. Training route shares include randomized
collection; read **greedy audit counts** to assess the learned router's preference.
Poor regret across all seeds falsifies this router's current predictive usefulness;
a stable global winner can be real competence, or evidence that the checkpoint has
little conditional specialization. This test cannot repair its experts.

### SSM execution and verification

All token-dependent projections are now batched. The portable `parallel_scan` uses
associative composition `(a2,b2) o (a1,b1) = (a2*a1, b2+a2*b1)` in O(log L) tensor
stages, with O(BLD log L) work. The optional C++/CUDA extension fuses the complete
recurrence into one forward launch, parallel across request/channel lanes, with the
time loop inside the kernel. It has O(BLD) work; it is not a parallel-time scan.
Backward uses the exact adjoint recurrence

$$q_t=g_t+a_{t+1}q_{t+1},\qquad
\frac{\partial L}{\partial b_t}=q_t,\qquad
\frac{\partial L}{\partial a_t}=q_t h_{t-1}.$$

It supports first-order gradients through the latent even when expert parameters
are frozen, uses the current CUDA stream/device, and accumulates in FP32. No decay
product divisions or fast-math flags are used. FP64 portable checks compare both
outputs and all parameter/input gradients against the original token loop. Reduced
precision rounding and changed operation order mean bitwise parity is not promised.
The CUDA custom backward is first-order only; higher-order differentiation requires
the portable backend.

`Auto` attempts a cached JIT build once per process on CUDA; missing tools/build
failure produce a warning and select the optimized PyTorch scan. Explicit `cuda`
fails rather than falling back. Runtime telemetry names the backend actually used.
Install `python -m pip install -e "./emc[test,kernels]"`; the fused backend also needs
a compatible CUDA toolkit (nvcc) and C++ compiler (MSVC Build Tools on Windows).
The JIT build occurs before training throughput timing. The CUDA kernel's speedup
and CUDA gradient tests still need verification on the user's GPU; neither is
established by CPU tests. Narrow request/channel grids may underutilize a GPU, so
compare against `parallel_scan` instead of assuming fusion always wins.

From `emc/`, run `python -m pytest tests/test_ssm_scan.py -q` and optionally
`python -m rayvan_emc.benchmark_ssm --device cuda --backend auto`. The benchmark
reports whole-module forward/backward milliseconds, excluding warmup/compilation;
both benchmark paths already batch projections. It is not LM generation throughput.

### Throughput units

Endpoint runs export `endpoints_per_second`, `context_tokens_per_second`,
`context_length`, `objective=prefix_endpoint`, and `throughput_unit=endpoint/s`.
The legacy `tokens_per_second` field remains the endpoint rate for compatibility.
For fixed context L, main-context tok/s = endpoint/s × L. This is primary input-token
exposure throughput, excluding repeated expert/probe visits. Elapsed training-loop
time includes periodic validation and audit overhead; initial setup/compilation and
the initial fixed-reference audit are excluded. It is neither generation tok/s nor
an equivalent all-position supervision rate. Live, history and comparison views
label target units explicitly and keep context tok/s separate. Existing logs with
known context length can derive this proxy; missing context yields no invented rate.


## Fixed-bank fitting diagnostic

Use this after router seeds perform near the random baseline. The diagnostic asks
whether the unchanged router can fit a small, stationary table of expert losses.
It does not retrain experts or establish end-to-end task improvement.

In History, open the **original trained model run**, click **Test router from
checkpoint**, then enable **Fixed-bank fitting diagnostic**. The toggle sets:

- Frozen experts/shared layers, fixed checkpoint continuation and suffix labels.
- 64 distinct training prefixes and 64 distinct held-out prefixes, with no exact
  raw-prefix overlap between the banks. All three trajectory depths are measured:
  192 states and eight candidate loss values per state in each bank.
- 1,000 full-bank router updates; audit every 50 updates, including update zero.
- FP32 and weight decay zero. Keep learning rate 0.0003 and the current need
  dimension 8. Use router seed 0 initially and the same source/data seed as before.

Only setup executes experts: both banks' states and full suffix targets are
measured once under the preserved source policy. Subsequent updates execute just
the router; the setup cost and zero fitting expert-work count are reported separately.
The cached banks (latents, prefix IDs, targets, expert losses and reference router)
are saved as `checkpoints/router-fit-bank.pt`. A hash covers prefixes, states,
targets and loss tables. Banks are deterministic across router seeds on the same
runtime/device/source checkpoint. The train/held-out split precedes fitting, and
held-out values do not enter the objective or optimizer.

For each depth t, let Y_t be the fixed expert losses and P their centering operator.
Each update minimizes the full-bank mean of `||A_phi(H_t)-P Y_t||^2`, with exactly
the existing need encoder, linear value head and AdamW. Training-bank and held-out
MSE are evaluated on all bank states. Normalized MSE divides by each bank's
`mean((P Y_t)^2)`: 1 means no gain over predicting equal competence, near 0 means
an excellent fit to the measured differences. A zero-denominator case is reported
as null. The train-mean baseline, fitted only from training labels, is also recorded.

The report includes train/held-out decision regret, normalized MSE, value-head and
encoder gradient norms, router parameter changes, zero expert update counts and
bank identity. Encoder gradients can be zero on the first update because the value
head starts at zero; they should become available after the head moves.

Interpretation:

- Training-bank MSE falls strongly: the router can fit this bank. If held-out error
  stays high, focus on generalization, coverage and predictability instead.
- Both errors fall: the old fitting budget/procedure was inadequate under these
  conditions. This test changes replay, batch size, precision and regularization;
  it does not isolate which individual change helped.
- Training-bank MSE remains near its baseline: investigate optimization/gradients,
  then compare a richer encoder or head on the exact same saved bank. This is not
  proof that expert differences contain no learnable signal.

The normal endpoint budget, batch multiplier, exploration, probe cap and calibration
schedule are unused in this mode; `value_fit_updates` controls duration. The live
x-axis counts **router updates**, throughput is **updates/s**, and generic loss
fields contain router MSE. Perplexity and context tok/s are null, and language
scaling projections/final capability generation are disabled. Do not compare these
losses with language-model cross entropy. Best checkpoint selection uses held-out
MSE, so reserve a new untouched panel for any later generalization claim. Optimizer
resume is deliberately unsupported for this diagnostic; repeat from the source
checkpoint instead. Initial bank measurement can be cancelled between sampling
batches and trajectory depths, and fitting between updates.

Fixed-bank setup measures prefixes in microbatches of at most four, reduced further
when necessary to respect a Delta expert's transition-tensor limit. This applies
to state collection and every counterfactual suffix. Results are concatenated in
prefix order before fitting: the router still uses every bank state in each
full-bank update. Increasing prefixes per split no longer increases the expert
execution batch size, and the transition safety guard remains enabled. If even
one prefix exceeds that guard, the normal error still applies.

This batching correction changes the order of random draws during state collection
relative to the initial unbatched diagnostic. Existing bank hashes may therefore
change; compare new runs using the same code, source checkpoint, data seed and bank
size. Router-seed comparisons remain deterministic under those fixed conditions.
