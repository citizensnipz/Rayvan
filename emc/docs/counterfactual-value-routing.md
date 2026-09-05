# Sequential counterfactual value routing: implementation and experiment protocol

This branch adds `counterfactual_value_emc` to the Python research console. It is
an experimental implementation of continuation-value geometry and controlled
expert development. Existing geometric, module-aware and parallel architectures
retain their existing behavior and checkpoint identities. Native C++ is unchanged.

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
shared parameters. A frozen run initialized from scratch is only an infrastructure
control. For a meaningful competence study, load a trained model using
`load_model_checkpoint`, replace `model.config.value_expert_training` with
`"frozen"` via `dataclasses.replace`, and call `train_model` with fresh optimizers
(`resume_from=None`). The console does not yet provide checkpoint warm-start UI.

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

For the first hypothesis pilot, choose these console settings:

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
