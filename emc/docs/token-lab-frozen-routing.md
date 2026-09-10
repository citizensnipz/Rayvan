# Frozen Expert Routing Test

This is a separate Token Lab experiment, not a change to the production EMC
router or the observational calibration/sweep modes.

## Run it from the Research Console UI

1. Pull `research/spectral-live-console` and restart the desktop development app.
2. Open **Token Lab** and reopen your **Specialization Validation Sweep**.
3. Click **Test routing with these saved experts**. Alternatively choose
   **Experiment mode → Frozen Expert Routing Test**, then select your saved sweep.
4. For the first run leave **GPT**, **All seeds in sweep**, and **Include paired
   0% controls** selected. Set **Device → cuda** on the CUDA machine.
5. Leave the defaults below, click **Validate**, then **Run Token Lab**.

There are no manually copied common-base paths or model dimensions. The UI
previews the family, original training seed, strength and completion status of
every bank. Reopening another result never overwrites the next-test form.
**All four families** can be selected to run the complete comparison in one launch.
Do not start the old sweep again; this test reuses its trained experts.

| Setting | Default |
|---|---:|
| Router / new sampling seed | 142 |
| Fresh training locations per bank | 1000 |
| Validation locations per bank | 300 |
| Untouched test locations per bank | 1000 |
| Fixed-teacher policy rounds | 3 |
| Preference fitting epochs per round | 80 |
| Router learning rate | 0.001 |
| Training exploration probability | 0.20 |
| Sequential comparison | 3 steps, plus automatic 1-step comparison |
| Add spectral features to full descriptors | Off |

For a three-seed GPT source this selects six banks (100% and 0% for each seed).
Every bank gets simple/full feature comparisons at one and three steps: 24
conditions, automatically. All four families select 24 banks / 96 conditions.
No expert retraining occurs. Jobs run serially; errors are reported, not hidden.
**Stop** cancels at a safe boundary and preserves completed files. Results reopen
without recomputation. Resume of incomplete routing jobs is not implemented;
select an individual family/seed to rerun a failed bank.

## What is actually executed?

Load the source `model.pt`, `specialization.json`, `reference-bank.pt`, tokenizer,
architecture settings and exact saved specialist hashes. Verify both experts and
all shared state. Freeze the entire `ProbeModel`: embedding, positional embedding,
common GPT context encoder, every expert, final normalization and readout.

The exact Token Lab expert semantics are retained:

    h0 = model.embed(prefix)
    h_next = h + expert_e(h)
    loss = CE(model.logits(h_final), next_answer_character)

This is **not** the production EMC learned Integrator. Changing application
semantics would invalidate the calibration. Repeated application is deliberately
experimental because the saved specialists were trained on one-step applications.
All windows are causal and recomputed on the transformed latent each step.

## Router inputs

Both arms get inference-available controls: absolute/context position, training
reference token frequency/rarity, predictive entropy, top-1 probability and logit
margin. The full arm adds existing core hidden-state, displacement, cosine,
local-dispersion, PCA/effective-rank, density, non-flatness and novelty descriptors.
Optional spectra reuse the existing feature utilities. Both arms receive remaining
steps and trajectory step. The saved common pre-expert reference bank stays fixed,
even at later stages; novelty relative to that bank can therefore rise off its
original state distribution.

The observer API receives no target or task identity. A backend allowlist also
excludes target probability/NLL/margin, post-expert features, expert internals and
expert-success history. No expert is executed to produce a routing preference.
Rows are imputed and scaled using router-training data only. Features require 80%
finite training availability; constants are dropped. Standardized values are
clipped to [-20,20] identically during fitting and inference.

## Objective and counterfactual continuation

There are exactly two saved homogeneous specialists per bank. A small
32-unit GELU MLP predicts a preference logit `f(state, remaining_steps)`.
Choose expert A if f > 0, otherwise B. These logits are **not predicted NLLs**.

For a temporarily fixed teacher policy pi, at the same state h and remaining
horizon r, measure:

    Q_pi(h, e, r) = final endpoint NLL after e, then pi for r-1 steps
    A_pi(h, r) = Q_pi(h, B, r) - Q_pi(h, A, r)

Fit the regret-weighted logistic objective:

    J = mean[ |A_pi| * softplus(-sign(A_pi) * f(h,r)) ]

Weights are divided by their training mean for stable optimization. Ties have
zero weight. AdamW uses weight decay .01, fixed learning rate, batch size 128 and
gradient norm clip 1. There is no absolute-loss MSE term. In the unrestricted
population optimum, `f = log(E[A_+ | features] / E[(-A)_+ | features])`, so its
sign agrees with expected expert advantage. Finite data, model restrictions,
regularization and optimization can still make the learned choice wrong.

Each round:

1. Freeze the currently accepted teacher (initially a training-selected fixed sequence).
2. Visit states from new training prefixes under that teacher with 20% uniform
   exploratory actions. At each visited state probe both experts with the same
   teacher continuation. Exploration affects visited states, not suffix targets.
3. Fit a fresh preference model. The teacher remains unchanged during collection.
4. Execute the candidate on validation prefixes; accept only if actual mean final
   NLL is lower. Otherwise keep the previous teacher. Never select using test loss.
5. Repeat. All-zero sampled advantages stop redundant rounds, notably exact 0% controls.

Raw labels and both teacher/candidate weights are retained for every round.
Validation re-use over rounds is explicitly part of model selection. There is no
claim of monotonic test improvement.

## Evaluation and interpretation

After model selection, execute the chosen policy on fresh held-out prefixes.
Only its selected expert runs each step. A separate audit measures both possible
first actions and follows the **final accepted policy**, recording suffix regret
on its visited states. Audits cannot change choices, weights or acceptance.

Baselines are evaluated on the same test inputs and at the same horizon:

- **Fixed sequence:** enumerate the 2^H sequences on training inputs and select the
  lowest mean training loss. No test-based selection.
- **Uniform random:** exact mean over all 2^H sequences, avoiding random-baseline
  sampling noise. This enumeration is diagnostic work, not learned inference.
- **Oracle sequence:** lowest test loss per prefix among all sequences; expensive
  diagnostic lower bound only, never a deployable policy.
- **Simple router:** paired comparison for the full-feature arm.

Primary gain is baseline loss minus executed router loss, in **nats per answer
endpoint**. Positive is better. Bootstrap intervals use 200 paired resamples of
whole generated examples. They are exploratory, pointwise, conditional on fitted
policies and not corrected for multiple comparisons. Aggregation reports every
seed, positive-seed counts and retained fixed policies, not just winners.

A successful result lowers test NLL relative to fixed and random baselines.
Full descriptors must additionally justify themselves against simple controls.
The 0% controls should have effectively zero differential benefit. Compare one
step to three steps too: routing cannot guarantee extra applications help.
This experiment does not establish generalization to unseen task families or
naturally emerging expertise.

## Data separation and persistence

New balanced answer-endpoint samples use the checkpoint's suite and tokenizer.
The router seed controls independent sampling streams. Exclusions include saved
training prefixes, reference/sanity prefixes and source evaluation observations.
Router train/validation/test generated-example groups and exact prefixes are
mutually disjoint; source analysis groups are also excluded. Original expert
training did not save all generated-example IDs, so its overlap protection is
exact-prefix based. Tasks share generator templates; this is not unseen-task testing.

Each new parent run saves:

- `config.json`, `metadata.json`, `status.json`, `metrics.jsonl`, `summary.json`
- `routing-test-analysis.json`, `analysis.json`, `report.md`
- `jobs/<source-key>/source.json`: provenance, hashes and restored architecture
- `jobs/<source-key>/sample-manifest.jsonl`: all new prefixes, targets, groups/splits
- per-condition `round-N-targets.jsonl`, teacher/candidate weights, `training.json`
- `router.pt`, `test-decisions.jsonl`, `result.json`

Model weights are not copied or mutated. Keep the source sweep's `jobs/` directory
when moving this experiment. Copied report/analysis JSON files alone are insufficient.
Relocated ordinary sweep children are found under `source_sweep/jobs/<key>` before
falling back to the original recorded path (useful for resumed sweeps).

## Performance and scope

This is a research execution path. Full descriptors use existing CPU SVD/graph
utilities; experts run on the selected device. No CUDA/C++ extension is required
by this addition. It is not optimized for production throughput.

Timing is an unaudited inference pass, excluding counterfactual verification.
**Endpoint/s** = answer targets scored / seconds. **Context tok/s** = endpoint/s
× prefix length, counting each input token once. It is not generated tokens/s,
training throughput, or the number of recurrent token exposures. Feature time
is shown separately. Exhaustive training/test baselines and audits add runtime
outside this inference measurement. A small CPU smoke test is functionality
verification, not evidence that routing helps trained specialists or a GPU benchmark.
