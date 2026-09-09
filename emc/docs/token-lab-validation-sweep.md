# Specialization Validation Sweep

## One launch from Token Lab

Select **Experiment mode → Specialization Validation Sweep**. This replaces the
manual expert/checkpoint/strength controls with the sweep form. No source run or
checkpoint selection and no separate discovery button are required.

Recommended first larger run:

| Setting | Value |
|---|---|
| Sweep name | Four-family calibration · 5 seeds |
| Number of training seeds | 5 |
| First training seed | 42 (runs 42–46) |
| Diagnostic seed | 42 |
| Common pretraining steps | 1000 |
| Specialist steps per expert | 1000 |
| Training batch size | 4 |
| Learning rate | 0.001 |
| Evaluation locations per calibration run | 2000 |
| Separate sanity-check locations per task | 50 |
| Latent / expert hidden dimension | 64 / 128 |
| Attention / Delta heads | 4 |
| Causal prefix length | 48 |
| Neighbourhood | 16 |
| Reference-bank size | 128 |
| Maximum selected / candidate features | 5 / 16 |
| Spectral measurements / probability | On / 1 |
| Deep probes | Off (post-response/target-dependent; not primary inputs) |
| Leave-one-task-out transfer checks | On |
| Device | cuda on a working CUDA machine |
| Resume previous sweep directory | Empty for a new sweep |

Click **Validate**, then **Run Token Lab** once. Five seeds mean 40 calibration
runs, 20 common pretraining stages, 80 individual specialist fine-tuning stages,
40 discovery analyses, and extra predictive checks. Three seeds (UI default)
mean 24 calibration runs. Budget units are optimizer steps; with batch size 4,
1000 specialist steps = 4000 answer-token targets and 192000 context-token
exposures per expert. This is the existing endpoint training protocol, not a
claim of 192000 independent loss targets.

Jobs execute sequentially, not simultaneously on one GPU. GPT, SSM, recurrent/GRU
and Delta are expert module families, not simply FFN activation alternatives.
They reuse the existing Token Lab module backends (parallel_scan SSM,
parallel_delta Delta), shared causal GPT context encoder and readout; no router
is constructed. This does not require a new custom CUDA kernel build.

## Exact experimental schedule

For each family and each seed:

1. Train one balanced common base on the ten registered tasks.
2. Clone its expert twice and fine-tune with the existing two-way 100% profile.
3. Freeze and evaluate the same states independently through both experts; save
   raw observations and the separate task sanity matrix.
4. Automatically run target-free saved-observation discovery.
5. Load that exact pre-specialization common base, clone again, and train the
   0% control with the identical minibatch stream.
6. Evaluate and analyze the control through the same paths.

Steps 1–4 are implemented as the existing 100% calibration runner. Its saved
`common-base.pt` is the source for step 5, not its final `model.pt`. Matching
checks compare common/shared hashes, initial expert hashes, parameter counts,
optimizer configuration and actual budgets. Clone identity and shared freezes
remain enforced in the existing trainer. A non-identical 0% final pair is
reported and excluded from successful matched-control evidence, not hidden.

Each seed changes both common pretraining and specialist sampling, deliberately
testing a broader replication than a fixed-base training-seed-only test. A
family/seed pair shares its common base across strengths; different families do
not share weights or necessarily parameter counts. Therefore family differences
are not isolated causal architecture effects.

## Analysis and report

Every child keeps the usual Token Lab and calibration analyses and full saved
discovery report. The parent `sweep-analysis.json` contains compact per-expert
improvement and pair-advantage results, selected feature recurrence, removal
intervals, control hashes, task specialization checks, and additional checks:

- **Pre-specified groups:** all admissible inputs; controls alone; without
  geometry; without spectral features; without novelty. Controls are position,
  token frequency/rarity, predictive entropy, top-1 probability and logit margin.
  The broad geometry block includes hidden statistics, local shape, temporal
  displacement and spectra. Novelty is evaluated separately. Nonexistent or
  insufficiently sampled measurements are excluded with the normal training-only
  availability check; a disabled feature group cannot establish its utility.
- **Ridge and small MLP:** model family and regularization/early stopping selected
  using validation data, never test performance. Numeric inputs use training-only
  imputation and standardization; task IDs and target-dependent/post-expert
  measurements never enter the feature matrix.
- **Offline decisions:** for `y = L_B - L_A`, choose A iff predicted y > 0.
  Regret is `max(y,0) - y * 1[prediction>0]`. The constant baseline chooses A iff
  mean training y > 0. Report mean regret, uniform expected regret `mean(|y|)/2`,
  paired example-group bootstrap choice-gain intervals and non-tie accuracy.
  These scores are not sequential-trajectory routing evaluations.
- **Shuffled-label sanity check:** independently permute pair-advantage labels
  within fitting/validation/test splits, then fit again. One permutation is not
  a formal significance test. An unexpectedly high null R² (>0.1) flags the
  combined verdict as inconclusive.
- **Leave-one-task-out:** exclude a task entirely from fitting/validation and
  predict its pair advantages. Compare all inputs with controls. Task identity
  defines the split only. The remaining tasks use example-group hashes for an
  80/20 fitting/validation division. Report every task, including failures,
  negative R² and insufficient-sample cases. This is transfer within the synthetic
  suite, not evidence of real-world generalization.

The report keeps null/negative findings. Selection frequencies use the number of
completed 100% runs as denominator; removal intervals are conditional on selection
and fitted models. Near-duplicate features can substitute for one another. Group
removal refits can also change optimization. Pointwise intervals are exploratory,
not adjusted for multiple comparisons.

Across strengths the held-out observations can differ because training-prefix
exclusions differ. Do not interpret differences of R² as a paired causal effect
size. Models are fitted separately per replicate: this tests repeatability of the
discovery procedure, not transfer of a frozen model/feature list between seeds.

Family verdicts require complete eligible pairs, task specialization formation,
at least three matched seeds, a passing shuffled-label sanity check and positive
offline choice gain across those seeds for the limited “SUPPORTS” label.
That label does not assert that geometry is uniquely responsible, that group or
cross-task checks passed, or that the learned predictor will work in EMC. Inspect
those separate checks before making that decision.

## Progress, failure, cancellation, reopening and resume

The parent run shows the current family/seed/strength/stage and completed jobs.
Stop uses the existing cancellation flag, checked inside child training/probes
and predictive fitting. Child runs are under `jobs/`, keeping the main saved-run
list uncluttered. Parent events retain the parent ID so the UI follows one job.

A failed child is recorded with its error and traceback; other jobs continue. A
failed 100% run without a verified base also blocks its paired 0% job. All
completed artifacts remain. The parent status is **partial** if any jobs failed.

Reopen the parent run to see its chart, per-job inspector and combined report.
**Prepare resume / reanalysis of this sweep** restores its settings and fills the
resume path. Then click Run Token Lab. Resume creates a new parent run, verifies
identical sweep settings, reuses completed child training and regenerates analyses.
Interrupted training stages restart from their common protocol; this is job-level
resume, not optimizer-step checkpointing. Keep the previous sweep directories:
resumed entries may reference their saved checkpoints/observations.

Parent resume metadata is replaced atomically. Child storage conventions remain
unchanged. Peak memory is one expert-training job plus one bounded analysis bank,
not all datasets/models simultaneously. Each child saves its raw observations,
models and banks, so ensure free disk space for the full sweep. Time the first
family/seed pair locally; no GPU runtime estimate is guaranteed by CPU smoke tests.

Share **report.md** and **sweep-analysis.json** from the parent sweep directory.
For diagnosing a failed job, include parent logs.txt and the relevant child config.

## Tests

`python -m pytest emc/tests/test_token_lab_sweep.py emc/tests/test_token_lab_discovery.py`

`node --import tsx --test tests/token-lab*.test.mjs`

Unit tests cover all family/seed combinations, same-base controls, stale-setting
exclusion, mismatched hashes, resume validation, failure continuation, parent
cancellation, decision signs, feature exclusions and custom task split
preprocessing. UI tests cover replaced controls and reopening combined reports.

Implementation smoke check: all four actual module families completed both
strengths on CPU with latent 16, hidden 32, prefix 48, window 4, 5 common steps,
2 specialist steps, batch 2 and 200 evaluation locations. Spectral and cross-task
checks were enabled. All four exact-stream controls had identical final expert
hashes and all strength pairs matched common hashes. This deliberately tiny
training budget checks plumbing, not specialization quality. It took about eight
minutes including all analyses on this CPU environment; it is not a forecast for
the larger CUDA run. CUDA/Windows execution must be verified on the local machine.
