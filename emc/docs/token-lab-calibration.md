# Forced Specialization Calibration

This extends Token Lab, not EMC routing. It is a controlled distribution-exposure
experiment, not a claim of emergent specialization. Standard runs and old saved
runs remain supported. No calibration task metadata is given to a routing model.

## Run from the desktop UI

1. Open **Token Lab → Experiment mode → Forced Specialization Calibration**.
2. Use these initial settings (do not run another GPU experiment concurrently):

| Setting | Value |
|---|---|
| Dataset / topology | 10-task capability suite / independent (locked) |
| Composition / experts | Homogeneous GPT / 2 |
| Common base source | Balanced common pretraining |
| Common pretraining steps | 1,000 |
| Training optimizer steps | 1,000 **per specialist** |
| Batch size / learning rate | 4 / 0.001 |
| Latent / hidden / heads | 64 / 128 / 4 |
| Maximum prefix / neighbourhood | 48 / 16 |
| Specialist profile | Two-way: calculation vs memory/sequence |
| Specialization strength | 1.0 (100%) |
| Seed | 42 |
| Sanity-check examples per task | 50 |
| Evaluation candidate locations | 2,000 (200 per task) |
| Analysis location cap | 2,000 or greater |
| Measurement rate | 1 (required for balanced analysis) |
| Reference bank | 128 |
| Spectral measurements / rate | ON / 1 |
| Deep probes | OFF for first run |
| Success top fraction | 0.20 |
| Save raw observations | ON |
| Device | CUDA locally, CPU also supported |

3. Validate, then Run. Progress distinguishes common pretraining, each specialist,
   the independent sanity-check panel, reference probes, evaluation and analysis.
4. Examine **Calibration · blind signals**, held-out predictors, and then press
   **Reveal calibration labels** to view the expert × task matrix.
5. Repeat with strengths **0.75, 0.5, 0.25, 0**. Keep every other setting identical.
   Keep **Identical minibatch stream at strength 0** checked. The zero run is the
   required negative control, not an optional basis for a strong conclusion.
6. Save time on subsequent strengths by selecting **Saved calibration
   common-base.pt** and entering the first run's `common-base.pt` path. Do not load
   its final specialists as a common base. This preserves the actual shared state
   and initial expert weights, not just their seed. Leave the common pretraining
   budget equal to the source's budget; compatibility is checked.

The first stage's adequacy is an empirical question: 1,000 steps is an initial
budget, not a guarantee of competent general representations or specialization.
If the sanity panel fails to show specialization, descriptor failure is
uninterpretable. Increase common/specialist training only after inspecting that
panel and the losses. These models are small and each sample supervises one
answer character, not all context tokens.

## Exact model and controls

The existing `ProbeModel` contains a common embedding, learned position embedding,
one causal GPT context block, LayerNorm and output head, plus existing EMC module
implementations. A model state is `h = embedding + position + context_delta`.
Every expert path, both during training and evaluation, is:

`h → h + expert(h) → common LayerNorm/readout at final position → target NLL`.

Common pretraining optimizes **one expert's endpoint loss** on the balanced suite,
with shared machinery trainable. Unlike standard Token Lab, there is no separate
baseline-loss term during common pretraining. Its endpoint budget must be divisible
by ten. The common expert's entire state dict is copied to every specialist.
Sorted names, dtypes, shapes and tensor bytes are SHA256 hashed to verify equality.

During specialization, only `experts.i.*` for the current specialist is trainable.
`embedding.*`, `position.*`, `context.*`, `norm.*`, `head.*`, and other specialists
are frozen. Shared forward computation runs without autograd. The frozen readout
still propagates gradients back to the current expert's output. State dicts are
compared after **each** expert trains; any unrelated mutation fails the run.
Expected expert updates must be nonempty. Parameter names, counts, hashes and
actual budgets are persisted. GPT/SSM/recurrent/Delta families are supported, but
mixed families are rejected for calibration rather than silently confounding it.

Every specialist uses AdamW (betas .9/.999, epsilon 1e-8), learning rate from UI,
constant schedule, weight decay zero, gradient clipping norm 1, float32, identical
batch size and update count. Training is gradient accumulation over independent
prefixes, reusing Token Lab's existing microbatch-one model semantics.

## Sampling and equal budgets

The task catalogue comes from `CAPABILITIES` in the existing generator:
language, associative_recall, fuzzy_recall, selective_copying, working_memory,
compression, arithmetic, symbolic, program_execution, stateful_action.

For expert i and task t, with normalized profile weights w:

`P_i(t) = (1-s)/10 + s*w_i(t)`.

Two-way profile uses arithmetic/symbolic/program_execution versus
associative_recall/fuzzy_recall/selective_copying/working_memory/compression/
stateful_action. Language remains in the general mixture and in evaluation;
at strength one this preset does not assign it to either specialist. These are
experimental partitions, not universal definitions of expertise. One-task mode
uses the first E registered tasks; Custom permits arbitrary overlap and weights.

Tasks are drawn independently using seeded categorical sampling. Actual counts,
fractions and L1 distance from requested probabilities are recorded. This means
equal **total** budgets, not identical realized per-task counts. At s=0 with the
identical-stream control, task choices, generated examples, order and stochastic
seed schedule are identical across specialists.

To match BOTH target-token and context-token budgets, every calibration sample
has exactly the configured prefix length and one supervised **answer** character.
The chosen character is uniformly sampled among eligible answer positions in an
existing generated example. Examples without enough preceding tokens are rejected
**within the already chosen task**. There is no padding or different training
forward. This conditions the sampled population on context eligibility; rejection
counts are recorded. It can omit short examples and truncate important prompt
content. This is an explicit experimental limitation, not perfect full-task
evaluation. Excessive rejection causes an error; it does not reduce budgets.

Targets: `steps * batch_size` per specialist. Context tokens:
`steps * batch_size * prefix_length` per specialist. The target is never included
in its own model input. Earlier answer characters can appear as causal context.

## Splits, verification and provenance

- Common and specialist training use the generator's train split.
- Separate balanced sanity panel uses its evaluation split.
- Blind observations use its validation split, with an exact number per task.
- Exact causal-prefix hashes exclude training/reference/sanity prefixes from
  blind observations and exclude duplicate held-out locations. Resampling stays
  within the requested task. Generated examples can still share templates.
- Reference frequencies and novelty banks use training/reference data only.
- Diagnostic train/validation/test split is SHA256 of the generated-example group
  (60/20/20). Endpoints from one example cannot span predictor splits.
- Seeded sampling is reproducible. CUDA kernel bitwise reproducibility is not
  guaranteed; the identical-stream control reports final state hash equality.

The common checkpoint loader currently accepts this experiment's verified
`common-base.pt` only. Arbitrary EMC or standard Token Lab checkpoints lack the
required balanced answer-pretraining provenance and are intentionally rejected.

## Blind feature policy

Each feature has an `origin`: `pre_expert`, `expert_internal`, `post_expert`, or
`expert_history`, and a separate `requires_target` flag. Legacy `role` remains for
saved-run compatibility. The backend creates an explicit allowlist; task IDs,
assignment weights, strength, task categories and unknown keys cannot become
predictors even if a caller asks for them.

Primary pre-expert features include token/frequency controls, baseline entropy,
confidence, hidden-state statistics, dynamics, dispersion, covariance/PCA,
density, generic training-reference novelty, spectral/graph measures, and local
non-flatness. Per the requested offline experiment, baseline loss/surprisal,
correct-target probability and target margin are also included, but they require
the true next token. **A separate target-free probe excludes these three.** A high
offline score is not evidence that the same inputs are available to a router.

Expert attention/FFN instrumentation, all expert changes/updates, gradient probes,
loss outcomes, ranks and expert-success familiarity are excluded. Existing
attention measurements are *inside the specialist*; they are not mislabelled as
pre-expert attention. Shared-trunk attention measurement is not currently added.
Deep measurements remain available solely in secondary standard analysis.

Outcomes include raw NLL, `I=L_before-L_after`, normalized improvement
`I/(abs(L_before)+1e-8)`, positive improvement indicator `I>0`, and secondary
same-state relative advantage. The success classifier uses
`I > quantile(I_analysis_train, 1-q)`, default q=.2. Threshold ties are excluded;
test prevalence is reported rather than forced to 20%.

## Predictive analysis and statistics

Feature ranking (Pearson, Spearman, five-quantile binned eta² and discrete MI)
uses analysis-train only. Top eight improvement associations are checked on the
untouched test split with 100 pointwise bootstrap replicates. Interactions are
limited to pairs of the top four train-selected candidates, with train-defined
quartile boundaries and test means/std/counts; cells below ten are suppressed.

For each specialist, all-input and target-free probes each fit:

- Ridge with alpha selected from 1,10,100,1000 on validation MSE.
- MLP: standardized inputs → 32 GELU → scalar, AdamW .003, decay .01,
  maximum 150 full-batch CPU epochs, patience 20 by validation loss.
- MLP binary classifier, same architecture, BCE loss and validation selection.

Imputation medians, standardization, feature availability (80% training coverage)
and constant-feature exclusion are fitted on analysis-train only. Outputs report
test R², MAE and prediction Spearman; success ROC-AUC, average precision (the
reported PR-AUC convention), balanced accuracy and prevalence. Score ties are
handled jointly. One-class test metrics are null. Baselines are train-mean
improvement and a one-variable linear fit to baseline loss. Baseline surprisal is
the *same quantity* and is explicitly not presented as an independent baseline.

MLP permutation importance uses only five candidate features ranked by training
ridge coefficients, five test permutations, and reports mean/std increase in test
MSE. Correlated features can mask or redistribute importance. Screening scores,
bootstrap intervals, interactions and coefficients are exploratory, not FDR-
corrected discoveries. Bootstrap units are generated examples, not task templates.
All calibration evaluation rows are analyzed; analysis-cap truncation is rejected.

## Reveal and detection verdict

Task matrix contains n, mean loss, mean/median improvement and within-tolerance
win rate. High-assignment tasks have normalized assigned weight greater than
1/10; other tasks are the comparison group. The gap equally weights task means.
To avoid mistaking universally easy tasks for specialist competence, formation
also requires the assigned-minus-other gap to exceed the corresponding gap of
the alternative experts. A paired within-task 300-replicate bootstrap estimates
this differential gap's interval.

Formation criterion: positive raw gap, differential gap above tie epsilon and
positive lower 95% differential interval. Failure means **specialization was not
demonstrated by this panel**, not proof that no specialization exists. Detection
uses held-out MLP R² and replicated top-feature associations: R²>.1 and at least
two test intervals excluding zero gives provisional DETECTABLE; R²>0 gives
WEAKLY DETECTABLE; otherwise NOT DETECTABLE. Without formation the verdict is
SPECIALIZATION DID NOT FORM. These are explicit exploratory thresholds, not
calibrated significance levels. No STRONGLY DETECTABLE claim is issued without
replicated control evidence.

The reveal also breaks down high-feature test regions by task, using training
quartile thresholds, including unassigned tasks. Those are descriptive cross-task
clues, not proof of transfer: a task-held-out predictive experiment is still needed.
Improvement prediction can reflect general token difficulty, so inspect target-
free probes, baseline-only performance, the differential gap and the s=0 control.

## Storage, reopen, and strength sweep

Normal Token Lab files remain. Additions:

- `common-base.pt`: trained shared model and identical starting expert states,
  tokenizer, config, train-prefix exclusions and provenance marker.
- `specialist-N.pt`: individual final expert, base/initial/final hashes, budget
  and distributions.
- `specialization.json`: controls, trainable/frozen/changed names, realized budgets,
  task panel, checkpoint hashes, profile and seed.
- `calibration-analysis.json`: blind rankings, diagnostics, interactions and reveal.
- `analysis.json.calibration`: same data for existing UI transport.
- `summary.json.calibration`: compact sweep metrics and per-feature test Spearman.
- `feature-schema.json`: origin and target-access classification, plus new outcomes.
- `model.pt`: complete final model, tokenizer/config and training-prefix exclusions.

Reopen restores saved results without training. To rerun measurements on frozen
specialists, reopen the run and set **Rerun saved specialists** to its run directory.
Model/training settings must match. Evaluation budgets/measurement controls may
change. This creates a new run with no retraining, not an overwrite. Checkpoints
are tensor-only loaded. Existing checkpoint trust and local filesystem permissions
still apply; use your own saved run files.

Strength Sweep automatically includes saved runs with matching common-model hash,
seed, training and analysis settings (excluding strength and source paths). It
shows actual/differential gap, blind and target-free R², success AUC, and feature
Spearman across strength. It does not interpolate a missing control or combine
unmatched seeds. Reopen older runs remains supported via optional fields.

## Verification and limitations

Tests: `python -m pytest emc/tests/test_token_lab.py emc/tests/test_token_lab_calibration.py`.
Frontend: `node --import tsx --test tests/token-lab.test.mjs`; TypeScript/Vite build.

The Python suite includes clone identity, unchanged other/shared groups, equal
budgets, s=0/1 distributions and sampling, seed reproduction, answer/split
semantics, hard blind exclusions (including malicious caller keys), classifier
ties, synthetic predictive signal, end-to-end same-state probes, individual
checkpoints, base reuse and zero-training specialist rerun.

Implementation verification (CPU, 2026-09-09): 24 Python tests passed, 44 frontend
and Math Visualizer regression tests passed, TypeScript checking and Vite build
passed. A nonzero-strength smoke run exercised spectral measurements, blind
ridge/MLP/classification, raw saving and report generation on 100 held-out
locations; the negative-control test also verified checkpoint-only replay.
Windows/CUDA and the native Rust build were not tested in this environment
(Cargo was unavailable). A smoke run is software verification, not evidence of
scientifically meaningful specialization.

No automatic research run or strength sweep is launched. No production router,
task-label predictor, mixed-family calibration, training resume, arbitrary EMC
checkpoint importer, confirmed cross-task transfer claim, multiple-testing
correction, or automatic control-adjusted strong verdict is implemented. Report
and UI disclose these boundaries. The existing shared GPT trunk remains even for
non-GPT specialists. It can make some tasks easy before a specialist acts; freezing
it isolates exposure but does not guarantee a useful or neutral representation.
