# Emergent Specialization: controlled development experiment

**Mixed-population extension:** GPT + SSM + GRU + Delta is now supported. See [heterogeneous studies](heterogeneous-token-lab.md) for the mixed-bank setup and changed control interpretation. The identical-clone descriptions below apply to homogeneous mode.

This Token Lab mode tests three separate hypotheses: neutral weight differences can develop into stable specialization (H1); the existing competence learner can recognize developing competence (H2); and competence-informed training improves final held-out loss compared with matched equal and random exposure (H3). A successful fit alone is not evidence for all three.

No production EMC router is replaced. There are no task assignments, specialist profiles or task-label inputs. This first developmental test uses the existing Token Lab operation `h_after = h + expert(h)`, one step, with the common encoder and readout frozen. It isolates the expert/router feedback loop; it does not yet test jointly changing shared representations or recurrent EMC trajectories.

## Running from the UI

1. Pull `research/spectral-live-console` and restart the application/Python backend.
2. Open **Token Lab → Experiment mode → Emergent Specialization**.
3. Select **General bases from saved validation sweep**, then your completed Specialization Validation Sweep. Select **GPT**. The runner loads each matching seed's **pre-specialization `common-base.pt`**, not its 100% specialist model. Model dimensions, tokenizer, context length and neighbourhood are restored automatically.
4. Click **Smoke test preset**. It selects two experts, the first available source seed, strengths `[0, .01]`, feedback `[0, .5]`, 10 training blocks × 2 endpoints, 20 training/validation/utility probe states, 200 separate diagnostic states, 20 reference states, 2 fitting epochs, and initialization/final checkpoints. Select your device, then **Validate → Run Token Lab**. This creates all five A–E jobs and should report INCONCLUSIVE hypotheses because it is only a pipeline check.
5. After a complete smoke run, select the same source and click **Narrow study preset**. Check the seed list against the displayed available seeds. The preset uses the first three available seeds when there are at least three; **Use all available seeds** explicitly selects the entire family seed set. Missing bases fail validation with the family/seed identified.
6. For the initial study leave the settings in the table below. Validate shows the job and application counts. Click Run. The existing Stop action preserves completed results/checkpoints. Reopen the saved run to inspect it without retraining.
7. Share `report.md` and `emergence-analysis.json`. Do not select another range because its held-out curve happened to look good without treating that as exploratory selection; replicate promising settings with fresh seeds.

| Narrow study setting | Value |
|---|---|
| Dataset | Existing balanced 10-task suite, automatic |
| Family / population | GPT / 4 homogeneous experts |
| Source | Saved validation sweep general bases, matched by seed |
| Seeds | First 3 matching saved seeds; increase to at least 5 for stronger replication |
| Weight perturbation α | 0, 0.001, 0.01, 0.05 |
| Competence feedback β | 0, 0.25, 0.5 |
| Design | Narrow |
| Training | 250 blocks × 4 mixed endpoints per block |
| Expert optimizer | AdamW, LR 0.001, weight decay 0, gradient clipping 1, float32 |
| Uniform exploration ε | 0.2 of redirected tickets |
| Router refresh | Every 50 blocks and at every checkpoint |
| Training / validation router panels | 256 / 200 states |
| Utility / diagnostic panels | 300 / 400 states |
| Reference bank | 128 training/reference states |
| Router fitting | Existing learner, LR 0.001, 80 epochs |
| Checkpoints | 0%, 10%, 25%, 50%, 75%, 100%, rounded to integer blocks |
| Descriptors | Full; spectral additions OFF, matching the validated baseline |
| Optional oracle F | OFF initially |
| Raw data and checkpoints | Always saved |

**Full grid preset** enables the Cartesian positive-asymmetry × positive-feedback grid. Narrow design crosses all positive asymmetries with the largest feedback and the middle positive asymmetry with all positive feedbacks. A/B/C are deduplicated controls. Repeat the informative range in SSM, recurrent/GRU and Delta by selecting another family; its own general base is required. Each population remains homogeneous.

An alternative source is a single verified calibration `common-base.pt`. It is reused for every seed, so these seeds replicate perturbation, data and learning randomness, **not common pretraining**. New study `seed-*/common-base.pt` copies are also reusable through this source option. Arbitrary EMC checkpoints and specialized `model.pt` files are rejected because they lack verified general-base provenance.

## Exact controls and mathematics

### Initialization

Load the shared model and one common expert from the general base, then clone the expert. SHA256 hashes of named tensor bytes verify equality before perturbation. For expert e, concatenate its trainable parameters into θ and draw independently seeded Gaussian direction z_e. Apply

`θ_e = θ + α ||θ||₂ z_e / ||z_e||₂`.

The seed is `study_seed + 104729*(expert_index+1)`. Strength α is a relative global parameter-vector L2 magnitude, not a per-layer standard deviation. α=0 changes nothing. For each expert record requested α, realized absolute/relative perturbation, original/final hashes; record pairwise absolute and normalized parameter distances throughout training. Buffers are not perturbed. Zero-norm parameters within a nonzero expert vector can acquire noise; this is intentionally a global, not layerwise, perturbation.

Freeze **every parameter except `experts.<index>.*`**: token/position embeddings, common contextual encoder, normalization and output head. The complete frozen/trainable names are saved. Shared state hashes are checked at each checkpoint; an unexpected change fails that job. A also fails if identical exposure causes its expert hashes to diverge. Deterministic Torch algorithms and cuBLAS workspace configuration are requested; unsupported deterministic kernels produce explicit failures rather than silently weakening the control. CPU/GPU numerics are not claimed bitwise equivalent.

### The router is reused

The existing `Observer.features` obtains inference-available controls, core state/neighbourhood geometry and training-reference novelty. Its API receives no target or task identity and executes no candidate expert. The full and simple descriptor options are unchanged; optional spectra are a secondary experiment. Target probability, baseline NLL/surprisal, target margin, post-expert changes and task labels are excluded from routing inputs. Baseline NLL appears only in measured outcomes and explicitly labelled difficulty-only diagnostic baselines.

For every pair (i,j), evaluate actual current expert NLLs on the same training probe states. Let `a_ij = L_j - L_i`. Reuse the existing network and loss:

`loss_ij = mean(|a_ij| softplus(-sign(a_ij) f_ij(x))) / mean(|a_ij|)`.

Implementation normalizes advantages before the existing preference loss, as in the frozen routing test. Median imputation, standardization, feature availability/variance filtering, clipping to ±20, the 32-unit GELU network, AdamW and gradient clipping are reused unchanged. Scaling is fitted on router-training data only. Current losses are remeasured at each refresh; stale targets are not mixed into a new fit.

For two experts the decision is exactly the validated sign rule, with B chosen at a zero score during deterministic evaluation. **For four/eight experts**, fit the same learner independently for each pair and choose the expert with the largest number of predicted pairwise wins (ties count one half), breaking equal win counts by mean preference:

`wins_i = sum_{j != i} [1(f_ij > 0) + 0.5 * 1(f_ij = 0)]`, with `f_ji = -f_ij`. Mean sigmoid preference breaks ties only. This preserves a consistent all-pairs winner even if another expert has larger margins in its other comparisons. Finite learned pairs can still form cycles. The stored scalar score is `(wins_i + .25 * mean_preference_i) / (E - 1 + .25)`; a half-win always outranks the entire soft tiebreak range. For E=2 the original sigmoid/sign decision is retained exactly.

This population aggregation is an explicit, unvalidated adaptation; the representation and pairwise training mathematics are preserved. Exactly identical pairs contribute 0.5, not an arbitrary win. During selective training, maximum-score ties are broken uniformly so initial label order does not assign all preferential tickets to one clone.

At every refresh, compare the previous router, the newly fitted candidate, and the training-panel best fixed expert on a separate router-validation panel. Retain the lowest validation loss. The utility and diagnostic panels are never consulted for this choice. Repeated validation reuse and candidate selection can overfit validation; held-out utility remains necessary. The initial refresh is allowed to discover the deliberately small initial performance difference, so initialization metrics are not an assertion of zero router fit.

### Equal total training application budget

For each block of B mixed input endpoints, create E training tickets per input. The original tickets allocate one to every expert. Independently redirect each ticket with probability β. On a redirected ticket, explore uniformly with probability ε, otherwise use the current competence preference (or privileged oracle for F). In aggregate the conditional expected expert allocation is

`p(e|x) = (1-β)/E + β[(1-ε)π(e|x) + ε/E]`.

All tickets are actually executed: there are exactly `E * B` expert-example applications per block, even when several tickets repeat the same expert/input. Each active expert takes one AdamW update per block using the **sum** of assigned ticket losses divided by the common B, then gradient clipping. Its own assignment count is not used as the denominator. These are equal optimizer configurations, not a claim that effective gradient sizes or active optimizer-update counts remain equal under selective exposure. Counts and timing are recorded. Repeated tickets can be computationally wasteful; the experiment makes that cost explicit.

| Condition | Initial α | Exposure |
|---|---:|---|
| A | 0 | One ticket/input/expert; identical minibatch stream and expert-local RNG |
| B | Positive | Same equal exposure as A |
| C | 0 | Competence feedback + mandatory exploration |
| D | Positive | Competence feedback + mandatory exploration |
| E | Same as D | D's ticket identities randomly permuted across input slots within each block |
| F, optional | Same as D | Counterfactual minimum-loss feedback + exploration; no learned router training |

E has **exactly D's per-expert ticket counts and active optimizer-update counts in every block**. It removes the input-to-expert competence association conditional on those marginal counts. It is not an independently sampled uniform control; its workload marginals intentionally follow D. Within small minibatches random permutations can coincide with D. B provides the separate equal-exposure comparison. A/B/E also fit observational routers at the same scheduled refreshes to compare final populations under the same evaluation method; their router predictions never allocate training tickets.

F supplies privileged one-step loss information. It is a reference opportunity estimate, not a mathematical upper bound on long-run developmental outcomes. Its oracle evaluation is separately labelled; it is not a deployable learned policy.

## Data and measurement protocol

Use the registered 10 capability tasks and existing full-context answer-endpoint sampler. New training, reference, router-training, router-validation, utility-evaluation and diagnostic panels have distinct prefixes and generated-example groups. Previously trained common-base prefixes are excluded. Training, validation, utility and diagnostic counts are exactly task-balanced; the 256-state router panel and 128-state reference bank differ by at most one sample per task. Task labels are retained only in metadata for later reveal. Context eligibility can condition the sampled population, as in the existing calibration experiment; this is not unrestricted natural-language evaluation.

Shared representations are frozen, so common states and descriptors are calculated once per seed and reused for every condition. The novelty reference bank comes only from new training/reference data. Matched panels are saved. The current implementation caches them in host RAM with an estimated 2 GiB tensor-cache limit (Python object overhead is additional), rather than repeatedly rerunning the common encoder. Reduce budgets if this limit is reached. This makes matched experiments practical but is not an inference-throughput benchmark.

Every utility checkpoint evaluates **all experts independently on exactly the same pre-expert states**, with frozen weights. For expert e:

- `I_ne = L0_n - L_ne`: token loss improvement.
- `A_ne = I_ne - mean_{j != e}(I_nj)`: relative competence advantage.
- `R_ne = I_ne - mean_n(I_ne) - mean_e(I_ne) + mean_{n,e}(I_ne)`: state × expert interaction residual.
- `interaction_rms = sqrt(mean(R²))`: removes both common state difficulty and constant global expert quality. It is a functional divergence measure, not sufficient evidence alone.
- Oracle opportunity: `min_e mean_n L_ne - mean_n min_e L_ne`.
- Wins, tie-adjusted win shares, mean/variance of improvements, mean advantages and pairwise improvement correlations.
- Output-update disagreement: RMS pairwise endpoint delta distance, averaged over matched states.
- Adjacent-checkpoint improvement, relative-advantage and router-preference correlations; top-improvement-tail Jaccard, winner and pairwise-ranking persistence. Constant correlations and degenerate top tails are unavailable. Ranking persistence in fully tied experts is not evidence for specialization.
- Parameter distances, expert application/update/probe counts, redirected/exploration ticket counts, task exposure for post-hoc interpretation.

A separate diagnostic panel reuses Token Lab's grouped 60/20/20 train/validation/test ridge, small MLP and success-classifier probes per expert. Only inference-available pre-expert descriptors enter the feature matrix. Report held-out R², MAE, Spearman, ROC-AUC, average precision and balanced accuracy, plus mean-improvement and target-aware baseline-NLL-only diagnostic baselines. Positive improvement predictability may merely reflect common difficulty: interpret it alongside relative competence, distinct winners and actual selection gains. Online preference–advantage Spearman is reported separately from these diagnostic regressors.

Utility reports actual endpoint NLL for the online-learned policy, training-selected fixed expert, exact expected uniform random expert, round-robin assignment, and the same-state oracle. No expert updates occur during these measurements. For this single-step experiment, indexing the matrix of executed counterfactual losses exactly equals executing the chosen expert; it is not a surrogate predicted loss. Baseline-selected expert identity comes only from training probes. Report selection gains versus fixed/random, oracle regret and best-expert agreement; pairwise sample-group bootstrap intervals use the existing routine.

## Comparisons and verdicts

Within each matched seed calculate B−A, C−A, D−B, D−E and optional F−D. Negative loss difference favors the left condition. Compare final populations with their own validation-selected router, not just raw oracle quality. F additionally has privileged oracle evaluation and is labelled accordingly.

Across seeds report all individual values, mean, median, sample SD, and a 1,000-resample seed-bootstrap interval. Bootstrap paired **differences**, not unmatched raw runs. Three seeds provide very uncertain intervals. Repeated checkpoints are correlated; displayed pointwise intervals are exploratory and not corrected for many conditions. Failed/incomplete seeds remain listed. Best recorded held-out loss is retrospective description only, never a selection gate.

The explicit exploratory verdict gates are conservative summaries, not statistical proofs:

- H1, positive-α conditions: interaction RMS increases beyond both initialization and A by the configured NLL tie tolerance; at least two experts each uniquely win at least `max(2, 5% of panel)` states; adjacent relative-advantage correlation exceeds 0.5.
- H2, learned-router conditions: the paired gain interval is positive against **every constant expert**, at least two experts are meaningfully selected (at least max(2, 5% of states) each), and at least two experts have meaningful distinct wins. This prevents retaining a generally strong constant choice from being labelled state-dependent detection.
- H3, D: final routed loss is lower than **both** matched B and E for that seed.
- Fewer than three seeds, any missing seed, mixed supporting/non-supporting seeds, or an inapplicable condition: INCONCLUSIVE. All seeds support the gate: WEAKLY SUPPORTED for 3–4 seeds, SUPPORTED for at least 5. No seeds support it: NOT SUPPORTED.

Always read magnitudes, seed intervals and trajectories alongside those labels. SUPPORTED is support within this controlled protocol, not validation of a production EMC architecture. Later fresh-seed replication is required after exploratory range selection. An unchanged A that allows accurate improvement prediction is a difficulty signal, not specialization. Warnings surface functional identity, inference/training concentration, starvation, weak probe counts, oversized preference logits without gain, and worsening routed loss.

## Artifacts and reopening

Existing Tauri commands dispatch the new CLI mode; existing run listing and reopening read its separate `analysis.emergence` schema. Ordinary Token Lab/calibration/sweep/routing runs are unchanged.

```
config.json, metadata.json, status.json, summary.json
analysis.json                 # {emergence: ...}, existing UI boundary
emergence-analysis.json       # protocol v1, settings, plan, failures, trajectories, seed comparisons
report.md, metrics.jsonl
seed-N/
  common-base.json            # source/hash, clone identity, frozen/trainable names, preparation time
  common-base.pt, reference-bank.pt, matched-data.pt, sample-manifest.jsonl
jobs/seed-N-CONDITION/
  initialization.json, training-assignments.jsonl, trajectory.json
  refresh-STEP/
    counterfactuals.jsonl, selection.json, accepted-router/
  checkpoint-STEP/
    model.pt                   # shared/expert state + optimizer states + condition/step
    expert-0.pt, expert-1.pt, ...
    router/router.json, router/pair-i-j.pt
    observations.jsonl, diagnostic-observations.json, metrics.json
  error.txt                    # if a job fails
```

Observation records preserve sample/group/prefix identity, task metadata for reveal, inference-available descriptors, baseline/expert loss, improvement, advantage, rank, chosen expert and preference. The per-seed manifest retains input/target token IDs. `PopulationRouter.load` restores pairwise routers and neutral ties. Reopening completed results restores charts/report without recomputation. Cancellation preserves files and completed checkpoints, but continuing interrupted optimizer trajectories is not implemented: restart the study for a complete matched comparison.

## Cost and limitations

Default narrow design: **14 conditions × 3 seeds = 42 jobs**. Total training: **168,000 expert-example applications**. Default checkpoint/refresh schedule adds approximately **1,471,680 counterfactual expert applications**, excluding optional F online oracle probes, common-state preparation and diagnostic predictor fitting. Up to 2,520 pairwise router fits and 1,008 per-expert diagnostic fits. Each diagnostic fit includes ridge, MLP and classifier work. Full grid has 54 jobs. Application counts are not FLOPs: expert families and backward passes differ. Common-state computation is cached; training-context token exposures are not generated tokens. The UI estimates remaining wall time after completed jobs; extrapolate on the actual model/device rather than from the tiny CPU smoke model.

Current limitations: expert-only development under a frozen general representation; one application per prediction; pairwise aggregation for E>2 needs its own validation; one homogeneous family per study; only weight perturbation (no LR/dropout/architecture asymmetry); bounded in-memory panels; full all-expert probes can dominate cost; repeated validation can overfit; task suite/context eligibility limits external validity; deterministic CUDA/Windows execution requires local verification. No networking, production routing or task-labelled specialist training is added.

## Implementation and checks

Added `token_lab_emergence.py` (protocol/runner), `token_lab_emergence_analysis.py` (metrics/comparisons/report), `EmergentSpecialization.tsx` (setup/live/results), Python and UI tests, and this protocol. Extended Token Lab CLI dispatch and mode/result selection. No Rust schema change is needed.

Tests cover exact relative perturbation and zero identity, all four families' loading/freeze behavior, sampling probabilities, exact D/E exposure counts, pairwise math and save/load, removal of common difficulty/global strength, disjoint balanced reproducible panels, input exclusion, real A–E pipeline, saved results, and UI settings persistence. See the implementation completion message for the actual test commands/results and smoke timings; smoke outcomes are not scientific evidence.

### Implementation verification

- Token Lab Python regression suite: 62 passed. After the final population-aggregation/oracle-label refinements, the updated emergence suite passed all 13 tests; the two numerical decision/roundtrip tests were rerun after the tiny-logit safeguard and passed.
- Token Lab UI tests: 16 passed, including real React form interaction with mocked Tauri IPC. The three emergence UI tests passed again after the final result-label changes.
- TypeScript `tsc --noEmit` and Vite production build passed. Vite retained the application's large-chunk advisory. Native Windows/CUDA execution was not available here.
- Real CLI smoke: four GPT experts, one seed, A–F, two checkpoints, 10 blocks × 2 mixed endpoints, tiny 8-dimensional common model, 48-token context. All six jobs completed in approximately 19.5 seconds on CPU. A/B each received `[20,20,20,20]` applications; D/E both received `[13,14,42,11]`; shared-state and A identity checks held. Every hypothesis remained INCONCLUSIVE, as required for a one-seed smoke run. This timing does not estimate a normal-size GPU study.
