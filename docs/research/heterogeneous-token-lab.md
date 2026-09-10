# Heterogeneous Token Lab studies

Both Specialization Validation Sweep and Emergent Specialization now support a **single mixed population: GPT + SSM + recurrent/GRU + Delta**. These are whole expert modules operating on the same contextual latent, not FFN activation variants. Existing homogeneous experiments remain available.

## 1. Build and validate a mixed bank in the UI

Open **Token Lab → Specialization Validation Sweep**.

| Setting | First study |
|---|---|
| Population | Mixed GPT + SSM + GRU + Delta |
| Number of training seeds | 4 |
| First seed | 42 (seeds 42–45) |
| Diagnostic seed | 42 |
| Common pretraining | 1000 steps |
| Specialist training | 1000 steps per expert |
| Batch size / learning rate | 4 / 0.001 |
| Latent / hidden dimensions / heads | 64 / 128 / 4 |
| Prefix / neighbourhood | 48 / 16 |
| Evaluation locations | 2000 |
| Separate sanity locations/task | 50 |
| Reference bank | 128 |
| Core / spectral measurements | On / On |
| Deep probes | Off |
| Maximum features / candidates | 5 / 16 |
| Leave-one-task-out checks | On |
| Device | cuda on your CUDA machine |
| Resume directory | Empty for a new study |

Validate, then Run Token Lab. This creates **8 child runs**: 0% and 100% for each of four seeds, with automatic discovery. Four consecutive seeds cover the full rotation of the calibration task partition across the four architectures. More seeds can be used, preferably in blocks of four for complete rotation.

For a quick execution check, use one seed, common/specialist steps 10, batch size 4, evaluation locations 100, sanity locations/task 2, reference size 16, spectral/deep/cross-task checks off. Keep dimensions unchanged if checking your GPU setup. This is only a smoke test, not research evidence.

### How the bank is trained

Each expert is initialized as its own supported architecture. You cannot clone GPT parameters into GRU/SSM/Delta, nor combine separate homogeneous checkpoints with unrelated encoders/readouts.

General pretraining uses **one shared encoder/readout and the same balanced examples for every expert**:

`J_common = mean_e CE(head(h + expert_e(h)), target)`.

All experts and shared parameters learn during this stage. Each expert receives the same number of example applications and optimizer steps. Architecture and parameter counts intentionally differ. Then save the complete general bank as `common-base.pt` and freeze the shared machinery.

The 100% child fine-tunes each expert on a deterministic four-way partition of the existing ten registered tasks: task index j belongs to slot `(j + seed) mod 4`. This rotates assignments across architectures over four consecutive seeds; no architecture is permanently labelled as a math/memory expert. Partitions contain two or three tasks, with normalized weights. This is deliberate **calibration**, not emergent specialization.

The 0% child restores the exact same **per-expert** general weights and shared state and gives all experts the same general minibatch stream and equal budgets. **0% does not mean identical experts or no competence differences in mixed mode.** Architecture-dependent differences are precisely what this baseline can reveal.

Calibration still evaluates all experts independently on the same states. Target-free discovery covers all four improvement outcomes and all six cross-family pair advantages. Full/simple feature groups, shuffled-label checks and optional leave-one-task-out analysis also run on **every pair**, not just the first pair.

### Reading results

Use the expert-pair selector to inspect choice-gain charts (both 0% and 100%), feature-group comparisons and task transfer. The prediction-outcome selector includes all four experts and six pairs. All pair results and failures are retained.

The overview's mean choice gain is a **mean of pairwise diagnostic choice gains**, not the gain of a four-way deployed router. The report explicitly labels architecture confounding and does not require identical 0% experts. Inspect individual pairs, both strengths and seed consistency. Differences at 0% can be useful architecture-dependent competence; 100% minus 0% asks what deliberate task bias adds. Existing task-gap heuristics alone cannot establish that training caused a difference already present at 0%.

## 2. Run mixed Emergent Specialization

After the sweep completes, reopen its parent results and click **Test emergence with this general bank**. This fills in the source, mixed family, four-expert count and source seed set. It loads **the general checkpoint before task-biased fine-tuning**, not the saved 100% or 0% specialist population. No calibration task profiles are used during emergence.

For an execution check click **Smoke test preset**: mixed mode keeps all four architectures, rather than switching to two GPT experts. For the study click **Narrow study preset** and confirm:

| Setting | First mixed emergence study |
|---|---|
| Expert family / count | Mixed GPT + SSM + GRU + Delta / 4 (fixed) |
| Source | The new heterogeneous validation sweep |
| Seeds | 42, 43, 44, 45, matching that sweep |
| Added weight perturbation | 0, 0.01 |
| Feedback strengths | 0, 0.25, 0.5 |
| Exploration | 0.2 |
| Training blocks / batch | 250 / 4 |
| Expert / router learning rate | 0.001 / 0.001 |
| Router refresh interval / epochs | 50 blocks / 80 |
| Router training / validation states | 256 / 200 |
| Utility / diagnostic states | 300 / 400 |
| Reference bank | 128 |
| Checkpoints | 0, .1, .25, .5, .75, 1 |
| Descriptors / spectral additions | Full / Off |
| Oracle reference | Off initially |
| Device | cuda |

This is **32 condition/seed jobs**. The architecture differences already exist at zero added perturbation. Additional 1% perturbations retain compatibility with the existing asymmetry/control design; large perturbations are not needed for this first architecture test.

A is mixed equal exposure with zero added noise. C is the same mixed starting bank with competence feedback and no added noise. **C−A is the most direct initial test of whether feedback benefits an already heterogeneous population.** B/D/E retain the small-perturbation, equal-exposure, competence-feedback and exactly exposure-matched random comparisons. Inspect D−E to test competence-informed selection beyond random allocation with the same per-expert counts.

All conditions start from the same per-architecture weights for that seed; shared parameters stay frozen. Cross-family weight-vector distances are explicitly unavailable because parameter coordinates are not comparable. Functional disagreement, loss differences, stability, detectability and held-out utility continue to be measured.

No competence-router representation, objective or aggregation was changed for mixed mode. This remains the prior one-step Token Lab residual operation, not a new production EMC router or full recurrent EMC training test. A useful initial architecture difference is not automatically newly emergent specialization; inspect initialization and the development curves separately.

## Controls, persistence and compatibility

- Mixed calibration requires explicit `architecture_confounded=True` internally; homogeneous clone checks remain strict.
- Mixed checkpoint loading requires the **exact ordered architecture sequence**, common dimensions and tokenizer. No silent expert-0 copying or cross-trunk checkpoint combination.
- Zero added perturbation verifies each expert's own hash is unchanged; it does not compare hashes across architectures.
- Shared freeze checks and per-expert equal-budget checks remain active. D/E training exposure counts remain exactly matched.
- Sweep configs add `population=heterogeneous`; entries use `family=mixed`. All pair checks are in `pair_extras` and aggregate `pair_replication`. The usual saved reports, checkpoints and raw observations remain in the same locations.
- Existing homogeneous saved configs default to `population=homogeneous` and `architecture_confounded=False`; resume comparison accepts those missing legacy defaults.
- The older Frozen Expert Routing Test still expects two homogeneous specialists. Its shortcut is hidden for mixed sweeps; use Emergent Specialization for the mixed population test.

Share **report.md + sweep-analysis.json** after the sweep and **report.md + emergence-analysis.json** after emergence. These are separate questions: detectable architecture/calibration differences versus useful competence feedback while experts train.

## Verification and limitations

The mixed backend smoke test runs real general pretraining, 0%/100% calibration, discovery and all six pair checks, then loads the resulting general bank into a real A–E mixed emergence run. Assertions check architecture order, all experts receiving common pretraining, budgets, shared freezes, distinct initial states, zero-noise preservation, incomparable parameter distances, six-pair coverage, correct general-bank loading and matched D/E exposure.

UI checks cover selectors, result views and actual React/Tauri request submission through the new source-bank button and smoke preset. Existing homogeneous regression tests remain in place. CPU execution is tested; Windows/CUDA execution must be checked locally. Equal application counts do not imply equal FLOPs across architectures. Larger discovery runs perform six pair analyses and can cost more analysis time than the old two-expert children; inspect progress by pair.

Verification results: 29 existing calibration/sweep/emergence Python tests passed; all 3 new mixed tests passed, including the real pipeline (~153 seconds for the tiny CPU sweep plus ~8 seconds for emergence). All 20 Token Lab UI tests, TypeScript checking and the Vite production build passed. These smoke timings do not predict normal-size CUDA runtime. Seed-rotated task partitions are counterbalanced across four seeds but are not a full seed-by-partition factorial design.
