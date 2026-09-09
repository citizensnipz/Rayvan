# Saved-observation feature discovery

## Run in Token Lab

1. Reopen a completed calibration or standard Token Lab run using the existing
   saved-run selector. Its `observations.jsonl` must still exist.
2. In **Find predictive features · saved observations**, keep:
   - Maximum selected features: **5**
   - Candidate feature count: **16**
   - Analysis location cap: **20,000** (all of a 2,000-location run)
   - Diagnostic seed: **42**
3. Click **Run saved-observation analysis**. Do not use the ordinary Run Token Lab
   button for this action. There is no checkpoint path to enter.
4. Use the result's Outcome selector for each expert's improvement and each pair's
   advantage. Compare mean, all-feature and compact-feature test scores. Inspect
   the feature-removal table and its intervals.
5. Repeat on the saved **100%, 50%, and 0%** runs using the same analysis settings.
   Share each resulting `report.md` and `discovery-analysis.json`.

This launches a CPU-only diagnostic process through the existing Tauri Token Lab
process manager. Stop and progress reporting use the normal controls. No model
checkpoint is loaded, expert is executed, dataset is generated, or expert is
trained. Source files are read-only. Results appear as a new saved Token Lab run
whose name ends in “feature discovery”; reopening does not recompute them. Its
saved settings restore in the analysis controls.

## Scientific protocol

The backend uses an explicit **target-free pre-expert** feature allowlist. It
excludes correct-target NLL/probability/margin, all outcomes, task IDs/assignment
labels, expert history, internal activation/attention measurements and post-expert
changes. Generic reference novelty, token controls, confidence, hidden statistics,
geometry and spectra remain eligible. There is no preference for spectral features.

Outcomes are expert improvement `I=L_before-L_after` and same-state pair advantage
`I_A-I_B=L_B-L_A`. Positive pair advantage favours A. Pair construction verifies
independent topology, equal prefix hashes and equal baseline losses; serial
experts receive individual improvement analyses only. Constant pair outcomes are
reported as having no state-dependent difference to explain, with no fabricated
R² or fitted feature ranking.

Raw JSONL is streamed, retaining only admissible measurements and required outcome
metadata. If the cap is exceeded, uniform reservoir sampling retains whole
locations including every expert row. Reports record source/retained counts and
the exact raw-file SHA256. Source Token Lab rows must be grouped by location, as
written by the existing runner. Full raw files, not display samples, are used.

The existing generated-example-group SHA256 **60/20/20** fitting/validation/test
split is preserved across experts, feature subsets and refits. For each outcome:

1. Feature availability (at least 80%), median imputation, centering/scaling and
   constant-column exclusion are fitted on training observations only.
2. Rank by absolute training Spearman correlation. Prune candidates with absolute
   training Pearson correlation above .98 with an earlier candidate; retain at most
   the configured candidate count.
3. Greedily add features using validation ridge MSE, stopping at the feature cap
   or when no addition improves validation MSE by at least 0.5%. The compact model
   can legitimately select no features.
4. Compare training-mean baseline, all-input ridge/MLP and compact ridge/MLP on the
   same test observations. Ridge alpha is selected from 1/10/100/1000 using
   validation MSE. The MLP uses 32 GELU hidden units, standardized targets,
   AdamW .003, weight decay .01, max 150 epochs and 20-epoch patience using
   validation MSE. Model fitting is CPU-only with one Torch thread.
5. Choose compact ridge versus compact MLP by validation MSE. Remove each selected
   feature in turn and refit that model family. Report test MAE increase relative
   to the full compact model and a 200-resample paired example-group bootstrap
   interval. Positive increase means omission harmed prediction. Report selected
   features' test Spearman, sample counts and pointwise intervals as well.

The model type and feature selection never use test outcomes. Test scores report
R², MAE and Spearman for improvement and pair differences separately. Prediction
of improvement alone does not establish specialist identity; the pair analysis
addresses differential competence.

## Interpretation and limitations

- This is a bounded search, not proof of the unique or optimal feature set.
  Univariate screening and ridge selection can miss purely nonlinear interactions.
  The all-input MLP is an explicit check for information lost by the compact set.
- Correlated measurements can substitute for one another. A weak individual
  ablation is not proof that a feature carries no information. Refitting a smaller
  MLP also changes its optimization; intervals condition on these fitted models.
- Pointwise intervals are exploratory, not multiple-testing-adjusted discoveries.
  Reusing test observations already inspected in earlier reports is not fresh
  confirmatory evidence, even with a correctly separated fitting pipeline.
- No new metrics or expert training are required. A failure here means this search
  and these predictors failed, not that the observations contain no information.
- Across-run scores are not paired if evaluation locations differ. Replicate
  useful compact sets on fresh observations before an architectural decision.

Files saved in the new run: `config.json` (source model configuration),
`discovery-config.json`, `discovery-analysis.json`, `analysis.json` (UI wrapper),
`summary.json`, `report.md`, `status.json`, `metrics.jsonl` and process logs. The
original observations and checkpoints are never modified or copied unnecessarily.

Tests: `python -m pytest emc/tests/test_token_lab_discovery.py` and
`node --import tsx --test tests/token-lab-discovery.test.mjs`.
