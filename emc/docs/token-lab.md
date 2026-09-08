# Token Lab

A standalone observational workspace on `research/spectral-live-console`. No router is instantiated, optimized or queried. No basins, utilization objective, measured-feature expert allocation, or semantic expert roles are used. The Math Visualizer and EMC run views are unchanged.

## Architecture and scientific scope

`token_lab.py` provides CLI/schema/validation and a fresh standalone probe model. A shared token/position embedding and one causal GPT-style contextual block produce H. All experts use the same LayerNorm/readout. Each expert's existing proposal is integrated as `H_after = H_before + delta`; this is not the EMC learned Integrator. GPT, SSM, GRU and Delta implementations come directly from `modules.py`; Delta includes its existing GELU FFN, and SSM uses parallel_scan to avoid a mandatory compiler. The shared GPT context block is present even in an all-SSM experiment: family comparisons concern the expert transformation, not pure end-to-end family architectures. No pre-existing EMC checkpoint is imported.

Training uses the mean endpoint cross-entropy of the baseline and every independent expert output (or every serial intermediate stage). Every expert receives every training example. The configured batch is gradient accumulation over variable-length prefixes, not padded all-position training. Experts have separate parameters; encoder/readout are shared. Serial losses backpropagate through earlier serial stages. Equal training exposure is part of this controlled observational experiment, not a load-balancing loss.

After training all model parameters freeze. For each sampled held-out location, only the observed prefix is supplied; the next token is withheld. One target is measured per context, not all positions in a single forward. Variable prefix endpoints cover source examples/blocks, truncating to maximum context. A Delta proposal may use all currently observed prefix states internally; previous-position states in a serial trajectory need not equal states computed on shorter prefixes. No computation sees the withheld target token except explicitly marked outcome/gradient diagnostics.

Independent mode evaluates every expert on exactly the same H:

- `L0 = -log softmax(readout(H_last))[target]`.
- `Li = -log softmax(readout((H+delta_i)_last))[target]`.
- `Ii = L0-Li`.
- `Ai = sum_{j!=i} Lj/(E-1)-Li` (null with one expert).
- Rank = `1+count(Lj < Li-tie_epsilon)`. Oracle index is zero-based and picks the first exact minimizer for ties; `is_best` marks every expert within epsilon of the minimum. `tied` means at least two experts lie within epsilon of the minimum. These are not exclusive win shares.

Serial mode evaluates each configured expert in order, using the previous stage's output. Improvement is `L_before_stage-L_after_stage`. Relative advantage, oracle expert, ranks, cross-expert tie/win comparisons and pairwise counterfactuals are null/unavailable because the experts did not receive the same state. Repetition/order is explicit in configuration.

## Datasets and splits

- TinyStories: reuses `load_tinystories`, its GPT-2 tokenizer and official train/validation splits. Configurable bounded story counts. Validation sample IDs are nonoverlapping packed token blocks, one sampled endpoint each. Blocks can share documents; this is not document-disjoint inference within the analysis train/test split. Exceeding available validation blocks raises an actionable error; it never silently repeats blocks.
- 10-task capability suite: reuses `CapabilityTaskSuite` and its split-aware deterministic generator. Tasks cycle through all ten lanes with their IDs stored outside model input. Targets are sampled next tokens from generated text, not exclusively task-answer tokens. The fixed printable character vocabulary is not fitted on validation. Training/reference frequencies use 1,000 training-split generated examples; this is recorded in metadata.
- Novelty references: bounded, deterministic post-training observations from TRAIN inputs only. They are not accumulated evaluation observations or a literal training-time history. Each serial stage has its own reference. Hidden references are L2 normalized. Descriptor references use `[radius,effective_rank,pca_entropy,nonflatness,pair_distance_mean,knn_mean,hidden_norm,centroid_distance]`, training-reference mean/std with a 0.001 std floor; incomplete descriptors are excluded. Success references retain training-reference rows with improvement > tie epsilon. Empty banks yield null.
- Analysis uses held-out expert observations only. With a cap, a uniform reservoir of complete locations retains all expert rows for each sampled location. Full raw observations stream independently of this cap. Inclusion probability is `min(1,cap/measured_locations)`. All data are analyzed when under the cap. Missing-feature analyses report their own n and are complete-case, not silently zero-imputed.
- Exact observed-prefix SHA256 matches to actual model-training prefixes or novelty-reference prefixes are excluded from evaluation, as are repeated evaluation prefixes. Counts of skipped candidate locations are recorded. This changes the analyzed population to novel exact prefixes; lexical/template/document overlap can still remain. Prefix hashes exclude the target token and are saved with raw rows and the reference-bank exclusion set.

## Feature catalogue

See [token-lab-features.md](token-lab-features.md) for every stable feature key, mathematical definition, units, category, role, availability and expense. Every run saves the same machine-readable `feature-schema.json`.

Categories: token/position/frequency; prediction uncertainty; hidden-state statistics; token-to-token dynamics; dispersion; PCA shape; local density; training-reference novelty/familiarity; GPT attention; GELU FFN statistics (GPT and Delta); expert transformations; sampled loss-gradient alignment; optional graph spectra/harmonic summaries; before/after feature changes.

Near-zero means `abs(value)<1e-5`; numeric epsilon is 1e-8. PCA uses float32 SVD of centered unscaled H; at most `min(m-1,D)` singular values. Nonflatness is the fraction of energy beyond two tangent components, not exact curvature. The condition ratio is null for numerical rank deficiency (`sigma_last <= 1e-6*sigma_first`). Population, not sample, standard deviations are used throughout features. Kurtosis is excess kurtosis. Degenerate cosines/ranks yield null, never fabricated zeros. Scalar nonfinite measurements are converted to null; invalid spectral computations fail explicitly rather than produce silently corrupted evidence.

Spectral features reuse `spectral_geometry.py` only as an observer. Center/RMS normalize, symmetric Gaussian kNN graph (k=4, whole distance ties), symmetric normalized Laplacian, eigendecomposition, first 8 nontrivial modes including tied boundary eigenspaces, 6 log-frequency bands and pooled HKS/WKS. The 48 harmonic values are normalized as one concatenated vector; they are not raw probabilities. Also save four nontrivial eigenvalues, second-smallest eigenvalue including trivial modes as gap, near-zero mode count, and entropy of eigenvalue mass. Before/after spectral differences share a location-level sampling decision. Spectral selection is random, never based on loss.

GPT attention is explicitly recomputed with returned head weights during sampled probes; it does not replace the expert's output computation. FFN hooks aggregate last-token GELU input/output then release arrays. Unsupported attention/FFN metrics are null. Attention before/after difference is null because there is no common attention operator across heterogeneous stages.

Deep probes compute the gradient of baseline endpoint loss with respect to the last pre-expert latent, and its cosine with each expert delta and negative delta direction. They do not form full parameter gradients/Jacobians. These are target-dependent post-hoc diagnostics, not deployable pre-target routing features.

## Saved schema and storage

Dedicated app-local `token-lab-runs/<run_id>/`, override `RAYVAN_TOKEN_LAB_DIR`. Uses the existing `RAYVAN_PYTHON` / `RAYVAN_EMC_ROOT` configuration. Dedicated Tauri state and event names prevent EMC live-monitor contamination.

- `config.json`, `metadata.json`: versions, seed, model protocol, tokenizer, training-frequency hash, parameter counts and split caveats.
- `observations.jsonl`: flat numeric feature keys plus run/sample/dataset/task/split, local/source position/context start, token/target IDs and decoded text, family/ID/stage, topology, all outcomes, `unavailable_features`. One line per expert per sampled location. `expert_id` is one-based `index:family`; oracle index is zero-based. Null means unavailable; there is no implicit zero filling.
- `feature-schema.json`: exhaustive definitions/roles.
- `summary.json`: full-stream mean performance, observations/location counts, analysis sample size, timing and throughput definitions.
- `analysis.json`: per-expert correlations, bootstrap intervals, quantile comparisons, ridge probes, redundancy, selected interactions, and same-state pair analysis.
- `display.json`: up to 10,000 rows from the stored analysis sample, explicitly display-only. Reopening does not rerun experts.
- `report.md`: deterministic report rendered as readable preformatted Markdown text in the workspace.
- `model.pt`, `reference-bank.pt`: frozen model/tokenizer/config and training-only reference banks/IDs/frequency counts.
- `metrics.jsonl`, `logs.txt`, `status.json`, optional `cancel.requested`: progress, diagnostics and safe cancellation state. Interrupted runs preserve already-written raw rows. No resume is currently exposed.

If raw saving is disabled, only analysis/display samples and summaries remain; there is no promise of recovering omitted rows. Cross-run schemas carry versions/configuration and IDs; cross-run comparison UI is deferred.

## Analysis methods and safeguards

- Pearson correlation and average-tie-rank Spearman per expert. 100 deterministic bootstrap resamples of locations per feature/outcome produce pointwise 2.5/97.5 percentiles when enough valid resamples exist. No p-values or claims of multiplicity correction. Nearby text/templates may remain dependent.
- Five empirical quantile bins (duplicate edges collapsed); mean/median/n, and descriptive `eta² = between-bin sum of squares / total sum of squares`. Tiny bins are flagged. No small-bin region claims.
- Discretized MI: empirical 5×5 quantile contingency table, `sum p_ab log(p_ab/(p_a p_b))` in nats, zeros omitted. This is a positively biased in-sample exploratory estimator, not continuous MI or significance evidence.
- Low/middle/high 20% outcome tails, with means, medians, population std, and high-minus-low standardized difference using the root mean of the two tail variances. UI tail fraction is configurable; saved analysis defaults to 20%. Ties can enlarge tail groups.
- Same-state `L_A-L_B` comparisons only in independent mode: pre-expert feature associations, A/tie/B group summaries, ridge model and selected feature-pair cells. Negative means A better. Shared raw difficulty is not treated as specialization.
- Predictive probes: deterministic SHA256 sample-ID grouped 70/30 split WITHIN expert-held-out observations. Keep only pre-expert input features available on >=80% of analysis-train observations. Median imputation, mean/std and constant-column exclusion fit only analysis-train. Solve ridge with alpha=10 and unpenalized target mean/intercept. Report train/held-out R², MAE, training-mean baseline MAE, standardized coefficients. Target surprisal/probability/margin, gradients, outcomes, attention/FFN and post-expert changes are excluded. A linear model with poor held-out R² cannot prove absence of nonlinear information.
- Rank univariate results by absolute effect size. Select up to four input candidates for modest pairwise 4×4 quantile heatmaps, suppress cells with n<10. This selection uses the same exploratory sample; do not call it confirmatory discovery. The UI also allows arbitrary numeric axis pairs.
- Pearson and Spearman feature redundancy tables. Matrix shows up to 30 category-selected features plus current axes; all pair entries remain accessible in saved analysis.
- Reports include positive, negative, small and unavailable associations, nonlinear candidates, pairwise regions, redundancy and predictive-probe results. A failed probe is `FAILS TO SUPPORT` that particular linear predictor, not evidence that geometry cannot work. No architectural recommendations are generated.

## UI and test recipe

Open **Token Lab** (separate primary nav). Validate then Run. Suggested first substantive exploratory run (not automatically launched): TinyStories, independent, homogeneous GPT ×2 (family selectable), quick 64/128 dimensions, 4 heads, seed 42, prefix 64, neighbourhood16, 1,000 training steps, batch4, LR0.001, 1,000 evaluation locations, reference256, analysis cap2,000, measurement rate1, spectral ON/rate1, deep ON/rate0.05, raw ON, CUDA if available. These are starting budgets, not a claim of expert convergence. Repeat changing only dataset to capability_10; do not directly compare losses across tokenizers/tasks. Repeat expert family/seed only after assessing whether the models learned useful predictions at all.

Charts/filters: grouped numeric X attributes, loss/improvement/advantage/baseline/rank Y, individual/compared experts, task, stage, position bounds, low/high loss quantiles, paired expert selection. Scatter uses ECharts canvas large mode, opacity, linear fit and binned medians. Filtered scatter, low/high table and manual heatmap statistics use the DISPLAY subset and are labelled as such. Saved overview/ridge/report use the documented analysis sample and are not recomputed by display filters. Reopen saved runs restores all stored results without execution. The automated report is always exploratory; target-dependent and response associations are marked by role.

## Performance and deliberate limits

Expert work is O(E) per measured location. Core pairwise distances cost O(m²D), SVD roughly O(m²D), small graph eigensolves O(m³); novelty distances are bounded by the configured training bank. Core and spectral calculations use vectorized Torch operations inside a prefix, on CPU float32; expert execution uses the selected device. Training uses variable-prefix microbatches of one with configurable gradient accumulation. This is an observational first version, not a high-throughput all-position LM trainer.

The summary separates instrumented expert seconds from feature/diagnostic seconds (FFN hook work remains in expert time). Both include reference and evaluation phases; these are not a paired uninstrumented overhead benchmark. End-to-end locations/s includes training/loading/analysis and must not be called generation tok/s. Raw rows stream; analysis memory is O(cap×E×features), display capped at10,000. Redundancy can cost O(features²×sample size); cancellation is checked during feature and redundancy work. Downloads and a single linear-algebra operation may delay cancellation.

Intentionally deferred: pretrained EMC import, resuming training, cross-run comparison UI, true training-time familiarity history, full parameter/Fisher/Jacobian probes, optional MLP predictor, formal multiple-testing p-values/FDR, document-cluster bootstrap, isolated A/B overhead benchmark, and a rich Markdown renderer. No fake attention or exact intrinsic-dimension/curvature claims are supplied in their place.

## Literature-informed motivation (not validation of these measurements)

- [Task2Vec](https://arxiv.org/abs/1902.03545) motivates using measured model-response/gradient information to describe task differences. This lab does not implement its Fisher embedding or claim equivalence.
- [Expert Choice Routing](https://arxiv.org/abs/2202.09368) discusses training allocation and specialization; Token Lab deliberately avoids routing/allocation interventions so observations are not caused by attribute-driven access to training.
- [TinyStories](https://arxiv.org/abs/2305.07759) supplies the existing text benchmark; this lab measures next-token outcomes, not that paper's generative evaluation.
- Entropy, PCA participation ratio, local kNN distance and graph harmonic summaries are explicit candidate statistics, not assumed universal descriptions of computational need. None of the sources establishes that this particular measurement catalogue predicts expert suitability.
