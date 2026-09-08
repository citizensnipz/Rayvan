# Spectral-geometric EMC implementation and verification report

**Verdict: INCONCLUSIVE.** The implementation and controlled comparison are working. The small synthetic experiment does not establish the research hypothesis, a spectral advantage, or reliable specialization.

## What changed

Router `spectral_geometric`; architecture `spectral_geometric_emc`; causal model `SpectralGeometricEMC`; dedicated checkpoint type `emc_spectral_geometric`. Baseline commit: `746f27f6c605145f76f8de13d82710047c3d9c9c`.

| Files | Purpose |
|---|---|
| `spectral_config.py` | Validated explicit configuration |
| `spectral_geometry.py` | Causal windows, graph, eigenspaces, Fourier energies, HKS/WKS, named geometry |
| `spectral_routing.py` | Training-only statistics, weighted resonance, multiple diagonal basins, initialization and objectives |
| `transformation_geometry.py` | Local fits, stretch/polar/escape/spectral changes, conditional complementarity |
| `spectral_model.py` | Full-sequence sequential routing, prefix safety, sampled probes, expert-only guarded complementarity |
| `spectral_experiment.py` | Existing-bank loader, descriptor caching, 13 matched fits, collision diagnostic, profiling |
| `model.py`, `nexus.py` | Config fields, new router dispatch, trace extensions; old router math unchanged |
| `architecture.py`, `research_config.py`, `research_runner.py` | Research registration, matched expert populations, config serialization |
| `checkpoint.py`, `training.py` | Separate checkpoint identity and endpoint training dispatch |
| Four `test_spectral_*` / `test_transformation_geometry.py` files | Mathematical, routing, gradient, causal, bank and training checks |
| `docs/spectral-geometric-routing.md` | Exact implemented equations, defaults, complexity and limitations |

## Implemented equations

The technical document is the detailed mathematical specification matching code. In brief: centre and RMS-normalize H; Gaussian symmetric kNN graph; normalized Laplacian; retained nontrivial eigenspaces with tied-boundary expansion; log-band occupancy and centered geometric-signal Fourier energy; mean/std pooled HKS/WKS; q=q_raw/(||q_raw||+eps). g contains log scale, SVD energies, participation-ratio dimension proxy, anisotropy, nearest-distance, cosine and local non-flatness summaries.

Basins use mean diagonal squared standardized distance and positive weighted cosine dissonance: S_im=b−αD_geo−β(1−R). Expert scores are τ_b[logsumexp(S_im/τ_b)−log M]. Router targets use detached measured losses, y=softmax(−loss/T_target), and KL(y||p), optional expected regret, and small conditional basin redundancy. Refractory and balance adjustments act after base scoring.

Transformations use solve(CᵀC+ridge I,CᵀC′), log stretch, log volume, polar features, off-axis stretch, tangent escape and spectral delta. Complementarity weights detached y_i y_j and squared similarity excess. Observational mode is default.

## Verification

- Full EMC suite: **283 passed, 9 skipped**. Skips are existing unavailable backend/GPU tests.
- New mathematical, routing, transformation and experiment tests: **34 passed** (included in the full suite).
- Python compileall and git diff whitespace checks passed.
- Existing seeded geometric and counterfactual-value models have **byte-identical complete state dictionaries and output logits** against the original commit; hashes are in `baseline-reproduction.json`.
- Research-console training, JSON trace persistence and checkpoint restoration passed.
- No NaN/Inf or failed eigensolve/SVD in the tests or smoke fits. Active complementarity gradient tests passed and showed expert gradients without Integrator/shared-latent gradients. Runtime guards still reject unstable repeated/zero stretch spectra.

## Controlled smoke experiment

Three unchanged expert families (GPT, recurrent, Delta), latent dimension 8, two trajectory steps, context 32. A fixed reference receives 30 brief uniform-expert updates on synthetic copy-next-token data. Train and held-out have 24 distinct prefixes each (48 observations per split). All 13 router fits share exactly the same measured fixed-reference suffix losses and states, use seed 17 and 80 updates, with no held-out early stopping. **No trained user checkpoint or real-data bank was available.**

Learned geometric baseline has 184 parameters. The spectral variants have 393 (one basin) or 1,572 (four basins); router parameter budgets are therefore not matched. The unchanged expert population and measured observations are matched.

| Variant | Held-out regret ↓ | Oracle top-1 | Oracle top-2 | Pairwise accuracy | Parameters |
|---|---:|---:|---:|---:|---:|
| learned_geometric | 0.08225 | 25.0% | 50.0% | 40.0% | 184 |
| spectral_only_w8_b4 | 0.06858 | 22.9% | 60.4% | 48.6% | 1572 |
| geometry_only_w8_b4 | 0.05131 | 41.7% | 72.9% | 54.3% | 1572 |
| combined_w8_b1 | 0.05659 | 43.8% | 66.7% | 55.7% | 393 |
| combined_w8_b4 | 0.06039 | 39.6% | 72.9% | 53.6% | 1572 |
| spectral_only_w16_b4 | 0.07984 | 16.7% | 60.4% | 44.3% | 1572 |
| geometry_only_w16_b4 | 0.06887 | 35.4% | 66.7% | 47.9% | 1572 |
| combined_w16_b1 | 0.05131 | 35.4% | 66.7% | 52.9% | 393 |
| combined_w16_b4 | 0.07443 | 33.3% | 66.7% | 45.7% | 1572 |
| spectral_only_w32_b4 | 0.06689 | 27.1% | 60.4% | 47.1% | 1572 |
| geometry_only_w32_b4 | 0.05529 | 45.8% | 64.6% | 55.7% | 1572 |
| combined_w32_b1 | 0.07338 | 25.0% | 60.4% | 50.7% | 393 |
| combined_w32_b4 | 0.05476 | 45.8% | 64.6% | 55.7% | 1572 |

Uniform/random expected regret: **0.06409**. Training-derived constant-per-step regret: **0.06532**.

Combined four-basin routing at window 16 has regret 0.07443: better than the learned baseline (0.08225), but **worse than uniform and constant-per-step**. One basin at window 16 is better (0.05131). At window 32 four basins improve on one (0.05476 vs 0.07338). Spectral-only does not beat uniform in any window. This is mixed evidence, not a successful architecture claim.

Numerical cost prediction error is not applicable: this router predicts preferences, not calibrated losses. Per-step regret, margins, ties, expected regret and utilization are in `report.json`.

## Basin occupancy and starvation

For combined window 16 / four basins, held-out occupancy is:

| Expert | Basin 0 | Basin 1 | Basin 2 | Basin 3 |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 4 | 18 |
| 1 | 2 | 0 | 3 | 0 |
| 2 | 0 | 2 | 19 | 0 |

**6 of 12 basins are unused** in these held-out observations. All three experts are selected. Window 32/four basins has 9 unused basins. Utilization alone does not establish specialization, and unused bins in a frozen-bank fit are not evidence of training-gradient starvation.

## Transformation complementarity

The observational audit standardizes features using training transformations only. Off-diagonal mean similarities are about 0.084–0.331 across experts/steps. L_comp is 0 at the configured similarity threshold 0.9 in both steps. This sample supplies no evidence that active complementarity is needed. Experts were not changed by this audit.

Active mode exists behind a positive config weight and has guarded gradient tests; no active-complementarity training benchmark is claimed. Its input basis and spectral delta stay detached, while local-fit/stretch/polar/escape features carry expert-only gradients.

## Descriptor-oracle disagreement

At window 16, mean nearest held-out descriptor distance is 0.447 / 0.445 for the two steps. Nearest-neighbour best-expert disagreement is 79.2% at each step; top-2 oracle overlap is 58.3% / 60.4%. **There are no neighbours within the near-identical threshold 0.05**, so near-identical disagreement is unavailable. The experiment cannot yet tell whether actual descriptor collisions explain routing errors. Distance-bin details are persisted.

## Latency and parameter counts

Default descriptor dimensions are Q=48 and G=17. Router parameters = E M (2G+2Q+1). At E=6, M=4: **3,144** trainable parameters, excluding normalization buffers.

CPU, one thread, batch one; milliseconds per routing event, ten repetitions. Random latent microbenchmarks, excluding probes and expert execution:

| Latent dim | Window | Graph/centring | Eigensolve | Full descriptor | Scoring | Total routing |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 8 | 0.235 | 0.133 | 1.150 | 0.230 | 1.415 |
| 32 | 16 | 0.203 | 0.113 | 1.300 | 0.211 | 1.546 |
| 32 | 32 | 0.220 | 0.196 | 2.033 | 0.217 | 2.288 |
| 256 | 8 | 0.204 | 0.094 | 1.109 | 0.195 | 1.336 |
| 256 | 16 | 0.272 | 0.138 | 2.345 | 0.243 | 2.633 |
| 256 | 32 | 0.490 | 0.240 | 7.977 | 0.272 | 8.300 |

At d=256/window 32, extraction dominates at roughly 8 ms on this CPU. This is material overhead; no GPU throughput claim is justified. Graph eigendecomposition is m×m; local SVDs dominate larger windows.

## Configuration defaults

All fields below are available in EMCConfig and research RoutingConfig, validated through SpectralConfig and saved in run metadata/checkpoints. Router/architecture names are explicit opt-ins. Existing refractory, balancing and counterfactual scheduling fields remain supported.

| Field | Default |
|---|---|
| `spectral_neighbourhood_size` | `16` |
| `spectral_knn_k` | `4` |
| `spectral_graph_mode` | `latent_knn` |
| `spectral_sequence_edge_weight` | `0.05` |
| `spectral_modes` | `8` |
| `spectral_bands` | `6` |
| `spectral_log_frequency_min` | `0.01` |
| `spectral_log_frequency_max` | `2.0` |
| `spectral_filter_width` | `0.8` |
| `spectral_zero_threshold` | `1e-05` |
| `spectral_eps` | `1e-08` |
| `hks_scales` | `6` |
| `hks_time_min` | `0.1` |
| `hks_time_max` | `100.0` |
| `wks_bands` | `6` |
| `geometry_pca_components` | `8` |
| `geometry_tangent_rank` | `2` |
| `geometry_statistics_frozen` | `False` |
| `geometry_std_floor` | `0.001` |
| `basins_per_expert` | `4` |
| `basin_temperature` | `0.1` |
| `spectral_router_temperature` | `0.25` |
| `spectral_target_temperature` | `0.25` |
| `geometry_score_weight` | `1.0` |
| `resonance_score_weight` | `1.0` |
| `basin_variance_floor` | `0.05` |
| `basin_variance_ceiling` | `100.0` |
| `transformation_rank` | `4` |
| `transformation_ridge` | `1e-05` |
| `transformation_complementarity_threshold` | `0.9` |
| `transformation_complementarity_weight` | `0.0` |
| `transformation_diagnostics_enabled` | `True` |
| `basin_redundancy_weight` | `0.001` |
| `basin_redundancy_distance` | `0.1` |
| `basin_redundancy_similarity` | `0.95` |
| `spectral_route_weight` | `1.0` |
| `spectral_regret_weight` | `0.0` |
| `geometry_temporal_delta_enabled` | `False` |

## Deferred or deliberately limited

- Real-data/user-checkpoint fixed-bank comparison, multiple seeds, uncertainty estimates, and GPU end-to-end throughput: no such bank/checkpoint/GPU was available.
- Stable fully differentiable graph eigenspaces and active complementarity across repeated/degenerate spectra; active mode is guarded and partially detached.
- Active complementarity architecture comparison: changing experts invalidates fixed-bank labels, so it requires its own controlled remeasurement run.
- End-to-end descriptor gradients, custom CUDA kernels, automatic basins pruning/revival, and semantic categorization are not implemented.
- Existing banks store input states/losses but not transformed outputs. The optional `--reference-checkpoint` path verifies source identity before remeasuring transformation diagnostics.
- Online calibration uses immediate endpoint losses; the fixed-bank comparison uses original fixed-reference suffix losses. This distinction is explicit in metadata and docs.
- Existing value replay is preserved; fixed-state descriptor caching is implemented, not a new moving-model replay protocol.

## Next meaningful test

Run the same suite against a source-identified saved bank from the trained EMC reference, with context at least 32. Use `--geometry-dim` to match the baseline need dimension, and optionally `--reference-checkpoint` for a verified transformation audit. Replicate seeds and compare against constant-per-step and uniform before drawing conclusions.

**Final judgement: INCONCLUSIVE for the full hypothesis.** The synthetic study fails to establish a distinct spectral benefit or a reliable advantage from multiple basins.
