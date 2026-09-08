# Experimental spectral-geometric EMC

Router: `spectral_geometric`. Model: `SpectralGeometricEMC`. Research architecture:
`spectral_geometric_emc`. Historical geometric, module-aware and value routers
retain their implementations and checkpoint types.

## Hypothesis and limits

A local latent neighbourhood may predict measured expert usefulness better than
a learned low-dimensional need vector. A graph spectral signature is an input
measurement, not a semantic label, unique shape identifier, or exact underlying
manifold Laplace–Beltrami spectrum. Resonance means weighted signature similarity,
not physical resonance. Held-out routing regret is the primary outcome.

## Descriptor (implementation version 1)

For each observed prefix and each trajectory step, take its last
`min(spectral_neighbourhood_size, prefix_length)` states without a learned
projection. The default window is 16; the controlled suite compares 8, 16, 32.
For the resulting matrix H, compute

- c = mean(H), X = H − c;
- s = sqrt(mean_j ||X_j||²), preserve log(s + eps);
- normalized coordinates Z = X / (s + eps).

Euclidean kNN distances in Z use k = min(spectral_knn_k, m−1). Retain all ties
at the kth distance, rather than arbitrarily breaking vertex relabeling symmetry.
Thus actual neighbourhood degree can exceed k. Bandwidth σ is the median positive
selected directed-edge distance, floored at eps. Directed affinity is
exp(−distance²/(2σ²)); unselected edges and the diagonal are zero. Symmetrize with
max(A, Aᵀ). `latent_knn_plus_sequence` optionally adds weak consecutive-position
edges using max(existing edge, configured weight); it intentionally loses arbitrary
relabeling invariance and is not the default.

L = I − D^(−1/2) A D^(−1/2), with zero diagonal/rows for isolated vertices.
`torch.linalg.eigh` acts on m×m L in float32 outside autocast. Numerical eigenvalues
are clipped to [0,2]. Values ≤ `spectral_zero_threshold` are trivial, including
multiple connected components. Retain the lowest `spectral_modes` nontrivial
modes, extending the cutoff to include a whole tied eigenspace within that
threshold. This tie extension prevents basis-dependent truncated signatures.

For B log-frequency Gaussian filters,

    G_kb = exp(−0.5 ((log(max(λ_k, eps)) − ω_b)/width)²)

centres ω are uniformly spaced from log(0.01) to log(2) by default; width is 0.8.
All frequencies, counts and width are configurable. Mask G to retained modes.
Occupancy is sum_k G_kb, normalized across bands.

Three vertex signals are radial distance, mean local kNN distance, and local
non-flatness. Center each scalar signal before applying Uᵀ. With
E_ks = |(Uᵀf_s)_k|² on retained nontrivial modes, its band energy is
sum_k G_kb E_ks / max(sum_k E_ks, eps). Local non-flatness uses each vertex's
directed kNN displacement matrix (including kth-distance ties), SVD energy,
and residual energy beyond `geometry_tangent_rank` divided by total energy.
It is a local non-flatness proxy, not differential-geometric curvature.

HKS_j(t) = sum_k exp(−λ_k t) U_jk² uses all trivial modes plus retained
nontrivial modes. Default diffusion times: six log-spaced values from 0.1 to 100.
WKS_j(b) = sum_k G_kb U_jk² / max(sum_k G_kb, eps), excluding trivial modes,
uses its separately configured number of log-energy bands. Both are pooled by
vertex mean and population standard deviation. Squared eigenvector weights and
whole tied eigenspaces remove eigenvector sign/basis ambiguity.

Concatenate occupancy, the three signal-energy vectors, HKS mean/std, and WKS
mean/std as nonnegative q_raw. q = q_raw / (||q_raw||₂ + eps).
Q = 4B + 2HKS_scales + 2WKS_bands = 48 by default. Feature names and order are
returned by `feature_names` and included in diagnostics.

The explicit geometry g contains, in order:

1. log(s + eps);
2. first P proportions σ_j² / max(sum σ², eps) from thin SVD(Z), zero-padded;
3. effective dimension proxy 1/sum p² (zero for an empty-energy cloud);
4. σ_max / max(mean σ, eps), using available thin singular values;
5. nearest-neighbour distance mean and population std;
6. pairwise centred-direction cosine mean/std, excluding self and zero vectors;
7. local non-flatness mean/std.

G = P + 9 = 17 by default. Participation ratio is an effective local dimension
proxy, not a ground-truth intrinsic dimension estimator. Graph geometry loses
translation and pure latent-kNN ordering information deliberately; g preserves
absolute RMS scale. Position information may still be present in the input latents.

## Normalization, basins, and scores

Persistent Welford population statistics standardize g. Updates require training
mode and an unfrozen flag. Evaluation cannot update statistics. The count, mean,
M2 and freeze flag save in checkpoints. Standard deviations have a configurable
floor (default 1e−3). q is never standardized through these statistics.

Each expert has M diagonal basins with centre μ, raw variance, positive harmonic
template and positive harmonic feature weights, and scalar bias b:

    v = min(softplus(raw_v) + variance_floor, variance_ceiling)
    r = normalize_L2(softplus(raw_r))
    w = softplus(raw_w) + eps
    D_geo = mean_d ((standardize(g)_d − μ_d)² / v_d)
    R = sum(w q r) / (sqrt(sum(w q²)) sqrt(sum(w r²)) + eps)
    D_res = 1 − clamp(R, 0, 1)
    S_im = b_im − α D_geo − β D_res
    S_i = τ_b (logsumexp_m(S_im / τ_b) − log(M))
    p = softmax(S / T_router)

Defaults: M=4, τ_b=0.1, T_router=0.25, α=β=1, variance floor=0.05,
ceiling=100. Parameter count is E M (2G + 2Q + 1); with six experts it is
3,144 parameters (excluding nontrainable statistics), or 1,572 with three experts.
One basin uses one quarter as many parameters. Spectral-only and geometry-only
ablations zero α and β respectively; inactive parameters remain allocated for
an identical parameter layout.

Base scores remain separate from load-balance bias and refractory penalties.
The latter change final expert action after geometric scoring, with request-local
refractory accumulation/decay. Top-1 chooses the effective minimum action. Traces
include base winner, pre-refractory winner, final winner, probabilities, all basin
distances/resonances/scores and winning basins. Utilization is not specialization.

Training-bank initialization uses only training observations. For each expert,
competitive observations have usefulness at least uniform mass. Deterministic
weighted farthest-point seeds and eight weighted Lloyd updates place centres;
cluster-average q initializes templates and cluster spread initializes widths.
No competitive examples falls back to the highest-usefulness training example.
Empty basins fall back to their deterministic seed; dead basins are reported,
not concealed. Bank fitting freezes fitted statistics. Without a bank, every
expert starts with the same small deterministic, distinct basin offsets and equal
expert scores, avoiding random expert identity preferences.

Within-expert redundancy penalty is

    mean_pairs relu(1 − mean_d(μ_a−μ_b)² / distance_threshold²)
               * relu(cos(r_a,r_b) − similarity_threshold)².

It is nonzero only for centres that are close and templates that are similar.
The default coefficient is 0.001.

## Supervision and causal execution

Measured continuation losses give y = softmax(−loss / T_target), detached.
Router supervision is stable KL(y || softmax(S/T_router)); optional expected
regret is mean sum_i p_i (loss_i − min loss). Weights and target temperature are
explicit. Scores are preference logits, not numerical cost predictions, so cost
prediction MSE is marked not applicable, rather than computing a misleading MSE.

The online sampled probe reuses existing expert execution, Integrator, endpoint
readout, probe schedule, and counterfactual trace/metric structures. It measures
immediate endpoint continuation loss, matching the legacy geometric probe's
one-insertion horizon but using only the causal endpoint. The offline comparison
reuses the existing `FitBank` schema and fixed-reference suffix losses unchanged.
These are separately labelled objectives and must not be conflated.

Routing geometry is computed from detached input states. Task cross-entropy still
trains executed experts, embeddings, and Integrator normally. Geometry is
recomputed after each expert transformation. `SpectralGeometricEMC.endpoint`
executes one complete observed prefix; generic training uses that endpoint path.
`forward` computes all-position logits using independent prefixes, following the
value model's prefix-safe convention, including for Delta experts. It never lets
a later routing decision alter earlier logits. Full all-position execution costs
more and is primarily a compatibility/evaluation path. Direct construction of a
legacy model with the new router is rejected to prevent accidental mean pooling.

Optional temporal geometry augments g with current−previous g and q, and an
explicit previous-valid mask. Cycle zero receives zero deltas/mask. Previous
descriptors live in the trajectory loop; the router has no cross-request cache.
Offline caches include matching previous descriptors from the same bank trajectory.

## Transformation geometry and complementarity

Center input X and output X′ separately. Retain an input PCA basis V_r whose
singular values exceed max(eps, largest_input_singular_value × 1e−5), capped by
`transformation_rank` (default 4). C = X V_r, C′ = X′ V_r. Solve

    J = solve(CᵀC + ridge I, CᵀC′)

with default ridge=1e−5; never explicitly invert. In J=U diag(a) Vᵀ, record padded
log(a+eps), mean/std log stretch, max/min stretch, and sum log stretch (local
log-volume proxy). R=UVᵀ gives normalized trace and Frobenius distance to identity
normalized by sqrt(r). P=V diag(a) Vᵀ gives off-diagonal Frobenius energy / ||P||.
This is shear-like/off-axis stretch, explicitly basis-dependent. Tangent escape
is ||X′−(X′V_r)V_rᵀ|| / (||X′||+eps). Append q_after−q_before and its norm.
Zero numerical rank has deterministic zero map features and reports rank zero;
interpret these with the rank mask, not as a measured identity transform.

The signature has transformation_rank + Q + 10 features (62 by default).
Training-only persistent statistics standardize it before L2 comparison.
For expert pairs, O_ij=stopgrad(y_i y_j), C_ij=cos(standardized ψ_i, standardized ψ_j):

    L_comp = mean_(samples,i<j) O_ij relu(C_ij − c0)².

Default c0=0.9. Default weight=0: observational detached outputs, with loss,
overlap and pairwise similarity reported. Positive weight explicitly enables
expert-only gradients. Integrator parameters and shared input states are detached.
Input PCA bases and graph-spectral changes stay fixed measurements even in active
mode; gradients pass through fitted map/stretch/polar/escape features. Active mode
rejects near-zero or repeated local stretch spectra rather than allowing unstable
SVD gradients. It is a guarded partial-gradient research mode, not a demonstrated
stable end-to-end spectral regularizer. No all-expert differentiable path runs
unless active mode is configured. Identity/high-symmetry cases are supported in
observational mode. No weight-space orthogonality is used.

## Controlled experiments and collision diagnostics

From the `emc` directory:

```bash
python -m rayvan_emc.spectral_experiment --bank /path/to/router-fit-bank.pt --output /path/to/comparison --updates 100 --geometry-dim 8
python -m rayvan_emc.spectral_experiment --smoke --output research/spectral-smoke --updates 80 --prefixes 24
```

The optional `--reference-checkpoint` verifies the original source identity and enables transformation remeasurement. `--geometry-dim` matches the learned baseline need dimension. Requested and actual window sizes are reported separately when bank contexts are shorter than an ablation window.

The first command accepts existing source-identified, fixed-reference train and
held-out banks. It validates finite tensor shapes, matching trajectories and
disjoint prefixes. The source identity and a shared bank fingerprint are persisted.
All fits use exactly those states, expert losses and split assignments. No experts
execute during fitting. Cached descriptors avoid repeated eigensolves. Existing
value replay is left unchanged; there is no new replay of stale latents against
moving expert targets. Use the fixed-bank cache for this initial research question.

Compare the original learned `GeometricNexusRouter` (same mean-pooling and need
encoder) with spectral-only, geometry-only, combined one-basin, and combined
four-basin at each window size. Use fixed update counts, no held-out early stopping
or hyperparameter selection. The original router receives the same KL gradients as
its existing soft-target cross-entropy, with balancing/refractory disabled for
base-score comparison. Training examples determine constant-per-step baselines.

Reports include held-out regret, oracle top-1/top-2, pairwise accuracy excluding
ties, margin/tie rate, expected regret, uniform/random expected regret,
constant-per-step regret, per-step utilization/unselected experts, basin occupancy,
dead basins, parameter counts and latency. Unselected experts in a frozen-bank
fit are utilization evidence; they do not establish gradient starvation.

Descriptor-oracle disagreement finds other held-out neighbours per trajectory
step in concatenated [training-standardized g, q], using Euclidean distance divided
by sqrt(feature count). It reports nearest distance, oracle disagreement within
0.05, distance-conditioned bins, and top-2 oracle-set overlap. No near examples
means the near-identical disagreement rate is unavailable, not zero. A disagreement
is diagnostic evidence, not proof: small counterfactual margins can also produce
oracle label instability. More basins cannot restore discarded information.

The observational transformation audit measures the same fixed experts and fits
transformation statistics on training states only. Existing banks do not contain
expert-transformed states; without the source reference, transformations must be
remeasured and the CLI reports them unavailable. The callable
`transformation_audit(reference, train, held, config)` performs that audit. The
synthetic smoke path has its reference and runs it. Active complementarity is
available through explicit model config, but is not enabled in frozen-bank fits,
where changing experts would invalidate the shared labels.

## Numerical and performance qualifications

Default eps=1e−8, zero-eigenvalue tolerance=1e−5, descriptor std floor=1e−3.
All eigensolves/SVDs/solves run in float32 outside autocast. Identical/tiny clouds,
lengths 1/2, disconnected graphs, repeated spectra and deficient maps have tests.
Eigenspace tie tolerance and nearest-neighbour boundary changes are still finite
numerical modelling choices. Eigenvalues alone do not determine a shape.

Pairwise geometry costs O(m²d), the graph eigensolve O(m³), and thin cloud SVD
O(md min(m,d)). The simple batched masked local SVD implementation can cost
O(m²d min(m,d)); it favours correctness over custom kernels. It never forms a
d×d covariance eigensolve. Scoring costs O(E M (G+Q)). Transformation probes add
all-expert work only for sampled requests; do not mistake offline cached scoring
latency for end-to-end online cost.

The profiler measures batch-one graph/centring, eigensolve, full extraction,
scoring, and combined routing separately, synchronizing CUDA when used. These
are microbenchmarks; allocations, probes and full language-model throughput are
separate. Use a representative trained bank and multiple seeds before drawing a
research conclusion. The included smoke results are explicitly synthetic and
scientifically **INCONCLUSIVE**.
