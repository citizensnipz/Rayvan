# Pairwise expert-advantage experiment

This is an optional training objective for `expert_geometric`. It does not change
expert execution, competence basins, state encoding, replay, or sequential argmin
inference. No new experts, kernels or checkpoint training are required.

## Exact objective

For a probed state let Y_i be the measured suffix endpoint cross-entropy after
inserting expert i and continuing with the frozen reference router. Let Q_i be the
new router's predicted cost. For every unordered pair i < j define:

    delta_ij = stop_gradient(Y_i - Y_j)
    w_ij = abs(delta_ij) if abs(delta_ij) > epsilon else 0
    m_ij = sign(delta_ij) * (Q_i - Q_j) / tau
    L_pair = mean_over_states_and_all_pairs(w_ij * softplus(-m_ij))

Defaults: tau = 0.05, epsilon = 0.001. A positive measured difference means i is
worse, so the objective encourages Q_i > Q_j. The implementation enumerates the
upper triangle once: 28 pairs for eight experts, with no diagonal or double
counting. Masked ties contribute zero but remain in the averaging denominator.
A single-expert population has zero pairwise loss and gradient. Computation uses
FP32 differences and numerically stable softplus.

The total configurable objective is:

    cost_mse_weight * L_MSE
    + geometry_regret_weight * L_soft_expected_regret
    + effect_weight * L_effect
    + pairwise_weight * L_pair

The pairwise weight defaults to zero, preserving existing experiments and old
checkpoints. Zero-weight objectives contribute no gradients. For the first new
test use pairwise weight 1 and all three other weights 0. Metrics for disabled
objectives remain visible. At least one objective weight must be positive.

Unlike the previous soft expected-regret loss, a pair's corrective gradient is not
multiplied by either candidate's policy probability. For a confidently wrong pair
its magnitude approaches abs(delta_ij)/tau (before averaging). Correct, well-separated
pairs have diminishing gradients. Comparisons are weighted by measured outcome
gaps, not expert utilization. This is logistic ranking with counterfactual gap
weights, not regression of exact numerical advantages and not a novelty claim.

## Interpretation and limitations

Scores remain geometric energies, but with pairwise-only training their margins
are preference strengths, not calibrated loss predictions. Bigger margins or
smaller training pairwise loss do not prove lower held-out top-choice regret.
All-pairs ranking can spend effort sorting poor experts that would never be
selected. Measured gaps can contain label noise; the tie tolerance is only a
simple filter, not a statistical confidence interval. Larger observed gaps may
also be noisy. This objective does not fix missing state information, guarantee
specialization, or eliminate every kind of saturation in the encoder/geometry.

The frozen reference policy and experts keep old replay labels valid. Every fresh
probe already measures all candidates, so no additional expert forwards are
needed. Pairwise loss costs O(B*E^2) during training; inference is unchanged and
has no pairwise work. For very large expert pools pair sampling would be needed.

Telemetry adds `expert_conditioned.pairwise_loss` for training and
`held_out.pairwise_loss` overall/per depth for audits. Retain regret, constant and
uniform baselines, shuffled-state regret, route counts and top-two margins as
primary decision diagnostics. Initial and later audits use the same settings.
The existing best-checkpoint selection is still ordinary validation loss;
model-latest.pt is the end-of-budget router.

## Console: reuse the heterogeneous checkpoint

Use the same original mixed-expert source used for the previous router run:

    .../research-runs/1788776474-f449f2ed/checkpoints/model-latest.pt

Do not use the failed-to-generalize router run's checkpoint as the new reference.
Clone the previous router experiment configuration, confirm the source path, then
select Expert-conditioned geometric + effects + replay. Set the objective weights
AFTER selecting the head, because head selection restores its defaults.

| Setting | Value |
|---|---|
| Experts | 2 GPT / 2 SSM / 2 recurrent / 2 Delta, unchanged |
| Dataset / model dimensions | Same source settings: Capability 10, latent 64, hidden 128, context 256, heads 4 |
| SSM backend | Parallel scan |
| Trajectory steps | 3 |
| Expert learning | Frozen/router only |
| Reset router / fixed continuation | On / on |
| Fixed-bank fitting | Off |
| Target | Final rerouted suffix |
| Pairwise advantage weight | 1 |
| Pairwise temperature | 0.05 |
| Pairwise tie tolerance | 0.001 |
| Cost MSE / Effect prediction / Decision regret weights | 0 / 0 / 0 |
| Need dimension / competence basins per expert | 32 / 4 |
| Effect signature dimension | 16 (auxiliary weight is zero) |
| Replay capacity / samples | 1024 / 64 |
| Endpoint budget | 4096 |
| Batch / multiplier | 4 / 1 |
| Probe probability / cap | 1 / 4 |
| Learning rate / weight decay | 0.0003 / 0 |
| Precision | FP32 |
| Data seed / router seed | 42 / 0 |
| Calibration blocks / minimum probes | 0 / 0 |
| Evaluation interval / batches | 128 / 16 |

Return summary.json, routing.jsonl, metrics.jsonl, and training-config.json. The
comparison is against the previous decision-only experiment on this exact source,
not raw regret from the eight-Delta population. Success needs sustained improvement
over the constant-per-depth baseline and worse performance after shuffling state
assignments. More diverse routes alone are not a success criterion.
