# Expert-conditioned geometric router experiment

Select **Expert-conditioned geometric + effects + replay** (`value_head_type=expert_geometric`). This is an additional router; linear, nonlinear and relational geometric checkpoints remain supported.

## Equations implemented

At trajectory depth t let H be the current observed-prefix latent sequence and r the remaining horizon divided by the configured horizon. With learned positional embeddings P:

- X = input_projection(LayerNorm(H + P)).
- q_e = u_e + C(concat(H_last, r)). Each independently trained expert has its own learned u_e; architecture family is not an input.
- v_e = LayerNorm(q_e + MultiHeadAttention(q_e, X, X)). All expert queries run together, four heads with attention width 32.
- z_e = normalize(F(v_e)), with shared two-layer GELU encoder F. Default z dimension is 32 **per expert**, not a shared 32-dimensional vector.
- d_em = ||z_e - normalize(p_em)||².
- Q_e = b_e - alpha*tau*(logsumexp_m(-d_em/tau) - log M), alpha=0.05, tau=0.25, M=4 initially.
- A = Q - mean_expert(Q). Select argmin A among available experts. Execute that expert and the identity-free Integrator, then recompute on the changed H. Repetition is permitted.

All candidates share the encoder, normalization, metric and scale. Expert biases and prototypes learn from comparable measured suffix costs. Independent queries imply initial score differences even though prototype sets start identical. Frozen, randomized data collection makes this initial asymmetry unable to change training opportunity or counterfactual labels. This is not a guarantee that all inference seeds converge identically.

For each sampled state, execute each candidate once to obtain its integrated first update H'_e. Continue from H'_e with the fixed source-checkpoint router to measure final endpoint cross-entropy Y_e. The same insertion supplies both labels; effect prediction does not add another candidate execution.

The compact effect target is a **structured linear projection**:

    d_e = concat(mean_tokens(H'_e-H), (H'_e-H)_last) @ R

R has shape (2*latent_width, effect_dim), fixed Gaussian entries scaled by 1/sqrt(2*latent_width). It uses a separate seed 1729 so changing the router seed does not change target definitions. This intentionally modest first signature preserves mean and endpoint update information; it does not preserve every position-specific transformation. The target includes Integrator acceptance, not just raw expert proposal.

A training-only decoder predicts d_e from concat(z_e, u_e). The objective is:

    cost_mse_weight * mean((A - center(Y))²)
    + regret_weight * mean(sum_e softmax(-A/0.05)_e * (Y_e - min_j Y_j))
    + effect_weight * mean((D(z_e,u_e) - d_e)²).

Defaults are cost_mse_weight=1, regret_weight=0.01 and effect_weight=0.01. A zero weight removes that objective from autograd while retaining its diagnostic error. At least one weight must be positive. State, costs and effects are detached targets; gradients train queries, shared encoder, basins, biases and decoder. The effect decoder never runs during normal routing/inference. Setting effect_weight=0 is an ablation, not the recommended initial test.

## Evidence replay and limits

This initial implementation requires **frozen experts/shared layers, a fixed source continuation, suffix targets, and fresh-probe mode**. Ordinary/controlled expert training and the old loss-only fixed bank are rejected because replay labels or effect targets would be invalid/missing. Joint training needs explicit label-version refresh before it can be supported.

A bounded CPU FIFO ring stores training latents, depth, all expert costs and all expert effect signatures. Default capacity is 1024 states. Every update includes fresh probes plus up to 64 uniformly sampled previous states without replacement. Sampling uses an independent RNG, so replay does not change the prefix/depth collection sequence. Groups of matching depth and sequence length run as batches, weighted by their number of examples. No experts execute for replay. States are FP32; approximate state storage is capacity * prefix_length * latent_width * 4 bytes, plus labels/metadata. Replay makes router updates heavier and checkpoints larger.

Checkpoint state includes replay, ring cursor, replay RNG, original continuation-router configuration and weights. Thus resuming an experiment can restore a reference of a different architecture from the new router. A new console test intentionally starts fresh optimizer/replay state.

No expert utilization penalty is added. A constant expert can legitimately win. The design does not promise that a useful conditional signal exists, or that effect prediction alone establishes usefulness. The bounded spherical energy contribution and mean/endpoint effect summary remain possible limitations.

## First console run

1. From History, use **Test router from checkpoint** on the same source run used in the previous frozen-router comparison. Verify the exact source checkpoint path. Preserve dataset, expert composition, context length and other model dimensions. For the existing eight-Delta source, retain eight Deltas.
2. Select **Expert-conditioned geometric + effects + replay**. This selects frozen experts, fixed continuation, suffix targets, reset router and fresh-probe mode automatically. Keep the defaults: need dimension 32, four basins, effect dimension 16, effect weight 0.01, regret weight 0.01, replay capacity 1024, replay samples 64.
3. Endpoint budget **4096**; batch size **4**; batch multiplier/gradient accumulation **1**; probe probability **1**; probe cap **4**.
4. Learning rate **0.0003**, weight decay **0**, precision **FP32**, router seed **0**, data/model seed **42** (or the exact seed of the comparison). Calibration blocks and minimum probes **0**; fixed-reference collection is already fully randomized.
5. Audit/evaluation interval **128**, evaluation batches **16**. With batch 4 this gives 64 held-out prefixes, 192 state/depth observations. This is a new training run, not a fixed-bank fit. The inference-sized expert measurement batch remains 4; replay batch 64 never enters DeltaNet.

A 4096-endpoint run at batch 4 performs 1024 router updates and, with probe rate 1/cap 4, 4096 fresh state/depth probes. Replay samples are not new data and are not counted as endpoints or context tokens. The console retains endpoint/s and context tok/s; context throughput is endpoint throughput times prefix length, not dense token supervision.

Pull/restart the console to load the new UI and Python backend. No new package or kernel build is required. Local checks:

    python -m pytest emc/tests/test_expert_conditioned.py -q

## Reading the result

Return summary.json, routing.jsonl, metrics.jsonl and training-config.json. Primary comparisons use the same fixed source:

- held_out.mean_regret versus held_out.constant_regret and held_out.uniform_regret;
- held_out.shuffled_state_regret versus held_out.mean_regret;
- held_out.gap_rmse and held_out.effect_mse over training;
- per-depth route counts and expert-update counts (must stay zero).

The shuffled-state audit cyclically shifts predictions among held-out states **within each depth** and evaluates them against the original labels. It preserves routing frequency while breaking state/prediction pairing; it is a diagnostic, not a significance test. It is null for one-prefix audits. A globally constant route gives identical actual and shuffled regret.

Training telemetry includes expert_conditioned.value_mse, effect_mse, soft_regret, router_gradient_norm, fresh_samples, replay_samples and replay_size. For this new head calibration_mse is the pure centered-cost MSE; router_objective is the combined objective. Historical geometric-head calibration_mse retains its previous composite meaning.

Useful evidence is sustained held-out regret improvement over the training-selected constant-per-depth baseline, with worse performance when state pairing is shuffled. Better training fit, lower effect error, or more varied routes alone do not establish successful routing. Confirm a promising result with another router seed; evaluate the new router's own full trajectories too, since fixed-reference action values do not guarantee its greedy rollout is better.


## Decision-only control

Start a fresh router test from the **same original expert checkpoint**, not the
latest trained router checkpoint. Select the expert-conditioned geometric head
first, then set **Cost MSE weight = 0**, **Effect prediction weight = 0**, and
**Decision regret weight = 1**. Do not change these weights on another head:
`value_cost_mse_weight` currently applies only to the expert-conditioned objective.

Keep reset router and fixed continuation enabled, experts frozen, suffix targets,
32 need dimensions, four basins, effect dimension 16, replay capacity 1024 and
64 replay samples. Use 4096 endpoints, batch 4, multiplier 1, probe probability 1,
probe cap 4, learning rate 0.0003, weight decay 0, FP32, router seed 0, data seed
42, calibration blocks/minimum probes 0, evaluation interval 128 and 16 evaluation
batches. The fixed-bank fitting diagnostic remains off.

This optimizes only soft expected measured regret. Inference still uses argmin;
MSE and effect errors are diagnostics, not training forces in this control.
An untrained effect decoder is expected to retain poor effect error. This does
not establish that hard argmin regret is optimized exactly.

Held-out audits now include `predicted_margin_mean`, `predicted_margin_min`, and
`predicted_margin_below_1e_3_fraction`, both overall and per depth. The margin is
second-lowest minus lowest predicted cost, in predicted loss units. A single
candidate has no second-best margin (null). The 0.001 cutoff is a diagnostic
reference, not a calibrated confidence threshold. Compare margins alongside
regret and shuffled-state regret; larger margins alone are not evidence of better
routing. Checkpoint selection is unchanged: model-best.pt follows ordinary
validation loss, not fixed-audit routing regret.
