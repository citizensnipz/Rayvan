# Sequential spectral test in Research Console

Branch `research/spectral-live-console` includes the Math Visualizer from `math-lab`, the spectral implementation, and the earlier 13-variant comparison. This new mode fits **one combined window-8, one-basin router** and evaluates actual sequential paths through the original frozen value-model experts and Integrator. It does not construct the separate fresh-training SpectralGeometricEMC architecture or replace the source Integrator.

## Console recipe

1. Open the original heterogeneous model run in History and click **Test router from checkpoint**. Keep the source suite, 2/2/2/2 experts, dimensions, context 256, three trajectory steps and model/data seed. Verify the source is the original `model-latest.pt` (or the exact model used for your bank), not a router-only state file.
2. Keep architecture **Sequential EMC — Counterfactual Value**, and enable **Spectral geometry — sequential learning + live trajectory charts**. The 13-variant toggle is disabled automatically.
3. Defaults: neighbourhood 8, one basin/expert, 256 bank prefixes/split, 100 bank warmup updates, 100 sequential learning updates, 128 fresh evaluation prefixes, router seed 17, bank warmup learning rate 0.01, sequential learning rate 0.001, FP32, zero weight decay, batch 4, validation cadence 20. The smaller online rate is a conservative starting choice for noisier single-probe updates, not a tuned optimum. Use CUDA for experts/evaluation; the small fitted router and bank descriptor cache use CPU.
4. Paste your previous `router-fit-bank.pt` into Saved bank path, keeping its source/model seed/prefix count. Or leave blank to build it. Bank reuse skips only bank measurements: online suffix probes and full-trajectory evaluations still execute experts.
5. Click Run. No CLI experiment command is needed. Normal token budget, gradient accumulation, old value MSE/effect/regret weights, old probe schedules, validation-batch count and capability-generation settings are unused in this mode.

## What is learned and measured

Initialize basin statistics, centres and templates using training-bank observations only. Statistics remain frozen. Step zero in the charts is **after this supervised initialization**, not an untrained baseline. Warmup minimizes mean KL between measured fixed-reference suffix preferences and spectral score preferences, plus the existing basin redundancy penalty. All selected-router defaults otherwise match the earlier combined candidate, with no numerical loss-MSE objective, balance or refractory term.

After warmup, reset the optimizer clock and draw fresh training prefixes. At each online update, uniformly sample one request and one trajectory depth. Reach that state using the current spectral policy, then measure every expert insertion and finish each branch using the same immutable current spectral policy. The expert changes the state and every following choice remeasures that changed state. No Top-K list is predicted ahead of time.

For depth t, expert e, policy pi_k and frozen transition F_e:

    Q_k(h_t,e) = CE(readout(rollout_pi_k(F_e(h_t), t+1)), target)
    y_k = softmax(-Q_k / 0.25)
    p_theta = softmax(S_theta(g(h_t),q(h_t)) / 0.25)
    L = KL(y_k || p_theta) + 0.001 * basin_redundancy

Targets, expert outputs and descriptor measurements are detached. Only basin centres, widths, templates, feature weights and biases receive optimizer updates. Repeats are allowed. One expert per step is executed in each branch; running all experts is restricted to sampled training probes. Online labels are freshly measured and never replayed across policy changes. Identical experts cannot be starved of training here because **all experts are frozen**; this experiment evaluates routing, not expert training opportunity.

The fresh evaluation panel excludes exact prefixes in both banks, and online training excludes that panel and the held-out bank. It is drawn from the same selected suite, not a new task distribution. Nearby packed-text contexts may overlap; this is not document-disjoint validation. Evaluation labels never train the router. Three policies run on the same panel: the current spectral router for every step; the exact original source router for every step; and a constant-per-step policy whose expert IDs come from training-bank mean suffix losses. The last baseline is not an exhaustively optimized constant expert sequence.

## Charts and interpretation

- **Actual final trajectory loss** is the primary chart: mean endpoint cross-entropy after all three steps. Compare spectral with original and constant policy curves. This is not fixed-bank regret or generation quality.
- **Fixed-bank routing regret** compares training/held-out choices under original-reference suffix labels. It is a stationary diagnostic even when online policy targets change.
- **Fitting versus generalization** shows train/held-out KL on that same stationary bank.
- **Fresh online training objective** is noisy because new states and suffix targets are measured each update. It is not comparable in absolute value to a full-bank average.
- Per-step expert counts show sequential choices; traffic diversity alone does not establish useful specialization.

The warmup boundary is stated above the charts. There is no held-out early stopping or best-checkpoint selection. Interpret the latest/final result, not the minimum of a repeatedly inspected curve. Per-prefix paired loss differences and descriptive normal intervals compare policies on identical evaluation inputs. These do not correct for repeated monitoring or overlapping text windows.

Trajectory evaluation endpoint/s and main-context tok/s count only that evaluation, excluding warmup, online probing, and setup. They are not autoregressive generation speed or ordinary all-position language-training tok/s. Warmup uses 100 optimizer updates and online uses another 100; the normal endpoint budget is ignored.

## Saved artifacts

The run contains `spectral-live-report.json`, `summary.json`, streaming `metrics.jsonl`, and a reusable bank. `checkpoints/spectral-live-evaluation.pt` preserves panel inputs/targets. `checkpoints/spectral-live-router-latest.pt` stores the fitted router, its config, optimizer state, source checkpoint path, frozen-source hash and bank hash. It is a router-only artifact, not a full-model checkpoint for the generic loader. Automatic resume is not exposed; repeat from the source for controlled comparisons. The source checkpoint remains unchanged.

Original checkpoints already trained on this suite can contain previous specialization biases; freezing does not undo them. Improvement would establish utility of the tested routing policy on this panel, not universally correct geometry or freedom from expert-training feedback. Failure can reflect descriptor information loss, optimization, sparse online supervision or limitations of the frozen experts.
