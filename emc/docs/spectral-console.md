# Spectral comparison in Research Console

Open the original heterogeneous value-model run in History and choose **Test router from checkpoint**. Keep the copied suite, model dimensions, expert composition, trajectory and data seed. Verify Source checkpoint path points to the original local model checkpoint.

Enable **Spectral geometry — frozen-bank comparison (13 variants)** under Routing & integration. This keeps architecture **Sequential EMC — Counterfactual Value** because it supplies the frozen reference. Do not select the separate fresh-training spectral architecture.

The toggle sets frozen experts/shared layers, fixed suffix continuation, reset router, linear placeholder head, FP32, zero weight decay, 256 prefixes per split, 100 updates per variant and learning rate 0.01. Use router seed 17 initially. Need dimension affects only the legacy learned geometric baseline; use 8 for the initial experiment. There are 13 fits, so 100 means 1300 total router updates. Normal token budget and probe schedules do not control this experiment.

Leave Saved bank path blank to measure both banks from the source checkpoint. Or paste the full local `router-fit-bank.pt` path to reuse one. Reuse requires the same source, data seed, context and prefix count; it validates these before fitting. The model checkpoint and bank are different files. Bank creation is microbatched using the existing Delta safety bound. Device selection applies to measurement; cached fitting runs on CPU.

Click Run. The live spectral panel reports bank preparation, variant/update progress, then train and held-out regret, uniform/constant-per-step baselines, oracle accuracy, pairwise accuracy and unused basins. Stop safely cancels at measurement boundaries or router updates. The completed panel is also available in History. The run saves `spectral-report.json`, `summary.json`, the reusable bank and all 13 router-only state files. Router-only state files are not deployable full-model checkpoints.

All variants use identical frozen states and suffix losses. The spectral defaults are documented in spectral-geometric-routing.md; windows 8/16/32 and one/four combined basins are automatic ablations. KL preference supervision uses target and router temperatures 0.25; expected-regret and active complementarity weights are zero, redundancy weight 0.001. No numerical cost MSE trains the spectral routers. No held-out early stopping is performed.

This compares against the legacy mean-pooled GeometricNexusRouter, not the recent expert-conditioned attention router. Prefer combined window 16/four basins as the predeclared primary variant and use other variants diagnostically. Reusing the same bank with another seed does not create new held-out evidence; spectral initialization is deterministic, so identical repeats are expected. An encouraging result should beat both uniform and training-derived constant-per-step regret. Full sequential deployment on the new policy remains a separate experiment.
