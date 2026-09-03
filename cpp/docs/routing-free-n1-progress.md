# Routing-Free Collective N1 implementation checkpoint

Last updated: 2026-09-03

## RFMoE-lambda rerun checkpoint

- Requested rerun changes no architecture or routing behavior.
- Only `routing_lambda_initial` changes from `1e-5` to `1e-10`.
- Retain `eta=0.02`, `mu=0.5`, target density `0.5`, threshold `1.0`,
  `Normal(0,0.02)` activation projections, bias `-1e-6`, seed 42, and the
  original data/batch/sequence/evaluation protocol.
- The retained initialization previously began at 90.625% activation density,
  so it already provides the requested near-all-on early participation.
- New run destination: `build/experiments/routing-free-50k-lambda1e-10`.
- Completed: CUDA release build and 38/38 native tests passed; the clean run
  reached 50,176 tokens (49 steps) without NaNs or CUDA errors.
- Result: training loss 10.171906, validation loss 10.158691, validation token
  accuracy 9.7107%, and cumulative throughput 2,899.96 tokens/s.
- Logged activation density moved from 0.90625 (3.625 active experts) at 1,024
  tokens to 0.4375 (1.75 experts) at 50,176 tokens. Recurrent participation was
  zero at the final observation; the starvation flag was set from 20,480 tokens
  onward. No monopoly, global all-on/all-off, scale, NaN, or CUDA warning fired.
- The adaptive coefficient moved from `1.02e-10` after step 1 to `2.2522e-10`
  at the final observation. Its density pressure was effectively negligible at
  this horizon, as intended by the RFMoE-matching setting.
- A 64-batch fixed-window audit gave natural activation counts
  `[556,256,9,1024]` out of 1,024 opportunities per GPT/SSM/recurrent/Delta
  expert: 45.04% density and 1.802 active experts per item. Temporal counts by
  chunk were `[256,256,9,256]`, `[256,0,0,256]`, `[44,0,0,256]`, and
  `[0,0,0,256]`, confirming severe late-context collapse rather than evaluation
  noise.
- Gate decision: **fail** due to expert starvation. Do not start the 1M run.

## Scope and repository state

- Goal: add `RoutingFreeCollective` beside the existing hard-Top-K `LegacyNexus`; never mutate the legacy path into the new design.
- Existing uncommitted work implements the native CUDA Delta family and its tests/tools. Treat all pre-existing modifications as user-owned prerequisite work.
- Experiment population is GPT, SSM, recurrent, Delta, each depth 2. Configuration must remain population-sized rather than hard-coded to four.
- Do not scale beyond four experts until the requested stability and diagnostic gates pass.

## Verified Routing-Free MoE reference

- Paper: Liu et al., *Routing-Free Mixture-of-Experts*, arXiv:2604.00801 (2026).
- Official code was inspected from `liuyilun2000/RoutingFreeMoE`, public-release commit `051b556`, cloned only into ignored `build/routing-free-reference`.
- Paper equations to preserve:
  - `G_i(z) = ReLU(||z A_i||_2 - b_i)`.
  - `f_i(z) = 1{G_i(z) - theta >= 0}`.
  - `rho = mean(f)`, `rho_proxy = mean(G)`.
  - `L_EB = mean_i(mean_items(f_i) * mean_items(G_i))`.
  - `L_TB = mean_items(mean_i(f_i) * mean_i(G_i))`.
  - `L_LB = mu * L_EB + (1-mu) * L_TB`.
  - `lambda_(t+1) = lambda_t * (1+eta)^sign(rho-rho_target)` (official code caps at 1.0).
- The official code batches expert-local `A_i` projections into one matmul, but parameters remain owned by individual experts. It stores bias as `-1e-6` and computes `norm - bias`, producing broad early activation. The paper text describes `b_i=1e-6`; this sign discrepancy must be documented.
- Required adaptation: routing items are chunks, not tokens, because SSM/GRU/Delta state semantics must remain intact.

## Planned native design

1. Add `N1Mode { LegacyNexus, RoutingFreeCollective }` plus routing/GQA/latent configuration fields and v3 config serialization with v1/v2 compatibility.
2. Add shared pre-norm causal GQA with RoPE. Use ATen `scaled_dot_product_attention(..., is_causal=true, enable_gqa=true)`; expose/report `_fused_sdp_choice` backend.
3. Add one local activation module per expert. Routing representation is RMS-normalized mean contextual chunk concatenated with RMS-normalized mean latent, projected back to width D; local `A_i: D -> rank` remains within expert module.
4. Add a routing-item API to `N1Node` so each selected chunk executes exactly two native blocks while carrying per-block recurrent state. Full-batch expert state is gathered/scattered by active request indices; inactive rows keep state unchanged.
5. Process chunks sequentially. Each expert independently masks active batch rows on GPU; no Top-K, activation Softmax, or CPU scalar decision. If an item is all-off, activate all experts for that item as a symmetric recovery path (no score comparison), and record the recovery. Expert outputs are weighted by absolute activation strengths.
6. Each active expert returns token and latent proposals. Normalize latent proposals before attention K/V while retaining raw and normalized norms in diagnostics. Shared latent attends only to active proposals, then takes a gated residual update. Token output uses residual plus a variance-stabilized sum scaled by `1/sqrt(number_of_experts)`; no competition normalization.
7. Add `routing_aux_loss` to model output and include it only during training. Keep lambda as a registered buffer and update it after each training forward without host synchronization.
8. Extend trace/telemetry for activation rates/strengths/biases, density, EB/TB losses, co-activation, correlations, proposal/latent norms, attention weights, inactivity, compute share, and collapse warnings. Parameter/update/gradient norms per expert are milestone-side CPU diagnostics, never hot-path synchronizations.

## Verification order

- First preserve and run the existing native suite before edits.
- Add CPU mathematical tests for RoPE/GQA/reference parity, independent activation/no all-off, balancing directions, latent scale robustness/gradients, sparse non-execution, and state carry.
- Add CUDA BF16 forward/backward and fused-backend reporting. Build with existing LibTorch/CUDA toolchain under `build/emc-delta10` or a new adjacent build directory.
- Run a short 50k-token gate only after unit tests pass. Run 1M only if no starvation/monopoly/all-on/all-off/scale/NaN/CUDA failure is observed.

## Current status

- Implementation and verification are complete. The CUDA release build passes
  38/38 native tests, including routing-free checkpoint round-trip/mode guarding,
  sparse execution, state carry, causal GQA/RoPE, balancing gradients, and BF16
  forward/backward.
- The retained calibrated run is `build/experiments/routing-free-50k-fused`; the
  controlled baseline is `build/experiments/legacy-50k-depth2`.
- The routing-free gate reached 50,176 tokens, validation loss 10.147350,
  validation accuracy 9.7656%, and 2,712.6 cumulative tokens/s. It showed expert
  starvation at multiple observations and ended with recurrent activation 0.
- The legacy gate reached the same token count, loss 10.291084, accuracy 6.8604%,
  and 4,310.5 cumulative tokens/s. Its Top-K routes also collapsed to recurrent
  and Delta.
- The requested 1M-token run was not started because the explicit stability gate
  failed. Full architecture, telemetry, causal, benchmark, memory, and decision
  details are in `cpp/docs/routing-free-n1.md`.
- The RFMoE-matching `lambda_0=1e-10` rerun also failed the same gate. It is
  retained at `build/experiments/routing-free-50k-lambda1e-10`; its checkpoint
  differs from the earlier controlled configuration only in
  `routing_lambda_initial`.
