# Snapshot-batch spectral learning

In the sequential spectral Console test, enable **Snapshot-batch spectral learning**. Keep the original heterogeneous source checkpoint and saved bank. Use:

- Window 8, one basin/expert; frozen 2/2/2/2 experts, context 256, three trajectory steps.
- Bank warmup: 100 updates, learning rate 0.01; 256 prefixes/split.
- Sequential learning: 1,000 updates, learning rate 0.001.
- Counterfactual states per snapshot: 20; fit updates per snapshot: 20.
- Evaluation cadence 100; evaluation prefixes 128; router seed 17, source/data seed unchanged.
- Execution batch 4, CUDA, FP32, weight decay 0.

This gives 50 rounds and 1,000 probed states, matching the previous 1,000-probe budget. Each optimizer update now averages 20 state/depth examples. Execution batch size is separate from the evidence batch size.

At each round, copy the current spectral router into an immutable snapshot. Collect the entire batch before fitting. The snapshot reaches each sampled state, and every counterfactual branch recomputes spectral geometry after every expert transformation. For each expert e, record final endpoint CE after inserting e and following the snapshot for the suffix. With fixed labels y = softmax(-CE/0.25), minimize mean KL(y || softmax(scores/0.25)) plus the existing basin regularizer over the whole batch. Only spectral router parameters change. After 20 updates discard the batch and refresh the snapshot. AdamW moments continue across rounds. Audits evaluate the current trained router and do not mutate the separate snapshot.

This is not a fixed expert sequence or an imitation penalty. It does not guarantee monotonic improvement. Repeated fitting may overfit a batch. Evaluate actual final trajectory loss; bank regret still assumes the original continuation, and low KL is not winner accuracy.

The toggle defaults off to preserve old configurations. Disabled means one state and one fit update per refresh, as before. Collection progress and cumulative probes are displayed; the optimizer counter pauses during collection. Reports include snapshot settings, rounds and actual probes. Partial final rounds collect the full configured batch but execute only the remaining optimizer updates. Held-out evaluation remains excluded from training; no best-checkpoint selection is used.

Send spectral-live-report.json and metrics.jsonl after testing. Source checkpoint and experts remain unchanged. The existing Math Visualizer is preserved.
