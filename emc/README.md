# Rayvan EMC research prototype

EMC (Emergent Modular Cognition) is an experimental language-model computation graph. Token embeddings form a shared latent state; a learned Nexus selects a sparse module set; those independent modules produce latent updates; a learned Integrator updates the shared state. The route-integrate cycle repeats a fixed number of times before normalization and vocabulary projection.

## Components

- **Nexus/Router:** scores every module independently at each causal token position, selects top-K, and applies a softmax to the selected scores. The hard selection is sparse while the selected routing weights remain differentiable.
- **EMC modules:** independent transformer-style blocks with separate weights. Each uses normalization, self-attention, feed-forward processing, and residual connections. Modules have no references to other modules and return only a latent update.
- **Integrator:** routing-weights the selected updates, then learns a candidate update and gate conditioned on the current shared state.
- **Cycles and output:** the integrated latent becomes the next cycle's router input. After the configured cycles, a final normalization and linear head produce `[batch, sequence, vocabulary]` logits.

Modules communicate only through the shared latent state and Integrator. `EMCModel.execute_selected_modules` is the local execution boundary: it runs the union of modules selected across a batch, gives each the same causally masked latent tensor, then gathers each token's top-K updates. No module can consume another module's output. The prototype uses a simple local Python loop—no multiprocessing and no claim of distributed execution. That boundary can be replaced experimentally later without coupling module internals.

This research area is deliberately independent from Rayvan's Rust networking code. It contains no networking, distributed execution, checkpoints, pretrained models, or CUDA-specific path. Modules have no manually assigned cognitive labels; useful specialization is intended to emerge through future training experiments.

## Language-learning experiments

The language-model path adds learned positions and causal attention to both EMC and the conventional decoder-only transformer baseline. The baseline is intentionally ordinary and is sized near EMC's total parameter count. Shared utilities provide character tokenization, next-token cross-entropy training, validation loss, perplexity, tokens/second, elapsed time, and autoregressive generation.

From this directory, install the research environment:

```sh
python -m pip install -e ".[test]"
```

### Tiny overfit sanity check

```sh
python -m rayvan_emc.experiments.overfit
```

This trains EMC on six repository-local sentences, prints loss periodically, reports routing diagnostics, and generates greedy continuations. It is a memorization test, not evidence of generalization.

### Small TinyStories experiment

TinyStories is optional so normal tests remain offline and dependency-light:

```sh
python -m pip install -e ".[test,data]"
python -m rayvan_emc.experiments.train --model emc --dataset tinystories
python -m rayvan_emc.experiments.train --model baseline --dataset tinystories
```

The adapter streams bounded subsets through the standard `datasets` API. Adjust `--steps`, `--batch-size`, `--sequence-length`, `--train-stories`, and `--validation-stories`. `--preset research` configures roughly 20M–50M parameters (about 27M for EMC and 25M for the baseline with a small character vocabulary); the default `quick` preset is much smaller.

### Reproducible comparison

```sh
python -m rayvan_emc.experiments.compare
python -m rayvan_emc.experiments.compare --dataset tinystories --preset research --steps 1000
```

Both models receive the same corpus batches, step count, context length, optimizer settings, and seed. The command prints total parameters, theoretical top-K active parameters per token-cycle, validation loss/perplexity, measured throughput, elapsed time, identical-prompt generations, and module utilization. Because this local prototype evaluates the union of modules selected across a batch, measured throughput—not the top-K count—is the honest cost of the current implementation.

## Interpreting results

Primary metrics are held-out validation loss and perplexity versus the honest transformer baseline. Throughput shows the actual cost of EMC's cycles and local union execution; theoretical active parameters describe the intended sparse per-token path. For EMC, inspect per-cycle module traffic, router entropy, route variation across batches and cycles, router gradient norm, and module update norms.

Encouraging evidence would be reliable tiny-corpus overfitting, continuing loss reduction on held-out real text, non-zero router/Integrator/module gradients, multiple used modules, input- or cycle-dependent routes, and validation quality competitive enough with the baseline to merit larger controlled runs.

Warnings or failures include inability to memorize the tiny corpus, stagnant or unstable validation loss, a persistent gap to the baseline at comparable scale, near-zero router gradients, only top-K modules ever receiving traffic, one module taking nearly all traffic, identical routes for every input and cycle, or modules receiving no distinct updates. Routing diversity alone does **not** demonstrate cognitive specialization.

Success does **NOT** mean EMC is a useful chatbot. The purpose is only to determine whether sparse routed circulating modules can learn language modeling competitively enough to justify further research.

## Tests and original forward example

```sh
python -m pytest
python example.py
```
