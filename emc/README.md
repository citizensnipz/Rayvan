# Rayvan EMC research prototype

EMC (Emergent Modular Cognition) is an experimental language-model computation graph. Token embeddings form a shared latent state; a learned Nexus selects a sparse module set; those independent modules produce latent updates; a learned Integrator updates the shared state. The route-integrate cycle repeats a fixed number of times before normalization and vocabulary projection.

## Components

- **Nexus/Router:** pools the current latent batch, scores every available module, selects top-K, and applies a softmax to the selected scores. The hard selection is sparse while the selected routing weights remain differentiable.
- **EMC modules:** independent transformer-style blocks with separate weights. Each uses normalization, self-attention, feed-forward processing, and residual connections. Modules have no references to other modules and return only a latent update.
- **Integrator:** routing-weights the selected updates, then learns a candidate update and gate conditioned on the current shared state.
- **Cycles and output:** the integrated latent becomes the next cycle's router input. After the configured cycles, a final normalization and linear head produce `[batch, sequence, vocabulary]` logits.

Modules communicate only through the shared latent state and Integrator. `EMCModel.execute_selected_modules` is the local execution boundary: every selected module receives the same tensor and cannot consume another module's output. The prototype uses a simple local Python loop—no multiprocessing and no claim of distributed execution. That boundary can be replaced experimentally later without coupling module internals.

This research area is deliberately independent from Rayvan's Rust networking code. It contains no networking, training pipeline, tokenizer, dataset, checkpoint, pretrained model, or CUDA-specific path. Modules have no manually assigned cognitive labels; useful specialization is intended to emerge through future training experiments.

## Run

From this directory, install the small research environment:

```sh
python -m pip install -e ".[test]"
```

Run the tests:

```sh
python -m pytest
```

Run the CPU example:

```sh
python example.py
```
