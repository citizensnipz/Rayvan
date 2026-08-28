# Rayvan core architecture

Rayvan divides a transformer model into contiguous layer ranges and assigns those ranges to participating machines. The Rust core owns the domain model and will own discovery, health, scheduling, transport, and wire formats. Model execution is isolated behind a runtime trait.

Layer ranges use half-open bounds: `[start_layer, end_layer)`. A model with three layers therefore spans `[0, 3)`.

## Domain model

### Node

A `Node` represents one participating machine. It currently contains a stable ID, availability status, and available-memory capability. These deliberately small fields are enough to identify an assignment and leave room for later capacity-aware placement without selecting a hardware or ML backend.

### ModelManifest

A `ModelManifest` identifies a model, states its transformer-layer count, and may carry a model revision. It is the source of truth for the layer range a swarm must cover. Weight files, tensor shapes, checksums, and tokenizer details remain future work.

### Shard

A `Shard` identifies one contiguous layer range for one model and records the node assigned to it. It describes placement only; it contains no CUDA, Python, PyTorch, Candle, llama.cpp, or model-file object.

### Swarm

A `Swarm` groups a manifest, nodes, and shard assignments. Construction sorts shards by layer and rejects:

- assignments for another model;
- assignments to unknown nodes;
- empty or out-of-bounds ranges;
- missing layers; and
- overlapping layers.

A successfully constructed swarm covers exactly `[0, manifest.total_layers)` once. Availability and future scheduling policy are separate from this structural invariant.

### Activation

An `Activation` carries transport-independent metadata (`model_id`, `request_id`, and the next layer to execute) plus an opaque byte payload. The payload is intentionally not a tensor wire format. A future transport codec can frame and validate it without exposing transport concerns to model runtimes.

## Runtime abstraction

`ShardRuntime` defines two operations:

1. `load_shard` prepares an assigned shard.
2. `forward` consumes an activation and returns the activation for the next shard.

The trait is synchronous and deliberately narrow for this milestone. A future backend may wrap an inference engine while preserving the boundary. `DummyRuntime` validates model and layer ordering, appends its shard ID to the dummy payload, and advances `next_layer`; it loads no weights and performs no inference.

The local simulation creates nodes A, B, and C, assigns one layer shard to each, loads one dummy runtime per shard, and passes an activation through the runtimes in layer order. It uses no sockets.

## Network membership

The prototype membership service listens on localhost TCP. A client sends a versioned `RegisterNode` message and becomes `Connected` only after receiving `JoinAccepted`. Both sides then exchange `Heartbeat` messages. A heartbeat timeout, closed socket, or stopped service removes the in-memory membership and returns the desktop client to `NotConnected`; the client retries from `Connecting`.

Membership messages are tagged JSON inside a bounded 16 KiB length-prefixed frame. JSON is an explicit, inspectable prototype wire format—not arbitrary Rust object serialization. The message enum is independent of TCP so a later transport can reuse or deliberately version the protocol.

The local installation ID is a random UUID generated once and stored in Tauri's application-local data directory. It identifies an installation only. It is not an account, credential, cryptographic identity, or authorization mechanism.

Network membership means only that the bootstrap service recognizes a live node connection. Joining does **not** assign a shard, add the node to a model `Swarm`, load a model, or make the node available for inference. Placement, scheduling, capabilities, and inference participation remain separate future decisions.

## Networking and execution boundary

Distributed networking and model execution have different failure and replacement cycles. WAN transport must handle untrusted bytes, timeouts, reconnection, backpressure, peer identity, and heterogeneous links. An execution backend must handle weights, devices, tensor layouts, and kernels. Keeping the Rust-owned distributed core independent means:

- no Python object or Python-specific serialization crosses a future network boundary;
- transport can reject malformed data before an ML runtime sees it;
- execution engines can be replaced without rewriting swarm or network logic; and
- a node only needs the weights for its assigned contiguous range, not the complete model.

Current networking is limited to localhost bootstrap membership. It provides no peer discovery, routing, NAT traversal, authentication, model loading, scheduling, or inference transport.

## Concepts borrowed from prior work

[Shard](https://github.com/leyten/shard) demonstrates the core pipeline shape used here: split transformer layers into contiguous blocks, retain only a block on each worker, and move activations through workers in order. Its architecture also reinforces an explicit model-runtime boundary, a manifest as model-level truth, and transport as a separately owned subsystem with supervision rather than an incidental socket call. Rayvan adopts those concepts, not Shard's implementation. Rayvan starts with Rust domain types, opaque activation bytes, and no Python wire objects, QUIC, NAT traversal, speculative decoding, payments, or production scheduler.

[Parallax](https://github.com/GradientHQ/parallax) separates contiguous layer allocation from request-path selection and accounts for heterogeneous node capacity and link latency. Rayvan retains the useful distinction between structurally valid model coverage and future dynamic placement/routing policy. It intentionally does not implement Parallax's allocator, scheduler, P2P stack, or framework-specific GPU and Apple Silicon backends yet.
