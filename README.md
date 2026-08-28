# Rayvan

Rayvan is intended to become a decentralized AI inference node for personal computers. In future versions, its Rust core will connect to a distributed network, contribute local compute, receive model shards, run inference workloads, manage its runtime, report local hardware and network health, and update the application.

## Current scope

This repository contains the desktop foundation, the first Rust distributed-inference domain model, and local network membership:

- a small Tauri desktop window;
- a minimal vanilla TypeScript UI;
- a Rust-owned live `NotConnected`, `Connecting`, or `Connected` status;
- a localhost bootstrap service with versioned registration and heartbeats;
- a persistent installation node ID;
- model manifests, nodes, contiguous layer shards, validated swarms, and transport-independent activations; and
- a replaceable shard-runtime trait with a three-node local dummy simulation.

There is no inference, shard assignment, peer-to-peer routing, decentralized discovery, NAT traversal, authentication, account, payment, model download, or GPU detection.

See [Core architecture](docs/architecture.md) for the domain boundaries and prior-work influences.

## Run locally

Prerequisites:

- [Rust](https://www.rust-lang.org/tools/install)
- [Node.js](https://nodejs.org/)
- the [Tauri system prerequisites](https://v2.tauri.app/start/prerequisites/) for your operating system

Install the frontend and Tauri CLI dependencies:

```sh
npm install
```

Start the local network service in one terminal:

```sh
cargo run --manifest-path src-tauri/Cargo.toml --bin rayvan-network
```

Launch Rayvan in development mode in another terminal:

```sh
npm run tauri dev
```

Both default to `127.0.0.1:7878`. Override the service with `RAYVAN_NETWORK_BIND` and the desktop client with `RAYVAN_NETWORK_ADDR`.

Create a production build:

```sh
npm run tauri build
```

## Replacing the icon

The logo shown inside the window is the documented placeholder at [`src/assets/rayvan-mark.svg`](src/assets/rayvan-mark.svg). Replace that file while keeping its name and `76 × 76` view box, or update the image path in [`index.html`](index.html).

The platform installer and executable icons in `src-tauri/icons/` are generated from the same placeholder. After replacing the SVG, regenerate them with:

```sh
npx tauri icon src/assets/rayvan-mark.svg --output src-tauri/icons
```
