# Rayvan

Rayvan is intended to become a decentralized AI inference node for personal computers. In future versions, its Rust core will connect to a distributed network, contribute local compute, receive model shards, run inference workloads, manage its runtime, report local hardware and network health, and update the application.

## Current scope

This repository contains the desktop foundation, the first Rust distributed-inference domain model, local network membership, and a graphical EMC Research Console:

- a resizable Tauri desktop application;
- a React and Apache ECharts research UI;
- a Rust-owned live `NotConnected`, `Connecting`, or `Connected` status;
- a localhost bootstrap service with versioned registration and heartbeats;
- a persistent installation node ID;
- model manifests, nodes, contiguous layer shards, validated swarms, and transport-independent activations; and
- a replaceable shard-runtime trait with a three-node local dummy simulation.

The distributed-node layer does not yet provide shard assignment, peer-to-peer routing, decentralized discovery, NAT traversal, authentication, accounts, payments, or model download. The EMC research workflow remains local and separate from that networking layer.

See [Core architecture](docs/architecture.md) for the domain boundaries and prior-work influences.

## EMC Research Console

The Research Console configures, launches, monitors, reopens, and compares EMC
experiments. Suites, architectures, and expert families come from the Python
backend. Runs are written incrementally under the app-local `research-runs`
directory (or `RAYVAN_RUNS_DIR`) with config, metadata, JSONL events,
checkpoints, projections, diagnostics, summary, and logs.

Every saved UI configuration is also a CLI configuration:

```powershell
cd emc
python -m rayvan_emc.research validate experiment.example.json
python -m rayvan_emc.research run experiment.example.json --runs-dir runs
```

Set `RAYVAN_PYTHON` if the desktop app should use a specific Python environment,
and `RAYVAN_EMC_ROOT` if the package is not located beside the desktop source.
TinyStories runs require the normal `rayvan-emc[data]` dependencies.

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

The logo shown inside the window is stored at [`src/assets/rayvan-logo.png`](src/assets/rayvan-logo.png).

The platform installer and executable icons in `src-tauri/icons/` are generated from the same placeholder. After replacing the SVG, regenerate them with:

```sh
npx tauri icon src/assets/rayvan-mark.svg --output src-tauri/icons
```
