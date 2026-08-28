# Rayvan

Rayvan is intended to become a decentralized AI inference node for personal computers. In future versions, its Rust core will connect to a distributed network, contribute local compute, receive model shards, run inference workloads, manage its runtime, report local hardware and network health, and update the application.

## Current scope

This repository is deliberately limited to the initial desktop foundation:

- a small Tauri desktop window;
- a minimal vanilla TypeScript UI;
- a Rust-owned application status with the single `NotConnected` network state; and
- a placeholder Rayvan logo.

There is no networking, inference, background service, account, payment, model download, node discovery, or update implementation yet.

## Run locally

Prerequisites:

- [Rust](https://www.rust-lang.org/tools/install)
- [Node.js](https://nodejs.org/)
- the [Tauri system prerequisites](https://v2.tauri.app/start/prerequisites/) for your operating system

Install the frontend and Tauri CLI dependencies:

```sh
npm install
```

Launch Rayvan in development mode:

```sh
npm run tauri dev
```

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
