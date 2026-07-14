# Architecture Overview

Rayvan is organized around five major runtime areas that work together locally.

## 1. Desktop frontend

The React application in `apps/desktop` presents Rayvan product concepts: projects, environments, configuration, deployments, findings, and actions. The UI does not call provider APIs directly or execute infrastructure mutations.

## 2. Native desktop backend

The Tauri Rust host in `apps/desktop/src-tauri` bridges the UI to native capabilities: filesystem access, secure credential storage, local database lifecycle, and plugin process hosting.

## 3. Plugin runtime

Provider plugins in `plugins/*` run as separate local Node.js processes managed by the Rust plugin host. Each plugin implements discovery, inspection, planning, and approved execution phases for one provider.

## 4. MCP server

The local MCP server in `apps/mcp-server` exposes Rayvan domain operations to coding agents. It uses the same domain packages and approval boundaries as the desktop UI.

## 5. Shared domain and analysis packages

TypeScript packages in `packages/*` define provider-independent domain types, configuration analysis, action workflows, AI orchestration boundaries, and plugin contracts. Rust crates in `crates/*` provide native services that should not be duplicated in TypeScript.

## Dependency direction

```text
desktop UI → Tauri commands → Rust native services → plugin host / credential store / local store

desktop UI, mcp server, ai harness, config engine, action engine, plugins → core domain
```

The `packages/core` package remains near the bottom of the TypeScript dependency graph.
