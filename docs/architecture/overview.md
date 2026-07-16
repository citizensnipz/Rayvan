# Architecture Overview

Rayvan is organized around six major runtime areas that work together locally.

## 1. Desktop frontend

The React application in `apps/desktop` presents Rayvan product concepts: projects, environments, configuration, deployments, findings, and actions. The UI does not call provider APIs directly or execute infrastructure mutations.

## 2. Local daemon

`rayvand` is the trusted local control plane and sole production owner of mutable SQLite access, migrations, credentials, operations, approvals, and plugin execution. Desktop and MCP clients use its versioned local IPC protocol.

## 3. Native desktop backend

The Tauri Rust host in `apps/desktop/src-tauri` launches or attaches to `rayvand`, proxies typed daemon commands, and forwards daemon events to the webview. Direct database ownership remains only as a temporary migration compatibility path and must be removed before the daemon cutover is complete.

## 4. Plugin runtime

Provider plugins in `plugins/*` run as separate local processes managed by the daemon through the Rust plugin host. Each plugin implements discovery, inspection, planning, and approved execution phases for one provider.

## 5. MCP server

The local MCP server in `apps/mcp-server` is a thin stdio adapter. It authenticates a registered client, calls `rayvand`, and maps daemon results to MCP tools, resources, prompts, progress, and safe errors. It does not import domain engines, repositories, or plugins.

## 6. Shared domain and analysis packages

TypeScript packages in `packages/*` define provider-independent domain types, configuration analysis, action workflows, AI orchestration boundaries, and plugin contracts. Rust crates in `crates/*` provide native services that should not be duplicated in TypeScript.

## Dependency direction

```text
desktop UI → Tauri daemon client ┐
                                ├→ rayvand → local store / keyring / plugin host
rayvan-mcp → daemon client ─────┘

daemon, desktop UI, ai harness, config engine, action engine, plugins → core domain
```

The `packages/core` package remains near the bottom of the TypeScript dependency graph.
