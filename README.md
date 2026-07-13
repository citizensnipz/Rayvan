# Rayvan

Rayvan is a local-first, cross-platform DevOps control plane for small teams, solo developers, and AI coding agents. It helps you discover infrastructure connected to a software project, compare environments, inspect configuration, detect drift, review findings, and propose safe infrastructure changes with explicit human approval.

Rayvan is **not** a traditional hosted frontend/backend SaaS application. Most functionality runs locally through a Tauri desktop app, Rust native services, local provider plugins, and a local MCP server.

## Status

Early development. The repository contains architectural scaffolding only:

- Domain types and package boundaries
- Placeholder provider plugins (GitHub, Vercel, Supabase, RunPod)
- A minimal desktop shell with an empty-state screen
- A local MCP server with placeholder tool registrations
- Documentation describing the intended architecture

No real provider API integrations, OAuth flows, hosted backend, billing, or analytics are implemented yet.

## Architecture

```text
apps/desktop        React UI + Tauri host
apps/mcp-server     Local MCP server for coding agents
packages/*          Shared TypeScript domain and engines
plugins/*           Provider plugins (placeholders)
crates/*            Rust native services
docs/*              Product and architecture documentation
```

Major runtime areas:

1. Desktop frontend
2. Native desktop backend (Tauri / Rust)
3. Plugin runtime
4. MCP server
5. Shared domain and analysis packages

See [docs/architecture/overview.md](docs/architecture/overview.md) for details.

## Repository structure

```text
rayvan/
├── apps/           Desktop app and MCP server
├── packages/       Core domain, engines, UI, plugin client
├── plugins/        Provider integrations
├── crates/         Rust native crates
├── docs/           Product and architecture docs
├── tooling/        Shared tooling configuration
└── tests/          Cross-cutting fixtures and integration tests
```

## Prerequisites

- Node.js 20+
- pnpm 10+
- Rust stable toolchain
- Platform dependencies for Tauri development ([Tauri prerequisites](https://v2.tauri.app/start/prerequisites/))

## Installation

```bash
pnpm install
```

## Development

```bash
# Run all dev tasks configured in the workspace
pnpm dev

# Desktop app (Vite + Tauri)
pnpm dev:desktop

# MCP server
pnpm dev:mcp
```

## Testing and quality checks

```bash
pnpm lint
pnpm typecheck
pnpm test
pnpm build
pnpm cargo:check
pnpm cargo:test
pnpm check
```

## Security warning

Do **not** commit real credentials, API tokens, private keys, or production `.env` files. Provider plugins are currently placeholders, but future integrations will store secrets in OS-backed secure storage rather than the repository.

Use `.env.example` as a template only. Keep real values in ignored local files or secure storage.

## License

Apache-2.0. See [LICENSE](LICENSE).
