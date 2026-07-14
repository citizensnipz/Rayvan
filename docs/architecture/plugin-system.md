# Plugin System

Provider integrations are implemented as plugins in `plugins/*` using `@rayvan/plugin-sdk`.

Rayvan Core must not contain provider-specific logic. Plugins describe how to authenticate, discover resources, inspect state, plan changes, apply approved changes, and verify results. Core owns projects, environments, resource organisation, approvals, audit history, secret access, permissions, and plugin lifecycle management.

## Plugin lifecycle

```text
authenticate → discover → inspect → plan → approve (host) → apply → verify → audit (host)
```

Not every plugin must support every capability. Approve and audit remain host/Core responsibilities.

A plugin must never treat inspect or plan requests as permission to mutate infrastructure.

## Manifest

Every plugin exports a versioned `PluginManifest` with:

- Stable plugin `id`
- Human-readable `name` and optional `description`
- `version` and `publisher`
- `rayvanApiVersion` (currently `"1"`)
- Optional `minimumRayvanVersion`
- Declared `capabilities` and `permissions`
- Declared `resourceTypes`

## Capabilities

```text
authenticate | discover | inspect | plan | apply | verify
```

- `authenticate` — connection/credential readiness check (not an OAuth broker)
- `discover` — find provider resources
- `inspect` — observe current resource state
- `plan` — produce a serializable `ChangePlan`
- `apply` — execute a host-approved plan
- `verify` — confirm post-apply state

## Resource model

Plugins use generic envelopes such as `PluginResource`, `DiscoveredResource`, `ObservedResourceState`, `DesiredResourceState`, and `ChangePlan`. Provider-specific schemas stay inside the plugin package.

## Registry

`InProcessPluginRegistry` registers built-in plugins explicitly at startup. It validates manifests, rejects duplicate IDs, enforces handler/capability consistency, and exposes manifest metadata without invoking plugin code.

Filesystem scanning, remote installation, and sandboxed third-party execution are out of scope for the current foundation.

## Process ownership

Rust owns future plugin process lifecycle through `crates/plugin-host`. TypeScript in `packages/plugin-client` will define request/response transport. The current SDK contract is intentionally serializable so it can move behind that boundary later.

## Built-in example

`plugins/example-local` (`@rayvan/plugin-example-local`) is an in-memory mock local environment plugin that demonstrates discover → inspect → plan → apply → verify without credentials or network access.

## Current status

- Foundational SDK, registry, validation, errors, and example plugin are implemented
- Official provider plugins (`github`, `vercel`, `supabase`, `runpod`) are placeholders with empty capabilities
- Dynamic loading, marketplace, signing, sandboxing, and OAuth are not implemented

See also:

- `docs/plugins/authoring.md`
- `docs/plugins/lifecycle.md`
- `docs/plugins/permissions.md`
