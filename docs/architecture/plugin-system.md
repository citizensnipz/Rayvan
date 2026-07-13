# Plugin System

Provider integrations are implemented as local plugins in `plugins/*` using `packages/plugin-sdk`.

## Plugin contract

Each plugin implements the `RayvanPlugin` interface with distinct phases:

```text
connect → discover → inspect → plan → approve → execute → audit
```

A plugin must never interpret inspect or plan requests as permission to mutate infrastructure.

## Manifest

Every plugin exports a manifest with:

- Stable plugin `id`
- Human-readable `name`
- `version`
- `protocolVersion`
- Declared `capabilities`

## Capabilities

Initial capability flags include:

- `resource-discovery`
- `configuration-read`
- `health-read`
- `action-plan`
- `action-execute`

Rayvan uses capabilities to decide which operations are available and which credentials may be requested.

## Process ownership

Rust owns plugin process lifecycle through `crates/plugin-host`. TypeScript in `packages/plugin-client` defines request and response messages, protocol versioning, timeouts, and error handling.

## Current status

All bundled provider plugins are placeholders. They export manifests and stub implementations that return not-implemented results.

See also:

- `docs/plugins/authoring.md`
- `docs/plugins/lifecycle.md`
- `docs/plugins/permissions.md`
