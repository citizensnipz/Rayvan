# Plugin System

Provider integrations are implemented as plugins in `plugins/*` using `@rayvan/plugin-sdk`.

Rayvan Core must not contain provider-specific logic. Plugins describe how to authenticate, discover resources, inspect state, plan changes, apply approved changes, and verify results. Core owns projects, environments, resource organisation, approvals, audit history, secret access, permissions, and plugin lifecycle management.

## Plugin lifecycle

```text
authenticate → discover → inspect → plan → approve (host) → apply → verify → audit (host)
```

Not every plugin must support every capability. Approve and audit remain host/Core responsibilities.

A plugin must never treat inspect or plan requests as permission to mutate infrastructure.

## Trusted execution boundary

All host-facing plugin capability calls go through `PluginExecutionService` in `@rayvan/plugin-sdk` (`src/execution/`). Callers must not invoke plugin handlers directly (except unit tests of the plugin itself).

`createPluginExecutionStack()` wires an `InProcessPluginRegistry`, `InProcessPluginRuntime`, permission resolver, event sink, and `PluginExecutionService` for local and test use.

### Pipeline

Every capability runs the same private pipeline:

1. Generate execution id
2. Record start time
3. Resolve plugin from registry
4. Confirm declared capability
5. Confirm handler exists
6. Validate execution request / capability context
7. Resolve granted permissions
8. Confirm required permissions (policy ∩ manifest ∩ grants)
9. Apply-guards (apply only)
10. Invoke via `PluginRuntime` with timeout + `AbortSignal`
11. Validate handler output
12. Normalize warnings/errors and redact secrets
13. Record finish time
14. Emit execution event (best-effort)
15. Return a typed `PluginExecutionResult` envelope (does not throw for expected failures)

### Approvals

`apply` accepts only an `ApprovedChangePlan` (never a raw `ChangePlan`). Guards require matching `pluginId` / `resourceId`, at least one operation, `approvedOperationIds` coverage for `requiresApproval` ops, and `destructiveApproval === true` when the plan or any operation is destructive.

### Timeouts and events

Default per-capability timeouts live in `DEFAULT_PLUGIN_TIMEOUTS`. External abort → `cancelled`; timer expiry → `timed_out`. Every attempt emits a `PluginExecutionEvent`; sink failures add a warning and must not replace the plugin result.

In-process handlers are not forcibly interrupted mid-flight; abort/timeout stop waiting at the service boundary. Treat timed-out or cancelled `apply` results cautiously until a sandboxed runtime can terminate the worker.

### Runtime

Today's runtime is in-process (`InProcessPluginRuntime`). The service depends on the `PluginRuntime` interface so a future subprocess host can replace the transport without changing callers.

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

`plugins/example-local` (`@rayvan/plugin-example-local`) is an in-memory mock local environment plugin that demonstrates discover → inspect → plan → apply → verify without credentials or network access. Lifecycle coverage includes both direct handler unit tests and an execution-service path test.

## Current status

- Foundational SDK, registry, validation, errors, execution service, and example plugin are implemented
- Official provider plugins (`github`, `vercel`, `supabase`, `runpod`) are placeholders with empty capabilities
- Dynamic loading, marketplace, signing, sandboxing, and OAuth are not implemented

See also:

- `docs/plugins/authoring.md`
- `docs/plugins/lifecycle.md`
- `docs/plugins/permissions.md`
