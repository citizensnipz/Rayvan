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

## Persistence

Plugin lifecycle data is stored locally in dedicated `plugin_*` SQLite tables (schema v3) via `@rayvan/local-database` and `crates/local-store`. Core `Integration` / `ActionPlan` remain separate product concepts and are not overloaded.

Records live in `@rayvan/local-database` (with `@rayvan/plugin-sdk` types). Repository interfaces are complete in-memory for domain tests. better-sqlite3 covers a representative subset (installed plugins, connections, grants, discovery, execution history); remaining entities use the same interfaces and can gain SQLite adapters later. A future cloud store can share the same contracts. Schema authority is the Rust SQL file under `crates/local-store/migrations/`.

### Installed plugin versus connection

An **installed plugin** is a package registration (built-in today) with a manifest snapshot, enablement, and compatibility status. A **plugin connection** is a configured provider account/instance under that installation (for example personal vs org GitHub). Connections may be project-scoped; workspace-wide reuse can be added later without a destructive migration.

### Credentials

Ordinary tables store only `CredentialReferenceRecord` rows (provider, storage key, type). Secret material goes through `CredentialStore`. This pass ships `development_memory` for tests/dev. Production keychain / encrypted store adapters can plug in later without changing connection models. Repository list methods never return secrets.

### Permissions

`PluginPermissionGrantRecord` rows are connection-scoped (optional project/environment narrowing). Active grants require `granted && !revokedAt` and a non-disconnected connection. `PersistentPluginPermissionResolver` (local-database) implements the SDK `PluginPermissionResolver` interface; `InMemoryPluginPermissionResolver` remains for SDK tests. Environment-scoped grants do not broaden to project-only requests.

### Discovery and bindings

**Discovered resources** are generic provider envelopes keyed uniquely by `(connectionId, providerResourceId, resourceType)`. Missing items from a sync are marked `missing` (not deleted). **Resource bindings** attach a discovered resource to a Rayvan project (optional environment). Bindings mean association, not exclusive ownership. Mapping suggestions are advisory and never auto-bind.

### Observed versus desired state

Latest observed state is upserted in `plugin_observed_resource_state` with append-only history in `plugin_observed_resource_state_history`. Desired state is stored separately per binding with monotonic `revision` and optimistic concurrency. Do not store plaintext secrets in either; use credential references.

### Plan, approval, apply, verify

Change plans are immutable versioned JSON envelopes (`planSchemaVersion`). Approvals and rejections are append-only. Apply and verify persist as distinct records. Hosts build `ApprovedChangePlan` from plan + latest approval via `ChangeApprovalService`. Execution history implements `PluginExecutionEventSink` without storing raw IO or secrets.

### Disconnect, disable, uninstall

- **Disconnect:** status `disconnected`, revoke/unusable grants, invalidate bindings, delete credential material, preserve discovered resources / plans / history. Never delete remote provider resources.
- **Disable / uninstall:** soft-status only (`PluginInstallationService.disable` / `uninstall`); block new executions via `PluginExecutionGuard`; preserve installation, connections, resources, and audit history. Permission grants are scoped: replacing grants for one project/environment does not wipe grants for other scopes.

### Startup reconciliation

`PluginInstallationService.reconcileBuiltIns()` creates missing built-in records, updates version/manifest snapshots, marks unavailable built-ins `missing`, and marks unsupported `rayvanApiVersion` as `incompatible` without auto-enabling.

### Host integration

Wire persistence outside the SDK:

```text
PluginExecutionGuard → createPluginExecutionStack({
  permissionResolver: PersistentPluginPermissionResolver,
  eventSink: PersistentPluginExecutionEventSink,
})
```

Do not import repositories from `@rayvan/plugin-sdk`.

### Example lifecycle

```text
Install built-in Vercel plugin
→ Create Vercel connection
→ Store credential reference
→ Grant read permissions
→ Discover projects
→ Bind project to Production
→ Inspect current state
→ Save desired state
→ Generate plan
→ Approve operations
→ Apply
→ Verify
→ Store audit history
```

### Future cloud repositories

The same repository interfaces can back a remote store. Local SQLite remains the default offline path.

## Current status

- Foundational SDK, registry, validation, errors, execution service, and example plugin are implemented
- Local plugin persistence (models, repos, services, adapters, migrations v3) is implemented in `@rayvan/local-database` / `rayvan-local-store`
- Official provider plugins (`github`, `vercel`, `supabase`, `runpod`) are placeholders with empty capabilities
- Dynamic loading, marketplace, signing, sandboxing, OAuth, Tauri command wiring, and environments FK are not implemented

See also:

- `docs/plugins/authoring.md`
- `docs/plugins/lifecycle.md`
- `docs/plugins/permissions.md`
