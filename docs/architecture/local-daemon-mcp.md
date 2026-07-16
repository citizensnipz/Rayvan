# Local daemon and MCP architecture

Rayvan's local control plane is split into three process roles:

```text
Rayvan Desktop ── local IPC ──┐
                              ▼
MCP host ── stdio ── rayvan-mcp ── local IPC ── rayvand
                                                ├─ SQLite and migrations
                                                ├─ OS credential store
                                                ├─ domain services
                                                ├─ operations and approvals
                                                └─ plugin execution boundary
```

`rayvand` is the authority. `rayvan-mcp` translates MCP requests into typed
daemon methods and does not import repositories, engines, plugins, or provider
credentials. MCP annotations improve host UX, but authorization is always
rechecked by the daemon.

## Local IPC

Protocol version `1` uses length-prefixed UTF-8 JSON-RPC messages.

- Windows: a user-scoped named pipe (`\\.\pipe\rayvan-<user-hash>`).
- macOS/Linux: a Unix socket in the user runtime directory, chmod `0600`.
- No network listener or localhost HTTP port is used.

The first request must be `system.handshake`. Production desktop, CLI, and MCP
sessions must present a registered client ID and credential. The credential is
stored in Windows Credential Manager, macOS Keychain, or Linux Secret Service
through `@napi-rs/keyring`; only a one-way hash is stored in Rayvan metadata.
The unauthenticated `test` client exists only when an isolated test runtime
explicitly enables it.

## Ownership and lifecycle

`rayvand serve` atomically creates a per-user lock before opening SQLite. A
second process probes the existing daemon and either reuses it, reports a
protocol conflict, or removes a stale lock. Lock release checks process
ownership. Interrupted apply operations are marked uncertain/failed and are
not automatically retried; their resource locks are released during recovery.

Operational commands:

```text
rayvand serve
rayvand status
rayvand diagnostics
rayvand stop
```

The daemon remains running after clients disconnect. Idle shutdown is not
currently enabled.

## Permissions and approvals

Built-in MCP permission profiles are Read only, Planner, Operator, and
Administrator. New MCP clients must select a profile and explicit project and
optional environment scopes. Administrator is never the default MCP profile.

Approval policies support:

- desktop approval for every remote apply;
- preapproved project/environment/plugin/permission scopes;
- client approval when the client has `plans:approve`.

Destructive approval is separate. Approval, permission, project scope,
environment scope, plugin scope, and plan freshness are daemon checks.

## Secrets

Ordinary configuration reads return presence/fingerprint/access metadata, not
plaintext secret values. Sensitive writes use the dedicated
`set_sensitive_configuration_value` command and never echo the submitted
value. Client credentials and configuration secrets use the OS keyring.
Legacy `.token` and `.bin` development files are migrated into the keyring and
deleted on first access.

## Operations and events

Operations and approvals are persistent SQLite records. Events are hints for
connected clients; canonical state remains queryable through daemon methods.
Events carry IDs, schema version, timestamps, actors, correlation IDs, and safe
payloads. Project and environment scopes are applied before forwarding events.

The current cancellation rule refuses to claim cancellation once an apply may
have crossed an irreversible provider boundary. Idempotency keys are bound to
the operation actor, project, and type before an existing result is replayed.

## MCP surface

Resources:

- `rayvan://projects`
- `rayvan://projects/{projectId}`
- `rayvan://projects/{projectId}/environments`
- `rayvan://projects/{projectId}/environments/{environmentId}`
- `rayvan://projects/{projectId}/integrations`
- `rayvan://projects/{projectId}/configuration`
- `rayvan://projects/{projectId}/findings`
- `rayvan://projects/{projectId}/operations`

Prompts:

- `diagnose_project`
- `diagnose_production`
- `compare_environments`
- `review_configuration_drift`
- `plan_finding_resolution`
- `review_recent_operations`

Tools are declared in `apps/mcp-server/src/tools/index.ts`. Each invocation
returns concise text plus structured output and maps safe daemon errors without
returning internal causes.

## Development

```text
pnpm daemon:dev
pnpm desktop:dev
pnpm mcp:dev -- --client-id <registered-client-id>
pnpm mcp:inspect
pnpm prepare:sidecars
pnpm test:integration
```

Real multi-process scenario coverage lives in
`tests/integration/daemon-mcp-e2e.test.ts` (spawned `rayvand`, isolated data
dirs, example-local apply/verify, optional MCP `list_projects`).

For MCP Inspector, configure the inspected command as:

```json
{
  "command": "rayvan-mcp",
  "args": ["serve", "--client-id", "<registered-client-id>"]
}
```

Set `RAYVAN_DAEMON_BIN` to a workspace or packaged `rayvand` executable when
`rayvand` is not on `PATH`. Do not put the client credential in this
configuration.

## Transitional constraints

The daemon IPC and MCP adapter are implemented. Status of the local cutover:

| Area | Status |
| --- | --- |
| Desktop → daemon IPC (no `LocalDatabase` in AppState) | **Done** — Tauri attaches/launches `rayvand`, authenticates as `rayvan-desktop` from the OS keyring |
| Agent / MCP workspace UI | **Done** — desktop Agent/MCP surface talks to the daemon |
| example-local in-process sync / plan / apply / verify | **Done** — real fixture mutations via the daemon-hosted TS plugin stack |
| Real multi-process E2E scenario | **Done** — `tests/integration/daemon-mcp-e2e.test.ts` (`pnpm test:integration`); successful verify auto-resolves findings linked to the change plan |
| Sidecar packaging scaffolding | **Done** — `pnpm prepare:sidecars` + `bundle.externalBin`; see `apps/desktop/PACKAGING.md` |
| Production SEA/pkg sidecar binaries | **Transitional** — wrappers / .NET launchers for dev; replace before shipping installers |
| Rust `crates/plugin-host` out-of-process plugin runtime | **Transitional** — not wired; daemon still hosts the TS stack in-process |
| SQLite-backed change-plan / binding persistence | **Transitional** — plugin domain records still use in-memory persistence |

### Plugin execution host (current)

Until `crates/plugin-host` is wired, **the daemon hosts the TypeScript plugin
execution stack in-process** (`createPluginExecutionStack` + built-in
`example-local`). That stack owns discover / inspect / plan / apply / verify for
the local fixture plugin. Plugin domain records for change plans, bindings, and
discovery currently use in-memory persistence (`createInMemoryPluginPersistence`)
even though SQLite tables exist — SQLite repositories for change plans are not
yet connected.

`plugins.list` reports `example-local` as **available** with `host: in_process`
when the built-in is registered. Sync, inspection, apply, and verification
execute against that host when a connection/resource exists. They **never**
report simulated remote success when the plugin, connection, or resource is
missing — callers receive `PLUGIN_UNAVAILABLE` or `NOT_FOUND` instead.

Configuration targets are derived from occurrence `resourceBindingId` (no
separate targets table). Adoption promotes discovered keys to managed
(`source: manual`); ignore marks occurrences with `scope: ignored`.

Isolated tests may set `RAYVAN_ALLOW_UNAUTHENTICATED_TEST_CLIENT=1` so the
unauthenticated `test` client can handshake without touching production keyring
credentials. Production desktop, CLI, and MCP clients must use provisioned
credentials.

Future Rayvan Cloud or a remote MCP endpoint can reuse the daemon
application-service contracts behind a different authenticated transport. It
must not replace the local stdio adapter or move local secret authority into a
hosted service.
