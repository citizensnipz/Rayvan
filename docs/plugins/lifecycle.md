# Plugin Lifecycle

## Control-plane lifecycle

Plugin operations follow:

```text
authenticate → discover → inspect → plan → approve → apply → verify → audit
```

Plugin capabilities cover authenticate, discover, inspect, plan, apply, and verify. Approve and audit are owned by Rayvan Core / the host.

Not every plugin must implement every capability.

## Execution service

Host code should invoke capabilities only through `PluginExecutionService` (or `createPluginExecutionStack().executionService`).

### Usage example

```ts
import {
  createPluginExecutionStack,
  type ApprovedChangePlan,
} from "@rayvan/plugin-sdk";
import { plugin as exampleLocal } from "@rayvan/plugin-example-local";

const { executionService } = createPluginExecutionStack({
  plugins: [exampleLocal],
});

const actor = { id: "user-1", type: "user" as const };

const discovered = await executionService.discover({
  pluginId: "example-local",
  actor,
  context: { pluginId: "example-local", integrationId: "int-1" },
});

if (discovered.status !== "succeeded") {
  throw new Error(discovered.error.message);
}

// After inspect + plan through the same service:
const planned = await executionService.plan({ /* … */ });
if (planned.status !== "succeeded") {
  throw new Error(planned.error.message);
}

const approvedPlan: ApprovedChangePlan = {
  plan: planned.data,
  approvalId: "approval-1",
  approvedAt: new Date().toISOString(),
  approvedOperationIds: planned.data.operations
    .filter((op) => op.requiresApproval)
    .map((op) => op.id),
  approvedBy: actor,
};

const applied = await executionService.apply({
  pluginId: "example-local",
  actor,
  resourceId: binding.resourceId,
  context: {
    pluginId: "example-local",
    integrationId: "int-1",
    resource: binding,
    approvedPlan,
  },
});
```

Results are envelopes (`succeeded` | `failed` | `cancelled` | `timed_out`) with redacted errors and warnings. Expected failure modes do not throw.

### Approvals and apply guards

Before `apply` invokes the plugin handler, the service checks:

1. `plan.pluginId === request.pluginId`
2. `plan.resourceId === resource binding resourceId`
3. `operations.length >= 1`
4. `validateChangePlan` + `validateApprovedChangePlan`
5. Every `requiresApproval` operation id is listed in `approvedOperationIds`
6. Every id in `approvedOperationIds` exists on the plan
7. Destructive plans/operations require `destructiveApproval === true`

### Timeouts and cancellation

`DEFAULT_PLUGIN_TIMEOUTS` apply unless `timeoutMs` is set on the request. Pass `signal` for external cancellation. Timer expiry yields `timed_out`; external abort yields `cancelled`. Timers and abort listeners are cleaned up after each attempt.

The in-process runtime checks `AbortSignal` before starting a handler but does not preempt a running handler. Callers must treat `timed_out` / `cancelled` on `apply` as “host stopped waiting,” not a guarantee that mutation stopped. A future subprocess runtime can kill the worker on abort.

### Events

Each attempt emits a `PluginExecutionEvent` to the configured sink (`NoopPluginExecutionEventSink` by default, `InMemoryPluginExecutionEventSink` for tests). Sink failures become warnings and never replace the plugin result.

## Process lifecycle

Plugins may eventually run as local processes managed by the Rust plugin host.

```text
stopped → starting → running → stopping → stopped
                     ↘ crashed
```

### Startup

1. Desktop host resolves the plugin package and executable entry point.
2. Plugin host starts the process with configured permissions.
3. Plugin client performs protocol handshake and initialization.

### Runtime

Rayvan invokes plugin methods according to declared capabilities through `PluginRuntime`. Requests are typed and time-bounded. The in-process runtime used today registers built-in plugins explicitly; process spawning is not required for the foundation SDK.

A future subprocess runtime can implement the same `PluginRuntime` interface so `PluginExecutionService` stays the single trusted boundary.

### Shutdown

Future process-hosted plugins should release resources on host stop. Application exit or integration disconnect stops plugin processes.

## Current status

Process spawning and sandboxing are not implemented yet. `crates/plugin-host` defines placeholder interfaces and documents the intended boundary. The TypeScript SDK contract is serializable so it can move behind that boundary later. The in-process execution service is the current control-plane entry point.
