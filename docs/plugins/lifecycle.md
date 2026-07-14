# Plugin Lifecycle

## Control-plane lifecycle

Plugin operations follow:

```text
authenticate → discover → inspect → plan → approve → apply → verify → audit
```

Plugin capabilities cover authenticate, discover, inspect, plan, apply, and verify. Approve and audit are owned by Rayvan Core / the host.

Not every plugin must implement every capability.

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

Rayvan invokes plugin methods according to declared capabilities. Requests are typed and time-bounded. The in-process registry used today registers built-in plugins explicitly; process spawning is not required for the foundation SDK.

### Shutdown

Future process-hosted plugins should release resources on host stop. Application exit or integration disconnect stops plugin processes.

## Current status

Process spawning and sandboxing are not implemented yet. `crates/plugin-host` defines placeholder interfaces and documents the intended boundary. The TypeScript SDK contract is serializable so it can move behind that boundary later.
