# Plugin Lifecycle

Plugins are local processes managed by the Rust plugin host.

## States

```text
stopped → starting → running → stopping → stopped
                     ↘ crashed
```

## Startup

1. Desktop host resolves the plugin package and executable entry point.
2. Plugin host starts the process with configured permissions.
3. Plugin client performs protocol handshake and initialization.

## Runtime

Rayvan invokes plugin methods according to declared capabilities. Requests are typed and time-bounded.

## Shutdown

Plugins should release resources in `dispose()`. The host stops processes on application exit or integration disconnect.

## Current status

Process spawning and sandboxing are not implemented yet. `crates/plugin-host` defines placeholder interfaces and documents the intended boundary.
