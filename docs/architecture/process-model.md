# Process Model

Rayvan is designed as a set of cooperating local processes:

```text
rayvand
└── provider plugin processes
    ├── GitHub plugin
    ├── Vercel plugin
    └── Supabase plugin

Rayvan desktop process
├── React webview
└── Tauri Rust daemon client

MCP host
└── rayvan-mcp stdio adapter
```

`rayvand` continues running when the desktop closes. The initial policy has no
automatic idle shutdown.

## Desktop shell

The desktop application hosts the React UI and Tauri backend in one user-facing
process tree. Tauri launches or attaches to `rayvand` and relays daemon commands
and events. It must not remain a writable production database owner after the
daemon migration.

## MCP server

The MCP server is a separate process launched by an MCP host. It communicates
with the host over stdio and with `rayvand` over user-scoped local IPC. It may
start the daemon when the configured local client identity is valid.

## Provider plugins

Each provider plugin runs as its own local process. `rayvand` owns process
lifecycle through `crates/plugin-host`. TypeScript in `packages/plugin-client`
defines the typed client protocol without owning process spawning.

## Evolution

Future versions may colocate some services or add supervised worker pools. The boundary principles remain: UI and MCP surfaces do not bypass approval workflows, and plugins do not receive unrestricted host access.
