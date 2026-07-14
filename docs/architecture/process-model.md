# Process Model

Rayvan is designed as a set of cooperating local processes. The exact process model may evolve, but the initial shape is:

```text
Rayvan desktop process
├── React webview
├── Tauri Rust host
├── local MCP server
└── provider plugin processes
    ├── GitHub plugin
    ├── Vercel plugin
    └── Supabase plugin
```

## Desktop shell

The desktop application hosts the React UI and Tauri backend in one user-facing process tree. Tauri commands mediate access to native services.

## MCP server

The MCP server may run as a separate local Node.js process started by the desktop host or an external tool configuration. It communicates over stdio and exposes placeholder tools during early development.

## Provider plugins

Each provider plugin is intended to run as its own local process. Rust owns process lifecycle through `crates/plugin-host`. TypeScript in `packages/plugin-client` defines the typed client protocol without owning process spawning.

## Evolution

Future versions may colocate some services or add supervised worker pools. The boundary principles remain: UI and MCP surfaces do not bypass approval workflows, and plugins do not receive unrestricted host access.
