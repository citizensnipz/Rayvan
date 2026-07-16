# Sidecar binaries for Tauri packaging

Tauri `bundle.externalBin` expects host-triple-suffixed binaries next to this
README:

```text
rayvand-<target-triple>[.exe]
rayvan-mcp-<target-triple>[.exe]
```

Generate them with:

```bash
pnpm prepare:sidecars
# or
pnpm --filter @rayvan/desktop prepare:sidecars
```

See [`../PACKAGING.md`](../PACKAGING.md) for Windows/macOS/Linux notes and the
SEA/pkg production path.

Development does **not** require real SEA binaries. In `tauri dev`, the Rust
shell resolves `rayvand` via packaged sidecar, `RAYVAN_DAEMON_BIN`, or a
workspace node/tsx launcher.

Never put client credentials on CLI args. Desktop authenticates as
`rayvan-desktop` using the OS keyring entry provisioned by the daemon.
