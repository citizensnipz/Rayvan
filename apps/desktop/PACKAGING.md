# Desktop sidecar packaging

Tauri bundles `rayvand` and `rayvan-mcp` via `bundle.externalBin` in
`src-tauri/tauri.conf.json`. At build time those resolve to host-triple files:

```text
src-tauri/binaries/rayvand-<target-triple>[.exe]
src-tauri/binaries/rayvan-mcp-<target-triple>[.exe]
```

## Prepare scaffolding

From the repo root:

```bash
pnpm prepare:sidecars
# or
pnpm --filter @rayvan/desktop prepare:sidecars
```

This:

1. Builds `@rayvan/daemon` and `@rayvan/mcp-server`
2. Writes host-triple launchers under `src-tauri/binaries/`
3. Copies `rayvand.mjs` / `rayvan-mcp.mjs` for local inspection
4. Replaces empty `build.rs` placeholders

Force rewrite:

```bash
RAYVAN_FORCE_SIDECAR_SCAFFOLD=1 pnpm prepare:sidecars
```

## Platform notes

### Windows

- Always writes `rayvand-<triple>.cmd` / `rayvan-mcp-<triple>.cmd` that run
  `node <apps/*/dist/main.js>`.
- Attempts to compile a small .NET Framework `.exe` launcher with PowerShell
  `Add-Type -OutputAssembly` so `externalBin` paths are real executables.
- If compilation fails, writes a non-empty text placeholder `.exe` and documents
  the sibling `.cmd`. Dev desktop still resolves via `RAYVAN_DAEMON_BIN` or the
  workspace node launcher in `state/mod.rs`.

### macOS / Linux

- Writes executable bash wrappers that `exec node` against the built dist entry.

## Production

Scaffolding launchers are **not** shippable single-file binaries. Before
installer release, replace each host-triple file with:

- Node [Single Executable Applications (SEA)](https://nodejs.org/api/single-executable-applications.html), or
- [`pkg`](https://github.com/vercel/pkg) / equivalent bundler output

Keep the same `rayvand-<triple>` / `rayvan-mcp-<triple>` names expected by Tauri.

## Dev without packaging

`tauri dev` does not require real sidecars. The Rust shell resolves `rayvand` as:

1. Packaged sidecar `binaries/rayvand` (release builds)
2. `RAYVAN_DAEMON_BIN`
3. Workspace `apps/daemon/dist/main.js` or `tsx src/main.ts` via a generated launcher
