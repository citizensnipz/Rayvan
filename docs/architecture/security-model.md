# Security Model

Rayvan is local-first, but local still requires explicit security boundaries.

## Credential storage

Credentials are stored through OS-backed secure storage in `crates/credential-store`:

- macOS Keychain
- Windows Credential Manager
- Linux Secret Service or an appropriate secure fallback

Credentials must not be stored in plaintext files or ordinary database tables.

## Secret handling in the UI and MCP

- UI code never receives raw credentials.
- MCP clients never receive raw credentials.
- Secret configuration values are redacted in inspection views.
- Drift detection should use fingerprints where possible instead of comparing secret values.

## Mutation safety

- Infrastructure mutations require explicit human approval.
- Every mutation produces an audit event.
- MCP tools must not bypass the action approval system.
- AI harness code must not approve its own actions.

## Plugin boundaries

Plugins receive only the capabilities and credentials they need for declared operations. The plugin host in `crates/plugin-host` is responsible for lifecycle management and will eventually enforce tighter permission boundaries.

Inspect and plan operations must never be treated as permission to mutate infrastructure.

## Development warning

Do not commit real credentials, tokens, or production environment files. Use local secure storage and ignored `.env` files for development only.
