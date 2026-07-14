# Plugin Permissions

Plugins receive only the capabilities and credentials required for declared operations.

## Initial permission model

`crates/plugin-host` defines a placeholder permission set:

- `network` — allow outbound provider API access when needed
- `filesystem_read` — read project files for discovery
- `filesystem_write` — disabled by default

## Credential access

Plugins request credentials through the native credential store rather than reading `.env` files directly. Secret values should not be logged or returned to the UI.

## Future hardening

Rayvan may add tighter sandboxing, syscall filtering, and per-plugin credential scopes. The current scaffold intentionally avoids overengineering sandbox behavior before real integrations exist.

## Capability gating

If a plugin does not declare `action-execute`, Rayvan must not invoke execution methods even if they exist in code.
