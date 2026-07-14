# Plugin Permissions

Plugins declare the permissions they need. These declarations are validated today; full sandbox enforcement is future work.

## Declared permissions

```text
network
read_secrets
write_remote_configuration
read_local_files
write_local_files
```

`crates/plugin-host` mirrors this vocabulary in `PluginPermissionSet`.

## Credential access

Plugins should request credentials through the native credential store rather than reading `.env` files directly. Secret values must not be logged or returned in errors.

## Capability gating

If a plugin does not declare a capability such as `apply`, Rayvan must not invoke the corresponding handler even if code exists. The in-process registry rejects registration when handlers and declared capabilities do not match.

## Future hardening

Rayvan may add tighter sandboxing, syscall filtering, and per-plugin credential scopes. The current foundation intentionally treats permissions as declarations only.
