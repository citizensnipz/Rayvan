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

## Execution-time permission checks

`PluginExecutionService` enforces three layers after plugin resolution:

1. Capability is declared and a handler exists
2. Required permissions for the capability (from an injectable `PluginCapabilityPermissionPolicy`) are a subset of `manifest.permissions`
3. Those same required permissions are a subset of grants from `PluginPermissionResolver`

`DEFAULT_CAPABILITY_PERMISSIONS` maps every capability to `[]`, so built-in plugins with empty grants work until the host injects a stricter policy. Resolvers such as `InMemoryPluginPermissionResolver` and `AllowAllPluginPermissionResolver` are available for tests and local stacks.

Manifest shape is unchanged: permissions remain declarative on `PluginManifest`. The execution service does not invent new manifest fields.

## Credential access

Plugins should request credentials through the native credential store rather than reading `.env` files directly. Secret values must not be logged or returned in errors. The execution service redacts nested keys such as `token`, `secret`, `password`, `authorization`, `apiKey`, `accessToken`, and `refreshToken` in result and event payloads.

## Capability gating

If a plugin does not declare a capability such as `apply`, Rayvan must not invoke the corresponding handler even if code exists. The in-process registry rejects registration when handlers and declared capabilities do not match. The execution service also rejects undeclared capabilities and missing handlers with typed result envelopes (`capability_unsupported`, `missing_handler`).

## Apply approvals

Permission grants are not a substitute for host approval. Mutating `apply` calls require an `ApprovedChangePlan` with `approvedOperationIds` and, for destructive work, `destructiveApproval`.

## Future hardening

Rayvan may add tighter sandboxing, syscall filtering, and per-plugin credential scopes. The current foundation intentionally treats permissions as declarations plus optional execution-time policy/resolver checks.
