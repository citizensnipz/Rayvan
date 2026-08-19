# Plugin Authoring

Plugins live in `plugins/*` and depend on `@rayvan/plugin-sdk`.

## What a Rayvan plugin is

A Rayvan plugin is a modular integration package that implements the `RayvanPlugin` contract. Official, community, local, and proprietary plugins should eventually share this same contract. Today, only explicitly registered built-in plugins are supported.

## Core vs plugin responsibilities

| Belongs in Core / host | Belongs in a plugin |
| --- | --- |
| Projects and environments | Provider authentication readiness |
| Resource organisation and bindings | Resource discovery and inspection |
| Approval and confirmation flows | Change plan generation |
| Audit history | Applying approved changes |
| Secret access mediation | Verifying applied changes |
| Plugin permissions and lifecycle | Provider-specific schemas and clients |

## Package layout

Suggested layout:

- `manifest.ts` — stable plugin identity, capabilities, permissions, resource types
- `index.ts` — `RayvanPlugin` implementation
- optional folders such as `discover/`, `inspect/`, `plan/`, `apply/`, `verify/`, `client/`

## Manifest example

```ts
import {
  RAYVAN_PLUGIN_API_VERSION,
  type PluginManifest,
} from "@rayvan/plugin-sdk";

export const manifest: PluginManifest = {
  id: "example-local",
  name: "Example Local",
  description: "Mock local environment plugin",
  version: "0.1.0",
  publisher: "rayvan",
  rayvanApiVersion: RAYVAN_PLUGIN_API_VERSION,
  capabilities: ["discover", "inspect", "plan", "apply", "verify"],
  permissions: [],
  resourceTypes: [
    {
      id: "local.service",
      name: "Local Service",
      schemaVersion: "1.0.0",
    },
  ],
};
```

## Implementing handlers

Handlers are optional. Declare a capability only when the matching handler exists.

```ts
import type { RayvanPlugin } from "@rayvan/plugin-sdk";
import { manifest } from "./manifest.js";

export const plugin: RayvanPlugin = {
  manifest,
  async discover(context) {
    return [
      {
        providerResourceId: "local-api",
        resourceType: "local.service",
        name: "Local API",
        metadata: { port: 3000 },
        schemaVersion: "1.0.0",
      },
    ];
  },
  async inspect(context) {
    return {
      resourceId: context.resource.resourceId,
      pluginId: manifest.id,
      resourceType: context.resource.resourceType,
      observedAt: new Date().toISOString(),
      status: "ready",
      attributes: { port: 3000 },
    };
  },
};
```

Contexts must remain plain serializable data. Do not accept database clients, UI state, or unrestricted service containers. `ApplyContext` and `VerifyContext` both include the `ResourceBinding` so plugins can resolve provider-native ids without assuming they equal Rayvan resource ids.

Change plans must be descriptive and serializable. They must not contain executable functions.

## Registering a built-in plugin

```ts
import { InProcessPluginRegistry } from "@rayvan/plugin-sdk";
import { plugin as exampleLocal } from "@rayvan/plugin-example-local";

const registry = new InProcessPluginRegistry();
registry.register(exampleLocal);
```

Registration validates the manifest, rejects duplicate plugin IDs, and fails if declared capabilities do not match handlers.

## Rules

- Do not depend on desktop, MCP server, AI harness, or other provider plugins.
- Keep provider-specific types inside the plugin package.
- Never mutate infrastructure during inspect or plan phases.
- Do not read secrets from `.env` files directly; request mediated secret access when that host API exists.
- Do not create approvals, persist audit events, or access the Core database from plugin code.
- Do not call `PluginExecutionService` or other `src/execution/` host APIs from plugin code. Export handlers only; the host invokes them through the execution service.

## Packaging an OOP plugin

1. Implement `RayvanPlugin` under `plugins/<name>` (developer-mode unpacked layout).
2. Provide a Node launcher entrypoint that calls `serveRayvanPlugin` from `@rayvan/plugin-client` (binary name e.g. `rayvan-plugin-github`).
3. Pack with `@rayvan/plugin-package` into a per-triple `.rayvan-plugin` ZIP.
4. Install via daemon `plugins.installFromPath` (Desktop **Add from file** or MCP).

Setup contributions are data-only (`manifest.setup`); Rayvan UI owns widgets. Auth secrets must go through `CredentialStore` — never plaintext in SQLite, logs, MCP, or UI.

## Current limitations

- No marketplace / remote catalog
- Signing is optional for local development only when explicitly opted in (`RAYVAN_ALLOW_UNSIGNED_PLUGINS=1` or `allowUnsignedPlugins: true`); mandatory signing UX deferred
- Packages are platform-specific (no universal multi-arch ZIP yet)
- No seccomp / full sandbox
- No plugin React components in the renderer
- Parallel `listPluginActions` catalog is deferred (capabilities remain SoT)

## Planned future support

- Public marketplace and mandatory signing UX
- Universal multi-arch packages
- Stronger process sandboxing
- Richer permission enforcement and secret scoping
