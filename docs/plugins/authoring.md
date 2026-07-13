# Plugin Authoring

Plugins live in `plugins/*` and depend on `@rayvan/plugin-sdk`, `@rayvan/core`, and `@rayvan/shared`.

## Package layout

Each plugin should include:

- `manifest.ts` — stable plugin identity and capabilities
- `client/` — provider API client (placeholder during scaffolding)
- `discovery/` — resource discovery
- `configuration/` — configuration inspection
- `health/` — health inspection
- `actions/` — plan and execute approved actions
- `index.ts` — `RayvanPlugin` implementation

## Rules

- Do not depend on desktop, MCP server, AI harness, or other provider plugins.
- Keep provider-specific types inside the plugin package.
- Throw or return explicit not-implemented results until real integrations are added.
- Never mutate infrastructure during inspect or plan phases.

## Manifest example

```ts
export const manifest = {
  id: "vercel",
  name: "Vercel",
  version: "0.0.1",
  protocolVersion: "1",
  capabilities: [
    "resource-discovery",
    "configuration-read",
    "health-read",
    "action-plan",
    "action-execute",
  ],
} as const;
```
