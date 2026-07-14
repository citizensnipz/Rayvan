# Domain Model

Rayvan's core domain is provider-independent. Initial types live in `packages/core`.

## Workspace and project

A **workspace** groups related projects. A **project** represents a software unit Rayvan can inspect, optionally linked to a local root path.

## Environment

An **environment** belongs to a project and describes where software runs: local, development, preview, staging, production, or custom.

## Integration

An **integration** connects a project to a provider plugin such as GitHub or Vercel. Integrations track connection status but do not embed provider-specific resource shapes in the core model.

## Configuration

**Configuration entries** store metadata about keys: whether they are secret or required, optional descriptions, and optional value fingerprints for drift detection. Secret values are not stored as plain `value` fields in ordinary tables.

## Findings

**Findings** represent issues Rayvan detects: missing configuration, drift, deployment failures, integration errors, security concerns, or health problems.

## Actions

**Action plans** describe proposed infrastructure mutations. An approved plan must include an **approval record** before execution can begin. The action engine enforces this lifecycle.

## Resources and deployments

**Resources** and **deployments** provide operational views discovered through plugins. Their metadata stays generic at the core layer; provider details remain in plugins.

See `packages/core/src/domain` for the current type definitions.
