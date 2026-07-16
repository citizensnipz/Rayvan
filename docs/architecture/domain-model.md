# Domain Model

Rayvan's core domain is provider-independent. Initial types live in `packages/core`.

## Workspace and project

A **workspace** groups related projects. A **project** represents a software unit Rayvan can inspect, optionally linked to a local root path.

## Environment

An **environment** belongs to a project and describes where software runs: local, development, preview, staging, production, or custom.

## Integration

An **integration** connects a project to a provider plugin such as GitHub or Vercel. Integrations track connection status but do not embed provider-specific resource shapes in the core model.

## Configuration

Logical **configuration keys** are project-scoped identities. **Occurrences** record discovered provider values (with value-access controls). **Desired configuration values** are per (key × environment); **applied configuration states** are per (key × environment × resource binding). Targets are derived from occurrences that have a `resourceBindingId` — there is no separate targets table.

Secret / sensitive desired values store only `secretValueRef` and fingerprints in ordinary tables — never plaintext. Editor draft dirty state is separate from persisted `ConfigurationSyncStatus`.

## Findings

**Findings** are durable issue records Rayvan detects and tracks over time: configuration gaps, drift, environment/resource mapping problems, integration health, and change apply failures.

- Identity is a stable **fingerprint** (`ruleId` + project + structural parts) — not titles or timestamps — so wording changes do not fork the same issue.
- Evaluators emit `{ detections, evaluatedRuleIds }`; the findings-engine matches, creates, updates, reopens, and resolves only for rule IDs that were fully evaluated (rules skipped for missing input data are not eligible for auto-resolve). Cancelled evaluation runs persist the run record only — they do not create, update, or resolve findings.
- Product categories include `configuration`, `drift`, `environment`, `resource`, `mapping`, `integration`, `permission`, `deployment`, and related taxonomy in `@rayvan/core`.
- Pure evaluation lives in `@rayvan/findings-engine`; persistence adapters belong in local-database (schema v6: `findings`, `finding_lifecycle_events`, `finding_evaluation_runs` — not dual product paths from config-engine derived findings).
- Desktop Findings and Environments workspaces read `FindingRecord` / `FindingSummary` via gateways — they must not compute or persist findings in React.

## Actions

**Action plans** describe proposed infrastructure mutations. An approved plan must include an **approval record** before execution can begin. The action engine enforces this lifecycle.

## Resources and deployments

**Resources** and **deployments** provide operational views discovered through plugins. Their metadata stays generic at the core layer; provider details remain in plugins.

See `packages/core/src/domain` for the current type definitions.
