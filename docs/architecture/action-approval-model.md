# Action Approval Model

Infrastructure mutations in Rayvan follow a strict lifecycle:

```text
requested → planned → awaiting approval → approved → executing → completed or failed
```

## Why the phases are separate

| Phase | Purpose |
| --- | --- |
| Requested | A human or agent asks for a change |
| Planned | Rayvan and the plugin produce a deterministic, reviewable plan |
| Awaiting approval | The plan is visible but cannot execute |
| Approved | A human explicitly approves the plan |
| Executing | The plugin performs only the approved operations |
| Completed / failed | Results and audit events are recorded |

Planning, approval, and execution are separate because infrastructure mistakes are expensive. Rayvan optimizes for reviewability and accountability rather than speed of mutation.

## Enforcement points

- `packages/action-engine` owns approval and execution guards.
- `packages/core` models approved plans with required approval records.
- MCP tools must route mutations through the action engine.
- AI harness code may prepare approval requests but must not approve its own plans.
- Desktop UI must not execute infrastructure mutations directly.

## Auditability

Every mutation should produce audit events capturing who approved the action, when execution started, and whether it succeeded or failed.

## Plugin responsibilities

Plugins may plan and execute only after receiving an approved plan object that includes an approval identifier. Inspect operations must remain read-only.
