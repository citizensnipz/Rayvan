# Rayvan AI Bootstrap

Canonical router for agents working in this repository. Read this file before meaningful work, then load at most two additional topical documents unless clearly necessary.

## Context-loading budget

1. `AGENTS.md`
2. `ai/BOOTSTRAP.md`
3. At most **two** topical documents from the routing table below
4. The nearest nested `AGENTS.md` when working in a subtree that defines one
5. `.agents/roles/<role>.md` when delegating to or embodying a role

Do **not** preload:

- Full `docs/` trees
- Generated build output, logs, or lockfile diffs
- Historical notes, retrospectives, or agent memories
- Package source unless required for the task

Prefer `rg`, targeted file reads, and subagent summaries over broad repository scans.

## Task-to-document routing

| Task area | Read first | Optional second doc |
| --- | --- | --- |
| Product intent, principles, terminology | `docs/product/vision.md` | `docs/product/principles.md` |
| Monorepo layout and runtime areas | `docs/architecture/overview.md` | `docs/architecture/process-model.md` |
| Domain types and core concepts | `docs/architecture/domain-model.md` | `docs/product/terminology.md` |
| Security, credentials, secrets handling | `docs/architecture/security-model.md` | `docs/architecture/action-approval-model.md` |
| Action plans, approvals, mutations | `docs/architecture/action-approval-model.md` | `docs/architecture/security-model.md` |
| Plugin system and provider boundaries | `docs/architecture/plugin-system.md` | `docs/plugins/authoring.md` |
| Plugin authoring | `docs/plugins/authoring.md` | `docs/plugins/permissions.md` |
| Plugin lifecycle and process hosting | `docs/plugins/lifecycle.md` | `docs/architecture/process-model.md` |
| AI harness product design | `docs/architecture/ai-harness.md` | `docs/architecture/overview.md` |
| Local-first architecture decision | `docs/decisions/0001-local-first-architecture.md` | `docs/architecture/overview.md` |
| Desktop app or Tauri shell | `docs/architecture/overview.md` | `docs/architecture/process-model.md` |
| MCP server behavior | `docs/architecture/ai-harness.md` | `docs/architecture/security-model.md` |
| TypeScript packages or plugins code | `docs/architecture/domain-model.md` | `docs/architecture/plugin-system.md` |
| Rust crates or native services | `docs/architecture/process-model.md` | `docs/architecture/security-model.md` |
| Harness, agents, or instruction files | `ai/harness-manifest.json` | `.agents/roles/` contract for the active role |

Every topical workflow document under `docs/` is reachable through this table. Do not maintain a second routing table elsewhere.

## Subagent routing

| Role | Use when |
| --- | --- |
| `code-explorer` | Relevant code path, behavior, ownership, or scope is unclear |
| `quick-implementer` | Change is small, well-defined, and limited to one or two files |
| `implementer` | Feature, bug fix, multi-file change, or proportionate validation is needed |
| `architecture-reviewer` | High-risk or cross-cutting design needs read-only architectural review |
| `code-reviewer` | Completed diff needs read-only review before merge or commit |
| `commit-pusher` | User explicitly requests staging, committing, and/or pushing |

Skip `architecture-reviewer` for text, styling, formatting, routine dependency maintenance, isolated low-risk fixes, and contained work with no architectural impact.

## Delegation rules

Every delegated prompt must include:

- Objective and acceptance criteria
- Relevant paths
- Supplied findings from prior steps
- Allowed reads and writes
- Topical harness documents to read (max two beyond bootstrap)
- Required validation tier
- Explicit exclusions
- Expected concise output format

Subagents must return summaries, decisions, file references, risks, and verification outcomes. Do not return raw search dumps or full logs.

## Preferred orchestration

### Simple work

Keep the task in the main agent when subagent overhead is not worthwhile.

### Unclear scope

1. `code-explorer` investigates and returns distilled findings.
2. Choose exactly one implementation role based on discovered scope:
   - one or two files → `quick-implementer`
   - broader change → `implementer`

### Substantial or high-risk work

1. `code-explorer` identifies affected areas.
2. `architecture-reviewer` evaluates the proposed design when architectural impact exists.
3. The implementation agent performs only work allowed by the architecture verdict.
4. `code-reviewer` reviews the completed diff.
5. For high-risk or cross-cutting work, `architecture-reviewer` compares implementation with the approved plan.
6. `commit-pusher` acts only after explicit user authorization.

Do not make architecture review mandatory for every task.

## Verification tiers

| Tier | Scope | Commands |
| --- | --- | --- |
| H0 Harness | Instruction or harness file changes | `pnpm check:harness` |
| H1 Focused | Single package or one-two file change | Targeted `pnpm --filter <pkg> lint typecheck test` |
| H2 Standard | Multi-file TS/Rust change | Affected package build + tests |
| H3 Full | Cross-cutting or release confidence | `pnpm check` |

Rust verification requires `cargo` on `PATH`. If unavailable, report the limitation explicitly.

## Role contracts

Native agent definitions in `.codex/agents/` and `.cursor/agents/` implement the editor-neutral contracts in `.agents/roles/`. The manifest at `ai/harness-manifest.json` is the source of expected permissions and model intent.
