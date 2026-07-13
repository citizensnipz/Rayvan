# Rayvan Agent Operating Rules

Read `ai/BOOTSTRAP.md` before meaningful work. When working below the repository root, also read the nearest nested `AGENTS.md` if one exists.

## Core rules

- Preserve unrelated worktree changes. Inspect `git status --short` before editing.
- Keep diffs focused. Do not expand scope with opportunistic refactors, dependency upgrades, or formatting sweeps.
- Do not stage, commit, or push unless the user's current prompt explicitly requests it.
- Do not expose secrets, credentials, tokens, or local environment files.
- Do not mutate infrastructure, provider APIs, or deployment state directly from the UI harness path.
- Load no more than two extra topical documents beyond `AGENTS.md` and `ai/BOOTSTRAP.md` unless clearly necessary.
- Do not preload indexes, logs, memories, retrospectives, or historical notes.
- Pass distilled findings between subagents. Do not repeat full discovery output.
- Use subagents only when context isolation or independent review justifies their startup cost.
- Parallelize only independent tasks. Never assign overlapping file writes.

## Repository verification

Proportionate validation for code changes:

| Tier | When | Commands |
| --- | --- | --- |
| Focused | One or two files, low risk | Targeted package `lint`, `typecheck`, or `test` |
| Standard | Multi-file TS/Rust changes | `pnpm build` in affected workspace and relevant tests |
| Full | Cross-cutting or pre-merge confidence | `pnpm check` |
| Harness | Harness or instruction changes | `pnpm check:harness` |

Full gate: `pnpm check` (lint, typecheck, test, integration, cargo check, cargo test).

Harness gate: `pnpm check:harness` (harness validator + `git diff --check`).

## Delegation

When delegating to a subagent, include objective, acceptance criteria, relevant paths, supplied findings, allowed reads/writes, topical documents to read, required validation, explicit exclusions, and expected concise output format.

Prefer the main agent for simple, well-scoped work. See `ai/BOOTSTRAP.md` for routing and orchestration.

## Canonical references

- Routing and orchestration: `ai/BOOTSTRAP.md`
- Role contracts: `.agents/roles/`
- Product and architecture docs: `docs/` (load selectively via bootstrap routing)
