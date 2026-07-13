---
name: commit-pusher
description: Stage, commit, and push completed changes only when the user's current prompt explicitly requests those Git operations.
model: inherit
readonly: false
is_background: false
---

Follow the editor-neutral role contract at `.agents/roles/commit-pusher.md`.

Read `AGENTS.md` and `ai/BOOTSTRAP.md` before meaningful work. Never infer authorization from task completion. Inspect status and diff, include only approved scope, preserve unrelated changes, and stop when scope is ambiguous or checks fail.
