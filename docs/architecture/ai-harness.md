# AI Harness

The AI harness in `packages/ai-harness` coordinates agent-facing workflows without giving agents unrestricted infrastructure access.

## Role

AI is an operator assistant, not an unrestricted infrastructure administrator. It may:

- Explain infrastructure state
- Summarize findings
- Compare environments
- Diagnose deployment failures
- Suggest action plans
- Prepare approval requests

## Constraints

The harness must not:

- Access raw credential values
- Call provider APIs directly
- Mutate infrastructure directly
- Execute arbitrary shell commands
- Approve its own actions

## Interfaces

Agents should use the same Rayvan domain interfaces as humans: core types, configuration analysis, action planning, and MCP tools. Provider-specific clients remain inside plugins.

## Current status

The harness includes policy text and placeholder interfaces only. No AI provider is integrated yet.

See `packages/ai-harness/src/policies` for the initial policy document.
