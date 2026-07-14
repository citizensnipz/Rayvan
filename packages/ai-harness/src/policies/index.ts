export const AI_HARNESS_POLICY = `
Rayvan AI Harness Policy

1. AI agents are operator assistants, not unrestricted infrastructure administrators.
2. AI must use Rayvan domain interfaces and MCP tools only.
3. AI must not access raw credential values.
4. AI must not call provider APIs directly.
5. AI must not mutate infrastructure directly.
6. AI must not execute arbitrary shell commands.
7. AI must not approve its own action plans.
8. All proposed mutations flow through planning, approval, and audit.
`.trim();

export interface AiHarnessPolicy {
  id: string;
  version: string;
  content: string;
}

export const defaultAiHarnessPolicy: AiHarnessPolicy = {
  id: "rayvan-ai-harness",
  version: "0.0.1",
  content: AI_HARNESS_POLICY,
};
