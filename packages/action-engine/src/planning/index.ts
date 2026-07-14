import type { ActionPlan, ProjectId } from "@rayvan/core";

export interface ActionRequest {
  projectId: ProjectId;
  pluginId: string;
  kind: string;
  parameters: Record<string, unknown>;
  requestedBy: string;
  requestedAt: string;
}

export function createActionRequest(
  input: Omit<ActionRequest, "requestedAt"> & { requestedAt?: string },
): ActionRequest {
  return {
    ...input,
    requestedAt: input.requestedAt ?? new Date().toISOString(),
  };
}

export interface DeterministicActionPlan extends ActionPlan {
  deterministicHash: string;
}

export function createDeterministicPlanHash(plan: ActionPlan): string {
  return JSON.stringify({
    pluginId: plan.pluginId,
    summary: plan.summary,
    operations: plan.operations,
  });
}
