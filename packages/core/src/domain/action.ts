import type { ActionPlanId, ProjectId } from "../ids/index.js";

export type ActionPlanStatus =
  | "draft"
  | "awaiting_approval"
  | "approved"
  | "executing"
  | "completed"
  | "failed"
  | "cancelled";

export interface PlannedOperation {
  id: string;
  kind: string;
  description: string;
  targetResourceId?: string;
}

export interface ActionPlan {
  id: ActionPlanId;
  projectId: ProjectId;
  pluginId: string;
  status: ActionPlanStatus;
  summary: string;
  operations: PlannedOperation[];
}

export interface ApprovalRecord {
  id: string;
  actionPlanId: ActionPlanId;
  approvedBy: string;
  approvedAt: string;
  notes?: string;
}

/**
 * An approved action plan must include an approval record.
 * Execution cannot proceed without explicit human approval.
 */
export interface ApprovedActionPlan extends ActionPlan {
  status: "approved" | "executing" | "completed" | "failed";
  approval: ApprovalRecord;
}

export function isApprovedActionPlan(
  plan: ActionPlan,
): plan is ApprovedActionPlan {
  return (
    (plan.status === "approved" ||
      plan.status === "executing" ||
      plan.status === "completed" ||
      plan.status === "failed") &&
    "approval" in plan &&
    (plan as ApprovedActionPlan).approval !== undefined
  );
}
