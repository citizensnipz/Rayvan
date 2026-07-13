import type { ActionPlanId, ApprovalRecord } from "@rayvan/core";

export interface ApprovalRequest {
  actionPlanId: ActionPlanId;
  requestedBy: string;
  requestedAt: string;
  rationale?: string;
}

export function createApprovalRecord(
  input: Omit<ApprovalRecord, "approvedAt"> & { approvedAt?: string },
): ApprovalRecord {
  return {
    ...input,
    approvedAt: input.approvedAt ?? new Date().toISOString(),
  };
}

export function canExecute(planStatus: string, approval?: ApprovalRecord): boolean {
  return planStatus === "approved" && approval !== undefined;
}
