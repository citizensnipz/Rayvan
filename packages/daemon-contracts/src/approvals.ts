import type { RayvanActor } from "./actor.js";

export type ApprovalRequestType =
  | "local_mutation"
  | "remote_apply"
  | "destructive_operation"
  | "sensitive_value_access";

export type ApprovalRequestStatus =
  "pending" | "approved" | "denied" | "expired" | "cancelled";

export interface ApprovalRequestRecord {
  id: string;
  projectId: string;
  operationId?: string;
  changePlanId?: string;
  requestedBy: RayvanActor;
  type: ApprovalRequestType;
  summary: string;
  safeDetails: Record<string, unknown>;
  status: ApprovalRequestStatus;
  createdAt: string;
  expiresAt?: string;
  decidedAt?: string;
  decidedBy?: RayvanActor;
}
