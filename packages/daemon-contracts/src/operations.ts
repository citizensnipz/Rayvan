import type { RayvanActor } from "./actor.js";
import type { DaemonSerializedError } from "./protocol.js";

export type OperationType =
  | "integration_sync"
  | "environment_sync"
  | "resource_inspection"
  | "findings_scan"
  | "change_plan_generation"
  | "change_apply"
  | "change_verification"
  | "plugin_operation";

export type OperationStatus =
  | "queued"
  | "running"
  | "waiting_for_approval"
  | "succeeded"
  | "partially_succeeded"
  | "failed"
  | "cancelled"
  | "timed_out";

export interface OperationProgress {
  percent?: number;
  message?: string;
  currentStep?: string;
  totalSteps?: number;
  completedSteps?: number;
}

export interface OperationRecord {
  id: string;
  projectId?: string;
  type: OperationType;
  status: OperationStatus;
  actor: RayvanActor;
  progress?: OperationProgress;
  startedAt?: string;
  finishedAt?: string;
  correlationId: string;
  idempotencyKey?: string;
  safeError?: DaemonSerializedError;
  resultSummary?: Record<string, unknown>;
  approvalRequestId?: string;
  changePlanId?: string;
}
