import type {
  FindingActor,
  FindingDetection,
  FindingLifecycleEventRecord,
  FindingRecord,
  FindingResolution,
  FindingStatus,
} from "@rayvan/core";
import {
  findingLifecycleEventId,
} from "@rayvan/core";

function newEventId(): string {
  return crypto.randomUUID();
}

export interface LifecycleMutationResult {
  record: FindingRecord;
  event: FindingLifecycleEventRecord;
}

function baseEvent(
  record: FindingRecord,
  type: FindingLifecycleEventRecord["type"],
  actor: FindingActor,
  now: string,
  previousStatus?: FindingStatus,
  nextStatus?: FindingStatus,
  reason?: string,
): FindingLifecycleEventRecord {
  return {
    id: findingLifecycleEventId(newEventId()),
    findingId: record.id,
    projectId: record.projectId,
    type,
    actor,
    createdAt: now,
    previousStatus,
    nextStatus,
    reason,
    metadata: {},
  };
}

/** Acknowledge an open finding — does not resolve. */
export function acknowledgeFinding(
  record: FindingRecord,
  actor: FindingActor,
  now: string,
): LifecycleMutationResult {
  if (record.status !== "open" && record.status !== "acknowledged") {
    throw new Error(
      `Cannot acknowledge finding in status ${record.status}`,
    );
  }
  const previousStatus = record.status;
  const next: FindingRecord = {
    ...record,
    status: "acknowledged",
    acknowledgedAt: now,
    acknowledgedBy: actor,
  };
  return {
    record: next,
    event: baseEvent(
      next,
      "acknowledged",
      actor,
      now,
      previousStatus,
      "acknowledged",
    ),
  };
}

export function dismissFinding(
  record: FindingRecord,
  actor: FindingActor,
  now: string,
  reason?: string,
): LifecycleMutationResult {
  if (record.status === "resolved") {
    throw new Error(
      `Cannot dismiss finding in status ${record.status}`,
    );
  }
  const previousStatus = record.status;
  const next: FindingRecord = {
    ...record,
    status: "dismissed",
    dismissedAt: now,
    dismissedBy: actor,
    dismissalReason: reason,
  };
  return {
    record: next,
    event: baseEvent(
      next,
      "dismissed",
      actor,
      now,
      previousStatus,
      "dismissed",
      reason,
    ),
  };
}

export function suppressFinding(
  record: FindingRecord,
  actor: FindingActor,
  now: string,
  suppressedUntil: string,
  reason?: string,
): LifecycleMutationResult {
  if (record.status === "resolved") {
    throw new Error(
      `Cannot suppress finding in status ${record.status}`,
    );
  }
  const previousStatus = record.status;
  const next: FindingRecord = {
    ...record,
    status: "suppressed",
    suppressedUntil,
  };
  return {
    record: next,
    event: baseEvent(
      next,
      "suppressed",
      actor,
      now,
      previousStatus,
      "suppressed",
      reason,
    ),
  };
}

export function resolveFinding(
  record: FindingRecord,
  actor: FindingActor,
  now: string,
  source: FindingResolution["source"] = "automatic",
  reason?: string,
): LifecycleMutationResult {
  const previousStatus = record.status;
  const next: FindingRecord = {
    ...record,
    status: "resolved",
    resolvedAt: now,
    resolution: {
      source,
      resolvedBy: actor,
      reason,
    },
  };
  return {
    record: next,
    event: baseEvent(
      next,
      "resolved",
      actor,
      now,
      previousStatus,
      "resolved",
      reason,
    ),
  };
}

/**
 * Reopen a resolved/dismissed finding when its fingerprint is detected again.
 * Acknowledged stays acknowledged (caller should use applyDetectionUpdate instead).
 */
export function reopenFinding(
  record: FindingRecord,
  detection: FindingDetection,
  actor: FindingActor,
  now: string,
  evaluationRunId?: string,
): LifecycleMutationResult {
  const previousStatus = record.status;
  const next: FindingRecord = {
    ...record,
    status: "open",
    title: detection.title,
    summary: detection.summary,
    description: detection.description,
    severity: detection.severity ?? record.severity,
    evidence: detection.evidence,
    remediation: detection.remediation ?? record.remediation,
    lastDetectedAt: now,
    occurrenceCount: record.occurrenceCount + 1,
    lastEvaluationRunId: evaluationRunId
      ? (evaluationRunId as FindingRecord["lastEvaluationRunId"])
      : record.lastEvaluationRunId,
    resolvedAt: undefined,
    resolution: undefined,
    dismissedAt: undefined,
    dismissedBy: undefined,
    dismissalReason: undefined,
    suppressedUntil: undefined,
    acknowledgedAt: undefined,
    acknowledgedBy: undefined,
    environmentId: detection.scope.environmentId
      ? (detection.scope.environmentId as FindingRecord["environmentId"])
      : record.environmentId,
    connectionId: detection.scope.connectionId ?? record.connectionId,
    discoveredResourceId:
      detection.scope.discoveredResourceId ?? record.discoveredResourceId,
    resourceBindingId:
      detection.scope.resourceBindingId ?? record.resourceBindingId,
    configurationKeyId: detection.scope.configurationKeyId
      ? (detection.scope
          .configurationKeyId as FindingRecord["configurationKeyId"])
      : record.configurationKeyId,
    changePlanId: detection.scope.changePlanId ?? record.changePlanId,
    deploymentId: detection.scope.deploymentId ?? record.deploymentId,
    metadata: { ...record.metadata, ...(detection.metadata ?? {}) },
  };
  return {
    record: next,
    event: baseEvent(next, "reopened", actor, now, previousStatus, "open"),
  };
}

/** Whether a suppressed finding's suppression window has elapsed. */
export function isSuppressionExpired(
  record: FindingRecord,
  now: string,
): boolean {
  if (record.status !== "suppressed" || !record.suppressedUntil) {
    return false;
  }
  const until = Date.parse(record.suppressedUntil);
  const nowMs = Date.parse(now);
  if (Number.isNaN(until) || Number.isNaN(nowMs)) {
    return false;
  }
  return nowMs >= until;
}
