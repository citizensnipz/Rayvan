import type {
  FindingActor,
  FindingDetection,
  FindingRecord,
  FindingRuleDefinition,
  FindingSeverity,
} from "@rayvan/core";
import {
  FINDING_SCHEMA_VERSION,
  findingId,
  findingLifecycleEventId,
  type FindingLifecycleEventRecord,
  type FindingEvaluationRunId,
  type ProjectId,
} from "@rayvan/core";

import { buildFindingFingerprint, fingerprintVersion } from "../fingerprint.js";
import { resolveDetectionSeverity } from "../validation.js";
import {
  isSuppressionExpired,
  reopenFinding,
  resolveFinding,
} from "./lifecycle.js";

export interface DedupeMatchResult {
  created: FindingRecord[];
  updated: FindingRecord[];
  reopened: FindingRecord[];
  resolved: FindingRecord[];
  events: FindingLifecycleEventRecord[];
}

function applyScopeFields(
  record: FindingRecord,
  detection: FindingDetection,
): FindingRecord {
  return {
    ...record,
    environmentId: detection.scope.environmentId
      ? (detection.scope.environmentId as FindingRecord["environmentId"])
      : record.environmentId,
    integrationId: detection.scope.integrationId
      ? (detection.scope.integrationId as FindingRecord["integrationId"])
      : record.integrationId,
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
  };
}

function createFromDetection(input: {
  detection: FindingDetection;
  rule: FindingRuleDefinition;
  severity: FindingSeverity;
  fingerprint: string;
  now: string;
  evaluationRunId: FindingEvaluationRunId;
}): FindingRecord {
  const { detection, rule, severity, fingerprint, now, evaluationRunId } =
    input;
  return {
    id: findingId(crypto.randomUUID()),
    projectId: detection.projectId as ProjectId,
    ruleId: detection.ruleId,
    source: rule.source,
    category: rule.category,
    severity,
    title: detection.title,
    summary: detection.summary,
    description: detection.description,
    status: "open",
    fingerprint,
    fingerprintVersion: fingerprintVersion(),
    environmentId: detection.scope.environmentId as FindingRecord["environmentId"],
    integrationId: detection.scope.integrationId as FindingRecord["integrationId"],
    connectionId: detection.scope.connectionId,
    discoveredResourceId: detection.scope.discoveredResourceId,
    resourceBindingId: detection.scope.resourceBindingId,
    configurationKeyId: detection.scope
      .configurationKeyId as FindingRecord["configurationKeyId"],
    changePlanId: detection.scope.changePlanId,
    deploymentId: detection.scope.deploymentId,
    evidence: detection.evidence,
    remediation: detection.remediation,
    firstDetectedAt: now,
    lastDetectedAt: now,
    occurrenceCount: 1,
    lastEvaluationRunId: evaluationRunId,
    metadata: detection.metadata ?? {},
    schemaVersion: FINDING_SCHEMA_VERSION,
  };
}

/**
 * Match detections to existing findings by fingerprint.
 * occurrenceCount increments once per evaluation that still sees the finding.
 */
export function matchAndApplyDetections(input: {
  detections: FindingDetection[];
  existing: FindingRecord[];
  rulesById: Map<string, FindingRuleDefinition>;
  severityOverrides: Map<string, FindingSeverity>;
  actor: FindingActor;
  now: string;
  evaluationRunId: FindingEvaluationRunId;
  /** Rule IDs whose evaluators succeeded — only these may auto-resolve. */
  succeededRuleIds: ReadonlySet<string>;
  /** When resolving stale findings, only consider findings in this project/scope set. */
  existingInScope: FindingRecord[];
}): DedupeMatchResult {
  const {
    detections,
    existing,
    rulesById,
    severityOverrides,
    actor,
    now,
    evaluationRunId,
    succeededRuleIds,
    existingInScope,
  } = input;

  const byFingerprint = new Map(
    existing.map((record) => [record.fingerprint, record]),
  );
  const seenFingerprints = new Set<string>();
  const created: FindingRecord[] = [];
  const updated: FindingRecord[] = [];
  const reopened: FindingRecord[] = [];
  const events: FindingLifecycleEventRecord[] = [];

  for (const detection of detections) {
    const rule = rulesById.get(detection.ruleId);
    if (!rule) {
      continue;
    }
    const fingerprint = buildFindingFingerprint({
      ruleId: detection.ruleId,
      projectId: String(detection.projectId),
      fingerprintParts: detection.fingerprintParts,
    });
    seenFingerprints.add(fingerprint);
    const severity = resolveDetectionSeverity(
      detection,
      rule.defaultSeverity,
      severityOverrides.get(detection.ruleId),
    );
    const current = byFingerprint.get(fingerprint);

    if (!current) {
      const record = createFromDetection({
        detection,
        rule,
        severity,
        fingerprint,
        now,
        evaluationRunId,
      });
      created.push(record);
      byFingerprint.set(fingerprint, record);
      events.push({
        id: findingLifecycleEventId(crypto.randomUUID()),
        findingId: record.id,
        projectId: record.projectId,
        type: "created",
        actor,
        createdAt: now,
        nextStatus: "open",
        metadata: {},
      });
      continue;
    }

    const stillActive =
      current.status === "open" ||
      current.status === "acknowledged" ||
      (current.status === "suppressed" && !isSuppressionExpired(current, now));

    if (
      current.status === "resolved" ||
      current.status === "dismissed" ||
      (current.status === "suppressed" && isSuppressionExpired(current, now))
    ) {
      const result = reopenFinding(
        current,
        detection,
        actor,
        now,
        evaluationRunId,
      );
      const withSeverity: FindingRecord = {
        ...applyScopeFields(result.record, detection),
        severity,
      };
      reopened.push(withSeverity);
      byFingerprint.set(fingerprint, withSeverity);
      events.push(result.event);
      continue;
    }

    if (stillActive) {
      const previousSeverity = current.severity;
      const next: FindingRecord = applyScopeFields(
        {
          ...current,
          title: detection.title,
          summary: detection.summary,
          description: detection.description ?? current.description,
          severity,
          evidence: detection.evidence,
          remediation: detection.remediation ?? current.remediation,
          lastDetectedAt: now,
          occurrenceCount: current.occurrenceCount + 1,
          lastEvaluationRunId: evaluationRunId,
          metadata: { ...current.metadata, ...(detection.metadata ?? {}) },
        },
        detection,
      );
      updated.push(next);
      byFingerprint.set(fingerprint, next);
      events.push({
        id: findingLifecycleEventId(crypto.randomUUID()),
        findingId: next.id,
        projectId: next.projectId,
        type: previousSeverity !== severity ? "severity_changed" : "updated",
        actor,
        createdAt: now,
        previousStatus: next.status,
        nextStatus: next.status,
        metadata:
          previousSeverity !== severity
            ? { previousSeverity, nextSeverity: severity }
            : {},
      });
    }
  }

  const resolved: FindingRecord[] = [];
  for (const record of existingInScope) {
    if (
      record.status !== "open" &&
      record.status !== "acknowledged" &&
      record.status !== "suppressed"
    ) {
      continue;
    }
    if (!succeededRuleIds.has(record.ruleId)) {
      continue;
    }
    if (seenFingerprints.has(record.fingerprint)) {
      continue;
    }
    const result = resolveFinding(
      record,
      actor,
      now,
      "automatic",
      "No longer detected in successful evaluation scope",
    );
    const withRun: FindingRecord = {
      ...result.record,
      lastEvaluationRunId: evaluationRunId,
    };
    resolved.push(withRun);
    events.push(result.event);
  }

  return { created, updated, reopened, resolved, events };
}
