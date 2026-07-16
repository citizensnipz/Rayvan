import {
  configurationKeyId,
  environmentId,
  findingEvaluationRunId,
  findingId,
  findingLifecycleEventId,
  integrationId,
  projectId,
  type FindingActor,
  type FindingCategory,
  type FindingEvaluationError,
  type FindingEvaluationRunRecord,
  type FindingEvaluationRunStatus,
  type FindingEvaluationScope,
  type FindingEvaluationTrigger,
  type FindingEvidence,
  type FindingLifecycleEventRecord,
  type FindingLifecycleEventType,
  type FindingRecord,
  type FindingRemediation,
  type FindingResolution,
  type FindingSeverity,
  type FindingSource,
  type FindingStatus,
} from "@rayvan/core";

export interface FindingRow {
  id: string;
  project_id: string;
  rule_id: string;
  source_json: string;
  category: FindingCategory;
  severity: FindingSeverity;
  title: string;
  summary: string;
  description: string | null;
  status: FindingStatus;
  fingerprint: string;
  fingerprint_version: string;
  environment_id: string | null;
  integration_id: string | null;
  connection_id: string | null;
  discovered_resource_id: string | null;
  resource_binding_id: string | null;
  configuration_key_id: string | null;
  change_plan_id: string | null;
  deployment_id: string | null;
  evidence_json: string;
  remediation_json: string | null;
  first_detected_at: string;
  last_detected_at: string;
  occurrence_count: number;
  acknowledged_at: string | null;
  acknowledged_by_json: string | null;
  dismissed_at: string | null;
  dismissed_by_json: string | null;
  dismissal_reason: string | null;
  resolved_at: string | null;
  resolution_json: string | null;
  suppressed_until: string | null;
  last_evaluation_run_id: string | null;
  metadata_json: string;
  schema_version: string;
}

export interface FindingLifecycleEventRow {
  id: string;
  finding_id: string;
  project_id: string;
  type: FindingLifecycleEventType;
  actor_json: string;
  created_at: string;
  previous_status: FindingStatus | null;
  next_status: FindingStatus | null;
  reason: string | null;
  metadata_json: string;
}

export interface FindingEvaluationRunRow {
  id: string;
  project_id: string;
  scope_json: string;
  trigger: FindingEvaluationTrigger;
  status: FindingEvaluationRunStatus;
  started_at: string;
  finished_at: string;
  evaluators_run: number;
  evaluators_failed: number;
  findings_created: number;
  findings_updated: number;
  findings_reopened: number;
  findings_resolved: number;
  safe_errors_json: string;
}

function parseJson<T>(value: string, label: string): T {
  try {
    return JSON.parse(value) as T;
  } catch (error) {
    throw new Error(`Failed to parse ${label}: ${String(error)}`);
  }
}

export function mapFindingRow(row: FindingRow): FindingRecord {
  return {
    id: findingId(row.id),
    projectId: projectId(row.project_id),
    ruleId: row.rule_id,
    source: parseJson<FindingSource>(row.source_json, "source_json"),
    category: row.category,
    severity: row.severity,
    title: row.title,
    summary: row.summary,
    description: row.description ?? undefined,
    status: row.status,
    fingerprint: row.fingerprint,
    fingerprintVersion: row.fingerprint_version,
    environmentId: row.environment_id
      ? environmentId(row.environment_id)
      : undefined,
    integrationId: row.integration_id
      ? integrationId(row.integration_id)
      : undefined,
    connectionId: row.connection_id ?? undefined,
    discoveredResourceId: row.discovered_resource_id ?? undefined,
    resourceBindingId: row.resource_binding_id ?? undefined,
    configurationKeyId: row.configuration_key_id
      ? configurationKeyId(row.configuration_key_id)
      : undefined,
    changePlanId: row.change_plan_id ?? undefined,
    deploymentId: row.deployment_id ?? undefined,
    evidence: parseJson<FindingEvidence[]>(row.evidence_json, "evidence_json"),
    remediation: row.remediation_json
      ? parseJson<FindingRemediation>(row.remediation_json, "remediation_json")
      : undefined,
    firstDetectedAt: row.first_detected_at,
    lastDetectedAt: row.last_detected_at,
    occurrenceCount: row.occurrence_count,
    acknowledgedAt: row.acknowledged_at ?? undefined,
    acknowledgedBy: row.acknowledged_by_json
      ? parseJson<FindingActor>(row.acknowledged_by_json, "acknowledged_by_json")
      : undefined,
    dismissedAt: row.dismissed_at ?? undefined,
    dismissedBy: row.dismissed_by_json
      ? parseJson<FindingActor>(row.dismissed_by_json, "dismissed_by_json")
      : undefined,
    dismissalReason: row.dismissal_reason ?? undefined,
    resolvedAt: row.resolved_at ?? undefined,
    resolution: row.resolution_json
      ? parseJson<FindingResolution>(row.resolution_json, "resolution_json")
      : undefined,
    suppressedUntil: row.suppressed_until ?? undefined,
    lastEvaluationRunId: row.last_evaluation_run_id
      ? findingEvaluationRunId(row.last_evaluation_run_id)
      : undefined,
    metadata: parseJson<Record<string, unknown>>(
      row.metadata_json,
      "metadata_json",
    ),
    schemaVersion: row.schema_version,
  };
}

export function findingRecordToBindParams(record: FindingRecord): unknown[] {
  return [
    record.id,
    record.projectId,
    record.ruleId,
    JSON.stringify(record.source),
    record.category,
    record.severity,
    record.title,
    record.summary,
    record.description ?? null,
    record.status,
    record.fingerprint,
    record.fingerprintVersion,
    record.environmentId ?? null,
    record.integrationId ?? null,
    record.connectionId ?? null,
    record.discoveredResourceId ?? null,
    record.resourceBindingId ?? null,
    record.configurationKeyId ?? null,
    record.changePlanId ?? null,
    record.deploymentId ?? null,
    JSON.stringify(record.evidence),
    record.remediation ? JSON.stringify(record.remediation) : null,
    record.firstDetectedAt,
    record.lastDetectedAt,
    record.occurrenceCount,
    record.acknowledgedAt ?? null,
    record.acknowledgedBy ? JSON.stringify(record.acknowledgedBy) : null,
    record.dismissedAt ?? null,
    record.dismissedBy ? JSON.stringify(record.dismissedBy) : null,
    record.dismissalReason ?? null,
    record.resolvedAt ?? null,
    record.resolution ? JSON.stringify(record.resolution) : null,
    record.suppressedUntil ?? null,
    record.lastEvaluationRunId ?? null,
    JSON.stringify(record.metadata ?? {}),
    record.schemaVersion,
  ];
}

export function mapLifecycleEventRow(
  row: FindingLifecycleEventRow,
): FindingLifecycleEventRecord {
  return {
    id: findingLifecycleEventId(row.id),
    findingId: findingId(row.finding_id),
    projectId: projectId(row.project_id),
    type: row.type,
    actor: parseJson<FindingActor>(row.actor_json, "actor_json"),
    createdAt: row.created_at,
    previousStatus: row.previous_status ?? undefined,
    nextStatus: row.next_status ?? undefined,
    reason: row.reason ?? undefined,
    metadata: parseJson<Record<string, unknown>>(
      row.metadata_json,
      "metadata_json",
    ),
  };
}

export function lifecycleEventToBindParams(
  event: FindingLifecycleEventRecord,
): unknown[] {
  return [
    event.id,
    event.findingId,
    event.projectId,
    event.type,
    JSON.stringify(event.actor),
    event.createdAt,
    event.previousStatus ?? null,
    event.nextStatus ?? null,
    event.reason ?? null,
    JSON.stringify(event.metadata ?? {}),
  ];
}

export function mapEvaluationRunRow(
  row: FindingEvaluationRunRow,
): FindingEvaluationRunRecord {
  return {
    id: findingEvaluationRunId(row.id),
    projectId: projectId(row.project_id),
    scope: parseJson<FindingEvaluationScope>(row.scope_json, "scope_json"),
    trigger: row.trigger,
    status: row.status,
    startedAt: row.started_at,
    finishedAt: row.finished_at,
    evaluatorsRun: row.evaluators_run,
    evaluatorsFailed: row.evaluators_failed,
    findingsCreated: row.findings_created,
    findingsUpdated: row.findings_updated,
    findingsReopened: row.findings_reopened,
    findingsResolved: row.findings_resolved,
    safeErrors: parseJson<FindingEvaluationError[]>(
      row.safe_errors_json,
      "safe_errors_json",
    ),
  };
}

export function evaluationRunToBindParams(
  run: FindingEvaluationRunRecord,
): unknown[] {
  return [
    run.id,
    run.projectId,
    JSON.stringify(run.scope),
    run.trigger,
    run.status,
    run.startedAt,
    run.finishedAt,
    run.evaluatorsRun,
    run.evaluatorsFailed,
    run.findingsCreated,
    run.findingsUpdated,
    run.findingsReopened,
    run.findingsResolved,
    JSON.stringify(run.safeErrors ?? []),
  ];
}
