import type {
  ConfigurationKeyId,
  EnvironmentId,
  FindingEvaluationRunId,
  FindingId,
  FindingLifecycleEventId,
  IntegrationId,
  ProjectId,
} from "../ids/index.js";

/**
 * Product taxonomy for Findings.
 * `configuration` covers missing/invalid/required keys.
 * `drift` covers desired≠applied / disagreeing observed values.
 */
export type FindingCategory =
  | "configuration"
  | "environment"
  | "integration"
  | "resource"
  | "deployment"
  | "security"
  | "availability"
  | "drift"
  | "permission"
  | "mapping"
  | "other";

export type FindingSeverity = "info" | "warning" | "error" | "critical";

export type FindingStatus =
  | "open"
  | "acknowledged"
  | "resolved"
  | "dismissed"
  | "suppressed";

/**
 * Who acted on a Finding. Uses `kind` for persistence-friendly JSON,
 * matching ConfigurationActorRef. Adapt PluginExecutionActor at the host boundary.
 */
export type FindingActor =
  | { kind: "user"; id: string; displayName?: string }
  | { kind: "system"; id: string }
  | { kind: "mcp_agent"; id: string; displayName?: string }
  | { kind: "rayvan"; id: string }
  | { kind: "plugin"; pluginId: string };

export type FindingObjectType =
  | "project"
  | "environment"
  | "integration"
  | "connection"
  | "resource"
  | "configuration_key"
  | "change_plan"
  | "deployment";

export type FindingSource =
  | { type: "rayvan" }
  | {
      type: "plugin";
      pluginId: string;
      pluginVersion?: string;
      connectionId?: string;
    };

/** Safe value representation — never plaintext secrets. */
export type SafeFindingValue =
  | {
      access: "readable";
      value: string;
      sensitive: false;
    }
  | {
      access: "fingerprint";
      fingerprint: string;
      sensitive: true;
    }
  | {
      access: "masked";
      maskedValue?: string;
      sensitive: true;
    }
  | {
      access: "locked" | "name_only" | "unknown";
      sensitive: boolean;
    };

export interface FindingObservedState {
  pluginId?: string;
  connectionId?: string;
  discoveredResourceId?: string;
  resourceBindingId?: string;
  label?: string;
  value: SafeFindingValue;
  inSync?: boolean;
  observedAt?: string;
}

export type FindingEvidence =
  | {
      type: "configuration_comparison";
      configurationKeyId: string;
      environmentId: string;
      expectedState?: SafeFindingValue;
      observedStates: FindingObservedState[];
    }
  | {
      type: "connection_error";
      connectionId: string;
      errorCode?: string;
      safeMessage: string;
    }
  | {
      type: "resource_state";
      resourceBindingId: string;
      state: string;
      observedAt?: string;
    }
  | {
      type: "deployment_state";
      deploymentId?: string;
      status: string;
      observedAt?: string;
    }
  | {
      type: "message";
      message: string;
    };

/** Serializable remediation descriptors — never executable callbacks. */
export type FindingRemediation =
  | {
      type: "generate_change_plan";
      pluginId?: string;
      connectionId?: string;
      resourceBindingId?: string;
      configurationKeyIds?: string[];
      label: string;
    }
  | {
      type: "open_environment";
      environmentId: string;
      label: string;
    }
  | {
      type: "open_integration";
      integrationId?: string;
      connectionId?: string;
      label: string;
    }
  | {
      type: "open_resource";
      resourceBindingId: string;
      label: string;
    }
  | {
      type: "reauthenticate";
      connectionId: string;
      label: string;
    }
  | {
      type: "resync";
      pluginId?: string;
      connectionId?: string;
      label: string;
    }
  | {
      type: "manual";
      label: string;
      instructions: string;
    };

export interface FindingResolution {
  source: "automatic" | "manual";
  resolvedBy: FindingActor;
  reason?: string;
}

export interface FindingRuleDefinition {
  id: string;
  name: string;
  description: string;
  source: FindingSource;
  category: FindingCategory;
  defaultSeverity: FindingSeverity;
  enabledByDefault: boolean;
  supportedObjectTypes: FindingObjectType[];
  documentationUrl?: string;
}

export interface FindingScope {
  environmentId?: EnvironmentId | string;
  integrationId?: IntegrationId | string;
  connectionId?: string;
  discoveredResourceId?: string;
  resourceBindingId?: string;
  configurationKeyId?: ConfigurationKeyId | string;
  changePlanId?: string;
  deploymentId?: string;
}

/**
 * Normalized detection returned by evaluators.
 * Evaluators must not assign record IDs or mutate Finding history.
 */
export interface FindingDetection {
  ruleId: string;
  projectId: ProjectId | string;
  severity?: FindingSeverity;
  title: string;
  summary: string;
  description?: string;
  scope: FindingScope;
  evidence: FindingEvidence[];
  remediation?: FindingRemediation;
  /** Parts used to build a stable fingerprint (not titles/timestamps). */
  fingerprintParts: string[];
  metadata?: Record<string, unknown>;
}

/**
 * Persistent Finding instance.
 * Fingerprint identity is stable across wording changes.
 */
export interface FindingRecord {
  id: FindingId;
  projectId: ProjectId;
  ruleId: string;
  source: FindingSource;
  category: FindingCategory;
  severity: FindingSeverity;
  title: string;
  summary: string;
  description?: string;
  status: FindingStatus;
  fingerprint: string;
  fingerprintVersion: string;

  environmentId?: EnvironmentId;
  integrationId?: IntegrationId;
  connectionId?: string;
  discoveredResourceId?: string;
  resourceBindingId?: string;
  configurationKeyId?: ConfigurationKeyId;
  changePlanId?: string;
  deploymentId?: string;

  evidence: FindingEvidence[];
  remediation?: FindingRemediation;

  firstDetectedAt: string;
  lastDetectedAt: string;
  occurrenceCount: number;

  acknowledgedAt?: string;
  acknowledgedBy?: FindingActor;

  dismissedAt?: string;
  dismissedBy?: FindingActor;
  dismissalReason?: string;

  resolvedAt?: string;
  resolution?: FindingResolution;

  suppressedUntil?: string;

  lastEvaluationRunId?: FindingEvaluationRunId;
  metadata: Record<string, unknown>;
  schemaVersion: string;
}

/**
 * @deprecated Prefer FindingRecord. Kept as a narrow view for AI/MCP summaries.
 */
export type Finding = Pick<
  FindingRecord,
  | "id"
  | "projectId"
  | "environmentId"
  | "integrationId"
  | "severity"
  | "category"
  | "title"
> & {
  description: string;
};

export type FindingLifecycleEventType =
  | "created"
  | "updated"
  | "reopened"
  | "acknowledged"
  | "dismissed"
  | "suppressed"
  | "resolved"
  | "severity_changed";

export interface FindingLifecycleEventRecord {
  id: FindingLifecycleEventId;
  findingId: FindingId;
  projectId: ProjectId;
  type: FindingLifecycleEventType;
  actor: FindingActor;
  createdAt: string;
  previousStatus?: FindingStatus;
  nextStatus?: FindingStatus;
  reason?: string;
  metadata: Record<string, unknown>;
}

export interface FindingRuleConfiguration {
  id: string;
  projectId: ProjectId | string;
  ruleId: string;
  enabled: boolean;
  severityOverride?: FindingSeverity;
  environmentSeverityOverrides?: Record<string, FindingSeverity>;
  updatedAt: string;
  updatedBy: FindingActor;
}

export type FindingEvaluationTrigger =
  | "manual"
  | "startup"
  | "environment_sync"
  | "integration_sync"
  | "configuration_inspection"
  | "apply_completed"
  | "verification_completed"
  | "connection_state_changed"
  | "desired_configuration_saved";

export type FindingEvaluationScope =
  | { type: "project"; projectId: string }
  | { type: "environment"; projectId: string; environmentId: string }
  | { type: "connection"; projectId: string; connectionId: string };

export type FindingEvaluationRunStatus =
  | "succeeded"
  | "partially_succeeded"
  | "failed"
  | "cancelled";

export interface FindingEvaluationError {
  evaluatorId: string;
  pluginId?: string;
  connectionId?: string;
  code?: string;
  safeMessage: string;
}

export interface FindingEvaluationRunRecord {
  id: FindingEvaluationRunId;
  projectId: ProjectId;
  scope: FindingEvaluationScope;
  trigger: FindingEvaluationTrigger;
  status: FindingEvaluationRunStatus;
  startedAt: string;
  finishedAt: string;
  evaluatorsRun: number;
  evaluatorsFailed: number;
  findingsCreated: number;
  findingsUpdated: number;
  findingsReopened: number;
  findingsResolved: number;
  safeErrors: FindingEvaluationError[];
}

export interface FindingSummary {
  openCount: number;
  acknowledgedCount: number;
  bySeverity: Record<FindingSeverity, number>;
  byCategory: Partial<Record<FindingCategory, number>>;
  highestSeverity?: FindingSeverity;
  hasRemediableFindings: boolean;
}

export const FINDING_SCHEMA_VERSION = "1";
export const FINDING_FINGERPRINT_VERSION = "1";

export const FINDING_CATEGORIES: readonly FindingCategory[] = [
  "configuration",
  "environment",
  "integration",
  "resource",
  "deployment",
  "security",
  "availability",
  "drift",
  "permission",
  "mapping",
  "other",
] as const;

export const FINDING_SEVERITIES: readonly FindingSeverity[] = [
  "info",
  "warning",
  "error",
  "critical",
] as const;

export const FINDING_STATUSES: readonly FindingStatus[] = [
  "open",
  "acknowledged",
  "resolved",
  "dismissed",
  "suppressed",
] as const;

/** Active statuses that should appear in default “open findings” views. */
export const ACTIVE_FINDING_STATUSES: readonly FindingStatus[] = [
  "open",
  "acknowledged",
  "suppressed",
] as const;

export function isFindingCategory(value: string): value is FindingCategory {
  return (FINDING_CATEGORIES as readonly string[]).includes(value);
}

export function isFindingSeverity(value: string): value is FindingSeverity {
  return (FINDING_SEVERITIES as readonly string[]).includes(value);
}

export function isFindingStatus(value: string): value is FindingStatus {
  return (FINDING_STATUSES as readonly string[]).includes(value);
}

/** Map a FindingRecord to the narrow Finding view used by AI summaries. */
export function toFindingView(record: FindingRecord): Finding {
  return {
    id: record.id,
    projectId: record.projectId,
    environmentId: record.environmentId,
    integrationId: record.integrationId,
    severity: record.severity,
    category: record.category,
    title: record.title,
    description: record.description ?? record.summary,
  };
}
