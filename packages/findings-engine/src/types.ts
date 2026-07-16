import type {
  AppliedConfigurationState,
  ConfigurationKey,
  ConfigurationOccurrence,
  DesiredConfigurationValue,
  Environment,
  FindingActor,
  FindingCategory,
  FindingDetection,
  FindingEvaluationError,
  FindingEvaluationRunRecord,
  FindingEvaluationTrigger,
  FindingId,
  FindingLifecycleEventRecord,
  FindingRecord,
  FindingRuleConfiguration,
  FindingSeverity,
  FindingStatus,
  ProjectId,
} from "@rayvan/core";

import type { FindingEvaluator } from "./evaluators/types.js";

/** Connection snapshot — structural mirror of plugin connection persistence. */
export interface FindingsConnectionSnapshot {
  id: string;
  pluginId: string;
  projectId?: string;
  name: string;
  status:
    | "pending"
    | "connected"
    | "expired"
    | "revoked"
    | "error"
    | "disconnected";
  credentialReferenceId?: string;
  lastAuthenticatedAt?: string;
  lastSuccessfulSyncAt?: string;
  lastFailedSyncAt?: string;
  lastErrorCode?: string;
}

export interface FindingsInstalledPluginSnapshot {
  id: string;
  pluginId: string;
  pluginVersion: string;
  status: "installed" | "disabled" | "incompatible" | "missing" | "error";
  enabled: boolean;
}

export interface FindingsDiscoveredResourceSnapshot {
  id: string;
  pluginId: string;
  connectionId: string;
  name: string;
  resourceType: string;
  discoveryStatus: "active" | "missing" | "inaccessible" | "archived";
}

export interface FindingsResourceBindingSnapshot {
  id: string;
  projectId: string;
  environmentId?: string;
  discoveredResourceId: string;
  pluginId: string;
  connectionId: string;
  bindingStatus: "active" | "suggested" | "detached" | "invalid";
  displayName?: string;
}

export interface FindingsMappingSuggestionSnapshot {
  id: string;
  projectId: string;
  connectionId: string;
  discoveredResourceId: string;
  suggestedEnvironmentId?: string;
  suggestedEnvironmentName?: string;
  status: "pending" | "accepted" | "rejected" | "superseded";
}

export interface FindingsChangeApplySnapshot {
  id: string;
  changePlanId: string;
  pluginId: string;
  connectionId: string;
  resourceBindingId: string;
  projectId?: string;
  environmentId?: string;
  status: "succeeded" | "failed" | "cancelled" | "timed_out";
  finishedAt: string;
  /** True when apply completed with some operations succeeding and others failing. */
  partial?: boolean;
  safeErrorMessage?: string;
}

export interface FindingsChangeVerificationSnapshot {
  id: string;
  changeApplyId: string;
  status: "verified" | "not_verified" | "failed";
  verifiedAt: string;
  safeErrorMessage?: string;
}

export interface FindingsChangePlanSnapshot {
  id: string;
  pluginId: string;
  connectionId: string;
  projectId: string;
  environmentId?: string;
  resourceBindingId: string;
  status: string;
  createdAt: string;
  expiresAt?: string;
  observedStateChecksum?: string;
  /** Current observed checksum; when set and differs from plan checksum → plan-stale. */
  currentObservedChecksum?: string;
}

/**
 * Serializable project snapshot for findings evaluation.
 * Avoids importing local-database; hosts map persistence records into this shape.
 */
export interface ProjectFindingsContext {
  projectId: string;
  environments: Environment[];
  keys: ConfigurationKey[];
  occurrences: ConfigurationOccurrence[];
  desired: DesiredConfigurationValue[];
  applied: AppliedConfigurationState[];
  connections: FindingsConnectionSnapshot[];
  installedPlugins: FindingsInstalledPluginSnapshot[];
  discoveredResources: FindingsDiscoveredResourceSnapshot[];
  resourceBindings: FindingsResourceBindingSnapshot[];
  mappingSuggestions: FindingsMappingSuggestionSnapshot[];
  changeApplies?: FindingsChangeApplySnapshot[];
  changeVerifications?: FindingsChangeVerificationSnapshot[];
  changePlans?: FindingsChangePlanSnapshot[];
}

export interface FindingQuery {
  projectId: ProjectId | string;
  statuses?: FindingStatus[];
  severities?: FindingSeverity[];
  categories?: FindingCategory[];
  environmentId?: string;
  connectionId?: string;
  resourceBindingId?: string;
  configurationKeyId?: string;
  ruleId?: string;
  search?: string;
  includeResolved?: boolean;
  limit?: number;
}

export interface FindingRepository {
  getById(id: FindingId | string): Promise<FindingRecord | undefined>;
  getByFingerprint(
    projectId: ProjectId | string,
    fingerprint: string,
  ): Promise<FindingRecord | undefined>;
  list(query: FindingQuery): Promise<FindingRecord[]>;
  save(record: FindingRecord): Promise<void>;
  saveMany?(records: FindingRecord[]): Promise<void>;
}

export interface FindingLifecycleEventRepository {
  append(event: FindingLifecycleEventRecord): Promise<void>;
  listByFindingId(
    findingId: FindingId | string,
  ): Promise<FindingLifecycleEventRecord[]>;
}

export interface FindingEvaluationRunRepository {
  save(run: FindingEvaluationRunRecord): Promise<void>;
  getById(
    id: string,
  ): Promise<FindingEvaluationRunRecord | undefined>;
  listByProject(
    projectId: ProjectId | string,
    limit?: number,
  ): Promise<FindingEvaluationRunRecord[]>;
}

export interface FindingEngineRepositories {
  findings: FindingRepository;
  lifecycleEvents: FindingLifecycleEventRepository;
  evaluationRuns: FindingEvaluationRunRepository;
}

export interface FindingEngineOptions {
  trigger: FindingEvaluationTrigger;
  actor?: FindingActor;
  abortSignal?: AbortSignal;
  /** ISO timestamp; defaults to Date.now(). */
  now?: string;
  /** Preloaded context; when omitted, `loadContext` is required. */
  context?: ProjectFindingsContext;
  loadContext?: (projectId: string) => Promise<ProjectFindingsContext>;
  pluginEvaluators?: FindingEvaluator[];
  /** In-memory rule enablement / severity overrides for now. */
  ruleConfigurations?: FindingRuleConfiguration[];
}

export interface FindingEvaluationResult {
  run: FindingEvaluationRunRecord;
  detections: FindingDetection[];
  created: FindingRecord[];
  updated: FindingRecord[];
  reopened: FindingRecord[];
  resolved: FindingRecord[];
  errors: FindingEvaluationError[];
}
