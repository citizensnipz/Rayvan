import type { PluginPermission } from "@rayvan/plugin-sdk";

import type {
  ChangeApplyRecord,
  ChangePlanApprovalRecord,
  ChangePlanRecord,
  ChangePlanRejectionRecord,
  ChangePlanStatus,
  ChangeVerificationRecord,
  CredentialReferenceRecord,
  DesiredResourceStateRecord,
  DiscoveredResourceRecord,
  EnvironmentMappingSuggestionRecord,
  InstalledPluginRecord,
  ObservedResourceStateHistoryRecord,
  ObservedResourceStateRecord,
  PluginConnectionRecord,
  PluginExecutionHistoryRecord,
  PluginPermissionGrantRecord,
  ResourceBindingRecord,
  ResourceDiscoveryStatus,
} from "../models.js";

export interface InstalledPluginRepository {
  save(record: InstalledPluginRecord): Promise<void>;
  getById(id: string): Promise<InstalledPluginRecord | undefined>;
  getByPluginId(pluginId: string): Promise<InstalledPluginRecord | undefined>;
  list(): Promise<InstalledPluginRecord[]>;
  setEnabled(id: string, enabled: boolean): Promise<void>;
  updateStatus(
    id: string,
    status: InstalledPluginRecord["status"],
  ): Promise<void>;
}

export interface PluginConnectionRepository {
  save(record: PluginConnectionRecord): Promise<void>;
  getById(id: string): Promise<PluginConnectionRecord | undefined>;
  listByPluginId(pluginId: string): Promise<PluginConnectionRecord[]>;
  listByProjectId(projectId: string): Promise<PluginConnectionRecord[]>;
  listByInstalledPluginId(
    installedPluginId: string,
  ): Promise<PluginConnectionRecord[]>;
}

export interface CredentialReferenceRepository {
  save(record: CredentialReferenceRecord): Promise<void>;
  getById(id: string): Promise<CredentialReferenceRecord | undefined>;
  listByConnectionId(
    connectionId: string,
  ): Promise<CredentialReferenceRecord[]>;
}

export interface PluginPermissionGrantRepository {
  save(record: PluginPermissionGrantRecord): Promise<void>;
  getById(id: string): Promise<PluginPermissionGrantRecord | undefined>;
  listByConnectionId(
    connectionId: string,
  ): Promise<PluginPermissionGrantRecord[]>;
  listActiveByConnectionId(
    connectionId: string,
  ): Promise<PluginPermissionGrantRecord[]>;
  /**
   * Revokes active grants matching the same connection + scope
   * (`projectId` / `environmentId`, including undefined = connection-wide),
   * then inserts the provided grants. Other scopes are left untouched.
   */
  replaceActiveGrants(input: {
    connectionId: string;
    projectId?: string;
    environmentId?: string;
    grants: PluginPermissionGrantRecord[];
  }): Promise<void>;
}

export interface DiscoverySyncItem {
  providerResourceId: string;
  resourceType: string;
  name: string;
  parentProviderResourceId?: string;
  metadata: Record<string, unknown>;
  pluginVersion: string;
  schemaVersion: string;
}

export interface DiscoveredResourceRepository {
  save(record: DiscoveredResourceRecord): Promise<void>;
  getById(id: string): Promise<DiscoveredResourceRecord | undefined>;
  getByProviderKey(input: {
    connectionId: string;
    providerResourceId: string;
    resourceType: string;
  }): Promise<DiscoveredResourceRecord | undefined>;
  listByConnectionId(connectionId: string): Promise<DiscoveredResourceRecord[]>;
  syncDiscovery(input: {
    pluginId: string;
    installedPluginId: string;
    connectionId: string;
    discoveredAt: string;
    items: DiscoverySyncItem[];
  }): Promise<DiscoveredResourceRecord[]>;
  markStatus(
    id: string,
    status: ResourceDiscoveryStatus,
    missingSince?: string,
  ): Promise<void>;
}

export interface ResourceBindingRepository {
  save(record: ResourceBindingRecord): Promise<void>;
  getById(id: string): Promise<ResourceBindingRecord | undefined>;
  listByProjectId(projectId: string): Promise<ResourceBindingRecord[]>;
  listByConnectionId(connectionId: string): Promise<ResourceBindingRecord[]>;
  listByDiscoveredResourceId(
    discoveredResourceId: string,
  ): Promise<ResourceBindingRecord[]>;
  invalidateByConnectionId(connectionId: string): Promise<void>;
}

export interface EnvironmentMappingSuggestionRepository {
  save(record: EnvironmentMappingSuggestionRecord): Promise<void>;
  getById(
    id: string,
  ): Promise<EnvironmentMappingSuggestionRecord | undefined>;
  listByProjectId(
    projectId: string,
  ): Promise<EnvironmentMappingSuggestionRecord[]>;
  listPendingByProjectId(
    projectId: string,
  ): Promise<EnvironmentMappingSuggestionRecord[]>;
}

export interface ObservedResourceStateRepository {
  upsertLatest(
    record: ObservedResourceStateRecord,
  ): Promise<ObservedResourceStateRecord>;
  getLatestByDiscoveredResourceId(
    discoveredResourceId: string,
  ): Promise<ObservedResourceStateRecord | undefined>;
  listHistory(
    discoveredResourceId: string,
  ): Promise<ObservedResourceStateHistoryRecord[]>;
}

export interface DesiredResourceStateRepository {
  getByBindingId(
    resourceBindingId: string,
  ): Promise<DesiredResourceStateRecord | undefined>;
  saveNew(record: DesiredResourceStateRecord): Promise<DesiredResourceStateRecord>;
  updateWithExpectedRevision(input: {
    resourceBindingId: string;
    expectedRevision: number;
    next: DesiredResourceStateRecord;
  }): Promise<DesiredResourceStateRecord>;
}

export interface ChangePlanRepository {
  save(record: ChangePlanRecord): Promise<void>;
  getById(id: string): Promise<ChangePlanRecord | undefined>;
  listByBindingId(resourceBindingId: string): Promise<ChangePlanRecord[]>;
  setStatus(id: string, status: ChangePlanStatus): Promise<void>;
  supersede(input: {
    planId: string;
    supersededByPlanId: string;
  }): Promise<void>;
}

export interface ChangePlanApprovalRepository {
  appendApproval(record: ChangePlanApprovalRecord): Promise<void>;
  appendRejection(record: ChangePlanRejectionRecord): Promise<void>;
  listApprovalsByPlanId(
    changePlanId: string,
  ): Promise<ChangePlanApprovalRecord[]>;
  listRejectionsByPlanId(
    changePlanId: string,
  ): Promise<ChangePlanRejectionRecord[]>;
  getLatestApproval(
    changePlanId: string,
  ): Promise<ChangePlanApprovalRecord | undefined>;
  approveAndTransitionPlan(input: {
    approval: ChangePlanApprovalRecord;
    planId: string;
  }): Promise<void>;
  rejectAndTransitionPlan(input: {
    rejection: ChangePlanRejectionRecord;
    planId: string;
  }): Promise<void>;
}

export interface ChangeApplyRepository {
  save(record: ChangeApplyRecord): Promise<void>;
  getById(id: string): Promise<ChangeApplyRecord | undefined>;
  listByPlanId(changePlanId: string): Promise<ChangeApplyRecord[]>;
  /** Transitions plan to applying before the host executes. */
  beginApply(planId: string): Promise<void>;
  completeApply(input: {
    planId: string;
    apply: ChangeApplyRecord;
    planStatus: Extract<ChangePlanStatus, "applied" | "failed">;
  }): Promise<void>;
}

export interface ChangeVerificationRepository {
  save(record: ChangeVerificationRecord): Promise<void>;
  getById(id: string): Promise<ChangeVerificationRecord | undefined>;
  listByApplyId(changeApplyId: string): Promise<ChangeVerificationRecord[]>;
}

export interface PluginExecutionHistoryRepository {
  append(record: PluginExecutionHistoryRecord): Promise<void>;
  getByExecutionId(
    executionId: string,
  ): Promise<PluginExecutionHistoryRecord | undefined>;
  listByPluginId(pluginId: string): Promise<PluginExecutionHistoryRecord[]>;
  listByConnectionId(
    connectionId: string,
  ): Promise<PluginExecutionHistoryRecord[]>;
}

export interface PluginPermissionResolveQuery {
  pluginId: string;
  connectionId?: string;
  projectId?: string;
  environmentId?: string;
  permission?: PluginPermission;
}
