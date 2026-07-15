import type {
  ApplyResult,
  ChangePlan,
  PluginCapability,
  PluginExecutionActor,
  PluginExecutionErrorCode,
  PluginExecutionStatus,
  PluginManifest,
  PluginPermission,
  SerializedPluginExecutionError,
  VerificationResult,
} from "@rayvan/plugin-sdk";

export type PluginInstallSource =
  | { type: "built_in" }
  | { type: "local"; path: string }
  | { type: "package"; packageId: string; registry?: string };

export type InstalledPluginStatus =
  | "installed"
  | "disabled"
  | "incompatible"
  | "missing"
  | "error";

export interface InstalledPluginRecord {
  id: string;
  pluginId: string;
  pluginVersion: string;
  manifestVersion: string;
  rayvanApiVersion: string;
  source: PluginInstallSource;
  status: InstalledPluginStatus;
  enabled: boolean;
  installedAt: string;
  updatedAt: string;
  lastLoadedAt?: string;
  manifestSnapshot: PluginManifest;
}

export type PluginConnectionStatus =
  | "pending"
  | "connected"
  | "expired"
  | "revoked"
  | "error"
  | "disconnected";

export interface PluginConnectionRecord {
  id: string;
  installedPluginId: string;
  pluginId: string;
  projectId?: string;
  name: string;
  status: PluginConnectionStatus;
  externalAccountId?: string;
  externalAccountName?: string;
  providerBaseUrl?: string;
  credentialReferenceId?: string;
  metadata: Record<string, unknown>;
  schemaVersion: string;
  createdAt: string;
  updatedAt: string;
  lastAuthenticatedAt?: string;
  lastSuccessfulSyncAt?: string;
  lastFailedSyncAt?: string;
  lastErrorCode?: string;
}

export type CredentialStorageProvider =
  | "os_keychain"
  | "encrypted_local_store"
  | "development_memory";

export interface CredentialReferenceRecord {
  id: string;
  pluginId: string;
  connectionId: string;
  provider: CredentialStorageProvider;
  storageKey: string;
  credentialType: string;
  displayName?: string;
  createdAt: string;
  updatedAt: string;
  lastUsedAt?: string;
}

export interface PluginPermissionGrantRecord {
  id: string;
  pluginId: string;
  connectionId: string;
  permission: PluginPermission;
  projectId?: string;
  environmentId?: string;
  granted: boolean;
  grantedBy: PluginExecutionActor;
  grantedAt: string;
  revokedBy?: PluginExecutionActor;
  revokedAt?: string;
  reason?: string;
}

export type ResourceDiscoveryStatus =
  | "active"
  | "missing"
  | "inaccessible"
  | "archived";

export interface DiscoveredResourceRecord {
  id: string;
  pluginId: string;
  installedPluginId: string;
  connectionId: string;
  providerResourceId: string;
  resourceType: string;
  name: string;
  parentProviderResourceId?: string;
  metadata: Record<string, unknown>;
  pluginVersion: string;
  schemaVersion: string;
  discoveryStatus: ResourceDiscoveryStatus;
  firstDiscoveredAt: string;
  lastDiscoveredAt: string;
  lastInspectedAt?: string;
  missingSince?: string;
}

export type ResourceBindingStatus =
  | "active"
  | "suggested"
  | "detached"
  | "invalid";

export interface ResourceBindingRecord {
  id: string;
  projectId: string;
  environmentId?: string;
  discoveredResourceId: string;
  pluginId: string;
  connectionId: string;
  role?: string;
  displayName?: string;
  bindingStatus: ResourceBindingStatus;
  createdBy: PluginExecutionActor;
  createdAt: string;
  updatedAt: string;
}

export type MappingSuggestionStatus =
  | "pending"
  | "accepted"
  | "rejected"
  | "superseded";

export interface EnvironmentMappingSuggestionRecord {
  id: string;
  projectId: string;
  connectionId: string;
  discoveredResourceId: string;
  suggestedEnvironmentId?: string;
  suggestedEnvironmentName?: string;
  confidence?: number;
  reasons: string[];
  status: MappingSuggestionStatus;
  createdAt: string;
  resolvedAt?: string;
  resolvedBy?: PluginExecutionActor;
}

export interface ObservedResourceStateRecord {
  id: string;
  discoveredResourceId: string;
  pluginId: string;
  connectionId: string;
  state: Record<string, unknown>;
  pluginVersion: string;
  schemaVersion: string;
  observedAt: string;
  sourceExecutionId?: string;
  checksum?: string;
}

export type ObservedResourceStateHistoryRecord = ObservedResourceStateRecord;

export interface DesiredResourceStateRecord {
  id: string;
  projectId: string;
  environmentId?: string;
  resourceBindingId: string;
  pluginId: string;
  connectionId: string;
  state: Record<string, unknown>;
  schemaVersion: string;
  revision: number;
  createdBy: PluginExecutionActor;
  createdAt: string;
  updatedAt: string;
}

export type ChangePlanStatus =
  | "pending"
  | "approved"
  | "rejected"
  | "applying"
  | "applied"
  | "failed"
  | "expired"
  | "superseded";

/** Current envelope version for persisted ChangePlan JSON. */
export const CHANGE_PLAN_SCHEMA_VERSION = "1";

export interface ChangePlanRecord {
  id: string;
  pluginId: string;
  connectionId: string;
  projectId: string;
  environmentId?: string;
  resourceBindingId: string;
  desiredStateRevision?: number;
  observedStateChecksum?: string;
  /** Versioned JSON envelope for the serializable plan. */
  planSchemaVersion: string;
  plan: ChangePlan;
  status: ChangePlanStatus;
  createdBy: PluginExecutionActor;
  createdAt: string;
  expiresAt?: string;
  supersededByPlanId?: string;
}

export interface ChangePlanApprovalRecord {
  id: string;
  changePlanId: string;
  approvedOperationIds: string[];
  destructiveApproval: boolean;
  approvedBy: PluginExecutionActor;
  approvedAt: string;
  comment?: string;
}

export interface ChangePlanRejectionRecord {
  id: string;
  changePlanId: string;
  rejectedBy: PluginExecutionActor;
  rejectedAt: string;
  reason?: string;
}

export type ChangeApplyStatus =
  | "succeeded"
  | "failed"
  | "cancelled"
  | "timed_out";

export interface ChangeApplyRecord {
  id: string;
  changePlanId: string;
  executionId: string;
  pluginId: string;
  connectionId: string;
  resourceBindingId: string;
  status: ChangeApplyStatus;
  result?: ApplyResult;
  error?: SerializedPluginExecutionError;
  startedAt: string;
  finishedAt: string;
}

export type ChangeVerificationStatus = "verified" | "not_verified" | "failed";

export interface ChangeVerificationRecord {
  id: string;
  changeApplyId: string;
  executionId: string;
  status: ChangeVerificationStatus;
  result?: VerificationResult;
  error?: SerializedPluginExecutionError;
  verifiedAt: string;
}

export interface PluginExecutionHistoryRecord {
  id: string;
  executionId: string;
  pluginId: string;
  pluginVersion: string;
  capability: PluginCapability;
  status: PluginExecutionStatus;
  startedAt: string;
  finishedAt: string;
  durationMs: number;
  actor: PluginExecutionActor;
  projectId?: string;
  environmentId?: string;
  resourceId?: string;
  connectionId?: string;
  reason?: string;
  errorCode?: PluginExecutionErrorCode;
  errorMessage?: string;
  warningCount: number;
  recordedAt: string;
}

/**
 * Core Integration / ActionPlan remain separate product concepts.
 * Plugin persistence uses dedicated `plugin_*` tables and these records.
 */
export const PLUGIN_PERSISTENCE_NOTE =
  "Plugin lifecycle data is stored in plugin_* tables, not Core Integration or ActionPlan.";
