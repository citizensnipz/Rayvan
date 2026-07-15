import type {
  AppliedConfigurationState,
  ConfigurationKeyId,
  ConfigurationSyncStatus,
  DesiredConfigurationValue,
  EnvironmentId,
  ProjectId,
} from "@rayvan/core";

/**
 * Reviewable apply plan for configuration (gateway stub — no provider calls).
 * Lighter than ActionPlan; approval-shaped for the Environments apply flow.
 */
export interface ConfigurationApplyPlanOperation {
  id: string;
  configurationKeyId: string;
  configurationKeyName: string;
  environmentId: string;
  resourceBindingId: string;
  pluginId: string;
  resourceName?: string;
  /** Redacted display value — never plaintext secrets. */
  displayValue: string;
  sensitive: boolean;
  desiredRevision: number;
  warnings: string[];
}

export interface ConfigurationApplyPlan {
  id: string;
  projectId: ProjectId;
  environmentId: EnvironmentId;
  status: "draft" | "awaiting_approval" | "approved" | "completed" | "partial" | "failed";
  summary: string;
  operations: ConfigurationApplyPlanOperation[];
  createdAt: string;
}

export interface ConfigurationApplyResultItem {
  operationId: string;
  configurationKeyId: string;
  resourceBindingId: string;
  status: AppliedConfigurationState["status"];
  applied?: AppliedConfigurationState;
  errorMessage?: string;
}

export interface ConfigurationApplyResult {
  planId: string;
  environmentId: EnvironmentId;
  status: "completed" | "partial" | "failed";
  items: ConfigurationApplyResultItem[];
  finishedAt: string;
}

export interface SaveDesiredValueInput {
  configurationKeyId: string;
  environmentId: string;
  projectId: string;
  desiredValue?: string;
  secretValueRef?: string;
  valueFingerprint?: string;
  expectedRevision?: number;
}

export interface AdoptDiscoveredKeyInput {
  configurationKeyId: string;
  environmentId: string;
  projectId: string;
  /** When true and occurrence is readable non-sensitive, copy observed value. */
  copyReadableValue?: boolean;
}

export interface EnvironmentConfigurationStatusSnapshot {
  environmentId: string;
  headlineLabel: string;
  hasUnsavedLocalChanges: boolean;
  hasChangesNotApplied: boolean;
  summary: {
    inSyncCount: number;
    localChangesCount: number;
    remoteChangedCount: number;
    mismatchedCount: number;
    missingRemoteCount: number;
    missingLocalCount: number;
    notManagedCount: number;
    partiallyAppliedCount: number;
    lockedCount: number;
    unknownCount: number;
    unsavedDraftCount: number;
    staleObservedCount: number;
  };
  keys: Array<{
    configurationKeyId: ConfigurationKeyId;
    configurationKeyName: string;
    syncStatus: ConfigurationSyncStatus;
    statusLabel: string;
    editorDirty: boolean;
    desired?: DesiredConfigurationValue;
  }>;
}

/** Static provider-effect warnings shown in apply review (no real provider calls). */
export const APPLY_EFFECT_WARNINGS: Record<string, string[]> = {
  vercel: [
    "Vercel may create a new deployment after environment variable changes.",
  ],
  supabase: [
    "Supabase project settings updates may require a brief reconnect for clients.",
  ],
  github: [
    "GitHub Actions secrets updates do not retrigger past workflow runs.",
  ],
  sentry: [
    "Sentry DSN or environment tag changes affect new events only.",
  ],
};
