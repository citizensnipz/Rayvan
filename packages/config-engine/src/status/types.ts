import type {
  AppliedConfigurationState,
  ConfigurationKey,
  ConfigurationKeyId,
  ConfigurationOccurrence,
  ConfigurationSyncStatus,
  DesiredConfigurationValue,
} from "@rayvan/core";

/** Optional editor draft for a key — never persisted as ConfigurationSyncStatus. */
export interface ConfigurationKeyDraft {
  configurationKeyId: ConfigurationKeyId | string;
  /** Non-sensitive draft plaintext, or undefined for sensitive. */
  draftValue?: string;
  draftSecretValueRef?: string;
  draftFingerprint?: string;
  /** True when draft differs from saved desired. */
  dirty: boolean;
}

export interface ConfigurationTargetRef {
  configurationKeyId: ConfigurationKeyId | string;
  environmentId: string;
  resourceBindingId: string;
  pluginId?: string;
  discoveredResourceId?: string;
}

export interface DeriveKeyStatusInput {
  key: ConfigurationKey;
  environmentId: string;
  desired?: DesiredConfigurationValue | null;
  occurrences: ConfigurationOccurrence[];
  applied: AppliedConfigurationState[];
  /** Targets derived from occurrences with resourceBindingId in this environment. */
  targets: ConfigurationTargetRef[];
  draft?: ConfigurationKeyDraft | null;
  /** ISO timestamp used for stale checks; defaults to now. */
  now?: string;
  /** Age threshold in ms for observedMayBeStale (default 7 days). */
  staleAfterMs?: number;
}

export interface ConfigurationKeyStatusViewModel {
  configurationKeyId: ConfigurationKeyId;
  configurationKeyName: string;
  environmentId: string;
  syncStatus: ConfigurationSyncStatus;
  statusLabel: string;
  /**
   * Editor draft dirty flag — separate from syncStatus.
   * When true, prefer UI label "Unsaved local changes" while syncStatus
   * still reflects persisted desired vs observed/applied.
   */
  editorDirty: boolean;
  hasUnsavedLocalChanges: boolean;
  observedMayBeStale: boolean;
  syncRequired: boolean;
  desiredRevision?: number;
  resourceStatuses: ConfigurationResourceStatusSummary[];
}

export interface ConfigurationResourceStatusSummary {
  resourceBindingId: string;
  pluginId?: string;
  syncStatus: ConfigurationSyncStatus;
  statusLabel: string;
  occurrenceId?: string;
  appliedStatus?: AppliedConfigurationState["status"];
}

export interface DeriveEnvironmentStatusInput {
  environmentId: string;
  keys: ConfigurationKey[];
  desired: DesiredConfigurationValue[];
  occurrences: ConfigurationOccurrence[];
  applied: AppliedConfigurationState[];
  drafts?: ConfigurationKeyDraft[];
  now?: string;
  staleAfterMs?: number;
}

export interface EnvironmentConfigurationStatusViewModel {
  environmentId: string;
  keyStatuses: ConfigurationKeyStatusViewModel[];
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
  /** Aggregate headline for cards — text, not colour alone. */
  headlineLabel: string;
  hasUnsavedLocalChanges: boolean;
  hasChangesNotApplied: boolean;
}

export interface DeriveResourceStatusInput {
  environmentId: string;
  resourceBindingId: string;
  keys: ConfigurationKey[];
  desired: DesiredConfigurationValue[];
  occurrences: ConfigurationOccurrence[];
  applied: AppliedConfigurationState[];
  now?: string;
  staleAfterMs?: number;
}

export interface ResourceConfigurationStatusViewModel {
  environmentId: string;
  resourceBindingId: string;
  keyStatuses: ConfigurationKeyStatusViewModel[];
  summary: EnvironmentConfigurationStatusViewModel["summary"];
  headlineLabel: string;
}

export interface EnvironmentConfigurationStatusService {
  getEnvironmentStatus(
    input: DeriveEnvironmentStatusInput,
  ): EnvironmentConfigurationStatusViewModel;
  getKeyStatus(input: DeriveKeyStatusInput): ConfigurationKeyStatusViewModel;
  getResourceStatus(
    input: DeriveResourceStatusInput,
  ): ResourceConfigurationStatusViewModel;
}
