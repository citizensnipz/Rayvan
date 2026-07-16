import type {
  AppliedConfigurationState,
  ConfigurationKey,
  ConfigurationOccurrence,
  ConfigurationValueType,
  DesiredConfigurationValue,
  EnvironmentColorToken,
  EnvironmentIconToken,
  EnvironmentKind,
  EnvironmentStatus,
  FindingSeverity,
} from "@rayvan/core";
import type {
  ConfigurationApplyPlan,
  ConfigurationApplyResult,
  ConfigurationKeyStatusViewModel,
  ConfigurationMatrixCellStatus,
  EnvironmentConfigurationStatusViewModel,
} from "@rayvan/config-engine";

export type EnvironmentWorkspaceTabKind =
  | "overview"
  | "matrix"
  | "resources"
  | "environment"
  | "configurationKey";

export type EnvironmentTab =
  | { kind: "overview" }
  | { kind: "matrix" }
  | { kind: "resources" }
  | { kind: "environment"; environmentId: string; label: string }
  | { kind: "configurationKey"; configurationKeyId: string; label: string };

export type EnvironmentCardActionId =
  | "open"
  | "sync"
  | "edit"
  | "archive"
  | "review_changes";

export interface EnvironmentCardAction {
  id: EnvironmentCardActionId;
  label: string;
  kind: "primary" | "secondary" | "overflow";
}

export interface EnvironmentHealthSummary {
  healthy: number;
  missing: number;
  mismatched: number;
  locked: number;
}

export interface EnvironmentConfigAggregateSummary {
  headlineLabel: string;
  inSyncCount: number;
  changesNotAppliedCount: number;
  missingRemoteCount: number;
  missingLocalCount: number;
  remoteChangedCount: number;
  lockedCount: number;
  hasUnsavedLocalChanges: boolean;
  hasChangesNotApplied: boolean;
}

export interface EnvironmentCardViewModel {
  environmentId: string;
  name: string;
  kind: EnvironmentKind;
  kindLabel: string;
  status: EnvironmentStatus;
  statusLabel: string;
  description?: string;
  color?: EnvironmentColorToken;
  icon?: EnvironmentIconToken;
  resourceCount: number;
  integrationCount: number;
  configurationKeyCount: number;
  findingsCount: number;
  /** Accessible text label — never colour-only severity. */
  findingsLabel: string;
  highestFindingSeverity?: FindingSeverity;
  lastSyncLabel: string;
  health: EnvironmentHealthSummary;
  configAggregate?: EnvironmentConfigAggregateSummary;
  actions: EnvironmentCardAction[];
}

export interface EnvironmentConfigurationDraftValue {
  configurationKeyId: string;
  draftValue: string;
  draftSecretValueRef?: string;
  draftFingerprint?: string;
  dirty: boolean;
}

export interface EnvironmentConfigurationEditorRowViewModel {
  key: ConfigurationKey;
  desired?: DesiredConfigurationValue;
  status: ConfigurationKeyStatusViewModel;
  draft: EnvironmentConfigurationDraftValue;
  occurrences: ConfigurationOccurrence[];
  applied: AppliedConfigurationState[];
  notManaged: boolean;
}

export interface EnvironmentConfigurationEditorViewModel {
  environmentId: string;
  environmentName: string;
  status: EnvironmentConfigurationStatusViewModel;
  rows: EnvironmentConfigurationEditorRowViewModel[];
  hasUnsavedLocalChanges: boolean;
  hasChangesNotApplied: boolean;
  headerStatusLabel: string;
}

export type { ConfigurationApplyPlan, ConfigurationApplyResult };

export interface ApplyReviewViewModel {
  plan: ConfigurationApplyPlan;
  groups: Array<{
    pluginId: string;
    resourceBindingId: string;
    resourceName?: string;
    operations: ConfigurationApplyPlan["operations"];
  }>;
}

export function inputTypeForValueType(
  valueType: ConfigurationValueType,
  revealed: boolean,
  options?: { sensitive?: boolean },
): "text" | "password" | "number" | "url" | "checkbox" {
  if ((valueType === "secret" || options?.sensitive) && !revealed) {
    return "password";
  }
  if (valueType === "number") {
    return "number";
  }
  if (valueType === "url") {
    return "url";
  }
  if (valueType === "boolean") {
    return "checkbox";
  }
  return "text";
}

export function fingerprintForDraft(value: string): string {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) | 0;
  }
  return `fp:draft-${Math.abs(hash).toString(16)}`;
}

export interface EnvironmentComparisonSummary {
  environmentCount: number;
  keyCount: number;
  missingCellCount: number;
  mismatchedCellCount: number;
  healthyCellCount: number;
}

export interface MappingSuggestionViewModel {
  suggestionId: string;
  resourceName: string;
  pluginId: string;
  suggestedEnvironmentId?: string;
  suggestedEnvironmentName?: string;
  confidence?: number;
  reasons: string[];
}

export interface ResourceListItemViewModel {
  discoveredResourceId: string;
  bindingId?: string;
  name: string;
  pluginId: string;
  resourceType: string;
  environmentId?: string;
  environmentName?: string;
  bindingStatus?: string;
  syncStatusLabel?: string;
}

export interface EnvironmentResourcesViewModel {
  groups: Array<{
    environmentId: string | null;
    title: string;
    items: ResourceListItemViewModel[];
  }>;
}

export interface ConfigurationCellSelection {
  configurationKeyId: string;
  configurationKeyName: string;
  environmentId: string;
  environmentName: string;
  status: ConfigurationMatrixCellStatus;
  statusLabel: string;
  safeVisibleValue?: string;
  accessLocked: boolean;
  requiredMissing: boolean;
  occurrenceIds: string[];
}

export const ENVIRONMENT_KIND_LABELS: Record<EnvironmentKind, string> = {
  local: "Local",
  development: "Development",
  preview: "Preview",
  staging: "Staging",
  production: "Production",
  test: "Test",
  custom: "Custom",
};

export const ENVIRONMENT_STATUS_LABELS: Record<EnvironmentStatus, string> = {
  local_only: "Local only",
  healthy: "Healthy",
  attention_required: "Attention required",
  syncing: "Syncing",
  error: "Error",
  archived: "Archived",
};

export const ENVIRONMENT_PRESETS: Array<{
  id: string;
  name: string;
  kind: EnvironmentKind;
  description: string;
}> = [
  {
    id: "development",
    name: "Development",
    kind: "development",
    description: "Shared development targets and local wiring.",
  },
  {
    id: "preview",
    name: "Preview",
    kind: "preview",
    description: "Ephemeral preview deployments.",
  },
  {
    id: "staging",
    name: "Staging",
    kind: "staging",
    description: "Pre-production validation environment.",
  },
  {
    id: "production",
    name: "Production",
    kind: "production",
    description: "Live customer-facing environment.",
  },
  {
    id: "test",
    name: "Test",
    kind: "test",
    description: "Automated or manual test environment.",
  },
  {
    id: "custom",
    name: "Custom",
    kind: "custom",
    description: "A custom environment for this project.",
  },
];
