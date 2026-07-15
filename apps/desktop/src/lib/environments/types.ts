import type {
  AppliedConfigurationState,
  ConfigurationKey,
  ConfigurationOccurrence,
  DesiredConfigurationValue,
  Environment,
  EnvironmentKind,
  EnvironmentPresentation,
  EnvironmentStatus,
} from "@rayvan/core";
import type {
  ConfigurationApplyPlan,
  ConfigurationApplyResult,
  ConfigurationDerivedFinding,
  ConfigurationMatrixViewModel,
  EnvironmentConfigurationStatusViewModel,
} from "@rayvan/config-engine";
import type {
  DiscoveredResourceRecord,
  EnvironmentMappingSuggestionRecord,
  ResourceBindingRecord,
} from "@rayvan/local-database";

export type EnvironmentSyncPhase =
  | "idle"
  | "starting"
  | "discovering"
  | "mapping"
  | "building_matrix"
  | "complete"
  | "cancelled"
  | "failed";

export interface EnvironmentSyncPluginResult {
  pluginId: string;
  connectionId: string;
  connectionName: string;
  status: "success" | "failed" | "skipped";
  message?: string;
  resourcesDiscovered?: number;
  suggestionsCreated?: number;
}

export interface EnvironmentSyncResult {
  projectId: string;
  environmentId?: string;
  phase: EnvironmentSyncPhase;
  cancelled: boolean;
  startedAt: string;
  finishedAt?: string;
  plugins: EnvironmentSyncPluginResult[];
  suggestionsCreated: number;
  resourcesTouched: number;
}

export interface EnvironmentSyncState {
  projectId: string;
  inProgress: boolean;
  phase: EnvironmentSyncPhase;
  cancelled: boolean;
  lastResult?: EnvironmentSyncResult;
  progressLabel?: string;
  environmentId?: string;
}

export interface CreateEnvironmentGatewayInput {
  projectId: string;
  name: string;
  slug?: string;
  kind: EnvironmentKind;
  description?: string;
  presentation?: EnvironmentPresentation;
  status?: EnvironmentStatus;
}

export interface UpdateEnvironmentGatewayInput {
  name?: string;
  slug?: string;
  description?: string;
  kind?: EnvironmentKind;
  presentation?: EnvironmentPresentation;
  status?: EnvironmentStatus;
}

export interface UpdateConfigurationKeyGatewayInput {
  description?: string;
  valueType?: ConfigurationKey["valueType"];
  required?: boolean;
  sensitive?: boolean;
}

export interface AttachResourceInput {
  projectId: string;
  environmentId: string;
  discoveredResourceId: string;
}

export interface MoveResourceInput {
  bindingId: string;
  environmentId: string;
}

export interface AcceptSuggestionInput {
  suggestionId: string;
  environmentId?: string;
  createEnvironment?: {
    name: string;
    kind: EnvironmentKind;
    description?: string;
  };
}

export interface AcceptSuggestionResult {
  suggestion: EnvironmentMappingSuggestionRecord;
  binding?: ResourceBindingRecord;
  environment?: Environment;
}

export interface SaveDesiredValueGatewayInput {
  configurationKeyId: string;
  environmentId: string;
  projectId: string;
  desiredValue?: string;
  secretValueRef?: string;
  valueFingerprint?: string;
  expectedRevision?: number;
}

export interface AdoptDiscoveredKeyGatewayInput {
  configurationKeyId: string;
  environmentId: string;
  projectId: string;
  copyReadableValue?: boolean;
}

/**
 * Host-side abstraction for the Environments workspace.
 * React-free; implementations must not call real provider APIs.
 */
export interface EnvironmentsGateway {
  ensureProjectSeeded(projectId: string): Promise<void>;

  listEnvironments(
    projectId: string,
    options?: { includeArchived?: boolean },
  ): Promise<Environment[]>;
  createEnvironment(input: CreateEnvironmentGatewayInput): Promise<Environment>;
  updateEnvironment(
    id: string,
    input: UpdateEnvironmentGatewayInput,
  ): Promise<Environment>;
  archiveEnvironment(id: string): Promise<Environment>;
  getEnvironment(id: string): Promise<Environment | null>;

  listConfigurationKeys(projectId: string): Promise<ConfigurationKey[]>;
  updateConfigurationKey(
    id: string,
    metadata: UpdateConfigurationKeyGatewayInput,
  ): Promise<ConfigurationKey>;
  listOccurrences(projectId: string): Promise<ConfigurationOccurrence[]>;

  listDesiredValues(projectId: string): Promise<DesiredConfigurationValue[]>;
  listDesiredValuesByEnvironment(
    environmentId: string,
  ): Promise<DesiredConfigurationValue[]>;
  listAppliedByEnvironment(
    environmentId: string,
  ): Promise<AppliedConfigurationState[]>;
  saveDesiredValues(
    inputs: SaveDesiredValueGatewayInput[],
  ): Promise<DesiredConfigurationValue[]>;
  getEnvironmentConfigurationStatus(
    projectId: string,
    environmentId: string,
    options?: {
      drafts?: Array<{
        configurationKeyId: string;
        draftValue?: string;
        draftSecretValueRef?: string;
        draftFingerprint?: string;
        dirty: boolean;
      }>;
    },
  ): Promise<EnvironmentConfigurationStatusViewModel>;

  buildApplyPlan(
    projectId: string,
    environmentId: string,
  ): Promise<ConfigurationApplyPlan>;
  approveApplyPlan(planId: string): Promise<ConfigurationApplyResult>;

  adoptDiscoveredKey(
    input: AdoptDiscoveredKeyGatewayInput,
  ): Promise<DesiredConfigurationValue>;

  listDiscoveredResources(projectId: string): Promise<DiscoveredResourceRecord[]>;
  listBindings(projectId: string): Promise<ResourceBindingRecord[]>;
  attachResource(input: AttachResourceInput): Promise<ResourceBindingRecord>;
  moveResource(input: MoveResourceInput): Promise<ResourceBindingRecord>;
  detachResource(bindingId: string): Promise<ResourceBindingRecord>;

  listPendingSuggestions(
    projectId: string,
  ): Promise<EnvironmentMappingSuggestionRecord[]>;
  acceptSuggestion(input: AcceptSuggestionInput): Promise<AcceptSuggestionResult>;
  rejectSuggestion(
    suggestionId: string,
  ): Promise<EnvironmentMappingSuggestionRecord>;
  chooseSuggestionEnvironment(
    suggestionId: string,
    environmentId: string,
  ): Promise<EnvironmentMappingSuggestionRecord>;

  getMatrix(projectId: string): Promise<ConfigurationMatrixViewModel>;
  getDerivedFindings(projectId: string): Promise<ConfigurationDerivedFinding[]>;

  syncWithIntegrations(
    projectId: string,
    options?: { environmentId?: string },
  ): Promise<EnvironmentSyncResult>;
  cancelSync(projectId: string): Promise<void>;
  getSyncState(projectId: string): Promise<EnvironmentSyncState>;
}
