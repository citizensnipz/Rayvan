import type {
  ConfigurationOccurrence,
  ConfigurationValueAccess,
} from "@rayvan/core";

export interface CreateConfigurationOccurrenceInput {
  configurationKeyId: string;
  projectId: string;
  environmentId?: string;
  pluginId: string;
  connectionId: string;
  discoveredResourceId: string;
  resourceBindingId?: string;
  providerKey: string;
  valueAccess: ConfigurationValueAccess;
  observedValue?: string;
  maskedValue?: string;
  valueFingerprint?: string;
  secretValueRef?: string;
  scope?: string;
  firstObservedAt?: string;
  lastObservedAt?: string;
}

export interface UpdateConfigurationOccurrenceInput {
  environmentId?: string | null;
  resourceBindingId?: string | null;
  providerKey?: string;
  valueAccess?: ConfigurationValueAccess;
  observedValue?: string | null;
  maskedValue?: string | null;
  valueFingerprint?: string | null;
  secretValueRef?: string | null;
  scope?: string | null;
  lastObservedAt?: string;
}

export interface ConfigurationOccurrenceRepository {
  listByProjectId(projectId: string): Promise<ConfigurationOccurrence[]>;
  listByKeyId(keyId: string): Promise<ConfigurationOccurrence[]>;
  listByEnvironmentId(
    environmentId: string,
  ): Promise<ConfigurationOccurrence[]>;
  getById(id: string): Promise<ConfigurationOccurrence | null>;
  findMatch(input: {
    configurationKeyId: string;
    connectionId: string;
    discoveredResourceId: string;
    providerKey: string;
    environmentId?: string;
  }): Promise<ConfigurationOccurrence | null>;
  create(
    input: CreateConfigurationOccurrenceInput,
  ): Promise<ConfigurationOccurrence>;
  update(
    id: string,
    input: UpdateConfigurationOccurrenceInput,
  ): Promise<ConfigurationOccurrence>;
  delete?(id: string): Promise<void>;
}
