import type {
  ConfigurationKey,
  ConfigurationKeySource,
  ConfigurationOccurrence,
  ConfigurationValueAccess,
  ConfigurationValueType,
} from "@rayvan/core";

import {
  ConfigurationKeyNotFoundError,
} from "./errors.js";
import type { ConfigurationKeyRepository } from "./key-repository.js";
import type { ConfigurationOccurrenceRepository } from "./occurrence-repository.js";
import {
  assertNoPlaintextSecretForKey,
  validateConfigurationKeyName,
} from "./validation.js";

export interface UpsertConfigurationKeyInput {
  description?: string;
  valueType?: ConfigurationValueType;
  required?: boolean;
  sensitive?: boolean;
  source?: ConfigurationKeySource;
}

export interface UpdateConfigurationKeyMetadataInput {
  description?: string;
  valueType?: ConfigurationValueType;
  required?: boolean;
  sensitive?: boolean;
}

export interface UpsertConfigurationOccurrenceInput {
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
}

export class ConfigurationService {
  constructor(
    private readonly keys: ConfigurationKeyRepository,
    private readonly occurrences: ConfigurationOccurrenceRepository,
  ) {}

  async upsertKeyByName(
    projectId: string,
    name: string,
    partial: UpsertConfigurationKeyInput = {},
  ): Promise<ConfigurationKey> {
    const validatedName = validateConfigurationKeyName(name);
    const existing = await this.keys.getByProjectAndName(
      projectId,
      validatedName,
    );

    if (existing) {
      return this.keys.update(existing.id, {
        description: partial.description,
        valueType: partial.valueType,
        required: partial.required,
        sensitive: partial.sensitive,
        source: partial.source,
      });
    }

    return this.keys.create({
      projectId,
      name: validatedName,
      description: partial.description,
      valueType: partial.valueType ?? "unknown",
      required: partial.required ?? false,
      sensitive: partial.sensitive ?? partial.valueType === "secret",
      source: partial.source ?? "manual",
    });
  }

  listKeys(projectId: string): Promise<ConfigurationKey[]> {
    return this.keys.listByProjectId(projectId);
  }

  getKey(id: string): Promise<ConfigurationKey | null> {
    return this.keys.getById(id);
  }

  async updateKeyMetadata(
    id: string,
    input: UpdateConfigurationKeyMetadataInput,
  ): Promise<ConfigurationKey> {
    const existing = await this.keys.getById(id);
    if (!existing) {
      throw new ConfigurationKeyNotFoundError(id);
    }
    return this.keys.update(id, input);
  }

  async upsertOccurrence(
    input: UpsertConfigurationOccurrenceInput,
  ): Promise<ConfigurationOccurrence> {
    const key = await this.keys.getById(input.configurationKeyId);
    if (!key) {
      throw new ConfigurationKeyNotFoundError(input.configurationKeyId);
    }

    assertNoPlaintextSecretForKey(key, input);

    const existing = await this.occurrences.findMatch({
      configurationKeyId: input.configurationKeyId,
      connectionId: input.connectionId,
      discoveredResourceId: input.discoveredResourceId,
      providerKey: input.providerKey,
      environmentId: input.environmentId,
    });

    const now = new Date().toISOString();

    if (existing) {
      return this.occurrences.update(existing.id, {
        environmentId: input.environmentId,
        resourceBindingId: input.resourceBindingId,
        providerKey: input.providerKey,
        valueAccess: input.valueAccess,
        observedValue: input.observedValue ?? null,
        maskedValue: input.maskedValue ?? null,
        valueFingerprint: input.valueFingerprint ?? null,
        secretValueRef: input.secretValueRef ?? null,
        scope: input.scope ?? null,
        lastObservedAt: now,
      });
    }

    return this.occurrences.create({
      ...input,
      firstObservedAt: now,
      lastObservedAt: now,
    });
  }

  listOccurrencesByProject(
    projectId: string,
  ): Promise<ConfigurationOccurrence[]> {
    return this.occurrences.listByProjectId(projectId);
  }

  listOccurrencesByKey(keyId: string): Promise<ConfigurationOccurrence[]> {
    return this.occurrences.listByKeyId(keyId);
  }

  listOccurrencesByEnvironment(
    environmentId: string,
  ): Promise<ConfigurationOccurrence[]> {
    return this.occurrences.listByEnvironmentId(environmentId);
  }
}
