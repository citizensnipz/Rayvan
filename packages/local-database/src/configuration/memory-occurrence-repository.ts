import {
  configurationKeyId,
  configurationOccurrenceId,
  environmentId,
  projectId,
  type ConfigurationOccurrence,
} from "@rayvan/core";

import { ConfigurationOccurrenceNotFoundError } from "./errors.js";
import type {
  ConfigurationOccurrenceRepository,
  CreateConfigurationOccurrenceInput,
  UpdateConfigurationOccurrenceInput,
} from "./occurrence-repository.js";

function sameOptional(left?: string, right?: string): boolean {
  return (left ?? undefined) === (right ?? undefined);
}

export class InMemoryConfigurationOccurrenceRepository
  implements ConfigurationOccurrenceRepository
{
  private readonly occurrences = new Map<string, ConfigurationOccurrence>();

  async listByProjectId(
    projectIdValue: string,
  ): Promise<ConfigurationOccurrence[]> {
    return [...this.occurrences.values()]
      .filter((occurrence) => occurrence.projectId === projectIdValue)
      .sort((left, right) =>
        right.lastObservedAt.localeCompare(left.lastObservedAt),
      );
  }

  async listByKeyId(keyId: string): Promise<ConfigurationOccurrence[]> {
    return [...this.occurrences.values()]
      .filter((occurrence) => occurrence.configurationKeyId === keyId)
      .sort((left, right) =>
        right.lastObservedAt.localeCompare(left.lastObservedAt),
      );
  }

  async listByEnvironmentId(
    environmentIdValue: string,
  ): Promise<ConfigurationOccurrence[]> {
    return [...this.occurrences.values()]
      .filter(
        (occurrence) => occurrence.environmentId === environmentIdValue,
      )
      .sort((left, right) =>
        right.lastObservedAt.localeCompare(left.lastObservedAt),
      );
  }

  async getById(id: string): Promise<ConfigurationOccurrence | null> {
    return this.occurrences.get(id) ?? null;
  }

  async findMatch(input: {
    configurationKeyId: string;
    connectionId: string;
    discoveredResourceId: string;
    providerKey: string;
    environmentId?: string;
  }): Promise<ConfigurationOccurrence | null> {
    return (
      [...this.occurrences.values()].find(
        (occurrence) =>
          occurrence.configurationKeyId === input.configurationKeyId &&
          occurrence.connectionId === input.connectionId &&
          occurrence.discoveredResourceId === input.discoveredResourceId &&
          occurrence.providerKey === input.providerKey &&
          sameOptional(occurrence.environmentId, input.environmentId),
      ) ?? null
    );
  }

  async create(
    input: CreateConfigurationOccurrenceInput,
  ): Promise<ConfigurationOccurrence> {
    const now = new Date().toISOString();
    const occurrence: ConfigurationOccurrence = {
      id: configurationOccurrenceId(crypto.randomUUID()),
      configurationKeyId: configurationKeyId(input.configurationKeyId),
      projectId: projectId(input.projectId),
      environmentId: input.environmentId
        ? environmentId(input.environmentId)
        : undefined,
      pluginId: input.pluginId,
      connectionId: input.connectionId,
      discoveredResourceId: input.discoveredResourceId,
      resourceBindingId: input.resourceBindingId,
      providerKey: input.providerKey,
      valueAccess: input.valueAccess,
      observedValue: input.observedValue,
      maskedValue: input.maskedValue,
      valueFingerprint: input.valueFingerprint,
      secretValueRef: input.secretValueRef,
      scope: input.scope,
      firstObservedAt: input.firstObservedAt ?? now,
      lastObservedAt: input.lastObservedAt ?? now,
    };
    this.occurrences.set(occurrence.id, occurrence);
    return occurrence;
  }

  async update(
    id: string,
    input: UpdateConfigurationOccurrenceInput,
  ): Promise<ConfigurationOccurrence> {
    const existing = this.occurrences.get(id);
    if (!existing) {
      throw new ConfigurationOccurrenceNotFoundError(id);
    }

    const updated: ConfigurationOccurrence = {
      ...existing,
      environmentId:
        input.environmentId === undefined
          ? existing.environmentId
          : input.environmentId === null
            ? undefined
            : environmentId(input.environmentId),
      resourceBindingId:
        input.resourceBindingId === undefined
          ? existing.resourceBindingId
          : input.resourceBindingId === null
            ? undefined
            : input.resourceBindingId,
      providerKey: input.providerKey ?? existing.providerKey,
      valueAccess: input.valueAccess ?? existing.valueAccess,
      observedValue:
        input.observedValue === undefined
          ? existing.observedValue
          : input.observedValue === null
            ? undefined
            : input.observedValue,
      maskedValue:
        input.maskedValue === undefined
          ? existing.maskedValue
          : input.maskedValue === null
            ? undefined
            : input.maskedValue,
      valueFingerprint:
        input.valueFingerprint === undefined
          ? existing.valueFingerprint
          : input.valueFingerprint === null
            ? undefined
            : input.valueFingerprint,
      secretValueRef:
        input.secretValueRef === undefined
          ? existing.secretValueRef
          : input.secretValueRef === null
            ? undefined
            : input.secretValueRef,
      scope:
        input.scope === undefined
          ? existing.scope
          : input.scope === null
            ? undefined
            : input.scope,
      lastObservedAt: input.lastObservedAt ?? new Date().toISOString(),
    };
    this.occurrences.set(id, updated);
    return updated;
  }

  async delete(id: string): Promise<void> {
    if (!this.occurrences.has(id)) {
      throw new ConfigurationOccurrenceNotFoundError(id);
    }
    this.occurrences.delete(id);
  }
}
