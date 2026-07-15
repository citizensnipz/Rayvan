import {
  configurationKeyId,
  desiredConfigurationValueId,
  environmentId,
  projectId,
  type DesiredConfigurationValue,
} from "@rayvan/core";

import {
  DesiredConfigurationRevisionConflictError,
  DesiredConfigurationValueNotFoundError,
} from "./errors.js";
import type {
  CreateDesiredConfigurationValueInput,
  DesiredConfigurationValueRepository,
  UpdateDesiredConfigurationValueInput,
} from "./desired-repository.js";

export class InMemoryDesiredConfigurationValueRepository
  implements DesiredConfigurationValueRepository
{
  private readonly values = new Map<string, DesiredConfigurationValue>();

  async getById(id: string): Promise<DesiredConfigurationValue | null> {
    return this.values.get(id) ?? null;
  }

  async getByKeyAndEnvironment(
    configurationKeyIdValue: string,
    environmentIdValue: string,
  ): Promise<DesiredConfigurationValue | null> {
    return (
      [...this.values.values()].find(
        (value) =>
          value.configurationKeyId === configurationKeyIdValue &&
          value.environmentId === environmentIdValue,
      ) ?? null
    );
  }

  async listByEnvironmentId(
    environmentIdValue: string,
  ): Promise<DesiredConfigurationValue[]> {
    return [...this.values.values()]
      .filter((value) => value.environmentId === environmentIdValue)
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
  }

  async listByProjectId(
    projectIdValue: string,
  ): Promise<DesiredConfigurationValue[]> {
    return [...this.values.values()]
      .filter((value) => value.projectId === projectIdValue)
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
  }

  async create(
    input: CreateDesiredConfigurationValueInput,
  ): Promise<DesiredConfigurationValue> {
    const now = new Date().toISOString();
    const value: DesiredConfigurationValue = {
      id: desiredConfigurationValueId(crypto.randomUUID()),
      configurationKeyId: configurationKeyId(input.configurationKeyId),
      environmentId: environmentId(input.environmentId),
      projectId: projectId(input.projectId),
      desiredValue: input.desiredValue,
      secretValueRef: input.secretValueRef,
      valueFingerprint: input.valueFingerprint,
      revision: 1,
      createdAt: now,
      updatedAt: now,
      updatedBy: input.updatedBy,
    };
    this.values.set(value.id, value);
    return value;
  }

  async updateWithExpectedRevision(
    id: string,
    input: UpdateDesiredConfigurationValueInput,
  ): Promise<DesiredConfigurationValue> {
    const existing = this.values.get(id);
    if (!existing) {
      throw new DesiredConfigurationValueNotFoundError(id);
    }
    if (existing.revision !== input.expectedRevision) {
      throw new DesiredConfigurationRevisionConflictError(
        input.expectedRevision,
        existing.revision,
      );
    }

    const updated: DesiredConfigurationValue = {
      ...existing,
      desiredValue:
        input.desiredValue === undefined
          ? existing.desiredValue
          : input.desiredValue === null
            ? undefined
            : input.desiredValue,
      secretValueRef:
        input.secretValueRef === undefined
          ? existing.secretValueRef
          : input.secretValueRef === null
            ? undefined
            : input.secretValueRef,
      valueFingerprint:
        input.valueFingerprint === undefined
          ? existing.valueFingerprint
          : input.valueFingerprint === null
            ? undefined
            : input.valueFingerprint,
      revision: existing.revision + 1,
      updatedAt: new Date().toISOString(),
      updatedBy: input.updatedBy,
    };
    this.values.set(id, updated);
    return updated;
  }

  async delete(id: string): Promise<void> {
    if (!this.values.has(id)) {
      throw new DesiredConfigurationValueNotFoundError(id);
    }
    this.values.delete(id);
  }
}
