import {
  configurationKeyId,
  projectId,
  type ConfigurationKey,
} from "@rayvan/core";

import { ConfigurationKeyNotFoundError } from "./errors.js";
import type {
  ConfigurationKeyRepository,
  CreateConfigurationKeyInput,
  UpdateConfigurationKeyInput,
} from "./key-repository.js";
import { validateConfigurationKeyName } from "./validation.js";

export class InMemoryConfigurationKeyRepository
  implements ConfigurationKeyRepository
{
  private readonly keys = new Map<string, ConfigurationKey>();

  async listByProjectId(projectIdValue: string): Promise<ConfigurationKey[]> {
    return [...this.keys.values()]
      .filter((key) => key.projectId === projectIdValue)
      .sort((left, right) => left.name.localeCompare(right.name));
  }

  async getById(id: string): Promise<ConfigurationKey | null> {
    return this.keys.get(id) ?? null;
  }

  async getByProjectAndName(
    projectIdValue: string,
    name: string,
  ): Promise<ConfigurationKey | null> {
    return (
      [...this.keys.values()].find(
        (key) => key.projectId === projectIdValue && key.name === name,
      ) ?? null
    );
  }

  async create(input: CreateConfigurationKeyInput): Promise<ConfigurationKey> {
    const now = new Date().toISOString();
    const key: ConfigurationKey = {
      id: configurationKeyId(crypto.randomUUID()),
      projectId: projectId(input.projectId),
      name: validateConfigurationKeyName(input.name),
      description: input.description?.trim() || undefined,
      valueType: input.valueType,
      required: input.required,
      sensitive: input.sensitive,
      source: input.source,
      createdAt: now,
      updatedAt: now,
    };
    this.keys.set(key.id, key);
    return key;
  }

  async update(
    id: string,
    input: UpdateConfigurationKeyInput,
  ): Promise<ConfigurationKey> {
    const existing = this.keys.get(id);
    if (!existing) {
      throw new ConfigurationKeyNotFoundError(id);
    }

    const updated: ConfigurationKey = {
      ...existing,
      description:
        input.description !== undefined
          ? input.description.trim() || undefined
          : existing.description,
      valueType: input.valueType ?? existing.valueType,
      required: input.required ?? existing.required,
      sensitive: input.sensitive ?? existing.sensitive,
      source: input.source ?? existing.source,
      updatedAt: new Date().toISOString(),
    };
    this.keys.set(id, updated);
    return updated;
  }

  async delete(id: string): Promise<void> {
    if (!this.keys.has(id)) {
      throw new ConfigurationKeyNotFoundError(id);
    }
    this.keys.delete(id);
  }
}
