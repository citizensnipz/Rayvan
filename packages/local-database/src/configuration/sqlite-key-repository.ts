import {
  configurationKeyId,
  projectId,
  type ConfigurationKey,
  type ConfigurationKeySource,
  type ConfigurationValueType,
} from "@rayvan/core";

import type { LocalDatabaseConnection } from "../database/connection.js";
import {
  ConfigurationKeyNotFoundError,
  ConfigurationPersistenceError,
  InvalidConfigurationKeyNameError,
} from "./errors.js";
import type {
  ConfigurationKeyRepository,
  CreateConfigurationKeyInput,
  UpdateConfigurationKeyInput,
} from "./key-repository.js";
import { validateConfigurationKeyName } from "./validation.js";

interface ConfigurationKeyRow {
  id: string;
  project_id: string;
  name: string;
  description: string | null;
  value_type: ConfigurationValueType;
  required: number;
  sensitive: number;
  source: ConfigurationKeySource;
  created_at: string;
  updated_at: string;
}

const SELECT_COLUMNS = `id, project_id, name, description, value_type, required, sensitive, source, created_at, updated_at`;

function mapRow(row: ConfigurationKeyRow): ConfigurationKey {
  return {
    id: configurationKeyId(row.id),
    projectId: projectId(row.project_id),
    name: row.name,
    description: row.description ?? undefined,
    valueType: row.value_type,
    required: row.required === 1,
    sensitive: row.sensitive === 1,
    source: row.source,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

export class SqliteConfigurationKeyRepository
  implements ConfigurationKeyRepository
{
  constructor(private readonly connection: LocalDatabaseConnection) {}

  async listByProjectId(projectIdValue: string): Promise<ConfigurationKey[]> {
    try {
      const rows = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM configuration_keys
           WHERE project_id = ?
           ORDER BY name ASC`,
        )
        .all(projectIdValue) as ConfigurationKeyRow[];
      return rows.map(mapRow);
    } catch (error) {
      throw new ConfigurationPersistenceError(
        "Failed to list configuration keys",
        error,
      );
    }
  }

  async getById(id: string): Promise<ConfigurationKey | null> {
    try {
      const row = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM configuration_keys WHERE id = ?`,
        )
        .get(id) as ConfigurationKeyRow | undefined;
      return row ? mapRow(row) : null;
    } catch (error) {
      throw new ConfigurationPersistenceError(
        "Failed to load configuration key",
        error,
      );
    }
  }

  async getByProjectAndName(
    projectIdValue: string,
    name: string,
  ): Promise<ConfigurationKey | null> {
    try {
      const row = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM configuration_keys
           WHERE project_id = ? AND name = ?`,
        )
        .get(projectIdValue, name) as ConfigurationKeyRow | undefined;
      return row ? mapRow(row) : null;
    } catch (error) {
      throw new ConfigurationPersistenceError(
        "Failed to load configuration key by name",
        error,
      );
    }
  }

  async create(
    input: CreateConfigurationKeyInput,
  ): Promise<ConfigurationKey> {
    const name = validateConfigurationKeyName(input.name);
    const now = new Date().toISOString();
    const id = crypto.randomUUID();

    try {
      this.connection.raw
        .prepare(
          `INSERT INTO configuration_keys (
             id, project_id, name, description, value_type,
             required, sensitive, source, created_at, updated_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        )
        .run(
          id,
          input.projectId,
          name,
          input.description?.trim() || null,
          input.valueType,
          input.required ? 1 : 0,
          input.sensitive ? 1 : 0,
          input.source,
          now,
          now,
        );

      const key = await this.getById(id);
      if (!key) {
        throw new ConfigurationPersistenceError(
          "Failed to read configuration key after create",
        );
      }
      return key;
    } catch (error) {
      if (
        error instanceof InvalidConfigurationKeyNameError ||
        error instanceof ConfigurationPersistenceError
      ) {
        throw error;
      }
      throw new ConfigurationPersistenceError(
        "Failed to create configuration key",
        error,
      );
    }
  }

  async update(
    id: string,
    input: UpdateConfigurationKeyInput,
  ): Promise<ConfigurationKey> {
    const existing = await this.getById(id);
    if (!existing) {
      throw new ConfigurationKeyNotFoundError(id);
    }

    const description =
      input.description !== undefined
        ? input.description.trim() || undefined
        : existing.description;
    const valueType = input.valueType ?? existing.valueType;
    const required = input.required ?? existing.required;
    const sensitive = input.sensitive ?? existing.sensitive;
    const source = input.source ?? existing.source;
    const updatedAt = new Date().toISOString();

    try {
      this.connection.raw
        .prepare(
          `UPDATE configuration_keys
           SET description = ?, value_type = ?, required = ?,
               sensitive = ?, source = ?, updated_at = ?
           WHERE id = ?`,
        )
        .run(
          description ?? null,
          valueType,
          required ? 1 : 0,
          sensitive ? 1 : 0,
          source,
          updatedAt,
          id,
        );

      const key = await this.getById(id);
      if (!key) {
        throw new ConfigurationKeyNotFoundError(id);
      }
      return key;
    } catch (error) {
      if (error instanceof ConfigurationKeyNotFoundError) {
        throw error;
      }
      throw new ConfigurationPersistenceError(
        "Failed to update configuration key",
        error,
      );
    }
  }
}
