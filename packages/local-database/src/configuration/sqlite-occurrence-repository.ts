import {
  configurationKeyId,
  configurationOccurrenceId,
  environmentId,
  projectId,
  type ConfigurationOccurrence,
  type ConfigurationValueAccess,
} from "@rayvan/core";

import type { LocalDatabaseConnection } from "../database/connection.js";
import {
  ConfigurationOccurrenceNotFoundError,
  ConfigurationPersistenceError,
} from "./errors.js";
import type {
  ConfigurationOccurrenceRepository,
  CreateConfigurationOccurrenceInput,
  UpdateConfigurationOccurrenceInput,
} from "./occurrence-repository.js";

interface ConfigurationOccurrenceRow {
  id: string;
  configuration_key_id: string;
  project_id: string;
  environment_id: string | null;
  plugin_id: string;
  connection_id: string;
  discovered_resource_id: string;
  resource_binding_id: string | null;
  provider_key: string;
  value_access: ConfigurationValueAccess;
  observed_value: string | null;
  masked_value: string | null;
  value_fingerprint: string | null;
  secret_value_ref: string | null;
  scope: string | null;
  first_observed_at: string;
  last_observed_at: string;
}

const SELECT_COLUMNS = `id, configuration_key_id, project_id, environment_id, plugin_id, connection_id, discovered_resource_id, resource_binding_id, provider_key, value_access, observed_value, masked_value, value_fingerprint, secret_value_ref, scope, first_observed_at, last_observed_at`;

function mapRow(row: ConfigurationOccurrenceRow): ConfigurationOccurrence {
  return {
    id: configurationOccurrenceId(row.id),
    configurationKeyId: configurationKeyId(row.configuration_key_id),
    projectId: projectId(row.project_id),
    environmentId: row.environment_id
      ? environmentId(row.environment_id)
      : undefined,
    pluginId: row.plugin_id,
    connectionId: row.connection_id,
    discoveredResourceId: row.discovered_resource_id,
    resourceBindingId: row.resource_binding_id ?? undefined,
    providerKey: row.provider_key,
    valueAccess: row.value_access,
    observedValue: row.observed_value ?? undefined,
    maskedValue: row.masked_value ?? undefined,
    valueFingerprint: row.value_fingerprint ?? undefined,
    secretValueRef: row.secret_value_ref ?? undefined,
    scope: row.scope ?? undefined,
    firstObservedAt: row.first_observed_at,
    lastObservedAt: row.last_observed_at,
  };
}

export class SqliteConfigurationOccurrenceRepository
  implements ConfigurationOccurrenceRepository
{
  constructor(private readonly connection: LocalDatabaseConnection) {}

  async listByProjectId(
    projectIdValue: string,
  ): Promise<ConfigurationOccurrence[]> {
    try {
      const rows = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM configuration_occurrences
           WHERE project_id = ?
           ORDER BY last_observed_at DESC`,
        )
        .all(projectIdValue) as ConfigurationOccurrenceRow[];
      return rows.map(mapRow);
    } catch (error) {
      throw new ConfigurationPersistenceError(
        "Failed to list configuration occurrences",
        error,
      );
    }
  }

  async listByKeyId(keyId: string): Promise<ConfigurationOccurrence[]> {
    try {
      const rows = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM configuration_occurrences
           WHERE configuration_key_id = ?
           ORDER BY last_observed_at DESC`,
        )
        .all(keyId) as ConfigurationOccurrenceRow[];
      return rows.map(mapRow);
    } catch (error) {
      throw new ConfigurationPersistenceError(
        "Failed to list configuration occurrences by key",
        error,
      );
    }
  }

  async listByEnvironmentId(
    environmentIdValue: string,
  ): Promise<ConfigurationOccurrence[]> {
    try {
      const rows = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM configuration_occurrences
           WHERE environment_id = ?
           ORDER BY last_observed_at DESC`,
        )
        .all(environmentIdValue) as ConfigurationOccurrenceRow[];
      return rows.map(mapRow);
    } catch (error) {
      throw new ConfigurationPersistenceError(
        "Failed to list configuration occurrences by environment",
        error,
      );
    }
  }

  async getById(id: string): Promise<ConfigurationOccurrence | null> {
    try {
      const row = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM configuration_occurrences WHERE id = ?`,
        )
        .get(id) as ConfigurationOccurrenceRow | undefined;
      return row ? mapRow(row) : null;
    } catch (error) {
      throw new ConfigurationPersistenceError(
        "Failed to load configuration occurrence",
        error,
      );
    }
  }

  async findMatch(input: {
    configurationKeyId: string;
    connectionId: string;
    discoveredResourceId: string;
    providerKey: string;
    environmentId?: string;
  }): Promise<ConfigurationOccurrence | null> {
    try {
      const row =
        input.environmentId === undefined
          ? (this.connection.raw
              .prepare(
                `SELECT ${SELECT_COLUMNS} FROM configuration_occurrences
                 WHERE configuration_key_id = ?
                   AND connection_id = ?
                   AND discovered_resource_id = ?
                   AND provider_key = ?
                   AND environment_id IS NULL`,
              )
              .get(
                input.configurationKeyId,
                input.connectionId,
                input.discoveredResourceId,
                input.providerKey,
              ) as ConfigurationOccurrenceRow | undefined)
          : (this.connection.raw
              .prepare(
                `SELECT ${SELECT_COLUMNS} FROM configuration_occurrences
                 WHERE configuration_key_id = ?
                   AND connection_id = ?
                   AND discovered_resource_id = ?
                   AND provider_key = ?
                   AND environment_id = ?`,
              )
              .get(
                input.configurationKeyId,
                input.connectionId,
                input.discoveredResourceId,
                input.providerKey,
                input.environmentId,
              ) as ConfigurationOccurrenceRow | undefined);
      return row ? mapRow(row) : null;
    } catch (error) {
      throw new ConfigurationPersistenceError(
        "Failed to find configuration occurrence",
        error,
      );
    }
  }

  async create(
    input: CreateConfigurationOccurrenceInput,
  ): Promise<ConfigurationOccurrence> {
    const now = new Date().toISOString();
    const id = crypto.randomUUID();
    const firstObservedAt = input.firstObservedAt ?? now;
    const lastObservedAt = input.lastObservedAt ?? now;

    try {
      this.connection.raw
        .prepare(
          `INSERT INTO configuration_occurrences (
             id, configuration_key_id, project_id, environment_id,
             plugin_id, connection_id, discovered_resource_id,
             resource_binding_id, provider_key, value_access,
             observed_value, masked_value, value_fingerprint,
             secret_value_ref, scope, first_observed_at, last_observed_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        )
        .run(
          id,
          input.configurationKeyId,
          input.projectId,
          input.environmentId ?? null,
          input.pluginId,
          input.connectionId,
          input.discoveredResourceId,
          input.resourceBindingId ?? null,
          input.providerKey,
          input.valueAccess,
          input.observedValue ?? null,
          input.maskedValue ?? null,
          input.valueFingerprint ?? null,
          input.secretValueRef ?? null,
          input.scope ?? null,
          firstObservedAt,
          lastObservedAt,
        );

      const occurrence = await this.getById(id);
      if (!occurrence) {
        throw new ConfigurationPersistenceError(
          "Failed to read configuration occurrence after create",
        );
      }
      return occurrence;
    } catch (error) {
      if (error instanceof ConfigurationPersistenceError) {
        throw error;
      }
      throw new ConfigurationPersistenceError(
        "Failed to create configuration occurrence",
        error,
      );
    }
  }

  async update(
    id: string,
    input: UpdateConfigurationOccurrenceInput,
  ): Promise<ConfigurationOccurrence> {
    const existing = await this.getById(id);
    if (!existing) {
      throw new ConfigurationOccurrenceNotFoundError(id);
    }

    const environmentIdValue =
      input.environmentId === undefined
        ? (existing.environmentId ?? null)
        : input.environmentId;
    const resourceBindingId =
      input.resourceBindingId === undefined
        ? (existing.resourceBindingId ?? null)
        : input.resourceBindingId;
    const providerKey = input.providerKey ?? existing.providerKey;
    const valueAccess = input.valueAccess ?? existing.valueAccess;
    const observedValue =
      input.observedValue === undefined
        ? (existing.observedValue ?? null)
        : input.observedValue;
    const maskedValue =
      input.maskedValue === undefined
        ? (existing.maskedValue ?? null)
        : input.maskedValue;
    const valueFingerprint =
      input.valueFingerprint === undefined
        ? (existing.valueFingerprint ?? null)
        : input.valueFingerprint;
    const secretValueRef =
      input.secretValueRef === undefined
        ? (existing.secretValueRef ?? null)
        : input.secretValueRef;
    const scope =
      input.scope === undefined ? (existing.scope ?? null) : input.scope;
    const lastObservedAt = input.lastObservedAt ?? new Date().toISOString();

    try {
      this.connection.raw
        .prepare(
          `UPDATE configuration_occurrences
           SET environment_id = ?, resource_binding_id = ?, provider_key = ?,
               value_access = ?, observed_value = ?, masked_value = ?,
               value_fingerprint = ?, secret_value_ref = ?, scope = ?,
               last_observed_at = ?
           WHERE id = ?`,
        )
        .run(
          environmentIdValue,
          resourceBindingId,
          providerKey,
          valueAccess,
          observedValue,
          maskedValue,
          valueFingerprint,
          secretValueRef,
          scope,
          lastObservedAt,
          id,
        );

      const occurrence = await this.getById(id);
      if (!occurrence) {
        throw new ConfigurationOccurrenceNotFoundError(id);
      }
      return occurrence;
    } catch (error) {
      if (error instanceof ConfigurationOccurrenceNotFoundError) {
        throw error;
      }
      throw new ConfigurationPersistenceError(
        "Failed to update configuration occurrence",
        error,
      );
    }
  }
}
