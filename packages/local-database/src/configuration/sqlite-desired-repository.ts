import {
  configurationKeyId,
  desiredConfigurationValueId,
  environmentId,
  projectId,
  type ConfigurationActorRef,
  type DesiredConfigurationValue,
} from "@rayvan/core";

import type { LocalDatabaseConnection } from "../database/connection.js";
import {
  ConfigurationPersistenceError,
  DesiredConfigurationRevisionConflictError,
  DesiredConfigurationValueNotFoundError,
} from "./errors.js";
import type {
  CreateDesiredConfigurationValueInput,
  DesiredConfigurationValueRepository,
  UpdateDesiredConfigurationValueInput,
} from "./desired-repository.js";

interface DesiredRow {
  id: string;
  configuration_key_id: string;
  environment_id: string;
  project_id: string;
  desired_value: string | null;
  secret_value_ref: string | null;
  value_fingerprint: string | null;
  revision: number;
  created_at: string;
  updated_at: string;
  updated_by_json: string;
}

const SELECT_COLUMNS = `id, configuration_key_id, environment_id, project_id, desired_value, secret_value_ref, value_fingerprint, revision, created_at, updated_at, updated_by_json`;

function mapRow(row: DesiredRow): DesiredConfigurationValue {
  return {
    id: desiredConfigurationValueId(row.id),
    configurationKeyId: configurationKeyId(row.configuration_key_id),
    environmentId: environmentId(row.environment_id),
    projectId: projectId(row.project_id),
    desiredValue: row.desired_value ?? undefined,
    secretValueRef: row.secret_value_ref ?? undefined,
    valueFingerprint: row.value_fingerprint ?? undefined,
    revision: row.revision,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    updatedBy: JSON.parse(row.updated_by_json) as ConfigurationActorRef,
  };
}

export class SqliteDesiredConfigurationValueRepository
  implements DesiredConfigurationValueRepository
{
  constructor(private readonly connection: LocalDatabaseConnection) {}

  async getById(id: string): Promise<DesiredConfigurationValue | null> {
    try {
      const row = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM desired_configuration_values WHERE id = ?`,
        )
        .get(id) as DesiredRow | undefined;
      return row ? mapRow(row) : null;
    } catch (error) {
      throw new ConfigurationPersistenceError(
        "Failed to get desired configuration value",
        error,
      );
    }
  }

  async getByKeyAndEnvironment(
    configurationKeyIdValue: string,
    environmentIdValue: string,
  ): Promise<DesiredConfigurationValue | null> {
    try {
      const row = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM desired_configuration_values
           WHERE configuration_key_id = ? AND environment_id = ?`,
        )
        .get(configurationKeyIdValue, environmentIdValue) as
        | DesiredRow
        | undefined;
      return row ? mapRow(row) : null;
    } catch (error) {
      throw new ConfigurationPersistenceError(
        "Failed to get desired configuration value by key and environment",
        error,
      );
    }
  }

  async listByEnvironmentId(
    environmentIdValue: string,
  ): Promise<DesiredConfigurationValue[]> {
    try {
      const rows = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM desired_configuration_values
           WHERE environment_id = ?
           ORDER BY updated_at DESC`,
        )
        .all(environmentIdValue) as DesiredRow[];
      return rows.map(mapRow);
    } catch (error) {
      throw new ConfigurationPersistenceError(
        "Failed to list desired configuration values by environment",
        error,
      );
    }
  }

  async listByProjectId(
    projectIdValue: string,
  ): Promise<DesiredConfigurationValue[]> {
    try {
      const rows = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM desired_configuration_values
           WHERE project_id = ?
           ORDER BY updated_at DESC`,
        )
        .all(projectIdValue) as DesiredRow[];
      return rows.map(mapRow);
    } catch (error) {
      throw new ConfigurationPersistenceError(
        "Failed to list desired configuration values by project",
        error,
      );
    }
  }

  async create(
    input: CreateDesiredConfigurationValueInput,
  ): Promise<DesiredConfigurationValue> {
    try {
      const now = new Date().toISOString();
      const id = crypto.randomUUID();
      this.connection.raw
        .prepare(
          `INSERT INTO desired_configuration_values (
            id, configuration_key_id, environment_id, project_id,
            desired_value, secret_value_ref, value_fingerprint,
            revision, created_at, updated_at, updated_by_json
          ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)`,
        )
        .run(
          id,
          input.configurationKeyId,
          input.environmentId,
          input.projectId,
          input.desiredValue ?? null,
          input.secretValueRef ?? null,
          input.valueFingerprint ?? null,
          now,
          now,
          JSON.stringify(input.updatedBy),
        );
      const created = await this.getById(id);
      if (!created) {
        throw new ConfigurationPersistenceError(
          "Failed to read desired configuration value after create",
        );
      }
      return created;
    } catch (error) {
      if (error instanceof ConfigurationPersistenceError) {
        throw error;
      }
      throw new ConfigurationPersistenceError(
        "Failed to create desired configuration value",
        error,
      );
    }
  }

  async updateWithExpectedRevision(
    id: string,
    input: UpdateDesiredConfigurationValueInput,
  ): Promise<DesiredConfigurationValue> {
    try {
      const existing = await this.getById(id);
      if (!existing) {
        throw new DesiredConfigurationValueNotFoundError(id);
      }
      if (existing.revision !== input.expectedRevision) {
        throw new DesiredConfigurationRevisionConflictError(
          input.expectedRevision,
          existing.revision,
        );
      }

      const desiredValue =
        input.desiredValue === undefined
          ? (existing.desiredValue ?? null)
          : input.desiredValue;
      const secretValueRef =
        input.secretValueRef === undefined
          ? (existing.secretValueRef ?? null)
          : input.secretValueRef;
      const valueFingerprint =
        input.valueFingerprint === undefined
          ? (existing.valueFingerprint ?? null)
          : input.valueFingerprint;
      const now = new Date().toISOString();

      const result = this.connection.raw
        .prepare(
          `UPDATE desired_configuration_values
           SET desired_value = ?, secret_value_ref = ?, value_fingerprint = ?,
               revision = ?, updated_at = ?, updated_by_json = ?
           WHERE id = ? AND revision = ?`,
        )
        .run(
          desiredValue,
          secretValueRef,
          valueFingerprint,
          existing.revision + 1,
          now,
          JSON.stringify(input.updatedBy),
          id,
          input.expectedRevision,
        );

      if (result.changes === 0) {
        const current = await this.getById(id);
        throw new DesiredConfigurationRevisionConflictError(
          input.expectedRevision,
          current?.revision ?? -1,
        );
      }

      const updated = await this.getById(id);
      if (!updated) {
        throw new DesiredConfigurationValueNotFoundError(id);
      }
      return updated;
    } catch (error) {
      if (
        error instanceof DesiredConfigurationValueNotFoundError ||
        error instanceof DesiredConfigurationRevisionConflictError
      ) {
        throw error;
      }
      throw new ConfigurationPersistenceError(
        "Failed to update desired configuration value",
        error,
      );
    }
  }

  async delete(id: string): Promise<void> {
    try {
      const result = this.connection.raw
        .prepare(`DELETE FROM desired_configuration_values WHERE id = ?`)
        .run(id);
      if (result.changes === 0) {
        throw new DesiredConfigurationValueNotFoundError(id);
      }
    } catch (error) {
      if (error instanceof DesiredConfigurationValueNotFoundError) {
        throw error;
      }
      throw new ConfigurationPersistenceError(
        "Failed to delete desired configuration value",
        error,
      );
    }
  }
}
