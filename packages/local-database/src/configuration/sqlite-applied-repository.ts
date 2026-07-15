import {
  appliedConfigurationStateId,
  configurationKeyId,
  environmentId,
  projectId,
  type AppliedConfigurationState,
  type AppliedConfigurationStatus,
} from "@rayvan/core";

import type { LocalDatabaseConnection } from "../database/connection.js";
import { ConfigurationPersistenceError } from "./errors.js";
import type {
  AppliedConfigurationStateRepository,
  UpsertAppliedConfigurationStateInput,
} from "./applied-repository.js";

interface AppliedRow {
  id: string;
  configuration_key_id: string;
  environment_id: string;
  project_id: string;
  resource_binding_id: string;
  desired_revision: number;
  applied_fingerprint: string | null;
  apply_execution_id: string;
  verification_execution_id: string | null;
  status: AppliedConfigurationStatus;
  applied_at: string;
  verified_at: string | null;
}

const SELECT_COLUMNS = `id, configuration_key_id, environment_id, project_id, resource_binding_id, desired_revision, applied_fingerprint, apply_execution_id, verification_execution_id, status, applied_at, verified_at`;

function mapRow(row: AppliedRow): AppliedConfigurationState {
  return {
    id: appliedConfigurationStateId(row.id),
    configurationKeyId: configurationKeyId(row.configuration_key_id),
    environmentId: environmentId(row.environment_id),
    projectId: projectId(row.project_id),
    resourceBindingId: row.resource_binding_id,
    desiredRevision: row.desired_revision,
    appliedFingerprint: row.applied_fingerprint ?? undefined,
    applyExecutionId: row.apply_execution_id,
    verificationExecutionId: row.verification_execution_id ?? undefined,
    status: row.status,
    appliedAt: row.applied_at,
    verifiedAt: row.verified_at ?? undefined,
  };
}

export class SqliteAppliedConfigurationStateRepository
  implements AppliedConfigurationStateRepository
{
  constructor(private readonly connection: LocalDatabaseConnection) {}

  async getById(id: string): Promise<AppliedConfigurationState | null> {
    try {
      const row = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM applied_configuration_states WHERE id = ?`,
        )
        .get(id) as AppliedRow | undefined;
      return row ? mapRow(row) : null;
    } catch (error) {
      throw new ConfigurationPersistenceError(
        "Failed to get applied configuration state",
        error,
      );
    }
  }

  async getByKeyEnvironmentBinding(
    configurationKeyIdValue: string,
    environmentIdValue: string,
    resourceBindingId: string,
  ): Promise<AppliedConfigurationState | null> {
    try {
      const row = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM applied_configuration_states
           WHERE configuration_key_id = ? AND environment_id = ? AND resource_binding_id = ?`,
        )
        .get(
          configurationKeyIdValue,
          environmentIdValue,
          resourceBindingId,
        ) as AppliedRow | undefined;
      return row ? mapRow(row) : null;
    } catch (error) {
      throw new ConfigurationPersistenceError(
        "Failed to get applied configuration state by binding",
        error,
      );
    }
  }

  async listByEnvironmentId(
    environmentIdValue: string,
  ): Promise<AppliedConfigurationState[]> {
    try {
      const rows = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM applied_configuration_states
           WHERE environment_id = ?
           ORDER BY applied_at DESC`,
        )
        .all(environmentIdValue) as AppliedRow[];
      return rows.map(mapRow);
    } catch (error) {
      throw new ConfigurationPersistenceError(
        "Failed to list applied configuration states by environment",
        error,
      );
    }
  }

  async listByProjectId(
    projectIdValue: string,
  ): Promise<AppliedConfigurationState[]> {
    try {
      const rows = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM applied_configuration_states
           WHERE project_id = ?
           ORDER BY applied_at DESC`,
        )
        .all(projectIdValue) as AppliedRow[];
      return rows.map(mapRow);
    } catch (error) {
      throw new ConfigurationPersistenceError(
        "Failed to list applied configuration states by project",
        error,
      );
    }
  }

  async upsert(
    input: UpsertAppliedConfigurationStateInput,
  ): Promise<AppliedConfigurationState> {
    try {
      const existing = await this.getByKeyEnvironmentBinding(
        input.configurationKeyId,
        input.environmentId,
        input.resourceBindingId,
      );
      const id = existing?.id ?? crypto.randomUUID();

      this.connection.raw
        .prepare(
          `INSERT INTO applied_configuration_states (
            id, configuration_key_id, environment_id, project_id, resource_binding_id,
            desired_revision, applied_fingerprint, apply_execution_id,
            verification_execution_id, status, applied_at, verified_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT(configuration_key_id, environment_id, resource_binding_id) DO UPDATE SET
            desired_revision = excluded.desired_revision,
            applied_fingerprint = excluded.applied_fingerprint,
            apply_execution_id = excluded.apply_execution_id,
            verification_execution_id = excluded.verification_execution_id,
            status = excluded.status,
            applied_at = excluded.applied_at,
            verified_at = excluded.verified_at`,
        )
        .run(
          id,
          input.configurationKeyId,
          input.environmentId,
          input.projectId,
          input.resourceBindingId,
          input.desiredRevision,
          input.appliedFingerprint ?? null,
          input.applyExecutionId,
          input.verificationExecutionId ?? null,
          input.status,
          input.appliedAt,
          input.verifiedAt ?? null,
        );

      const saved = await this.getByKeyEnvironmentBinding(
        input.configurationKeyId,
        input.environmentId,
        input.resourceBindingId,
      );
      if (!saved) {
        throw new ConfigurationPersistenceError(
          "Failed to read applied configuration state after upsert",
        );
      }
      return saved;
    } catch (error) {
      if (error instanceof ConfigurationPersistenceError) {
        throw error;
      }
      throw new ConfigurationPersistenceError(
        "Failed to upsert applied configuration state",
        error,
      );
    }
  }
}
