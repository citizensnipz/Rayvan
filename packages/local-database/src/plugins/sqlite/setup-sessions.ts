import type { LocalDatabaseConnection } from "../../database/connection.js";
import type { PluginSetupSessionRecord } from "../models-setup.js";
import type { PluginSetupSessionRepository } from "../repositories/setup-sessions.js";

export class SqlitePluginSetupSessionRepository
  implements PluginSetupSessionRepository
{
  constructor(private readonly connection: LocalDatabaseConnection) {}

  async save(record: PluginSetupSessionRecord): Promise<void> {
    this.connection.raw
      .prepare(
        `INSERT INTO plugin_setup_sessions (
          id, plugin_id, installed_plugin_id, project_id, status,
          current_step_id, auth_method, state_json, created_at, updated_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          plugin_id = excluded.plugin_id,
          installed_plugin_id = excluded.installed_plugin_id,
          project_id = excluded.project_id,
          status = excluded.status,
          current_step_id = excluded.current_step_id,
          auth_method = excluded.auth_method,
          state_json = excluded.state_json,
          created_at = excluded.created_at,
          updated_at = excluded.updated_at,
          completed_at = excluded.completed_at`,
      )
      .run(
        record.id,
        record.pluginId,
        record.installedPluginId,
        record.projectId ?? null,
        record.status,
        record.currentStepId ?? null,
        record.authMethod ?? null,
        JSON.stringify(record.state),
        record.createdAt,
        record.updatedAt,
        record.completedAt ?? null,
      );
  }

  async getById(id: string): Promise<PluginSetupSessionRecord | undefined> {
    const row = this.connection.raw
      .prepare(`SELECT * FROM plugin_setup_sessions WHERE id = ?`)
      .get(id) as Record<string, unknown> | undefined;
    return row ? mapRow(row) : undefined;
  }

  async listByPluginId(pluginId: string): Promise<PluginSetupSessionRecord[]> {
    const rows = this.connection.raw
      .prepare(
        `SELECT * FROM plugin_setup_sessions WHERE plugin_id = ? ORDER BY created_at DESC`,
      )
      .all(pluginId) as Record<string, unknown>[];
    return rows.map(mapRow);
  }

  async listActiveByProjectId(
    projectId: string,
  ): Promise<PluginSetupSessionRecord[]> {
    const rows = this.connection.raw
      .prepare(
        `SELECT * FROM plugin_setup_sessions
         WHERE project_id = ? AND status = 'active'
         ORDER BY created_at DESC`,
      )
      .all(projectId) as Record<string, unknown>[];
    return rows.map(mapRow);
  }
}

function mapRow(row: Record<string, unknown>): PluginSetupSessionRecord {
  return {
    id: String(row.id),
    pluginId: String(row.plugin_id),
    installedPluginId: String(row.installed_plugin_id),
    projectId: row.project_id ? String(row.project_id) : undefined,
    status: row.status as PluginSetupSessionRecord["status"],
    currentStepId: row.current_step_id
      ? String(row.current_step_id)
      : undefined,
    authMethod: row.auth_method
      ? (String(row.auth_method) as PluginSetupSessionRecord["authMethod"])
      : undefined,
    state: JSON.parse(String(row.state_json)) as Record<string, unknown>,
    createdAt: String(row.created_at),
    updatedAt: String(row.updated_at),
    completedAt: row.completed_at ? String(row.completed_at) : undefined,
  };
}
