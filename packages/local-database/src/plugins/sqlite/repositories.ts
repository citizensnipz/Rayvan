import type { LocalDatabaseConnection } from "../../database/connection.js";
import { PluginPersistenceError } from "../errors.js";
import type {
  InstalledPluginRecord,
  PluginConnectionRecord,
  PluginExecutionHistoryRecord,
  PluginInstallSource,
  PluginPermissionGrantRecord,
  DiscoveredResourceRecord,
} from "../models.js";
import type {
  DiscoveredResourceRepository,
  DiscoverySyncItem,
  InstalledPluginRepository,
  PluginConnectionRepository,
  PluginExecutionHistoryRepository,
  PluginPermissionGrantRepository,
} from "../repositories/types.js";
import type {
  PluginExecutionActor,
  PluginManifest,
  PluginPermission,
} from "@rayvan/plugin-sdk";

function parseJson<T>(value: string, label: string): T {
  try {
    return JSON.parse(value) as T;
  } catch (error) {
    throw new PluginPersistenceError(`Failed to parse ${label}`, error);
  }
}

export class SqliteInstalledPluginRepository
  implements InstalledPluginRepository
{
  constructor(private readonly connection: LocalDatabaseConnection) {}

  async save(record: InstalledPluginRecord): Promise<void> {
    this.connection.raw
      .prepare(
        `INSERT INTO plugin_installed (
          id, plugin_id, plugin_version, manifest_version, rayvan_api_version,
          source_json, status, enabled, installed_at, updated_at, last_loaded_at,
          manifest_snapshot_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          plugin_id = excluded.plugin_id,
          plugin_version = excluded.plugin_version,
          manifest_version = excluded.manifest_version,
          rayvan_api_version = excluded.rayvan_api_version,
          source_json = excluded.source_json,
          status = excluded.status,
          enabled = excluded.enabled,
          installed_at = excluded.installed_at,
          updated_at = excluded.updated_at,
          last_loaded_at = excluded.last_loaded_at,
          manifest_snapshot_json = excluded.manifest_snapshot_json`,
      )
      .run(
        record.id,
        record.pluginId,
        record.pluginVersion,
        record.manifestVersion,
        record.rayvanApiVersion,
        JSON.stringify(record.source),
        record.status,
        record.enabled ? 1 : 0,
        record.installedAt,
        record.updatedAt,
        record.lastLoadedAt ?? null,
        JSON.stringify(record.manifestSnapshot),
      );
  }

  async getById(id: string): Promise<InstalledPluginRecord | undefined> {
    const row = this.connection.raw
      .prepare(`SELECT * FROM plugin_installed WHERE id = ?`)
      .get(id) as Record<string, unknown> | undefined;
    return row ? mapInstalled(row) : undefined;
  }

  async getByPluginId(
    pluginId: string,
  ): Promise<InstalledPluginRecord | undefined> {
    const row = this.connection.raw
      .prepare(`SELECT * FROM plugin_installed WHERE plugin_id = ?`)
      .get(pluginId) as Record<string, unknown> | undefined;
    return row ? mapInstalled(row) : undefined;
  }

  async list(): Promise<InstalledPluginRecord[]> {
    const rows = this.connection.raw
      .prepare(`SELECT * FROM plugin_installed ORDER BY plugin_id`)
      .all() as Record<string, unknown>[];
    return rows.map(mapInstalled);
  }

  async setEnabled(id: string, enabled: boolean): Promise<void> {
    const existing = await this.getById(id);
    if (!existing) {
      throw new PluginPersistenceError(`Installed plugin not found: ${id}`);
    }
    await this.save({
      ...existing,
      enabled,
      status: enabled ? "installed" : "disabled",
      updatedAt: new Date().toISOString(),
    });
  }

  async updateStatus(
    id: string,
    status: InstalledPluginRecord["status"],
  ): Promise<void> {
    const existing = await this.getById(id);
    if (!existing) {
      throw new PluginPersistenceError(`Installed plugin not found: ${id}`);
    }
    await this.save({
      ...existing,
      status,
      enabled: status === "installed" ? existing.enabled : false,
      updatedAt: new Date().toISOString(),
    });
  }
}

function mapInstalled(row: Record<string, unknown>): InstalledPluginRecord {
  return {
    id: String(row.id),
    pluginId: String(row.plugin_id),
    pluginVersion: String(row.plugin_version),
    manifestVersion: String(row.manifest_version),
    rayvanApiVersion: String(row.rayvan_api_version),
    source: parseJson<PluginInstallSource>(
      String(row.source_json),
      "source_json",
    ),
    status: row.status as InstalledPluginRecord["status"],
    enabled: Boolean(row.enabled),
    installedAt: String(row.installed_at),
    updatedAt: String(row.updated_at),
    lastLoadedAt: row.last_loaded_at
      ? String(row.last_loaded_at)
      : undefined,
    manifestSnapshot: parseJson<PluginManifest>(
      String(row.manifest_snapshot_json),
      "manifest_snapshot_json",
    ),
  };
}

export class SqlitePluginConnectionRepository
  implements PluginConnectionRepository
{
  constructor(private readonly connection: LocalDatabaseConnection) {}

  async save(record: PluginConnectionRecord): Promise<void> {
    this.connection.raw
      .prepare(
        `INSERT INTO plugin_connections (
          id, installed_plugin_id, plugin_id, project_id, name, status,
          external_account_id, external_account_name, provider_base_url,
          credential_reference_id, metadata_json, schema_version,
          created_at, updated_at, last_authenticated_at,
          last_successful_sync_at, last_failed_sync_at, last_error_code
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          installed_plugin_id = excluded.installed_plugin_id,
          plugin_id = excluded.plugin_id,
          project_id = excluded.project_id,
          name = excluded.name,
          status = excluded.status,
          external_account_id = excluded.external_account_id,
          external_account_name = excluded.external_account_name,
          provider_base_url = excluded.provider_base_url,
          credential_reference_id = excluded.credential_reference_id,
          metadata_json = excluded.metadata_json,
          schema_version = excluded.schema_version,
          created_at = excluded.created_at,
          updated_at = excluded.updated_at,
          last_authenticated_at = excluded.last_authenticated_at,
          last_successful_sync_at = excluded.last_successful_sync_at,
          last_failed_sync_at = excluded.last_failed_sync_at,
          last_error_code = excluded.last_error_code`,
      )
      .run(
        record.id,
        record.installedPluginId,
        record.pluginId,
        record.projectId ?? null,
        record.name,
        record.status,
        record.externalAccountId ?? null,
        record.externalAccountName ?? null,
        record.providerBaseUrl ?? null,
        record.credentialReferenceId ?? null,
        JSON.stringify(record.metadata),
        record.schemaVersion,
        record.createdAt,
        record.updatedAt,
        record.lastAuthenticatedAt ?? null,
        record.lastSuccessfulSyncAt ?? null,
        record.lastFailedSyncAt ?? null,
        record.lastErrorCode ?? null,
      );
  }

  async getById(id: string): Promise<PluginConnectionRecord | undefined> {
    const row = this.connection.raw
      .prepare(`SELECT * FROM plugin_connections WHERE id = ?`)
      .get(id) as Record<string, unknown> | undefined;
    return row ? mapConnection(row) : undefined;
  }

  async listByPluginId(pluginId: string): Promise<PluginConnectionRecord[]> {
    const rows = this.connection.raw
      .prepare(`SELECT * FROM plugin_connections WHERE plugin_id = ?`)
      .all(pluginId) as Record<string, unknown>[];
    return rows.map(mapConnection);
  }

  async listByProjectId(projectId: string): Promise<PluginConnectionRecord[]> {
    const rows = this.connection.raw
      .prepare(`SELECT * FROM plugin_connections WHERE project_id = ?`)
      .all(projectId) as Record<string, unknown>[];
    return rows.map(mapConnection);
  }

  async listByInstalledPluginId(
    installedPluginId: string,
  ): Promise<PluginConnectionRecord[]> {
    const rows = this.connection.raw
      .prepare(
        `SELECT * FROM plugin_connections WHERE installed_plugin_id = ?`,
      )
      .all(installedPluginId) as Record<string, unknown>[];
    return rows.map(mapConnection);
  }
}

function mapConnection(row: Record<string, unknown>): PluginConnectionRecord {
  return {
    id: String(row.id),
    installedPluginId: String(row.installed_plugin_id),
    pluginId: String(row.plugin_id),
    projectId: row.project_id ? String(row.project_id) : undefined,
    name: String(row.name),
    status: row.status as PluginConnectionRecord["status"],
    externalAccountId: row.external_account_id
      ? String(row.external_account_id)
      : undefined,
    externalAccountName: row.external_account_name
      ? String(row.external_account_name)
      : undefined,
    providerBaseUrl: row.provider_base_url
      ? String(row.provider_base_url)
      : undefined,
    credentialReferenceId: row.credential_reference_id
      ? String(row.credential_reference_id)
      : undefined,
    metadata: parseJson<Record<string, unknown>>(
      String(row.metadata_json),
      "metadata_json",
    ),
    schemaVersion: String(row.schema_version),
    createdAt: String(row.created_at),
    updatedAt: String(row.updated_at),
    lastAuthenticatedAt: row.last_authenticated_at
      ? String(row.last_authenticated_at)
      : undefined,
    lastSuccessfulSyncAt: row.last_successful_sync_at
      ? String(row.last_successful_sync_at)
      : undefined,
    lastFailedSyncAt: row.last_failed_sync_at
      ? String(row.last_failed_sync_at)
      : undefined,
    lastErrorCode: row.last_error_code
      ? String(row.last_error_code)
      : undefined,
  };
}

export class SqlitePluginPermissionGrantRepository
  implements PluginPermissionGrantRepository
{
  constructor(private readonly connection: LocalDatabaseConnection) {}

  async save(record: PluginPermissionGrantRecord): Promise<void> {
    this.connection.raw
      .prepare(
        `INSERT INTO plugin_permission_grants (
          id, plugin_id, connection_id, permission, project_id, environment_id,
          granted, granted_by_json, granted_at, revoked_by_json, revoked_at, reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          plugin_id = excluded.plugin_id,
          connection_id = excluded.connection_id,
          permission = excluded.permission,
          project_id = excluded.project_id,
          environment_id = excluded.environment_id,
          granted = excluded.granted,
          granted_by_json = excluded.granted_by_json,
          granted_at = excluded.granted_at,
          revoked_by_json = excluded.revoked_by_json,
          revoked_at = excluded.revoked_at,
          reason = excluded.reason`,
      )
      .run(
        record.id,
        record.pluginId,
        record.connectionId,
        record.permission,
        record.projectId ?? null,
        record.environmentId ?? null,
        record.granted ? 1 : 0,
        JSON.stringify(record.grantedBy),
        record.grantedAt,
        record.revokedBy ? JSON.stringify(record.revokedBy) : null,
        record.revokedAt ?? null,
        record.reason ?? null,
      );
  }

  async getById(
    id: string,
  ): Promise<PluginPermissionGrantRecord | undefined> {
    const row = this.connection.raw
      .prepare(`SELECT * FROM plugin_permission_grants WHERE id = ?`)
      .get(id) as Record<string, unknown> | undefined;
    return row ? mapGrant(row) : undefined;
  }

  async listByConnectionId(
    connectionId: string,
  ): Promise<PluginPermissionGrantRecord[]> {
    const rows = this.connection.raw
      .prepare(
        `SELECT * FROM plugin_permission_grants WHERE connection_id = ?`,
      )
      .all(connectionId) as Record<string, unknown>[];
    return rows.map(mapGrant);
  }

  async listActiveByConnectionId(
    connectionId: string,
  ): Promise<PluginPermissionGrantRecord[]> {
    const rows = this.connection.raw
      .prepare(
        `SELECT * FROM plugin_permission_grants
         WHERE connection_id = ? AND granted = 1 AND revoked_at IS NULL`,
      )
      .all(connectionId) as Record<string, unknown>[];
    return rows.map(mapGrant);
  }

  async replaceActiveGrants(input: {
    connectionId: string;
    projectId?: string;
    environmentId?: string;
    grants: PluginPermissionGrantRecord[];
  }): Promise<void> {
    const run = this.connection.raw.transaction(() => {
      const now = new Date().toISOString();
      this.connection.raw
        .prepare(
          `UPDATE plugin_permission_grants
           SET granted = 0, revoked_at = ?, reason = COALESCE(reason, 'replaced')
           WHERE connection_id = ?
             AND granted = 1
             AND revoked_at IS NULL
             AND ((project_id IS NULL AND ? IS NULL) OR project_id = ?)
             AND ((environment_id IS NULL AND ? IS NULL) OR environment_id = ?)`,
        )
        .run(
          now,
          input.connectionId,
          input.projectId ?? null,
          input.projectId ?? null,
          input.environmentId ?? null,
          input.environmentId ?? null,
        );

      const insert = this.connection.raw.prepare(
        `INSERT INTO plugin_permission_grants (
          id, plugin_id, connection_id, permission, project_id, environment_id,
          granted, granted_by_json, granted_at, revoked_by_json, revoked_at, reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          plugin_id = excluded.plugin_id,
          connection_id = excluded.connection_id,
          permission = excluded.permission,
          project_id = excluded.project_id,
          environment_id = excluded.environment_id,
          granted = excluded.granted,
          granted_by_json = excluded.granted_by_json,
          granted_at = excluded.granted_at,
          revoked_by_json = excluded.revoked_by_json,
          revoked_at = excluded.revoked_at,
          reason = excluded.reason`,
      );

      for (const grant of input.grants) {
        insert.run(
          grant.id,
          grant.pluginId,
          grant.connectionId,
          grant.permission,
          grant.projectId ?? null,
          grant.environmentId ?? null,
          grant.granted ? 1 : 0,
          JSON.stringify(grant.grantedBy),
          grant.grantedAt,
          grant.revokedBy ? JSON.stringify(grant.revokedBy) : null,
          grant.revokedAt ?? null,
          grant.reason ?? null,
        );
      }
    });
    run();
  }
}

function mapGrant(row: Record<string, unknown>): PluginPermissionGrantRecord {
  return {
    id: String(row.id),
    pluginId: String(row.plugin_id),
    connectionId: String(row.connection_id),
    permission: String(row.permission) as PluginPermission,
    projectId: row.project_id ? String(row.project_id) : undefined,
    environmentId: row.environment_id
      ? String(row.environment_id)
      : undefined,
    granted: Boolean(row.granted),
    grantedBy: parseJson<PluginExecutionActor>(
      String(row.granted_by_json),
      "granted_by_json",
    ),
    grantedAt: String(row.granted_at),
    revokedBy: row.revoked_by_json
      ? parseJson<PluginExecutionActor>(
          String(row.revoked_by_json),
          "revoked_by_json",
        )
      : undefined,
    revokedAt: row.revoked_at ? String(row.revoked_at) : undefined,
    reason: row.reason ? String(row.reason) : undefined,
  };
}

export class SqliteDiscoveredResourceRepository
  implements DiscoveredResourceRepository
{
  constructor(private readonly connection: LocalDatabaseConnection) {}

  async save(record: DiscoveredResourceRecord): Promise<void> {
    this.connection.raw
      .prepare(
        `INSERT INTO plugin_discovered_resources (
          id, plugin_id, installed_plugin_id, connection_id, provider_resource_id,
          resource_type, name, parent_provider_resource_id, metadata_json,
          plugin_version, schema_version, discovery_status, first_discovered_at,
          last_discovered_at, last_inspected_at, missing_since
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          plugin_id = excluded.plugin_id,
          installed_plugin_id = excluded.installed_plugin_id,
          connection_id = excluded.connection_id,
          provider_resource_id = excluded.provider_resource_id,
          resource_type = excluded.resource_type,
          name = excluded.name,
          parent_provider_resource_id = excluded.parent_provider_resource_id,
          metadata_json = excluded.metadata_json,
          plugin_version = excluded.plugin_version,
          schema_version = excluded.schema_version,
          discovery_status = excluded.discovery_status,
          first_discovered_at = excluded.first_discovered_at,
          last_discovered_at = excluded.last_discovered_at,
          last_inspected_at = excluded.last_inspected_at,
          missing_since = excluded.missing_since`,
      )
      .run(
        record.id,
        record.pluginId,
        record.installedPluginId,
        record.connectionId,
        record.providerResourceId,
        record.resourceType,
        record.name,
        record.parentProviderResourceId ?? null,
        JSON.stringify(record.metadata),
        record.pluginVersion,
        record.schemaVersion,
        record.discoveryStatus,
        record.firstDiscoveredAt,
        record.lastDiscoveredAt,
        record.lastInspectedAt ?? null,
        record.missingSince ?? null,
      );
  }

  async getById(id: string): Promise<DiscoveredResourceRecord | undefined> {
    const row = this.connection.raw
      .prepare(`SELECT * FROM plugin_discovered_resources WHERE id = ?`)
      .get(id) as Record<string, unknown> | undefined;
    return row ? mapDiscovered(row) : undefined;
  }

  async getByProviderKey(input: {
    connectionId: string;
    providerResourceId: string;
    resourceType: string;
  }): Promise<DiscoveredResourceRecord | undefined> {
    const row = this.connection.raw
      .prepare(
        `SELECT * FROM plugin_discovered_resources
         WHERE connection_id = ? AND provider_resource_id = ? AND resource_type = ?`,
      )
      .get(
        input.connectionId,
        input.providerResourceId,
        input.resourceType,
      ) as Record<string, unknown> | undefined;
    return row ? mapDiscovered(row) : undefined;
  }

  async listByConnectionId(
    connectionId: string,
  ): Promise<DiscoveredResourceRecord[]> {
    const rows = this.connection.raw
      .prepare(
        `SELECT * FROM plugin_discovered_resources WHERE connection_id = ?`,
      )
      .all(connectionId) as Record<string, unknown>[];
    return rows.map(mapDiscovered);
  }

  async syncDiscovery(input: {
    pluginId: string;
    installedPluginId: string;
    connectionId: string;
    discoveredAt: string;
    items: DiscoverySyncItem[];
  }): Promise<DiscoveredResourceRecord[]> {
    const sync = this.connection.raw.transaction(() => {
      const results: DiscoveredResourceRecord[] = [];
      const seen = new Set<string>();

      for (const item of input.items) {
        const key = `${item.providerResourceId}|${item.resourceType}`;
        if (seen.has(key)) {
          throw new PluginPersistenceError(
            `Duplicate provider resource in discovery batch: ${key}`,
          );
        }
        seen.add(key);

        const existingRow = this.connection.raw
          .prepare(
            `SELECT * FROM plugin_discovered_resources
             WHERE connection_id = ? AND provider_resource_id = ? AND resource_type = ?`,
          )
          .get(
            input.connectionId,
            item.providerResourceId,
            item.resourceType,
          ) as Record<string, unknown> | undefined;

        if (existingRow) {
          const existing = mapDiscovered(existingRow);
          const updated: DiscoveredResourceRecord = {
            ...existing,
            name: item.name,
            parentProviderResourceId: item.parentProviderResourceId,
            metadata: item.metadata,
            pluginVersion: item.pluginVersion,
            schemaVersion: item.schemaVersion,
            discoveryStatus: "active",
            lastDiscoveredAt: input.discoveredAt,
            missingSince: undefined,
          };
          void this.save(updated);
          results.push(updated);
        } else {
          const created: DiscoveredResourceRecord = {
            id: crypto.randomUUID(),
            pluginId: input.pluginId,
            installedPluginId: input.installedPluginId,
            connectionId: input.connectionId,
            providerResourceId: item.providerResourceId,
            resourceType: item.resourceType,
            name: item.name,
            parentProviderResourceId: item.parentProviderResourceId,
            metadata: item.metadata,
            pluginVersion: item.pluginVersion,
            schemaVersion: item.schemaVersion,
            discoveryStatus: "active",
            firstDiscoveredAt: input.discoveredAt,
            lastDiscoveredAt: input.discoveredAt,
          };
          void this.save(created);
          results.push(created);
        }
      }

      const existing = this.connection.raw
        .prepare(
          `SELECT * FROM plugin_discovered_resources WHERE connection_id = ?`,
        )
        .all(input.connectionId) as Record<string, unknown>[];

      for (const row of existing) {
        const record = mapDiscovered(row);
        const key = `${record.providerResourceId}|${record.resourceType}`;
        if (!seen.has(key) && record.discoveryStatus === "active") {
          void this.save({
            ...record,
            discoveryStatus: "missing",
            missingSince: input.discoveredAt,
          });
        }
      }

      return results;
    });

    return sync();
  }

  async markStatus(
    id: string,
    status: DiscoveredResourceRecord["discoveryStatus"],
    missingSince?: string,
  ): Promise<void> {
    const existing = await this.getById(id);
    if (!existing) {
      throw new PluginPersistenceError(`Discovered resource not found: ${id}`);
    }
    await this.save({
      ...existing,
      discoveryStatus: status,
      missingSince:
        status === "missing"
          ? (missingSince ?? new Date().toISOString())
          : undefined,
    });
  }
}

function mapDiscovered(row: Record<string, unknown>): DiscoveredResourceRecord {
  return {
    id: String(row.id),
    pluginId: String(row.plugin_id),
    installedPluginId: String(row.installed_plugin_id),
    connectionId: String(row.connection_id),
    providerResourceId: String(row.provider_resource_id),
    resourceType: String(row.resource_type),
    name: String(row.name),
    parentProviderResourceId: row.parent_provider_resource_id
      ? String(row.parent_provider_resource_id)
      : undefined,
    metadata: parseJson<Record<string, unknown>>(
      String(row.metadata_json),
      "metadata_json",
    ),
    pluginVersion: String(row.plugin_version),
    schemaVersion: String(row.schema_version),
    discoveryStatus:
      row.discovery_status as DiscoveredResourceRecord["discoveryStatus"],
    firstDiscoveredAt: String(row.first_discovered_at),
    lastDiscoveredAt: String(row.last_discovered_at),
    lastInspectedAt: row.last_inspected_at
      ? String(row.last_inspected_at)
      : undefined,
    missingSince: row.missing_since ? String(row.missing_since) : undefined,
  };
}

export class SqlitePluginExecutionHistoryRepository
  implements PluginExecutionHistoryRepository
{
  constructor(private readonly connection: LocalDatabaseConnection) {}

  async append(record: PluginExecutionHistoryRecord): Promise<void> {
    this.connection.raw
      .prepare(
        `INSERT INTO plugin_execution_history (
          id, execution_id, plugin_id, plugin_version, capability, status,
          started_at, finished_at, duration_ms, actor_json, project_id,
          environment_id, resource_id, connection_id, reason, error_code,
          error_message, warning_count, recorded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        record.id,
        record.executionId,
        record.pluginId,
        record.pluginVersion,
        record.capability,
        record.status,
        record.startedAt,
        record.finishedAt,
        record.durationMs,
        JSON.stringify(record.actor),
        record.projectId ?? null,
        record.environmentId ?? null,
        record.resourceId ?? null,
        record.connectionId ?? null,
        record.reason ?? null,
        record.errorCode ?? null,
        record.errorMessage ?? null,
        record.warningCount,
        record.recordedAt,
      );
  }

  async getByExecutionId(
    executionId: string,
  ): Promise<PluginExecutionHistoryRecord | undefined> {
    const row = this.connection.raw
      .prepare(
        `SELECT * FROM plugin_execution_history WHERE execution_id = ?`,
      )
      .get(executionId) as Record<string, unknown> | undefined;
    return row ? mapHistory(row) : undefined;
  }

  async listByPluginId(
    pluginId: string,
  ): Promise<PluginExecutionHistoryRecord[]> {
    const rows = this.connection.raw
      .prepare(
        `SELECT * FROM plugin_execution_history WHERE plugin_id = ? ORDER BY started_at DESC`,
      )
      .all(pluginId) as Record<string, unknown>[];
    return rows.map(mapHistory);
  }

  async listByConnectionId(
    connectionId: string,
  ): Promise<PluginExecutionHistoryRecord[]> {
    const rows = this.connection.raw
      .prepare(
        `SELECT * FROM plugin_execution_history WHERE connection_id = ? ORDER BY started_at DESC`,
      )
      .all(connectionId) as Record<string, unknown>[];
    return rows.map(mapHistory);
  }
}

function mapHistory(row: Record<string, unknown>): PluginExecutionHistoryRecord {
  return {
    id: String(row.id),
    executionId: String(row.execution_id),
    pluginId: String(row.plugin_id),
    pluginVersion: String(row.plugin_version),
    capability: row.capability as PluginExecutionHistoryRecord["capability"],
    status: row.status as PluginExecutionHistoryRecord["status"],
    startedAt: String(row.started_at),
    finishedAt: String(row.finished_at),
    durationMs: Number(row.duration_ms),
    actor: parseJson<PluginExecutionActor>(
      String(row.actor_json),
      "actor_json",
    ),
    projectId: row.project_id ? String(row.project_id) : undefined,
    environmentId: row.environment_id
      ? String(row.environment_id)
      : undefined,
    resourceId: row.resource_id ? String(row.resource_id) : undefined,
    connectionId: row.connection_id ? String(row.connection_id) : undefined,
    reason: row.reason ? String(row.reason) : undefined,
    errorCode: row.error_code
      ? (String(row.error_code) as PluginExecutionHistoryRecord["errorCode"])
      : undefined,
    errorMessage: row.error_message ? String(row.error_message) : undefined,
    warningCount: Number(row.warning_count),
    recordedAt: String(row.recorded_at),
  };
}
