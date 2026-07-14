use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum PluginStoreError {
    #[error("installed plugin not found: {0}")]
    NotFound(String),
    #[error("persistence failure: {0}")]
    Persistence(String),
}

impl From<rusqlite::Error> for PluginStoreError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Persistence(error.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct InstalledPlugin {
    pub id: String,
    pub plugin_id: String,
    pub plugin_version: String,
    pub manifest_version: String,
    pub rayvan_api_version: String,
    pub source_json: String,
    pub status: String,
    pub enabled: bool,
    pub installed_at: String,
    pub updated_at: String,
    pub last_loaded_at: Option<String>,
    pub manifest_snapshot_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PluginConnection {
    pub id: String,
    pub installed_plugin_id: String,
    pub plugin_id: String,
    pub project_id: Option<String>,
    pub name: String,
    pub status: String,
    pub credential_reference_id: Option<String>,
    pub metadata_json: String,
    pub schema_version: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PluginPermissionGrant {
    pub id: String,
    pub plugin_id: String,
    pub connection_id: String,
    pub permission: String,
    pub project_id: Option<String>,
    pub environment_id: Option<String>,
    pub granted: bool,
    pub granted_by_json: String,
    pub granted_at: String,
    pub revoked_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PluginExecutionHistory {
    pub id: String,
    pub execution_id: String,
    pub plugin_id: String,
    pub plugin_version: String,
    pub capability: String,
    pub status: String,
    pub started_at: String,
    pub finished_at: String,
    pub duration_ms: i64,
    pub actor_json: String,
    pub connection_id: Option<String>,
    pub warning_count: i64,
    pub recorded_at: String,
}

pub struct PluginRepository<'conn> {
    connection: &'conn Connection,
}

impl<'conn> PluginRepository<'conn> {
    pub fn new(connection: &'conn Connection) -> Self {
        Self { connection }
    }

    pub fn save_installed_plugin(
        &self,
        record: &InstalledPlugin,
    ) -> Result<(), PluginStoreError> {
        self.connection.execute(
            "INSERT INTO plugin_installed (
                id, plugin_id, plugin_version, manifest_version, rayvan_api_version,
                source_json, status, enabled, installed_at, updated_at, last_loaded_at,
                manifest_snapshot_json
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
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
                manifest_snapshot_json = excluded.manifest_snapshot_json",
            params![
                record.id,
                record.plugin_id,
                record.plugin_version,
                record.manifest_version,
                record.rayvan_api_version,
                record.source_json,
                record.status,
                if record.enabled { 1 } else { 0 },
                record.installed_at,
                record.updated_at,
                record.last_loaded_at,
                record.manifest_snapshot_json,
            ],
        )?;
        Ok(())
    }

    pub fn get_installed_by_plugin_id(
        &self,
        plugin_id: &str,
    ) -> Result<Option<InstalledPlugin>, PluginStoreError> {
        let mut statement = self.connection.prepare(
            "SELECT id, plugin_id, plugin_version, manifest_version, rayvan_api_version,
                    source_json, status, enabled, installed_at, updated_at, last_loaded_at,
                    manifest_snapshot_json
             FROM plugin_installed WHERE plugin_id = ?1",
        )?;
        let record = statement
            .query_row(params![plugin_id], map_installed)
            .optional()?;
        Ok(record)
    }

    pub fn list_installed(&self) -> Result<Vec<InstalledPlugin>, PluginStoreError> {
        let mut statement = self.connection.prepare(
            "SELECT id, plugin_id, plugin_version, manifest_version, rayvan_api_version,
                    source_json, status, enabled, installed_at, updated_at, last_loaded_at,
                    manifest_snapshot_json
             FROM plugin_installed ORDER BY plugin_id",
        )?;
        let rows = statement
            .query_map([], map_installed)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn save_connection(&self, record: &PluginConnection) -> Result<(), PluginStoreError> {
        self.connection.execute(
            "INSERT INTO plugin_connections (
                id, installed_plugin_id, plugin_id, project_id, name, status,
                credential_reference_id, metadata_json, schema_version, created_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(id) DO UPDATE SET
                installed_plugin_id = excluded.installed_plugin_id,
                plugin_id = excluded.plugin_id,
                project_id = excluded.project_id,
                name = excluded.name,
                status = excluded.status,
                credential_reference_id = excluded.credential_reference_id,
                metadata_json = excluded.metadata_json,
                schema_version = excluded.schema_version,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at",
            params![
                record.id,
                record.installed_plugin_id,
                record.plugin_id,
                record.project_id,
                record.name,
                record.status,
                record.credential_reference_id,
                record.metadata_json,
                record.schema_version,
                record.created_at,
                record.updated_at,
            ],
        )?;
        Ok(())
    }

    pub fn get_connection(
        &self,
        id: &str,
    ) -> Result<Option<PluginConnection>, PluginStoreError> {
        let mut statement = self.connection.prepare(
            "SELECT id, installed_plugin_id, plugin_id, project_id, name, status,
                    credential_reference_id, metadata_json, schema_version, created_at, updated_at
             FROM plugin_connections WHERE id = ?1",
        )?;
        let record = statement
            .query_row(params![id], map_connection)
            .optional()?;
        Ok(record)
    }

    pub fn list_connections_by_plugin(
        &self,
        plugin_id: &str,
    ) -> Result<Vec<PluginConnection>, PluginStoreError> {
        let mut statement = self.connection.prepare(
            "SELECT id, installed_plugin_id, plugin_id, project_id, name, status,
                    credential_reference_id, metadata_json, schema_version, created_at, updated_at
             FROM plugin_connections WHERE plugin_id = ?1",
        )?;
        let rows = statement
            .query_map(params![plugin_id], map_connection)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn save_permission_grant(
        &self,
        record: &PluginPermissionGrant,
    ) -> Result<(), PluginStoreError> {
        self.connection.execute(
            "INSERT INTO plugin_permission_grants (
                id, plugin_id, connection_id, permission, project_id, environment_id,
                granted, granted_by_json, granted_at, revoked_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(id) DO UPDATE SET
                granted = excluded.granted,
                revoked_at = excluded.revoked_at",
            params![
                record.id,
                record.plugin_id,
                record.connection_id,
                record.permission,
                record.project_id,
                record.environment_id,
                if record.granted { 1 } else { 0 },
                record.granted_by_json,
                record.granted_at,
                record.revoked_at,
            ],
        )?;
        Ok(())
    }

    pub fn list_active_grants(
        &self,
        connection_id: &str,
    ) -> Result<Vec<PluginPermissionGrant>, PluginStoreError> {
        let mut statement = self.connection.prepare(
            "SELECT id, plugin_id, connection_id, permission, project_id, environment_id,
                    granted, granted_by_json, granted_at, revoked_at
             FROM plugin_permission_grants
             WHERE connection_id = ?1 AND granted = 1 AND revoked_at IS NULL",
        )?;
        let rows = statement
            .query_map(params![connection_id], map_grant)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn append_execution_history(
        &self,
        record: &PluginExecutionHistory,
    ) -> Result<(), PluginStoreError> {
        self.connection.execute(
            "INSERT INTO plugin_execution_history (
                id, execution_id, plugin_id, plugin_version, capability, status,
                started_at, finished_at, duration_ms, actor_json, connection_id,
                warning_count, recorded_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                record.id,
                record.execution_id,
                record.plugin_id,
                record.plugin_version,
                record.capability,
                record.status,
                record.started_at,
                record.finished_at,
                record.duration_ms,
                record.actor_json,
                record.connection_id,
                record.warning_count,
                record.recorded_at,
            ],
        )?;
        Ok(())
    }

    pub fn get_execution_by_id(
        &self,
        execution_id: &str,
    ) -> Result<Option<PluginExecutionHistory>, PluginStoreError> {
        let mut statement = self.connection.prepare(
            "SELECT id, execution_id, plugin_id, plugin_version, capability, status,
                    started_at, finished_at, duration_ms, actor_json, connection_id,
                    warning_count, recorded_at
             FROM plugin_execution_history WHERE execution_id = ?1",
        )?;
        let record = statement
            .query_row(params![execution_id], map_history)
            .optional()?;
        Ok(record)
    }

    pub fn create_installed_builtin(
        &self,
        plugin_id: &str,
        version: &str,
        manifest_json: &str,
    ) -> Result<InstalledPlugin, PluginStoreError> {
        let now = Utc::now().to_rfc3339();
        let record = InstalledPlugin {
            id: Uuid::new_v4().to_string(),
            plugin_id: plugin_id.to_string(),
            plugin_version: version.to_string(),
            manifest_version: version.to_string(),
            rayvan_api_version: "1".to_string(),
            source_json: r#"{"type":"built_in"}"#.to_string(),
            status: "installed".to_string(),
            enabled: true,
            installed_at: now.clone(),
            updated_at: now,
            last_loaded_at: None,
            manifest_snapshot_json: manifest_json.to_string(),
        };
        self.save_installed_plugin(&record)?;
        Ok(record)
    }
}

fn map_installed(row: &rusqlite::Row<'_>) -> rusqlite::Result<InstalledPlugin> {
    let enabled: i64 = row.get(7)?;
    Ok(InstalledPlugin {
        id: row.get(0)?,
        plugin_id: row.get(1)?,
        plugin_version: row.get(2)?,
        manifest_version: row.get(3)?,
        rayvan_api_version: row.get(4)?,
        source_json: row.get(5)?,
        status: row.get(6)?,
        enabled: enabled != 0,
        installed_at: row.get(8)?,
        updated_at: row.get(9)?,
        last_loaded_at: row.get(10)?,
        manifest_snapshot_json: row.get(11)?,
    })
}

fn map_connection(row: &rusqlite::Row<'_>) -> rusqlite::Result<PluginConnection> {
    Ok(PluginConnection {
        id: row.get(0)?,
        installed_plugin_id: row.get(1)?,
        plugin_id: row.get(2)?,
        project_id: row.get(3)?,
        name: row.get(4)?,
        status: row.get(5)?,
        credential_reference_id: row.get(6)?,
        metadata_json: row.get(7)?,
        schema_version: row.get(8)?,
        created_at: row.get(9)?,
        updated_at: row.get(10)?,
    })
}

fn map_grant(row: &rusqlite::Row<'_>) -> rusqlite::Result<PluginPermissionGrant> {
    let granted: i64 = row.get(6)?;
    Ok(PluginPermissionGrant {
        id: row.get(0)?,
        plugin_id: row.get(1)?,
        connection_id: row.get(2)?,
        permission: row.get(3)?,
        project_id: row.get(4)?,
        environment_id: row.get(5)?,
        granted: granted != 0,
        granted_by_json: row.get(7)?,
        granted_at: row.get(8)?,
        revoked_at: row.get(9)?,
    })
}

fn map_history(row: &rusqlite::Row<'_>) -> rusqlite::Result<PluginExecutionHistory> {
    Ok(PluginExecutionHistory {
        id: row.get(0)?,
        execution_id: row.get(1)?,
        plugin_id: row.get(2)?,
        plugin_version: row.get(3)?,
        capability: row.get(4)?,
        status: row.get(5)?,
        started_at: row.get(6)?,
        finished_at: row.get(7)?,
        duration_ms: row.get(8)?,
        actor_json: row.get(9)?,
        connection_id: row.get(10)?,
        warning_count: row.get(11)?,
        recorded_at: row.get(12)?,
    })
}
