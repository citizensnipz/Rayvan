use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rusqlite::{params, Connection};
use thiserror::Error;

use crate::migrations::MIGRATIONS;
use crate::plugins::{PluginRepository, PluginStoreError};
use crate::projects::{ProjectError, ProjectRepository};
use crate::settings::{SettingsError, SettingsRepository};

#[derive(Debug, Error)]
pub enum LocalStoreError {
    #[error("database path is unavailable")]
    PathUnavailable,
    #[error("database is not initialized")]
    NotInitialized,
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),
    #[error("project error: {0}")]
    Project(#[from] ProjectError),
    #[error("settings error: {0}")]
    Settings(#[from] SettingsError),
    #[error("plugin store error: {0}")]
    Plugin(#[from] PluginStoreError),
}

#[derive(Debug, Clone)]
pub struct LocalDatabaseConfig {
    pub path: PathBuf,
}

pub struct LocalDatabase {
    config: LocalDatabaseConfig,
    connection: Mutex<Option<Connection>>,
}

impl LocalDatabase {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            config: LocalDatabaseConfig {
                path: path.as_ref().to_path_buf(),
            },
            connection: Mutex::new(None),
        }
    }

    pub fn initialize(&self) -> Result<(), LocalStoreError> {
        if let Some(parent) = self.config.path.parent() {
            std::fs::create_dir_all(parent).map_err(|_| LocalStoreError::PathUnavailable)?;
        }

        let connection = Connection::open(&self.config.path)?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        connection.pragma_update(None, "synchronous", "NORMAL")?;
        Self::run_migrations(&connection)?;

        let mut guard = self
            .connection
            .lock()
            .map_err(|_| LocalStoreError::NotInitialized)?;
        *guard = Some(connection);
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.config.path
    }

    pub fn is_initialized(&self) -> bool {
        self.connection
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|_| true))
            .unwrap_or(false)
    }

    pub fn with_repository<T>(
        &self,
        operation: impl FnOnce(&ProjectRepository) -> Result<T, ProjectError>,
    ) -> Result<T, LocalStoreError> {
        let guard = self
            .connection
            .lock()
            .map_err(|_| LocalStoreError::NotInitialized)?;
        let connection = guard.as_ref().ok_or(LocalStoreError::NotInitialized)?;
        let repository = ProjectRepository::new(connection);
        operation(&repository).map_err(LocalStoreError::Project)
    }

    pub fn mutate_repository<T>(
        &self,
        operation: impl FnOnce(&ProjectRepository) -> Result<T, ProjectError>,
    ) -> Result<T, LocalStoreError> {
        let result = self.with_repository(operation)?;
        self.checkpoint()?;
        Ok(result)
    }

    pub fn with_settings<T>(
        &self,
        operation: impl FnOnce(&SettingsRepository) -> Result<T, SettingsError>,
    ) -> Result<T, LocalStoreError> {
        let guard = self
            .connection
            .lock()
            .map_err(|_| LocalStoreError::NotInitialized)?;
        let connection = guard.as_ref().ok_or(LocalStoreError::NotInitialized)?;
        let repository = SettingsRepository::new(connection);
        operation(&repository).map_err(LocalStoreError::Settings)
    }

    pub fn mutate_settings<T>(
        &self,
        operation: impl FnOnce(&SettingsRepository) -> Result<T, SettingsError>,
    ) -> Result<T, LocalStoreError> {
        let result = self.with_settings(operation)?;
        self.checkpoint()?;
        Ok(result)
    }

    pub fn with_plugins<T>(
        &self,
        operation: impl FnOnce(&PluginRepository) -> Result<T, PluginStoreError>,
    ) -> Result<T, LocalStoreError> {
        let guard = self
            .connection
            .lock()
            .map_err(|_| LocalStoreError::NotInitialized)?;
        let connection = guard.as_ref().ok_or(LocalStoreError::NotInitialized)?;
        let repository = PluginRepository::new(connection);
        operation(&repository).map_err(LocalStoreError::Plugin)
    }

    pub fn mutate_plugins<T>(
        &self,
        operation: impl FnOnce(&PluginRepository) -> Result<T, PluginStoreError>,
    ) -> Result<T, LocalStoreError> {
        let result = self.with_plugins(operation)?;
        self.checkpoint()?;
        Ok(result)
    }

    fn checkpoint(&self) -> Result<(), LocalStoreError> {
        let guard = self
            .connection
            .lock()
            .map_err(|_| LocalStoreError::NotInitialized)?;
        let connection = guard.as_ref().ok_or(LocalStoreError::NotInitialized)?;
        connection
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
            .map_err(LocalStoreError::Database)?;
        Ok(())
    }

    fn run_migrations(connection: &Connection) -> Result<(), rusqlite::Error> {
        let current_version: u32 = connection
            .query_row(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'",
                [],
                |row| row.get::<_, String>(0),
            )
            .ok()
            .and_then(|_| {
                connection
                    .query_row(
                        "SELECT COALESCE(MAX(version), 0) FROM schema_migrations",
                        [],
                        |row| row.get::<_, u32>(0),
                    )
                    .ok()
            })
            .unwrap_or(0);

        for migration in MIGRATIONS {
            if migration.version <= current_version {
                continue;
            }
            connection.execute_batch(migration.sql)?;
            connection.execute(
                "INSERT OR REPLACE INTO schema_migrations (version, applied_at) VALUES (?1, ?2)",
                params![migration.version, chrono::Utc::now().to_rfc3339()],
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projects::CreateProjectInput;

    #[test]
    fn initializes_local_database_and_persists_projects() {
        let temp_dir = std::env::temp_dir().join(format!(
            "rayvan-local-store-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let db_path = temp_dir.join("rayvan.db");
        let database = LocalDatabase::new(&db_path);
        database.initialize().expect("initialize");

        let created = database
            .mutate_repository(|repository| {
                repository.create(CreateProjectInput {
                    name: "Native".to_string(),
                    description: Some("From Rust".to_string()),
                })
            })
            .expect("create project");

        database
            .mutate_settings(|settings| settings.set_current_project_id(Some(&created.id)))
            .expect("save preference");

        drop(database);

        let reopened = LocalDatabase::new(&db_path);
        reopened.initialize().expect("reinitialize");
        let loaded = reopened
            .with_repository(|repository| repository.get_by_id(&created.id))
            .expect("load project")
            .expect("project exists");
        let restored_id = reopened
            .with_settings(|settings| settings.get_current_project_id())
            .expect("load preference");

        assert_eq!(loaded.name, "Native");
        assert_eq!(restored_id.as_deref(), Some(created.id.as_str()));
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn initializes_plugin_tables_and_persists_installed_plugin() {
        let temp_dir = std::env::temp_dir().join(format!(
            "rayvan-local-store-plugins-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let db_path = temp_dir.join("rayvan.db");
        let database = LocalDatabase::new(&db_path);
        database.initialize().expect("initialize");

        let installed = database
            .mutate_plugins(|repository| {
                repository.create_installed_builtin(
                    "example.local",
                    "0.0.1",
                    r#"{"id":"example.local","name":"Example","version":"0.0.1","publisher":"rayvan","rayvanApiVersion":"1","capabilities":[],"permissions":[],"resourceTypes":[]}"#,
                )
            })
            .expect("create installed plugin");

        let loaded = database
            .with_plugins(|repository| repository.get_installed_by_plugin_id("example.local"))
            .expect("load")
            .expect("exists");

        assert_eq!(loaded.plugin_id, installed.plugin_id);
        assert_eq!(loaded.status, "installed");
        assert!(loaded.enabled);

        let version: u32 = {
            let guard = database
                .connection
                .lock()
                .expect("lock");
            let connection = guard.as_ref().expect("conn");
            connection
                .query_row(
                    "SELECT COALESCE(MAX(version), 0) FROM schema_migrations",
                    [],
                    |row| row.get(0),
                )
                .expect("version")
        };
        assert_eq!(version, 3);

        std::fs::remove_dir_all(temp_dir).ok();
    }
}
