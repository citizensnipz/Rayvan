use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rusqlite::{params, Connection};
use thiserror::Error;

use crate::migrations::INITIAL_MIGRATION;
use crate::projects::{ProjectError, ProjectRepository};

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

        if current_version < INITIAL_MIGRATION.version {
            connection.execute_batch(INITIAL_MIGRATION.sql)?;
            connection.execute(
                "INSERT OR REPLACE INTO schema_migrations (version, applied_at) VALUES (?1, ?2)",
                params![
                    INITIAL_MIGRATION.version,
                    chrono::Utc::now().to_rfc3339()
                ],
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
            .with_repository(|repository| {
                repository.create(CreateProjectInput {
                    name: "Native".to_string(),
                    description: Some("From Rust".to_string()),
                })
            })
            .expect("create project");

        drop(database);

        let reopened = LocalDatabase::new(&db_path);
        reopened.initialize().expect("reinitialize");
        let loaded = reopened
            .with_repository(|repository| repository.get_by_id(&created.id))
            .expect("load project")
            .expect("project exists");

        assert_eq!(loaded.name, "Native");
        std::fs::remove_dir_all(temp_dir).ok();
    }
}
