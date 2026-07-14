use rusqlite::{params, Connection, OptionalExtension};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SettingsError {
    #[error("persistence failure: {0}")]
    Persistence(String),
}

impl From<rusqlite::Error> for SettingsError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Persistence(error.to_string())
    }
}

pub const CURRENT_PROJECT_ID_KEY: &str = "current_project_id";

pub struct SettingsRepository<'conn> {
    connection: &'conn Connection,
}

impl<'conn> SettingsRepository<'conn> {
    pub fn new(connection: &'conn Connection) -> Self {
        Self { connection }
    }

    pub fn get(&self, key: &str) -> Result<Option<String>, SettingsError> {
        let mut statement = self
            .connection
            .prepare("SELECT value FROM app_settings WHERE key = ?1")?;
        let value = statement
            .query_row(params![key], |row| row.get::<_, String>(0))
            .optional()?;
        Ok(value)
    }

    pub fn set(&self, key: &str, value: &str) -> Result<(), SettingsError> {
        self.connection.execute(
            "INSERT INTO app_settings (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    pub fn delete(&self, key: &str) -> Result<(), SettingsError> {
        self.connection
            .execute("DELETE FROM app_settings WHERE key = ?1", params![key])?;
        Ok(())
    }

    pub fn get_current_project_id(&self) -> Result<Option<String>, SettingsError> {
        self.get(CURRENT_PROJECT_ID_KEY)
    }

    pub fn set_current_project_id(&self, project_id: Option<&str>) -> Result<(), SettingsError> {
        match project_id {
            Some(id) => self.set(CURRENT_PROJECT_ID_KEY, id),
            None => self.delete(CURRENT_PROJECT_ID_KEY),
        }
    }
}
