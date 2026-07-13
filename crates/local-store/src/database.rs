use std::path::{Path, PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LocalStoreError {
    #[error("database path is unavailable")]
    PathUnavailable,
    #[error("database is not initialized")]
    NotInitialized,
}

#[derive(Debug, Clone)]
pub struct LocalDatabaseConfig {
    pub path: PathBuf,
}

pub struct LocalDatabase {
    config: LocalDatabaseConfig,
    initialized: bool,
}

impl LocalDatabase {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            config: LocalDatabaseConfig {
                path: path.as_ref().to_path_buf(),
            },
            initialized: false,
        }
    }

    pub fn initialize(&mut self) -> Result<(), LocalStoreError> {
        self.initialized = true;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.config.path
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initializes_local_database_placeholder() {
        let mut database = LocalDatabase::new("rayvan.db");
        assert!(!database.is_initialized());
        database.initialize().expect("initialize");
        assert!(database.is_initialized());
    }
}
