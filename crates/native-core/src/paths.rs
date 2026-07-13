use std::path::{Path, PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PathError {
    #[error("unable to resolve application data directory")]
    AppDataUnavailable,
}

pub fn normalize_path(path: &Path) -> PathBuf {
    path.to_path_buf()
}

#[cfg(feature = "tauri")]
pub fn app_data_dir(app: &tauri::AppHandle) -> Result<PathBuf, PathError> {
    use tauri::Manager;

    app.path()
        .app_data_dir()
        .map_err(|_| PathError::AppDataUnavailable)
}

#[cfg(not(feature = "tauri"))]
pub fn app_data_dir_placeholder() -> Result<PathBuf, PathError> {
    Err(PathError::AppDataUnavailable)
}
