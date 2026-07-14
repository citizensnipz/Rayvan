use std::sync::Arc;

use rayvan_local_store::LocalDatabase;
use tauri::AppHandle;

pub struct AppState {
    pub database: Arc<LocalDatabase>,
}

pub fn initialize_app_state(app: &AppHandle) -> AppState {
    let data_dir = rayvan_native_core::paths::app_data_dir(app)
        .expect("application data directory should be available");
    std::fs::create_dir_all(&data_dir).expect("application data directory should be writable");

    let database_path = data_dir.join("rayvan.db");
    let database = Arc::new(LocalDatabase::new(database_path));
    database
        .initialize()
        .expect("local database should initialize on startup");

    tracing::info!(
        path = %database.path().display(),
        "Rayvan desktop state initialized"
    );

    AppState { database }
}
