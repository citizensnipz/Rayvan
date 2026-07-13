use tauri::AppHandle;

pub fn initialize_app_state(_app: &AppHandle) {
    tracing::info!("Rayvan desktop state initialized");
}
