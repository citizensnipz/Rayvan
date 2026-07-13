use serde::Serialize;

#[derive(Serialize)]
pub struct AppInfo {
    pub name: String,
    pub version: String,
    pub data_dir: String,
}

#[tauri::command]
pub fn greet(name: &str) -> String {
    format!("Hello, {name}! Welcome to Rayvan.")
}

#[tauri::command]
pub fn get_app_info(app: tauri::AppHandle) -> Result<AppInfo, String> {
    let data_dir = rayvan_native_core::paths::app_data_dir(&app)
        .map_err(|error| error.to_string())?
        .display()
        .to_string();

    Ok(AppInfo {
        name: "Rayvan".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        data_dir,
    })
}
