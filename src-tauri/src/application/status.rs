use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum NetworkStatus {
    NotConnected,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApplicationStatus {
    network_status: NetworkStatus,
}

impl Default for ApplicationStatus {
    fn default() -> Self {
        Self {
            network_status: NetworkStatus::NotConnected,
        }
    }
}

#[tauri::command]
pub fn get_application_status() -> ApplicationStatus {
    ApplicationStatus::default()
}
