use serde::Serialize;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum NetworkStatus {
    NotConnected,
    Connecting,
    Connected,
}

#[derive(Debug, Clone)]
pub(crate) struct NetworkStatusStore {
    status: Arc<RwLock<NetworkStatus>>,
}

impl NetworkStatusStore {
    pub(crate) fn get(&self) -> NetworkStatus {
        *self
            .status
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(crate) fn set(&self, status: NetworkStatus) {
        *self
            .status
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = status;
    }
}

impl Default for NetworkStatusStore {
    fn default() -> Self {
        Self {
            status: Arc::new(RwLock::new(NetworkStatus::NotConnected)),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApplicationStatus {
    network_status: NetworkStatus,
}

#[tauri::command]
pub fn get_application_status(
    network_status: tauri::State<'_, NetworkStatusStore>,
) -> ApplicationStatus {
    ApplicationStatus {
        network_status: network_status.get(),
    }
}
