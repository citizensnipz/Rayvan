use rayvan_daemon_ipc::{DaemonConnectionError, DaemonStatusSnapshot};
use serde::Serialize;
use serde_json::Value;
use tauri::{AppHandle, State};

use crate::state::{self, AppState};

#[derive(Serialize)]
pub struct CommandError {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl From<DaemonConnectionError> for CommandError {
    fn from(error: DaemonConnectionError) -> Self {
        match error {
            DaemonConnectionError::Rpc { code, message, data } => Self {
                code,
                message,
                id: None,
                data,
            },
            DaemonConnectionError::Unavailable(message) => Self {
                code: "DAEMON_UNAVAILABLE".into(),
                message,
                id: None,
                data: None,
            },
            DaemonConnectionError::Unauthenticated(message) => Self {
                code: "UNAUTHENTICATED".into(),
                message,
                id: None,
                data: None,
            },
            DaemonConnectionError::Timeout(method) => Self {
                code: "DAEMON_UNAVAILABLE".into(),
                message: format!("Timed out waiting for daemon response to {method}"),
                id: None,
                data: None,
            },
            other => Self {
                code: "INTERNAL_ERROR".into(),
                message: other.to_string(),
                id: None,
                data: None,
            },
        }
    }
}

#[tauri::command]
pub fn daemon_status(state: State<'_, AppState>) -> Result<DaemonStatusSnapshot, CommandError> {
    let mut snapshot = {
        let session = state.daemon.lock().unwrap();
        session.status_snapshot()
    };

    if snapshot.connected {
        match state::daemon_request(&state, "system.status", Value::Object(Default::default())) {
            Ok(status) => snapshot.status = Some(status),
            Err(error) => {
                snapshot.last_error = Some(error.to_string());
            }
        }
    }

    Ok(snapshot)
}

#[tauri::command]
pub fn daemon_request(
    method: String,
    params: Option<Value>,
    state: State<'_, AppState>,
) -> Result<Value, CommandError> {
    state::daemon_request(
        &state,
        &method,
        params.unwrap_or_else(|| Value::Object(Default::default())),
    )
    .map_err(CommandError::from)
}

#[tauri::command]
pub fn daemon_reconnect(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<DaemonStatusSnapshot, CommandError> {
    state::reconnect(&app, &state).map_err(CommandError::from)
}
