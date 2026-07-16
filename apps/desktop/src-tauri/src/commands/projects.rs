use serde_json::{json, Value};
use tauri::State;

use crate::state::AppState;

use super::daemon::CommandError;

fn map_project_error(error: CommandError) -> CommandError {
    // Normalize common daemon codes for the existing TS repository.
    let code = match error.code.as_str() {
        "NOT_FOUND" => "not_found".to_string(),
        "VALIDATION_FAILED" => "validation_failed".to_string(),
        other => other.to_string(),
    };
    CommandError {
        code,
        message: error.message,
        id: error.id,
        data: error.data,
    }
}

fn request(state: &AppState, method: &str, params: Value) -> Result<Value, CommandError> {
    crate::state::daemon_request(state, method, params).map_err(|error| {
        map_project_error(CommandError::from(error))
    })
}

#[tauri::command]
pub fn list_projects(
    include_archived: Option<bool>,
    state: State<'_, AppState>,
) -> Result<Value, CommandError> {
    request(
        &state,
        "projects.list",
        json!({ "includeArchived": include_archived.unwrap_or(false) }),
    )
}

#[tauri::command]
pub fn get_project(id: String, state: State<'_, AppState>) -> Result<Value, CommandError> {
    request(&state, "projects.get", json!({ "projectId": id }))
}

#[tauri::command]
pub fn create_project(
    name: String,
    description: Option<String>,
    state: State<'_, AppState>,
) -> Result<Value, CommandError> {
    request(
        &state,
        "projects.create",
        json!({ "name": name, "description": description }),
    )
}

#[tauri::command]
pub fn update_project(
    id: String,
    name: Option<String>,
    description: Option<String>,
    state: State<'_, AppState>,
) -> Result<Value, CommandError> {
    request(
        &state,
        "projects.update",
        json!({
            "projectId": id,
            "name": name,
            "description": description,
        }),
    )
}

#[tauri::command]
pub fn set_project_archived(
    id: String,
    archived: bool,
    state: State<'_, AppState>,
) -> Result<Value, CommandError> {
    request(
        &state,
        "projects.update",
        json!({
            "projectId": id,
            "archived": archived,
        }),
    )
}

#[tauri::command]
pub fn delete_project(id: String, state: State<'_, AppState>) -> Result<(), CommandError> {
    // Daemon does not expose project delete yet — surface a clear error.
    let _ = id;
    let _ = state;
    Err(CommandError {
        code: "METHOD_NOT_FOUND".into(),
        message: "Project delete is not available through the daemon yet".into(),
        id: None,
        data: None,
    })
}
