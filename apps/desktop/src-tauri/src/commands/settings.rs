use tauri::State;

use crate::state::AppState;

use super::projects::CommandError;

#[tauri::command]
pub fn get_current_project_id(
    state: State<'_, AppState>,
) -> Result<Option<String>, CommandError> {
    state
        .database
        .with_settings(|settings| settings.get_current_project_id())
        .map_err(CommandError::from)
}

#[tauri::command]
pub fn set_current_project_id(
    project_id: Option<String>,
    state: State<'_, AppState>,
) -> Result<(), CommandError> {
    state
        .database
        .mutate_settings(|settings| settings.set_current_project_id(project_id.as_deref()))
        .map_err(CommandError::from)
}
