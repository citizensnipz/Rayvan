use tauri::State;

use crate::state::AppState;

use super::daemon::CommandError;

#[tauri::command]
pub fn get_current_project_id(
    state: State<'_, AppState>,
) -> Result<Option<String>, CommandError> {
    let prefs = state.prefs.lock().unwrap();
    Ok(prefs.current_project_id.clone())
}

#[tauri::command]
pub fn set_current_project_id(
    project_id: Option<String>,
    state: State<'_, AppState>,
) -> Result<(), CommandError> {
    let mut prefs = state.prefs.lock().unwrap();
    prefs
        .set_current_project_id(project_id)
        .map_err(|message| CommandError {
            code: "INTERNAL_ERROR".into(),
            message,
            id: None,
            data: None,
        })
}
