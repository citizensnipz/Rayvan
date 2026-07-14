use rayvan_local_store::{
    CreateProjectInput, Project, ProjectError, UpdateProjectInput,
};
use tauri::State;

use crate::state::AppState;

#[derive(serde::Serialize)]
pub struct CommandError {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

impl From<ProjectError> for CommandError {
    fn from(error: ProjectError) -> Self {
        match &error {
            ProjectError::NotFound(id) => Self {
                code: error.code().to_string(),
                message: error.to_string(),
                id: Some(id.clone()),
            },
            _ => Self {
                code: error.code().to_string(),
                message: error.to_string(),
                id: None,
            },
        }
    }
}

impl From<rayvan_local_store::LocalStoreError> for CommandError {
    fn from(error: rayvan_local_store::LocalStoreError) -> Self {
        match error {
            rayvan_local_store::LocalStoreError::Project(project_error) => {
                CommandError::from(project_error)
            }
            other => Self {
                code: "internal".to_string(),
                message: other.to_string(),
                id: None,
            },
        }
    }
}

#[tauri::command]
pub fn list_projects(
    include_archived: Option<bool>,
    state: State<'_, AppState>,
) -> Result<Vec<Project>, CommandError> {
    state
        .database
        .with_repository(|repository| repository.list(include_archived.unwrap_or(false)))
        .map_err(CommandError::from)
}

#[tauri::command]
pub fn get_project(id: String, state: State<'_, AppState>) -> Result<Option<Project>, CommandError> {
    state
        .database
        .with_repository(|repository| repository.get_by_id(&id))
        .map_err(CommandError::from)
}

#[tauri::command]
pub fn create_project(
    name: String,
    description: Option<String>,
    state: State<'_, AppState>,
) -> Result<Project, CommandError> {
    state
        .database
        .mutate_repository(|repository| {
            repository.create(CreateProjectInput { name, description })
        })
        .map_err(CommandError::from)
}

#[tauri::command]
pub fn update_project(
    id: String,
    name: Option<String>,
    description: Option<String>,
    state: State<'_, AppState>,
) -> Result<Project, CommandError> {
    state
        .database
        .mutate_repository(|repository| {
            repository.update(&id, UpdateProjectInput { name, description })
        })
        .map_err(CommandError::from)
}

#[tauri::command]
pub fn set_project_archived(
    id: String,
    archived: bool,
    state: State<'_, AppState>,
) -> Result<Project, CommandError> {
    state
        .database
        .mutate_repository(|repository| repository.set_archived(&id, archived))
        .map_err(CommandError::from)
}

#[tauri::command]
pub fn delete_project(id: String, state: State<'_, AppState>) -> Result<(), CommandError> {
    state
        .database
        .mutate_repository(|repository| repository.delete(&id))
        .map_err(CommandError::from)
}
