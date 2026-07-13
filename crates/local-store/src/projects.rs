use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ProjectStatus {
    Active,
    Archived,
}

impl ProjectStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Archived => "archived",
        }
    }

    fn from_db(value: &str) -> Result<Self, ProjectError> {
        match value {
            "active" => Ok(Self::Active),
            "archived" => Ok(Self::Archived),
            _ => Err(ProjectError::Persistence(format!(
                "invalid project status: {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Project {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub status: ProjectStatus,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateProjectInput {
    pub name: String,
    pub description: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateProjectInput {
    pub name: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ListProjectsInput {
    pub include_archived: Option<bool>,
}

#[derive(Debug, Error)]
pub enum ProjectError {
    #[error("project not found: {0}")]
    NotFound(String),
    #[error("invalid project name")]
    InvalidName,
    #[error("persistence failure: {0}")]
    Persistence(String),
}

impl ProjectError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::NotFound(_) => "not_found",
            Self::InvalidName => "validation_failed",
            Self::Persistence(_) => "internal",
        }
    }
}

impl From<rusqlite::Error> for ProjectError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Persistence(error.to_string())
    }
}

pub struct ProjectRepository<'conn> {
    connection: &'conn Connection,
}

impl<'conn> ProjectRepository<'conn> {
    pub fn new(connection: &'conn Connection) -> Self {
        Self { connection }
    }

    pub fn list(&self, include_archived: bool) -> Result<Vec<Project>, ProjectError> {
        let sql = if include_archived {
            "SELECT id, name, description, status, created_at, updated_at
             FROM projects
             ORDER BY updated_at DESC"
        } else {
            "SELECT id, name, description, status, created_at, updated_at
             FROM projects
             WHERE status = 'active'
             ORDER BY updated_at DESC"
        };

        let mut statement = self.connection.prepare(sql)?;
        let projects = statement
            .query_map([], map_project_row)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| ProjectError::Persistence(error.to_string()))?;

        Ok(projects)
    }

    pub fn get_by_id(&self, id: &str) -> Result<Option<Project>, ProjectError> {
        let mut statement = self.connection.prepare(
            "SELECT id, name, description, status, created_at, updated_at
             FROM projects
             WHERE id = ?1",
        )?;

        let project = statement
            .query_row(params![id], map_project_row)
            .optional()
            .map_err(|error| ProjectError::Persistence(error.to_string()))?;

        Ok(project)
    }

    pub fn create(&self, input: CreateProjectInput) -> Result<Project, ProjectError> {
        let name = validate_name(&input.name)?;
        let description = normalize_optional_text(input.description);
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();

        self.connection
            .execute(
                "INSERT INTO projects (id, name, description, status, created_at, updated_at)
                 VALUES (?1, ?2, ?3, 'active', ?4, ?5)",
                params![id, name, description, now, now],
            )
            .map_err(|error| ProjectError::Persistence(error.to_string()))?;

        self.get_by_id(&id)?
            .ok_or_else(|| ProjectError::Persistence("failed to read created project".to_string()))
    }

    pub fn update(&self, id: &str, input: UpdateProjectInput) -> Result<Project, ProjectError> {
        let existing = self
            .get_by_id(id)?
            .ok_or_else(|| ProjectError::NotFound(id.to_string()))?;

        let name = match input.name {
            Some(name) => validate_name(&name)?,
            None => existing.name,
        };
        let description = match input.description {
            Some(description) => normalize_optional_text(Some(description)),
            None => existing.description,
        };
        let updated_at = Utc::now().to_rfc3339();

        self.connection
            .execute(
                "UPDATE projects SET name = ?1, description = ?2, updated_at = ?3 WHERE id = ?4",
                params![name, description, updated_at, id],
            )
            .map_err(|error| ProjectError::Persistence(error.to_string()))?;

        self.get_by_id(id)?
            .ok_or_else(|| ProjectError::NotFound(id.to_string()))
    }

    pub fn set_archived(&self, id: &str, archived: bool) -> Result<Project, ProjectError> {
        if self.get_by_id(id)?.is_none() {
            return Err(ProjectError::NotFound(id.to_string()));
        }

        let status = if archived {
            ProjectStatus::Archived
        } else {
            ProjectStatus::Active
        };
        let updated_at = Utc::now().to_rfc3339();

        self.connection
            .execute(
                "UPDATE projects SET status = ?1, updated_at = ?2 WHERE id = ?3",
                params![status.as_str(), updated_at, id],
            )
            .map_err(|error| ProjectError::Persistence(error.to_string()))?;

        self.get_by_id(id)?
            .ok_or_else(|| ProjectError::NotFound(id.to_string()))
    }

    pub fn delete(&self, id: &str) -> Result<(), ProjectError> {
        if self.get_by_id(id)?.is_none() {
            return Err(ProjectError::NotFound(id.to_string()));
        }

        self.connection
            .execute("DELETE FROM projects WHERE id = ?1", params![id])
            .map_err(|error| ProjectError::Persistence(error.to_string()))?;

        Ok(())
    }
}

fn map_project_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Project> {
    let status_value: String = row.get(3)?;
    Ok(Project {
        id: row.get(0)?,
        name: row.get(1)?,
        description: row.get(2)?,
        status: ProjectStatus::from_db(&status_value)
            .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?,
        created_at: row.get(4)?,
        updated_at: row.get(5)?,
    })
}

fn validate_name(name: &str) -> Result<String, ProjectError> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err(ProjectError::InvalidName);
    }
    Ok(trimmed.to_string())
}

fn normalize_optional_text(value: Option<String>) -> Option<String> {
    value.and_then(|text| {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}
