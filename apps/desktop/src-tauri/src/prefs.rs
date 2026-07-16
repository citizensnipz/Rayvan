use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionPrefs {
    pub current_project_id: Option<String>,
    #[serde(skip)]
    path: PathBuf,
}

impl SessionPrefs {
    pub fn load(path: &Path) -> Self {
        let mut prefs = if path.exists() {
            std::fs::read_to_string(path)
                .ok()
                .and_then(|raw| serde_json::from_str::<SessionPrefs>(&raw).ok())
                .unwrap_or_default()
        } else {
            SessionPrefs::default()
        };
        prefs.path = path.to_path_buf();
        prefs
    }

    pub fn save(&self) -> Result<(), String> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let raw = serde_json::to_string_pretty(self).map_err(|error| error.to_string())?;
        std::fs::write(&self.path, raw).map_err(|error| error.to_string())
    }

    pub fn set_current_project_id(&mut self, project_id: Option<String>) -> Result<(), String> {
        self.current_project_id = project_id;
        self.save()
    }
}
