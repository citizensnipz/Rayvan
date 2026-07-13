use thiserror::Error;

#[derive(Debug, Error)]
pub enum PluginHostError {
    #[error("plugin process failed to start: {0}")]
    StartFailed(String),
    #[error("plugin process is not running")]
    NotRunning,
}

#[derive(Debug, Clone)]
pub struct PluginProcessConfig {
    pub plugin_id: String,
    pub executable: String,
    pub args: Vec<String>,
}

pub trait PluginProcessSpawner {
    fn start(&self, config: &PluginProcessConfig) -> Result<u32, PluginHostError>;
    fn stop(&self, plugin_id: &str) -> Result<(), PluginHostError>;
}

pub struct PlaceholderPluginProcessSpawner;

impl PluginProcessSpawner for PlaceholderPluginProcessSpawner {
    fn start(&self, config: &PluginProcessConfig) -> Result<u32, PluginHostError> {
        Err(PluginHostError::StartFailed(format!(
            "plugin host not implemented for {}",
            config.plugin_id
        )))
    }

    fn stop(&self, _plugin_id: &str) -> Result<(), PluginHostError> {
        Err(PluginHostError::NotRunning)
    }
}
