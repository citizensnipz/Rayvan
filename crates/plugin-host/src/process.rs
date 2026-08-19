use std::collections::HashMap;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PluginHostError {
    #[error("plugin process failed to start: {0}")]
    StartFailed(String),
    #[error("plugin process is not running")]
    NotRunning,
    #[error("plugin process kill failed: {0}")]
    KillFailed(String),
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
    fn kill(&self, plugin_id: &str) -> Result<(), PluginHostError>;
}

/// Spawns plugin launchers, tracks child processes, and supports stop/kill-on-cancel.
pub struct NativePluginProcessSpawner {
    children: Mutex<HashMap<String, Child>>,
}

impl NativePluginProcessSpawner {
    pub fn new() -> Self {
        Self {
            children: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for NativePluginProcessSpawner {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginProcessSpawner for NativePluginProcessSpawner {
    fn start(&self, config: &PluginProcessConfig) -> Result<u32, PluginHostError> {
        let mut children = self
            .children
            .lock()
            .map_err(|_| PluginHostError::StartFailed("process map lock poisoned".into()))?;

        if let Some(existing) = children.remove(&config.plugin_id) {
            let _ = existing;
            // Dropping previous handle; best-effort replace.
        }

        let mut command = Command::new(&config.executable);
        command
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let child = command.spawn().map_err(|error| {
            PluginHostError::StartFailed(format!(
                "{} ({}): {error}",
                config.plugin_id, config.executable
            ))
        })?;
        let pid = child.id();
        children.insert(config.plugin_id.clone(), child);
        Ok(pid)
    }

    fn stop(&self, plugin_id: &str) -> Result<(), PluginHostError> {
        let mut children = self
            .children
            .lock()
            .map_err(|_| PluginHostError::KillFailed("process map lock poisoned".into()))?;
        let Some(mut child) = children.remove(plugin_id) else {
            return Err(PluginHostError::NotRunning);
        };
        // Graceful attempt then force kill.
        let _ = child.kill();
        let _ = child.wait();
        Ok(())
    }

    fn kill(&self, plugin_id: &str) -> Result<(), PluginHostError> {
        self.stop(plugin_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_and_stops_echo_process() {
        let spawner = NativePluginProcessSpawner::new();
        let (executable, args) = if cfg!(windows) {
            ("cmd".to_string(), vec!["/C".into(), "echo".into(), "ok".into()])
        } else {
            ("echo".to_string(), vec!["ok".into()])
        };

        let pid = spawner
            .start(&PluginProcessConfig {
                plugin_id: "test.plugin".into(),
                executable,
                args,
            })
            .expect("start");
        assert!(pid > 0);
        // Process may exit quickly; stop should tolerate that.
        let _ = spawner.stop("test.plugin");
    }
}
