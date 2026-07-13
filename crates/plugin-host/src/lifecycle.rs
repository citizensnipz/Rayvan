#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginLifecycleState {
    Stopped,
    Starting,
    Running,
    Stopping,
    Crashed,
}
