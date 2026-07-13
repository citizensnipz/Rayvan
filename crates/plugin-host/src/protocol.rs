pub const PLUGIN_PROTOCOL_VERSION: &str = "1";

#[derive(Debug, Clone)]
pub struct PluginMessage {
    pub id: String,
    pub method: String,
}
