pub const PLUGIN_PROTOCOL_VERSION: &str = "1";

/// Capability-aligned JSON-RPC methods (framing owned by TypeScript host/client).
pub const PLUGIN_RPC_METHODS: &[&str] = &[
    "initialize",
    "shutdown",
    "authenticate",
    "discover",
    "inspect",
    "plan",
    "apply",
    "verify",
    "evaluate_findings",
    "cancel",
];

#[derive(Debug, Clone)]
pub struct PluginMessage {
    pub id: String,
    pub method: String,
}
