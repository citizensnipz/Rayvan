#[derive(Debug, Clone)]
pub struct PluginPermissionSet {
    pub network: bool,
    pub filesystem_read: bool,
    pub filesystem_write: bool,
}

impl Default for PluginPermissionSet {
    fn default() -> Self {
        Self {
            network: true,
            filesystem_read: true,
            filesystem_write: false,
        }
    }
}
