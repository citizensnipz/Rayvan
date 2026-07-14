#[derive(Debug, Clone)]
pub struct PluginPermissionSet {
    pub network: bool,
    pub read_secrets: bool,
    pub write_remote_configuration: bool,
    pub read_local_files: bool,
    pub write_local_files: bool,
}

impl Default for PluginPermissionSet {
    fn default() -> Self {
        Self {
            network: true,
            read_secrets: false,
            write_remote_configuration: false,
            read_local_files: true,
            write_local_files: false,
        }
    }
}
