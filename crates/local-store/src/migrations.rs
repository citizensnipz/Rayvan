pub const CURRENT_SCHEMA_VERSION: u32 = 5;

pub const V1_PROJECTS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY NOT NULL,
  applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
  id TEXT PRIMARY KEY NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  status TEXT NOT NULL CHECK (status IN ('active', 'archived')),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_projects_status_updated
  ON projects (status, updated_at DESC);
"#;

pub const V2_APP_SETTINGS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS app_settings (
  key TEXT PRIMARY KEY NOT NULL,
  value TEXT NOT NULL
);
"#;

/// Authoritative plugin persistence SQL (also mirrored in TypeScript).
/// Edit `migrations/v3_plugin_persistence.sql`, then sync TS via
/// `node packages/local-database/scripts/sync-v3-sql-from-rust.mjs`.
pub const V3_PLUGIN_PERSISTENCE_SQL: &str =
    include_str!("../migrations/v3_plugin_persistence.sql");

/// Environments + configuration persistence SQL (also mirrored in TypeScript).
pub const V4_ENVIRONMENTS_CONFIGURATION_SQL: &str =
    include_str!("../migrations/v4_environments_configuration.sql");

/// Desired + applied configuration persistence SQL (also mirrored in TypeScript).
pub const V5_DESIRED_APPLIED_CONFIGURATION_SQL: &str =
    include_str!("../migrations/v5_desired_applied_configuration.sql");

pub struct MigrationSet {
    pub version: u32,
    pub description: &'static str,
    pub sql: &'static str,
}

pub const MIGRATIONS: &[MigrationSet] = &[
    MigrationSet {
        version: 1,
        description: "Initial Rayvan projects schema",
        sql: V1_PROJECTS_SQL,
    },
    MigrationSet {
        version: 2,
        description: "App settings for session preferences",
        sql: V2_APP_SETTINGS_SQL,
    },
    MigrationSet {
        version: 3,
        description: "Plugin persistence tables",
        sql: V3_PLUGIN_PERSISTENCE_SQL,
    },
    MigrationSet {
        version: 4,
        description: "Environments and configuration persistence",
        sql: V4_ENVIRONMENTS_CONFIGURATION_SQL,
    },
    MigrationSet {
        version: 5,
        description: "Desired and applied configuration state",
        sql: V5_DESIRED_APPLIED_CONFIGURATION_SQL,
    },
];

/// Backward-compatible alias used by earlier call sites.
pub const INITIAL_MIGRATION: MigrationSet = MigrationSet {
    version: 1,
    description: "Initial Rayvan projects schema",
    sql: V1_PROJECTS_SQL,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v3_sql_defines_plugin_tables() {
        assert!(V3_PLUGIN_PERSISTENCE_SQL.contains("plugin_installed"));
        assert!(V3_PLUGIN_PERSISTENCE_SQL.contains("plugin_connections"));
        assert!(V3_PLUGIN_PERSISTENCE_SQL.contains("plugin_execution_history"));
        assert!(V3_PLUGIN_PERSISTENCE_SQL.contains(
            "UNIQUE (connection_id, provider_resource_id, resource_type)"
        ));
    }

    #[test]
    fn v4_sql_defines_environments_and_configuration() {
        assert!(V4_ENVIRONMENTS_CONFIGURATION_SQL.contains("environments"));
        assert!(V4_ENVIRONMENTS_CONFIGURATION_SQL.contains("configuration_keys"));
        assert!(V4_ENVIRONMENTS_CONFIGURATION_SQL.contains("configuration_occurrences"));
        assert!(V4_ENVIRONMENTS_CONFIGURATION_SQL.contains("secret_value_ref"));
    }

    #[test]
    fn v5_sql_defines_desired_and_applied() {
        assert!(V5_DESIRED_APPLIED_CONFIGURATION_SQL.contains("desired_configuration_values"));
        assert!(V5_DESIRED_APPLIED_CONFIGURATION_SQL.contains("applied_configuration_states"));
        assert!(V5_DESIRED_APPLIED_CONFIGURATION_SQL.contains("secret_value_ref"));
        assert!(V5_DESIRED_APPLIED_CONFIGURATION_SQL.contains(
            "UNIQUE (configuration_key_id, environment_id)"
        ));
        assert!(V5_DESIRED_APPLIED_CONFIGURATION_SQL.contains(
            "UNIQUE (configuration_key_id, environment_id, resource_binding_id)"
        ));
        assert_eq!(CURRENT_SCHEMA_VERSION, 5);
    }
}
