pub const CURRENT_SCHEMA_VERSION: u32 = 3;

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
        assert_eq!(CURRENT_SCHEMA_VERSION, 3);
    }
}
