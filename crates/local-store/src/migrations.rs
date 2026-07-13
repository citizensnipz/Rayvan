pub const CURRENT_SCHEMA_VERSION: u32 = 1;

pub const PROJECTS_MIGRATION_SQL: &str = r#"
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

pub struct MigrationSet {
    pub version: u32,
    pub description: &'static str,
    pub sql: &'static str,
}

pub const INITIAL_MIGRATION: MigrationSet = MigrationSet {
    version: CURRENT_SCHEMA_VERSION,
    description: "Initial Rayvan projects schema",
    sql: PROJECTS_MIGRATION_SQL,
};
