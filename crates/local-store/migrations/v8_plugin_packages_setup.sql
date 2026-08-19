
-- Package install metadata lives primarily in plugin_installed.source_json.
-- Setup sessions hold non-secret wizard state; secrets stay in CredentialStore.

CREATE TABLE IF NOT EXISTS plugin_setup_sessions (
  id TEXT PRIMARY KEY NOT NULL,
  plugin_id TEXT NOT NULL,
  installed_plugin_id TEXT NOT NULL REFERENCES plugin_installed(id) ON DELETE CASCADE,
  project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
  status TEXT NOT NULL CHECK (status IN ('active', 'completed', 'cancelled', 'expired')),
  current_step_id TEXT,
  auth_method TEXT,
  state_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_plugin_setup_sessions_plugin
  ON plugin_setup_sessions (plugin_id, status);

CREATE INDEX IF NOT EXISTS idx_plugin_setup_sessions_project
  ON plugin_setup_sessions (project_id);
