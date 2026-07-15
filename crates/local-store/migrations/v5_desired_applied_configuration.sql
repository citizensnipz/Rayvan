
CREATE TABLE IF NOT EXISTS desired_configuration_values (
  id TEXT PRIMARY KEY NOT NULL,
  configuration_key_id TEXT NOT NULL REFERENCES configuration_keys(id) ON DELETE CASCADE,
  environment_id TEXT NOT NULL REFERENCES environments(id) ON DELETE CASCADE,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  desired_value TEXT,
  secret_value_ref TEXT,
  value_fingerprint TEXT,
  revision INTEGER NOT NULL CHECK (revision >= 1),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  updated_by_json TEXT NOT NULL,
  UNIQUE (configuration_key_id, environment_id)
);

CREATE INDEX IF NOT EXISTS idx_desired_configuration_values_project
  ON desired_configuration_values (project_id);
CREATE INDEX IF NOT EXISTS idx_desired_configuration_values_environment
  ON desired_configuration_values (environment_id);
CREATE INDEX IF NOT EXISTS idx_desired_configuration_values_key
  ON desired_configuration_values (configuration_key_id);

CREATE TABLE IF NOT EXISTS applied_configuration_states (
  id TEXT PRIMARY KEY NOT NULL,
  configuration_key_id TEXT NOT NULL REFERENCES configuration_keys(id) ON DELETE CASCADE,
  environment_id TEXT NOT NULL REFERENCES environments(id) ON DELETE CASCADE,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  resource_binding_id TEXT NOT NULL,
  desired_revision INTEGER NOT NULL CHECK (desired_revision >= 1),
  applied_fingerprint TEXT,
  apply_execution_id TEXT NOT NULL,
  verification_execution_id TEXT,
  status TEXT NOT NULL CHECK (status IN (
    'applied', 'verified', 'failed', 'verification_failed'
  )),
  applied_at TEXT NOT NULL,
  verified_at TEXT,
  UNIQUE (configuration_key_id, environment_id, resource_binding_id)
);

CREATE INDEX IF NOT EXISTS idx_applied_configuration_states_project
  ON applied_configuration_states (project_id);
CREATE INDEX IF NOT EXISTS idx_applied_configuration_states_environment
  ON applied_configuration_states (environment_id);
CREATE INDEX IF NOT EXISTS idx_applied_configuration_states_key
  ON applied_configuration_states (configuration_key_id);
CREATE INDEX IF NOT EXISTS idx_applied_configuration_states_binding
  ON applied_configuration_states (resource_binding_id);
