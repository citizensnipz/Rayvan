
CREATE TABLE IF NOT EXISTS environments (
  id TEXT PRIMARY KEY NOT NULL,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  slug TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN (
    'local', 'development', 'preview', 'staging', 'production', 'test', 'custom'
  )),
  description TEXT,
  presentation_json TEXT,
  status TEXT NOT NULL CHECK (status IN (
    'local_only', 'healthy', 'attention_required', 'syncing', 'error', 'archived'
  )),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE (project_id, name),
  UNIQUE (project_id, slug)
);

CREATE INDEX IF NOT EXISTS idx_environments_project_updated
  ON environments (project_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_environments_project_status
  ON environments (project_id, status);

CREATE TABLE IF NOT EXISTS configuration_keys (
  id TEXT PRIMARY KEY NOT NULL,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  description TEXT,
  value_type TEXT NOT NULL CHECK (value_type IN (
    'string', 'number', 'boolean', 'url', 'json', 'secret', 'unknown'
  )),
  required INTEGER NOT NULL CHECK (required IN (0, 1)),
  sensitive INTEGER NOT NULL CHECK (sensitive IN (0, 1)),
  source TEXT NOT NULL CHECK (source IN ('discovered', 'manual', 'imported')),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE (project_id, name)
);

CREATE INDEX IF NOT EXISTS idx_configuration_keys_project
  ON configuration_keys (project_id);

CREATE TABLE IF NOT EXISTS configuration_occurrences (
  id TEXT PRIMARY KEY NOT NULL,
  configuration_key_id TEXT NOT NULL REFERENCES configuration_keys(id) ON DELETE CASCADE,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  environment_id TEXT REFERENCES environments(id) ON DELETE SET NULL,
  plugin_id TEXT NOT NULL,
  connection_id TEXT NOT NULL,
  discovered_resource_id TEXT NOT NULL,
  resource_binding_id TEXT,
  provider_key TEXT NOT NULL,
  value_access TEXT NOT NULL CHECK (value_access IN (
    'readable', 'masked', 'locked', 'name_only', 'missing'
  )),
  observed_value TEXT,
  masked_value TEXT,
  value_fingerprint TEXT,
  secret_value_ref TEXT,
  scope TEXT,
  first_observed_at TEXT NOT NULL,
  last_observed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_configuration_occurrences_project
  ON configuration_occurrences (project_id);
CREATE INDEX IF NOT EXISTS idx_configuration_occurrences_key
  ON configuration_occurrences (configuration_key_id);
CREATE INDEX IF NOT EXISTS idx_configuration_occurrences_environment
  ON configuration_occurrences (environment_id);
CREATE INDEX IF NOT EXISTS idx_configuration_occurrences_match
  ON configuration_occurrences (
    configuration_key_id,
    connection_id,
    discovered_resource_id,
    provider_key,
    environment_id
  );
