
CREATE TABLE IF NOT EXISTS plugin_installed (
  id TEXT PRIMARY KEY NOT NULL,
  plugin_id TEXT NOT NULL UNIQUE,
  plugin_version TEXT NOT NULL,
  manifest_version TEXT NOT NULL,
  rayvan_api_version TEXT NOT NULL,
  source_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('installed', 'disabled', 'incompatible', 'missing', 'error')),
  enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
  installed_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_loaded_at TEXT,
  manifest_snapshot_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_plugin_installed_status
  ON plugin_installed (status);

CREATE TABLE IF NOT EXISTS plugin_connections (
  id TEXT PRIMARY KEY NOT NULL,
  installed_plugin_id TEXT NOT NULL REFERENCES plugin_installed(id) ON DELETE RESTRICT,
  plugin_id TEXT NOT NULL,
  project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
  name TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pending', 'connected', 'expired', 'revoked', 'error', 'disconnected')),
  external_account_id TEXT,
  external_account_name TEXT,
  provider_base_url TEXT,
  credential_reference_id TEXT,
  metadata_json TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_authenticated_at TEXT,
  last_successful_sync_at TEXT,
  last_failed_sync_at TEXT,
  last_error_code TEXT
);

CREATE INDEX IF NOT EXISTS idx_plugin_connections_plugin
  ON plugin_connections (plugin_id);
CREATE INDEX IF NOT EXISTS idx_plugin_connections_project
  ON plugin_connections (project_id);
CREATE INDEX IF NOT EXISTS idx_plugin_connections_installed
  ON plugin_connections (installed_plugin_id);

CREATE TABLE IF NOT EXISTS plugin_credential_references (
  id TEXT PRIMARY KEY NOT NULL,
  plugin_id TEXT NOT NULL,
  connection_id TEXT NOT NULL REFERENCES plugin_connections(id) ON DELETE RESTRICT,
  provider TEXT NOT NULL CHECK (provider IN ('os_keychain', 'encrypted_local_store', 'development_memory')),
  storage_key TEXT NOT NULL,
  credential_type TEXT NOT NULL,
  display_name TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_used_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_plugin_credential_refs_connection
  ON plugin_credential_references (connection_id);

CREATE TABLE IF NOT EXISTS plugin_permission_grants (
  id TEXT PRIMARY KEY NOT NULL,
  plugin_id TEXT NOT NULL,
  connection_id TEXT NOT NULL REFERENCES plugin_connections(id) ON DELETE RESTRICT,
  permission TEXT NOT NULL,
  project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
  environment_id TEXT,
  granted INTEGER NOT NULL CHECK (granted IN (0, 1)),
  granted_by_json TEXT NOT NULL,
  granted_at TEXT NOT NULL,
  revoked_by_json TEXT,
  revoked_at TEXT,
  reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_plugin_permission_grants_connection
  ON plugin_permission_grants (connection_id);
CREATE INDEX IF NOT EXISTS idx_plugin_permission_grants_active
  ON plugin_permission_grants (connection_id, granted);

CREATE TABLE IF NOT EXISTS plugin_discovered_resources (
  id TEXT PRIMARY KEY NOT NULL,
  plugin_id TEXT NOT NULL,
  installed_plugin_id TEXT NOT NULL REFERENCES plugin_installed(id) ON DELETE RESTRICT,
  connection_id TEXT NOT NULL REFERENCES plugin_connections(id) ON DELETE RESTRICT,
  provider_resource_id TEXT NOT NULL,
  resource_type TEXT NOT NULL,
  name TEXT NOT NULL,
  parent_provider_resource_id TEXT,
  metadata_json TEXT NOT NULL,
  plugin_version TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  discovery_status TEXT NOT NULL CHECK (discovery_status IN ('active', 'missing', 'inaccessible', 'archived')),
  first_discovered_at TEXT NOT NULL,
  last_discovered_at TEXT NOT NULL,
  last_inspected_at TEXT,
  missing_since TEXT,
  UNIQUE (connection_id, provider_resource_id, resource_type)
);

CREATE INDEX IF NOT EXISTS idx_plugin_discovered_connection
  ON plugin_discovered_resources (connection_id);

CREATE TABLE IF NOT EXISTS plugin_resource_bindings (
  id TEXT PRIMARY KEY NOT NULL,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
  environment_id TEXT,
  discovered_resource_id TEXT NOT NULL REFERENCES plugin_discovered_resources(id) ON DELETE RESTRICT,
  plugin_id TEXT NOT NULL,
  connection_id TEXT NOT NULL REFERENCES plugin_connections(id) ON DELETE RESTRICT,
  role TEXT,
  display_name TEXT,
  binding_status TEXT NOT NULL CHECK (binding_status IN ('active', 'suggested', 'detached', 'invalid')),
  created_by_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_plugin_bindings_project
  ON plugin_resource_bindings (project_id);
CREATE INDEX IF NOT EXISTS idx_plugin_bindings_resource
  ON plugin_resource_bindings (discovered_resource_id);

CREATE TABLE IF NOT EXISTS plugin_environment_mapping_suggestions (
  id TEXT PRIMARY KEY NOT NULL,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
  connection_id TEXT NOT NULL REFERENCES plugin_connections(id) ON DELETE RESTRICT,
  discovered_resource_id TEXT NOT NULL REFERENCES plugin_discovered_resources(id) ON DELETE RESTRICT,
  suggested_environment_id TEXT,
  suggested_environment_name TEXT,
  confidence REAL,
  reasons_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pending', 'accepted', 'rejected', 'superseded')),
  created_at TEXT NOT NULL,
  resolved_at TEXT,
  resolved_by_json TEXT
);

CREATE TABLE IF NOT EXISTS plugin_observed_resource_state (
  id TEXT PRIMARY KEY NOT NULL,
  discovered_resource_id TEXT NOT NULL UNIQUE REFERENCES plugin_discovered_resources(id) ON DELETE RESTRICT,
  plugin_id TEXT NOT NULL,
  connection_id TEXT NOT NULL REFERENCES plugin_connections(id) ON DELETE RESTRICT,
  state_json TEXT NOT NULL,
  plugin_version TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  observed_at TEXT NOT NULL,
  source_execution_id TEXT,
  checksum TEXT
);

CREATE TABLE IF NOT EXISTS plugin_observed_resource_state_history (
  id TEXT PRIMARY KEY NOT NULL,
  discovered_resource_id TEXT NOT NULL REFERENCES plugin_discovered_resources(id) ON DELETE RESTRICT,
  plugin_id TEXT NOT NULL,
  connection_id TEXT NOT NULL REFERENCES plugin_connections(id) ON DELETE RESTRICT,
  state_json TEXT NOT NULL,
  plugin_version TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  observed_at TEXT NOT NULL,
  source_execution_id TEXT,
  checksum TEXT
);

CREATE INDEX IF NOT EXISTS idx_plugin_observed_history_resource
  ON plugin_observed_resource_state_history (discovered_resource_id, observed_at DESC);

CREATE TABLE IF NOT EXISTS plugin_desired_resource_state (
  id TEXT PRIMARY KEY NOT NULL,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
  environment_id TEXT,
  resource_binding_id TEXT NOT NULL UNIQUE REFERENCES plugin_resource_bindings(id) ON DELETE RESTRICT,
  plugin_id TEXT NOT NULL,
  connection_id TEXT NOT NULL REFERENCES plugin_connections(id) ON DELETE RESTRICT,
  state_json TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  revision INTEGER NOT NULL,
  created_by_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plugin_change_plans (
  id TEXT PRIMARY KEY NOT NULL,
  plugin_id TEXT NOT NULL,
  connection_id TEXT NOT NULL REFERENCES plugin_connections(id) ON DELETE RESTRICT,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
  environment_id TEXT,
  resource_binding_id TEXT NOT NULL REFERENCES plugin_resource_bindings(id) ON DELETE RESTRICT,
  desired_state_revision INTEGER,
  observed_state_checksum TEXT,
  plan_schema_version TEXT NOT NULL,
  plan_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected', 'applying', 'applied', 'failed', 'expired', 'superseded')),
  created_by_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT,
  superseded_by_plan_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_plugin_change_plans_binding
  ON plugin_change_plans (resource_binding_id, status);

CREATE TABLE IF NOT EXISTS plugin_change_plan_approvals (
  id TEXT PRIMARY KEY NOT NULL,
  change_plan_id TEXT NOT NULL REFERENCES plugin_change_plans(id) ON DELETE RESTRICT,
  approved_operation_ids_json TEXT NOT NULL,
  destructive_approval INTEGER NOT NULL CHECK (destructive_approval IN (0, 1)),
  approved_by_json TEXT NOT NULL,
  approved_at TEXT NOT NULL,
  comment TEXT
);

CREATE TABLE IF NOT EXISTS plugin_change_plan_rejections (
  id TEXT PRIMARY KEY NOT NULL,
  change_plan_id TEXT NOT NULL REFERENCES plugin_change_plans(id) ON DELETE RESTRICT,
  rejected_by_json TEXT NOT NULL,
  rejected_at TEXT NOT NULL,
  reason TEXT
);

CREATE TABLE IF NOT EXISTS plugin_change_applies (
  id TEXT PRIMARY KEY NOT NULL,
  change_plan_id TEXT NOT NULL REFERENCES plugin_change_plans(id) ON DELETE RESTRICT,
  execution_id TEXT NOT NULL,
  plugin_id TEXT NOT NULL,
  connection_id TEXT NOT NULL REFERENCES plugin_connections(id) ON DELETE RESTRICT,
  resource_binding_id TEXT NOT NULL REFERENCES plugin_resource_bindings(id) ON DELETE RESTRICT,
  status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed', 'cancelled', 'timed_out')),
  result_json TEXT,
  error_json TEXT,
  started_at TEXT NOT NULL,
  finished_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plugin_change_verifications (
  id TEXT PRIMARY KEY NOT NULL,
  change_apply_id TEXT NOT NULL REFERENCES plugin_change_applies(id) ON DELETE RESTRICT,
  execution_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('verified', 'not_verified', 'failed')),
  result_json TEXT,
  error_json TEXT,
  verified_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plugin_execution_history (
  id TEXT PRIMARY KEY NOT NULL,
  execution_id TEXT NOT NULL,
  plugin_id TEXT NOT NULL,
  plugin_version TEXT NOT NULL,
  capability TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT NOT NULL,
  duration_ms INTEGER NOT NULL,
  actor_json TEXT NOT NULL,
  project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
  environment_id TEXT,
  resource_id TEXT,
  connection_id TEXT,
  reason TEXT,
  error_code TEXT,
  error_message TEXT,
  warning_count INTEGER NOT NULL,
  recorded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_plugin_execution_history_plugin
  ON plugin_execution_history (plugin_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_plugin_execution_history_execution
  ON plugin_execution_history (execution_id);
