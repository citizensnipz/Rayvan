export const PROJECTS_TABLE_MIGRATION = `
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
`;

export const APP_SETTINGS_TABLE_MIGRATION = `
CREATE TABLE IF NOT EXISTS app_settings (
  key TEXT PRIMARY KEY NOT NULL,
  value TEXT NOT NULL
);
`;

/**
 * Plugin persistence schema (v3).
 * Authoritative source: `crates/local-store/migrations/v3_plugin_persistence.sql`
 * Keep this mirror in sync with `node packages/local-database/scripts/sync-v3-sql-from-rust.mjs`.
 */
export const V3_PLUGIN_PERSISTENCE_SQL = `
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
`;

/**
 * Environments + configuration persistence schema (v4).
 * Authoritative source: `crates/local-store/migrations/v4_environments_configuration.sql`
 */
export const V4_ENVIRONMENTS_AND_CONFIGURATION_SQL = `
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
`;

/**
 * Desired + applied configuration persistence schema (v5).
 * Authoritative source: `crates/local-store/migrations/v5_desired_applied_configuration.sql`
 * Keep this mirror in sync with `node packages/local-database/scripts/sync-v5-sql-from-rust.mjs`.
 */
export const V5_DESIRED_APPLIED_CONFIGURATION_SQL = `
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
`;

/**
 * Findings persistence schema (v6).
 * Authoritative source: `crates/local-store/migrations/v6_findings.sql`
 * Keep this mirror in sync with `node packages/local-database/scripts/sync-v6-sql-from-rust.mjs`.
 */
export const V6_FINDINGS_SQL = `
CREATE TABLE IF NOT EXISTS findings (
  id TEXT PRIMARY KEY NOT NULL,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  rule_id TEXT NOT NULL,
  source_json TEXT NOT NULL,
  category TEXT NOT NULL CHECK (category IN (
    'configuration', 'environment', 'integration', 'resource', 'deployment',
    'security', 'availability', 'drift', 'permission', 'mapping', 'other'
  )),
  severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'error', 'critical')),
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  description TEXT,
  status TEXT NOT NULL CHECK (status IN (
    'open', 'acknowledged', 'resolved', 'dismissed', 'suppressed'
  )),
  fingerprint TEXT NOT NULL,
  fingerprint_version TEXT NOT NULL,
  environment_id TEXT REFERENCES environments(id) ON DELETE SET NULL,
  integration_id TEXT,
  connection_id TEXT,
  discovered_resource_id TEXT,
  resource_binding_id TEXT,
  configuration_key_id TEXT REFERENCES configuration_keys(id) ON DELETE SET NULL,
  change_plan_id TEXT,
  deployment_id TEXT,
  evidence_json TEXT NOT NULL,
  remediation_json TEXT,
  first_detected_at TEXT NOT NULL,
  last_detected_at TEXT NOT NULL,
  occurrence_count INTEGER NOT NULL CHECK (occurrence_count >= 0),
  acknowledged_at TEXT,
  acknowledged_by_json TEXT,
  dismissed_at TEXT,
  dismissed_by_json TEXT,
  dismissal_reason TEXT,
  resolved_at TEXT,
  resolution_json TEXT,
  suppressed_until TEXT,
  last_evaluation_run_id TEXT,
  metadata_json TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  UNIQUE (project_id, fingerprint)
);

CREATE INDEX IF NOT EXISTS idx_findings_project_status
  ON findings (project_id, status);
CREATE INDEX IF NOT EXISTS idx_findings_project_severity
  ON findings (project_id, severity);
CREATE INDEX IF NOT EXISTS idx_findings_project_category
  ON findings (project_id, category);
CREATE INDEX IF NOT EXISTS idx_findings_project_environment
  ON findings (project_id, environment_id);
CREATE INDEX IF NOT EXISTS idx_findings_project_connection
  ON findings (project_id, connection_id);
CREATE INDEX IF NOT EXISTS idx_findings_resource_binding
  ON findings (resource_binding_id);
CREATE INDEX IF NOT EXISTS idx_findings_configuration_key
  ON findings (configuration_key_id);
CREATE INDEX IF NOT EXISTS idx_findings_last_detected
  ON findings (last_detected_at);

CREATE TABLE IF NOT EXISTS finding_lifecycle_events (
  id TEXT PRIMARY KEY NOT NULL,
  finding_id TEXT NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  type TEXT NOT NULL CHECK (type IN (
    'created', 'updated', 'reopened', 'acknowledged', 'dismissed',
    'suppressed', 'resolved', 'severity_changed'
  )),
  actor_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  previous_status TEXT CHECK (previous_status IS NULL OR previous_status IN (
    'open', 'acknowledged', 'resolved', 'dismissed', 'suppressed'
  )),
  next_status TEXT CHECK (next_status IS NULL OR next_status IN (
    'open', 'acknowledged', 'resolved', 'dismissed', 'suppressed'
  )),
  reason TEXT,
  metadata_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_finding_lifecycle_events_finding_created
  ON finding_lifecycle_events (finding_id, created_at);

CREATE TABLE IF NOT EXISTS finding_evaluation_runs (
  id TEXT PRIMARY KEY NOT NULL,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  scope_json TEXT NOT NULL,
  trigger TEXT NOT NULL CHECK (trigger IN (
    'manual', 'startup', 'environment_sync', 'integration_sync',
    'configuration_inspection', 'apply_completed', 'verification_completed',
    'connection_state_changed', 'desired_configuration_saved'
  )),
  status TEXT NOT NULL CHECK (status IN (
    'succeeded', 'partially_succeeded', 'failed', 'cancelled'
  )),
  started_at TEXT NOT NULL,
  finished_at TEXT NOT NULL,
  evaluators_run INTEGER NOT NULL CHECK (evaluators_run >= 0),
  evaluators_failed INTEGER NOT NULL CHECK (evaluators_failed >= 0),
  findings_created INTEGER NOT NULL CHECK (findings_created >= 0),
  findings_updated INTEGER NOT NULL CHECK (findings_updated >= 0),
  findings_reopened INTEGER NOT NULL CHECK (findings_reopened >= 0),
  findings_resolved INTEGER NOT NULL CHECK (findings_resolved >= 0),
  safe_errors_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_finding_evaluation_runs_project_started
  ON finding_evaluation_runs (project_id, started_at);
`;

export const V7_DAEMON_CONTROL_PLANE_SQL = `
-- Daemon control plane: local clients, operations, approvals, audit, locks.

CREATE TABLE IF NOT EXISTS daemon_metadata (
  key TEXT PRIMARY KEY NOT NULL,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS local_clients (
  id TEXT PRIMARY KEY NOT NULL,
  name TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('desktop', 'mcp', 'cli')),
  status TEXT NOT NULL CHECK (status IN ('active', 'revoked', 'expired')),
  permission_profile_id TEXT NOT NULL,
  project_scopes_json TEXT NOT NULL,
  environment_scopes_json TEXT,
  approval_policy_json TEXT NOT NULL,
  credential_reference_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  last_connected_at TEXT,
  last_activity_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_local_clients_status
  ON local_clients (status);

CREATE TABLE IF NOT EXISTS mcp_permission_profiles (
  id TEXT PRIMARY KEY NOT NULL,
  name TEXT NOT NULL,
  built_in INTEGER NOT NULL CHECK (built_in IN (0, 1)),
  permissions_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS operations (
  id TEXT PRIMARY KEY NOT NULL,
  project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
  type TEXT NOT NULL CHECK (type IN (
    'integration_sync', 'environment_sync', 'resource_inspection',
    'findings_scan', 'change_plan_generation', 'change_apply',
    'change_verification', 'plugin_operation'
  )),
  status TEXT NOT NULL CHECK (status IN (
    'queued', 'running', 'waiting_for_approval', 'succeeded',
    'partially_succeeded', 'failed', 'cancelled', 'timed_out'
  )),
  actor_json TEXT NOT NULL,
  progress_json TEXT,
  started_at TEXT,
  finished_at TEXT,
  correlation_id TEXT NOT NULL,
  idempotency_key TEXT,
  safe_error_json TEXT,
  result_summary_json TEXT,
  approval_request_id TEXT,
  change_plan_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_operations_idempotency
  ON operations (idempotency_key)
  WHERE idempotency_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_operations_project_created
  ON operations (project_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_operations_status
  ON operations (status);

CREATE TABLE IF NOT EXISTS operation_events (
  id TEXT PRIMARY KEY NOT NULL,
  operation_id TEXT NOT NULL REFERENCES operations(id) ON DELETE CASCADE,
  type TEXT NOT NULL,
  message TEXT,
  progress_json TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_operation_events_operation_created
  ON operation_events (operation_id, created_at);

CREATE TABLE IF NOT EXISTS approval_requests (
  id TEXT PRIMARY KEY NOT NULL,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  operation_id TEXT REFERENCES operations(id) ON DELETE SET NULL,
  change_plan_id TEXT,
  requested_by_json TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN (
    'local_mutation', 'remote_apply', 'destructive_operation', 'sensitive_value_access'
  )),
  summary TEXT NOT NULL,
  safe_details_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN (
    'pending', 'approved', 'denied', 'expired', 'cancelled'
  )),
  created_at TEXT NOT NULL,
  expires_at TEXT,
  decided_at TEXT,
  decided_by_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_approval_requests_status_created
  ON approval_requests (status, created_at DESC);

CREATE TABLE IF NOT EXISTS mcp_audit_events (
  id TEXT PRIMARY KEY NOT NULL,
  client_id TEXT,
  tool_name TEXT,
  daemon_method TEXT,
  project_id TEXT,
  environment_id TEXT,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL CHECK (status IN (
    'succeeded', 'failed', 'denied', 'cancelled', 'pending_approval'
  )),
  operation_id TEXT,
  approval_id TEXT,
  affected_object_ids_json TEXT NOT NULL,
  safe_summary TEXT NOT NULL,
  error_code TEXT,
  contacted_remote INTEGER NOT NULL CHECK (contacted_remote IN (0, 1)),
  mutated_remote INTEGER NOT NULL CHECK (mutated_remote IN (0, 1)),
  correlation_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mcp_audit_events_client_started
  ON mcp_audit_events (client_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_mcp_audit_events_project_started
  ON mcp_audit_events (project_id, started_at DESC);

CREATE TABLE IF NOT EXISTS resource_locks (
  id TEXT PRIMARY KEY NOT NULL,
  resource_key TEXT NOT NULL UNIQUE,
  operation_id TEXT NOT NULL REFERENCES operations(id) ON DELETE CASCADE,
  holder_actor_json TEXT NOT NULL,
  acquired_at TEXT NOT NULL,
  expires_at TEXT
);

CREATE TABLE IF NOT EXISTS idempotency_records (
  idempotency_key TEXT PRIMARY KEY NOT NULL,
  method TEXT NOT NULL,
  request_hash TEXT NOT NULL,
  response_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT
);

CREATE TABLE IF NOT EXISTS plugin_mcp_actions (
  id TEXT PRIMARY KEY NOT NULL,
  plugin_id TEXT NOT NULL,
  namespaced_id TEXT NOT NULL UNIQUE,
  description TEXT NOT NULL,
  input_schema_json TEXT NOT NULL,
  output_schema_json TEXT NOT NULL,
  risk TEXT NOT NULL CHECK (risk IN (
    'read', 'local_write', 'remote_read', 'plan', 'remote_write', 'destructive'
  )),
  required_plugin_permissions_json TEXT NOT NULL,
  required_mcp_permissions_json TEXT NOT NULL,
  destructive INTEGER NOT NULL CHECK (destructive IN (0, 1)),
  requires_approval INTEGER NOT NULL CHECK (requires_approval IN (0, 1)),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
`;

export const MIGRATION_VERSION = 7;
