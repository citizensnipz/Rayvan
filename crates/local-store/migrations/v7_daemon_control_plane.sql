
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
