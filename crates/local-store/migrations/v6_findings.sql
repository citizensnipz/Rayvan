
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
