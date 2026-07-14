/**
 * Tables that currently exist in the local SQLite schema (v3).
 * Aspirational Core entities (environments, integrations, action_plans, …)
 * are intentionally omitted until they have migrations.
 */
export const PHYSICAL_TABLES = [
  "schema_migrations",
  "projects",
  "app_settings",
  "plugin_installed",
  "plugin_connections",
  "plugin_credential_references",
  "plugin_permission_grants",
  "plugin_discovered_resources",
  "plugin_resource_bindings",
  "plugin_environment_mapping_suggestions",
  "plugin_observed_resource_state",
  "plugin_observed_resource_state_history",
  "plugin_desired_resource_state",
  "plugin_change_plans",
  "plugin_change_plan_approvals",
  "plugin_change_plan_rejections",
  "plugin_change_applies",
  "plugin_change_verifications",
  "plugin_execution_history",
] as const;

/** @deprecated Prefer PHYSICAL_TABLES — kept as an alias for existing imports. */
export const ENTITY_TABLES = PHYSICAL_TABLES;

export interface DatabaseSchemaVersion {
  version: number;
  description: string;
}

export const CURRENT_SCHEMA_VERSION: DatabaseSchemaVersion = {
  version: 3,
  description: "Projects, app settings, and plugin persistence",
};
