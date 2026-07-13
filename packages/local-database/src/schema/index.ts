export interface DatabaseSchemaVersion {
  version: number;
  description: string;
}

export const CURRENT_SCHEMA_VERSION: DatabaseSchemaVersion = {
  version: 1,
  description: "Initial Rayvan local metadata schema",
};

export const ENTITY_TABLES = [
  "workspaces",
  "projects",
  "environments",
  "integrations",
  "configuration_entries",
  "findings",
  "action_plans",
  "approval_records",
  "audit_events",
  "plugin_installations",
] as const;
