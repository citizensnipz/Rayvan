export const PLACEHOLDER_TOOL_NAMES = [
  "list_projects",
  "get_project",
  "list_environments",
  "inspect_environment",
  "list_integrations",
  "inspect_integration",
  "compare_configuration",
  "list_findings",
  "explain_finding",
  "create_action_plan",
  "get_action_plan",
  "request_action_approval",
  "execute_approved_action",
] as const;

export type PlaceholderToolName = (typeof PLACEHOLDER_TOOL_NAMES)[number];
