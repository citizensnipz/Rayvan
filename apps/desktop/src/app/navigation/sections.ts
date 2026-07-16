export const APP_SECTIONS = [
  "overview",
  "environments",
  "integrations",
  "deployments",
  "findings",
  "actions",
  "agent-mcp",
  "settings",
] as const;

export type AppSection = (typeof APP_SECTIONS)[number];

export interface AppSectionDefinition {
  id: AppSection;
  label: string;
  requiresProject: boolean;
}

export const APP_SECTION_DEFINITIONS: AppSectionDefinition[] = [
  { id: "overview", label: "Overview", requiresProject: false },
  { id: "environments", label: "Environments", requiresProject: true },
  { id: "integrations", label: "Integrations", requiresProject: true },
  { id: "deployments", label: "Deployments", requiresProject: true },
  { id: "findings", label: "Findings", requiresProject: true },
  { id: "actions", label: "Actions", requiresProject: true },
  { id: "agent-mcp", label: "Agent / MCP", requiresProject: true },
  { id: "settings", label: "Settings", requiresProject: false },
];
