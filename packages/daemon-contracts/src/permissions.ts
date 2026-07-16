export const MCP_PERMISSIONS = [
  "projects:read",
  "environments:read",
  "environments:write",
  "integrations:read",
  "integrations:sync",
  "resources:read",
  "configuration:read",
  "configuration:write",
  "configuration:read_sensitive",
  "findings:read",
  "findings:scan",
  "findings:manage",
  "plans:read",
  "plans:create",
  "plans:approve",
  "changes:apply",
  "changes:verify",
  "operations:read",
  "operations:cancel",
  "plugins:read",
  "plugins:execute_actions",
  "daemon:read",
  "mcp_clients:manage",
] as const;

export type McpPermission = (typeof MCP_PERMISSIONS)[number];

export type BuiltInPermissionProfileId =
  "read_only" | "planner" | "operator" | "administrator" | "custom";

export const BUILT_IN_PERMISSION_PROFILES: Record<
  Exclude<BuiltInPermissionProfileId, "custom">,
  readonly McpPermission[]
> = {
  read_only: [
    "projects:read",
    "environments:read",
    "integrations:read",
    "resources:read",
    "configuration:read",
    "findings:read",
    "plans:read",
    "operations:read",
    "plugins:read",
    "daemon:read",
  ],
  planner: [
    "projects:read",
    "environments:read",
    "integrations:read",
    "integrations:sync",
    "resources:read",
    "configuration:read",
    "findings:read",
    "findings:scan",
    "plans:read",
    "plans:create",
    "operations:read",
    "plugins:read",
    "daemon:read",
  ],
  operator: [
    "projects:read",
    "environments:read",
    "environments:write",
    "integrations:read",
    "integrations:sync",
    "resources:read",
    "configuration:read",
    "configuration:write",
    "findings:read",
    "findings:scan",
    "findings:manage",
    "plans:read",
    "plans:create",
    "plans:approve",
    "changes:apply",
    "changes:verify",
    "operations:read",
    "operations:cancel",
    "plugins:read",
    "plugins:execute_actions",
    "daemon:read",
  ],
  administrator: [...MCP_PERMISSIONS],
};

export type McpApprovalPolicy =
  | {
      type: "always_require_desktop_approval";
    }
  | {
      type: "allow_preapproved_scope";
      projectIds: string[];
      environmentIds?: string[];
      pluginIds?: string[];
      permissions: McpPermission[];
      allowDestructive: boolean;
    }
  | {
      type: "client_may_approve";
      allowDestructive: boolean;
    };

export type McpToolRisk =
  "read" | "local_write" | "remote_read" | "plan" | "remote_write" | "destructive";

export interface RayvanMcpToolDefinition {
  name: string;
  description: string;
  risk: McpToolRisk;
  requiredPermissions: McpPermission[];
  readOnly: boolean;
  destructive: boolean;
  idempotent: boolean;
  openWorld: boolean;
  /** Daemon method invoked by this tool. */
  daemonMethod: string;
}
