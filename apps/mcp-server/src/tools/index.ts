import type { DaemonClient } from "@rayvan/daemon-client";
import { DaemonMethods, type DaemonSerializedError } from "@rayvan/daemon-contracts";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ToolAnnotations } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

type DaemonCaller = Pick<DaemonClient, "call">;
type Risk = "read" | "local_mutation" | "sync" | "approval" | "remote_mutation";

interface ToolDefinition {
  name: string;
  method: string;
  description: string;
  inputSchema: z.ZodTypeAny;
  risk: Risk;
}

const empty = z.object({}).strict();
const project = z.object({ project_id: z.string().min(1) }).strict();
const environment = z.object({ environment_id: z.string().min(1) }).strict();
const finding = z.object({ finding_id: z.string().min(1) }).strict();
const plan = z.object({ change_plan_id: z.string().min(1) }).strict();
const operation = z.object({ operation_id: z.string().min(1) }).strict();

export const RAYVAN_TOOLS: readonly ToolDefinition[] = [
  {
    name: "get_daemon_status",
    method: DaemonMethods.status,
    description: "Get rayvand health and version status.",
    inputSchema: empty,
    risk: "read",
  },
  {
    name: "get_daemon_diagnostics",
    method: DaemonMethods.diagnostics,
    description: "Get redacted local daemon diagnostics.",
    inputSchema: empty,
    risk: "read",
  },
  {
    name: "list_projects",
    method: DaemonMethods.listProjects,
    description: "List projects visible to this MCP client.",
    inputSchema: z.object({ include_archived: z.boolean().optional() }).strict(),
    risk: "read",
  },
  {
    name: "get_project",
    method: DaemonMethods.getProject,
    description: "Get one project by id.",
    inputSchema: project,
    risk: "read",
  },
  {
    name: "get_project_overview",
    method: DaemonMethods.getProjectOverview,
    description: "Get a project's environment and finding overview.",
    inputSchema: project,
    risk: "read",
  },
  {
    name: "create_project",
    method: DaemonMethods.createProject,
    description: "Create a project in Rayvan's local store.",
    inputSchema: z
      .object({ name: z.string().min(1), description: z.string().optional() })
      .strict(),
    risk: "local_mutation",
  },
  {
    name: "update_project",
    method: DaemonMethods.updateProject,
    description: "Update local project metadata.",
    inputSchema: project.extend({
      name: z.string().min(1).optional(),
      description: z.string().optional(),
      expected_revision: z.number().int().nonnegative().optional(),
    }),
    risk: "local_mutation",
  },
  {
    name: "list_environments",
    method: DaemonMethods.listEnvironments,
    description: "List environments in a project.",
    inputSchema: project.extend({ include_archived: z.boolean().optional() }),
    risk: "read",
  },
  {
    name: "get_environment",
    method: DaemonMethods.getEnvironment,
    description: "Get one environment by id.",
    inputSchema: environment,
    risk: "read",
  },
  {
    name: "create_environment",
    method: DaemonMethods.createEnvironment,
    description: "Create an environment in Rayvan's local store.",
    inputSchema: z
      .object({
        project_id: z.string().min(1),
        name: z.string().min(1),
        kind: z.string().min(1),
        slug: z.string().optional(),
        description: z.string().optional(),
      })
      .strict(),
    risk: "local_mutation",
  },
  {
    name: "update_environment",
    method: DaemonMethods.updateEnvironment,
    description: "Update local environment metadata.",
    inputSchema: environment.extend({
      name: z.string().optional(),
      description: z.string().optional(),
      kind: z.string().optional(),
      status: z.string().optional(),
    }),
    risk: "local_mutation",
  },
  {
    name: "archive_environment",
    method: DaemonMethods.archiveEnvironment,
    description: "Archive an environment locally.",
    inputSchema: environment,
    risk: "local_mutation",
  },
  {
    name: "compare_environments",
    method: DaemonMethods.compareEnvironments,
    description: "Compare two environments in one project.",
    inputSchema: z
      .object({
        project_id: z.string().min(1),
        left_environment_id: z.string().min(1),
        right_environment_id: z.string().min(1),
      })
      .strict(),
    risk: "read",
  },
  {
    name: "list_configuration_keys",
    method: DaemonMethods.listConfigurationKeys,
    description: "List managed configuration keys for a project.",
    inputSchema: project,
    risk: "read",
  },
  {
    name: "get_configuration_key",
    method: DaemonMethods.getConfigurationKey,
    description: "Get a sanitized configuration key.",
    inputSchema: z.object({ configuration_key_id: z.string().min(1) }).strict(),
    risk: "read",
  },
  {
    name: "find_configuration_usage",
    method: DaemonMethods.findConfigurationUsage,
    description:
      "Find sanitized provider and environment usage for a configuration key.",
    inputSchema: z.object({ configuration_key_id: z.string().min(1) }).strict(),
    risk: "read",
  },
  {
    name: "get_configuration_status",
    method: DaemonMethods.getConfigurationStatus,
    description: "Get desired, applied, and observed configuration status.",
    inputSchema: z
      .object({ project_id: z.string().min(1), environment_id: z.string().optional() })
      .strict(),
    risk: "read",
  },
  {
    name: "list_unmanaged_configuration",
    method: DaemonMethods.listUnmanagedConfiguration,
    description: "List discovered configuration that is not yet managed.",
    inputSchema: project.extend({ environment_id: z.string().optional() }),
    risk: "read",
  },
  {
    name: "get_environment_configuration",
    method: DaemonMethods.getEnvironmentConfiguration,
    description: "Get sanitized desired configuration for an environment.",
    inputSchema: z
      .object({ project_id: z.string().min(1), environment_id: z.string().min(1) })
      .strict(),
    risk: "read",
  },
  {
    name: "set_configuration_value",
    method: DaemonMethods.setConfigurationValue,
    description: "Set a non-sensitive desired value locally.",
    inputSchema: z
      .object({
        project_id: z.string().min(1),
        environment_id: z.string().min(1),
        configuration_key_id: z.string().min(1),
        value: z.string(),
        expected_revision: z.number().int().nonnegative().optional(),
      })
      .strict(),
    risk: "local_mutation",
  },
  {
    name: "set_sensitive_configuration_value",
    method: DaemonMethods.setSensitiveConfigurationValue,
    description:
      "Store a sensitive desired value through rayvand; the value is never returned.",
    inputSchema: z
      .object({
        project_id: z.string().min(1),
        environment_id: z.string().min(1),
        configuration_key_id: z.string().min(1),
        secret_value: z.string().min(1),
        expected_revision: z.number().int().nonnegative().optional(),
      })
      .strict(),
    risk: "local_mutation",
  },
  {
    name: "clear_configuration_value",
    method: DaemonMethods.clearConfigurationValue,
    description: "Clear a desired configuration value locally.",
    inputSchema: z
      .object({
        project_id: z.string().min(1),
        environment_id: z.string().min(1),
        configuration_key_id: z.string().min(1),
        expected_revision: z.number().int().nonnegative(),
      })
      .strict(),
    risk: "local_mutation",
  },
  {
    name: "set_configuration_metadata",
    method: DaemonMethods.setConfigurationMetadata,
    description: "Update local metadata for a configuration key.",
    inputSchema: z
      .object({
        project_id: z.string().min(1),
        configuration_key_id: z.string().min(1),
        description: z.string().optional(),
        value_type: z
          .enum(["string", "number", "boolean", "url", "json", "secret", "unknown"])
          .optional(),
        required: z.boolean().optional(),
        sensitive: z.boolean().optional(),
      })
      .strict(),
    risk: "local_mutation",
  },
  {
    name: "set_configuration_targets",
    method: DaemonMethods.setConfigurationTargets,
    description:
      "Bind configuration occurrences to a resource binding (targets via resourceBindingId).",
    inputSchema: z
      .object({
        project_id: z.string().min(1),
        configuration_key_id: z.string().min(1),
        resource_binding_id: z.string().min(1),
        occurrence_ids: z.array(z.string().min(1)).optional(),
      })
      .strict(),
    risk: "local_mutation",
  },
  {
    name: "remove_configuration_target",
    method: DaemonMethods.removeConfigurationTarget,
    description: "Clear the resource binding target from a configuration occurrence.",
    inputSchema: z
      .object({
        project_id: z.string().min(1),
        occurrence_id: z.string().min(1),
      })
      .strict(),
    risk: "local_mutation",
  },
  {
    name: "adopt_discovered_configuration",
    method: DaemonMethods.adoptDiscoveredConfiguration,
    description: "Adopt discovered configuration into managed desired state.",
    inputSchema: z
      .object({
        project_id: z.string().min(1),
        occurrence_id: z.string().min(1),
        environment_id: z.string().optional(),
        resource_binding_id: z.string().optional(),
      })
      .strict(),
    risk: "local_mutation",
  },
  {
    name: "ignore_discovered_configuration",
    method: DaemonMethods.ignoreDiscoveredConfiguration,
    description: "Ignore a discovered configuration occurrence.",
    inputSchema: z
      .object({
        project_id: z.string().min(1),
        occurrence_id: z.string().min(1),
      })
      .strict(),
    risk: "local_mutation",
  },
  {
    name: "list_findings",
    method: DaemonMethods.listFindings,
    description: "List findings for a project.",
    inputSchema: project.extend({
      status: z.string().optional(),
      environment_id: z.string().optional(),
    }),
    risk: "read",
  },
  {
    name: "get_finding",
    method: DaemonMethods.getFinding,
    description: "Get one finding.",
    inputSchema: finding,
    risk: "read",
  },
  {
    name: "explain_finding",
    method: DaemonMethods.explainFinding,
    description: "Get a finding with its safe explanation and evidence.",
    inputSchema: finding,
    risk: "read",
  },
  {
    name: "get_finding_summary",
    method: DaemonMethods.getFindingSummary,
    description: "Summarize findings for a project.",
    inputSchema: project,
    risk: "read",
  },
  {
    name: "scan_findings",
    method: DaemonMethods.scanFindings,
    description: "Run the daemon's local findings scan.",
    inputSchema: project.extend({ idempotency_key: z.string().optional() }),
    risk: "local_mutation",
  },
  {
    name: "acknowledge_finding",
    method: DaemonMethods.acknowledgeFinding,
    description: "Acknowledge a finding locally.",
    inputSchema: finding.extend({ comment: z.string().optional() }),
    risk: "local_mutation",
  },
  {
    name: "dismiss_finding",
    method: DaemonMethods.dismissFinding,
    description: "Dismiss a finding locally.",
    inputSchema: finding.extend({ reason: z.string().optional() }),
    risk: "local_mutation",
  },
  {
    name: "suppress_finding",
    method: DaemonMethods.suppressFinding,
    description: "Suppress a finding locally for the daemon-defined period.",
    inputSchema: finding.extend({ reason: z.string().optional() }),
    risk: "local_mutation",
  },
  {
    name: "reopen_finding",
    method: DaemonMethods.reopenFinding,
    description: "Reopen a dismissed, resolved, or suppressed finding.",
    inputSchema: finding,
    risk: "local_mutation",
  },
  {
    name: "generate_change_plan",
    method: DaemonMethods.generateChangePlan,
    description:
      "Generate a change plan for a resource binding via the daemon plugin host.",
    inputSchema: z
      .object({
        project_id: z.string().min(1),
        environment_id: z.string().optional(),
        resource_binding_id: z.string().optional(),
        desired_attributes: z.record(z.unknown()).optional(),
      })
      .strict(),
    risk: "local_mutation",
  },
  {
    name: "generate_plan_from_finding",
    method: DaemonMethods.generatePlanFromFinding,
    description: "Generate a local change plan from a finding.",
    inputSchema: finding,
    risk: "local_mutation",
  },
  {
    name: "list_change_plans",
    method: DaemonMethods.listChangePlans,
    description: "List change plans for a project.",
    inputSchema: project,
    risk: "read",
  },
  {
    name: "get_change_plan",
    method: DaemonMethods.getChangePlan,
    description: "Get one change plan.",
    inputSchema: plan,
    risk: "read",
  },
  {
    name: "approve_change_plan",
    method: DaemonMethods.approveChangePlan,
    description: "Approve a plan or enqueue daemon-controlled desktop approval.",
    inputSchema: plan,
    risk: "approval",
  },
  {
    name: "reject_change_plan",
    method: DaemonMethods.rejectChangePlan,
    description: "Reject a change plan.",
    inputSchema: plan.extend({ reason: z.string().optional() }),
    risk: "approval",
  },
  {
    name: "apply_change_plan",
    method: DaemonMethods.applyChangePlan,
    description: "Apply an already-approved plan through the daemon security boundary.",
    inputSchema: plan.extend({ idempotency_key: z.string().optional() }),
    risk: "remote_mutation",
  },
  {
    name: "verify_change_plan",
    method: DaemonMethods.verifyChangePlan,
    description: "Verify a previously applied change plan.",
    inputSchema: plan,
    risk: "sync",
  },
  {
    name: "retry_failed_change",
    method: DaemonMethods.retryFailedChange,
    description:
      "Retry a failed change apply after verification rules (never blind retry of interrupted apply).",
    inputSchema: plan.extend({ idempotency_key: z.string().optional() }),
    risk: "remote_mutation",
  },
  {
    name: "list_operations",
    method: DaemonMethods.listOperations,
    description: "List daemon operations.",
    inputSchema: z
      .object({ project_id: z.string().optional(), status: z.string().optional() })
      .strict(),
    risk: "read",
  },
  {
    name: "get_operation",
    method: DaemonMethods.getOperation,
    description: "Get one daemon operation.",
    inputSchema: operation,
    risk: "read",
  },
  {
    name: "cancel_operation",
    method: DaemonMethods.cancelOperation,
    description: "Request cancellation of a cancellable daemon operation.",
    inputSchema: operation,
    risk: "local_mutation",
  },
  {
    name: "list_approvals",
    method: DaemonMethods.listApprovals,
    description: "List approval requests.",
    inputSchema: z
      .object({ project_id: z.string().optional(), status: z.string().optional() })
      .strict(),
    risk: "read",
  },
  {
    name: "decide_approval",
    method: DaemonMethods.decideApproval,
    description: "Approve or deny a pending daemon approval request.",
    inputSchema: z
      .object({
        approval_id: z.string().min(1),
        decision: z.enum(["approved", "denied"]),
        remember_scope: z.boolean().optional(),
      })
      .strict(),
    risk: "approval",
  },
  {
    name: "list_integrations",
    method: DaemonMethods.listIntegrations,
    description: "List integrations visible to this client.",
    inputSchema: z.object({ project_id: z.string().optional() }).strict(),
    risk: "read",
  },
  {
    name: "get_integration",
    method: DaemonMethods.getIntegration,
    description: "Get an integration snapshot.",
    inputSchema: z
      .object({ project_id: z.string().optional(), integration_id: z.string().min(1) })
      .strict(),
    risk: "read",
  },
  {
    name: "list_integration_resources",
    method: DaemonMethods.listIntegrationResources,
    description: "List resources discovered for an integration.",
    inputSchema: z
      .object({ project_id: z.string().optional(), integration_id: z.string().min(1) })
      .strict(),
    risk: "read",
  },
  {
    name: "get_integration_health",
    method: DaemonMethods.getIntegrationHealth,
    description: "Get integration health.",
    inputSchema: z
      .object({ project_id: z.string().optional(), integration_id: z.string().min(1) })
      .strict(),
    risk: "read",
  },
  {
    name: "get_environment_resources",
    method: DaemonMethods.getEnvironmentResources,
    description: "List resources in an environment.",
    inputSchema: z
      .object({ project_id: z.string().optional(), environment_id: z.string().min(1) })
      .strict(),
    risk: "read",
  },
  {
    name: "sync_project",
    method: DaemonMethods.syncProject,
    description: "Synchronize integrations for a project.",
    inputSchema: project.extend({ idempotency_key: z.string().optional() }),
    risk: "sync",
  },
  {
    name: "sync_environment",
    method: DaemonMethods.syncEnvironment,
    description: "Synchronize an environment.",
    inputSchema: z
      .object({
        project_id: z.string().min(1),
        environment_id: z.string().min(1),
        idempotency_key: z.string().optional(),
      })
      .strict(),
    risk: "sync",
  },
  {
    name: "sync_integration",
    method: DaemonMethods.syncIntegration,
    description: "Synchronize an integration.",
    inputSchema: z
      .object({
        project_id: z.string().min(1),
        integration_id: z.string().min(1),
        idempotency_key: z.string().optional(),
      })
      .strict(),
    risk: "sync",
  },
  {
    name: "inspect_resource",
    method: DaemonMethods.inspectResource,
    description: "Inspect a resource through the daemon.",
    inputSchema: z
      .object({ project_id: z.string().optional(), resource_id: z.string().min(1) })
      .strict(),
    risk: "read",
  },
  {
    name: "list_plugins",
    method: DaemonMethods.listPlugins,
    description: "List locally available daemon plugins.",
    inputSchema: empty,
    risk: "read",
  },
  {
    name: "list_plugin_actions",
    method: DaemonMethods.listPluginActions,
    description: "List actions exposed by daemon plugins.",
    inputSchema: z.object({ plugin_id: z.string().optional() }).strict(),
    risk: "read",
  },
  {
    name: "get_mcp_client_scope",
    method: DaemonMethods.getMcpClientScope,
    description: "Get this client's effective permissions and scopes.",
    inputSchema: empty,
    risk: "read",
  },
  {
    name: "list_available_capabilities",
    method: DaemonMethods.listAvailableCapabilities,
    description: "List daemon methods and permissions available to this client.",
    inputSchema: empty,
    risk: "read",
  },
] as const;

export const RAYVAN_TOOL_NAMES = RAYVAN_TOOLS.map((tool) => tool.name);
const outputSchema = z.object({ data: z.unknown() });

export function registerRayvanTools(server: McpServer, daemon: DaemonCaller): void {
  for (const tool of RAYVAN_TOOLS) {
    server.registerTool(
      tool.name,
      {
        description: tool.description,
        inputSchema: tool.inputSchema,
        outputSchema,
        annotations: annotationsFor(tool),
        _meta: { "rayvan/risk": tool.risk, "rayvan/daemon-method": tool.method },
      },
      async (rawInput, extra) => {
        const params = toDaemonParams(rawInput as Record<string, unknown>);
        try {
          const result = await callWithCancellation(
            daemon,
            tool.method,
            params,
            extra.signal,
          );
          const data = result === undefined ? null : result;
          return {
            structuredContent: { data },
            content: [{ type: "text", text: summarize(tool.name, data) }],
          };
        } catch (error) {
          return daemonErrorResult(error);
        }
      },
    );
  }
}

function annotationsFor(tool: ToolDefinition): ToolAnnotations {
  const readOnly = tool.risk === "read";
  return {
    title: tool.name.replaceAll("_", " "),
    readOnlyHint: readOnly,
    destructiveHint: tool.risk === "remote_mutation",
    idempotentHint:
      readOnly || tool.name.startsWith("get_") || tool.name.startsWith("list_"),
    openWorldHint: tool.risk === "sync" || tool.risk === "remote_mutation",
  };
}

function toDaemonParams(input: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(input).map(([key, value]) => [
      key.replace(/_([a-z])/g, (_, letter: string) => letter.toUpperCase()),
      value,
    ]),
  );
}

async function callWithCancellation(
  daemon: DaemonCaller,
  method: string,
  params: Record<string, unknown>,
  signal: AbortSignal,
): Promise<unknown> {
  let operationId =
    typeof params.operationId === "string" ? params.operationId : undefined;
  const cancel = () => {
    if (operationId && method !== DaemonMethods.cancelOperation) {
      void daemon
        .call(DaemonMethods.cancelOperation, { operationId })
        .catch(() => undefined);
    }
  };
  signal.addEventListener("abort", cancel, { once: true });
  try {
    const result = await daemon.call(method, params);
    operationId ??= extractOperationId(result);
    if (signal.aborted) cancel();
    return result;
  } finally {
    signal.removeEventListener("abort", cancel);
  }
}

function extractOperationId(value: unknown): string | undefined {
  if (!value || typeof value !== "object") return undefined;
  const record = value as Record<string, unknown>;
  if (typeof record.id === "string" && typeof record.status === "string")
    return record.id;
  if (record.operation && typeof record.operation === "object") {
    const id = (record.operation as Record<string, unknown>).id;
    return typeof id === "string" ? id : undefined;
  }
  return undefined;
}

function summarize(toolName: string, data: unknown): string {
  if (Array.isArray(data))
    return `${toolName} returned ${data.length} item${data.length === 1 ? "" : "s"}.`;
  if (data === null) return `${toolName} returned no matching result.`;
  return `${toolName} completed through rayvand.`;
}

function daemonErrorResult(error: unknown) {
  const candidate = error as Error & { code?: string; data?: DaemonSerializedError };
  const daemonError = candidate.data;
  const code = daemonError?.code ?? candidate.code ?? "DAEMON_UNAVAILABLE";
  const message =
    daemonError?.message ??
    (candidate instanceof Error ? candidate.message : "Daemon request failed");
  const safeError = {
    code,
    message,
    retryable: daemonError?.retryable ?? false,
    ...(daemonError?.correlationId ? { correlationId: daemonError.correlationId } : {}),
  };
  return {
    isError: true as const,
    structuredContent: { data: { error: safeError } },
    content: [
      { type: "text" as const, text: `Rayvan daemon error (${code}): ${message}` },
    ],
  };
}
