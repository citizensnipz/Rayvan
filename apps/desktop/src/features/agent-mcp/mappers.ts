import type {
  ApprovalRequestRecord,
  DaemonDiagnostics,
  DaemonStatus,
  LocalClientPublicView,
  OperationRecord,
} from "@rayvan/daemon-contracts";

import type { DaemonStatusSnapshot } from "../../lib/daemon/index.js";
import type {
  AgentActivityRowViewModel,
  AgentApprovalRowViewModel,
  AgentCapabilityViewModel,
  AgentClientRowViewModel,
  AgentDaemonViewModel,
  AgentOverviewViewModel,
  AgentSetupViewModel,
} from "./view-models.js";

function formatTimestamp(value?: string): string {
  if (!value) {
    return "—";
  }
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
}

export function mapOverview(input: {
  clients: LocalClientPublicView[];
  approvals: ApprovalRequestRecord[];
  operations: OperationRecord[];
  snapshot: DaemonStatusSnapshot | null;
  status: DaemonStatus | null;
}): AgentOverviewViewModel {
  const mcpClients = input.clients.filter((client) => client.type === "mcp");
  return {
    mcpClientCount: mcpClients.length,
    activeMcpClientCount: mcpClients.filter((client) => client.status === "active")
      .length,
    pendingApprovalCount: input.approvals.filter(
      (approval) => approval.status === "pending",
    ).length,
    recentOperationCount: input.operations.length,
    daemonConnected: input.snapshot?.connected === true,
    daemonVersion: input.snapshot?.daemonVersion ?? input.status?.version,
    daemonState: input.status?.state,
  };
}

export function mapClientRows(
  clients: LocalClientPublicView[],
): AgentClientRowViewModel[] {
  return clients.map((client) => ({
    id: client.id,
    name: client.name,
    type: client.type,
    status: client.status,
    profile: client.permissionProfileId,
    connected: client.connected,
    projectScopeLabel:
      client.projectScopes.length === 0
        ? "All projects"
        : client.projectScopes.join(", "),
    createdAtLabel: formatTimestamp(client.createdAt),
  }));
}

export function mapApprovalRows(
  approvals: ApprovalRequestRecord[],
): AgentApprovalRowViewModel[] {
  return approvals.map((approval) => ({
    id: approval.id,
    summary: approval.summary,
    type: approval.type,
    status: approval.status,
    projectId: approval.projectId,
    createdAtLabel: formatTimestamp(approval.createdAt),
  }));
}

export function mapActivityRows(input: {
  operations: OperationRecord[];
  auditEvents: Array<{
    id?: string;
    type?: string;
    action?: string;
    timestamp?: string;
    createdAt?: string;
    summary?: string;
  }>;
}): AgentActivityRowViewModel[] {
  const operations = input.operations.map((operation) => ({
    id: operation.id,
    kind: "operation" as const,
    title: `${operation.type} · ${operation.projectId ?? "global"}`,
    status: operation.status,
    timestampLabel: formatTimestamp(
      operation.finishedAt ?? operation.startedAt,
    ),
  }));
  const audits = input.auditEvents.map((event, index) => ({
    id: event.id ?? `audit-${index}`,
    kind: "audit" as const,
    title: event.summary ?? event.action ?? event.type ?? "Audit event",
    timestampLabel: formatTimestamp(event.timestamp ?? event.createdAt),
  }));
  return [...operations, ...audits].sort((a, b) =>
    b.timestampLabel.localeCompare(a.timestampLabel),
  );
}

export function mapCapabilities(input: {
  permissions: string[];
  methods: string[];
}): AgentCapabilityViewModel {
  return {
    permissions: [...input.permissions].sort(),
    methods: [...input.methods].sort(),
  };
}

export function mapDaemonView(input: {
  snapshot: DaemonStatusSnapshot | null;
  status: DaemonStatus | null;
  diagnostics: DaemonDiagnostics | null;
}): AgentDaemonViewModel {
  return {
    connected: input.snapshot?.connected === true,
    endpoint: input.snapshot?.endpoint ?? "",
    daemonVersion: input.snapshot?.daemonVersion ?? input.status?.version,
    sessionId: input.snapshot?.sessionId,
    authenticatedClientId: input.snapshot?.authenticatedClientId,
    spawned: input.snapshot?.spawned ?? false,
    lastError: input.snapshot?.lastError,
    state: input.status?.state,
    connectedClients: input.status?.connectedClients,
    pendingApprovals: input.status?.pendingApprovals,
    activeOperations: input.status?.activeOperations,
    pluginHostStatus: input.status?.pluginHostStatus,
    diagnosticsJson: input.diagnostics
      ? JSON.stringify(input.diagnostics, null, 2)
      : undefined,
  };
}

export function mapSetupView(exampleClientId?: string): AgentSetupViewModel {
  const clientId = exampleClientId ?? "<registered-client-id>";
  return {
    exampleClientId: clientId,
    commandSnippet: [
      "rayvan-mcp serve --client-id " + clientId,
      "",
      "# MCP Inspector / host config (credential stays in OS keyring):",
      "{",
      '  "command": "rayvan-mcp",',
      `  "args": ["serve", "--client-id", "${clientId}"]`,
      "}",
    ].join("\n"),
    notes: [
      "Credentials are stored in the OS keyring by the daemon — never paste raw tokens into host config.",
      "Use mcpClients.create in the Clients tab to register a new MCP client, then copy only the client id.",
      "Set RAYVAN_DAEMON_BIN when rayvand is not on PATH.",
    ],
  };
}
