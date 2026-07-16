export type AgentMcpTabId =
  | "overview"
  | "clients"
  | "tools"
  | "approvals"
  | "activity"
  | "setup"
  | "daemon";

export interface AgentMcpTab {
  id: AgentMcpTabId;
  label: string;
}

export const AGENT_MCP_TABS: readonly AgentMcpTab[] = [
  { id: "overview", label: "Overview" },
  { id: "clients", label: "Clients" },
  { id: "tools", label: "Tools / Capabilities" },
  { id: "approvals", label: "Approvals" },
  { id: "activity", label: "Activity" },
  { id: "setup", label: "Setup" },
  { id: "daemon", label: "Daemon" },
] as const;

export interface AgentOverviewViewModel {
  mcpClientCount: number;
  activeMcpClientCount: number;
  pendingApprovalCount: number;
  recentOperationCount: number;
  daemonConnected: boolean;
  daemonVersion?: string;
  daemonState?: string;
}

export interface AgentClientRowViewModel {
  id: string;
  name: string;
  type: string;
  status: string;
  profile: string;
  connected: boolean;
  projectScopeLabel: string;
  createdAtLabel: string;
}

export interface AgentApprovalRowViewModel {
  id: string;
  summary: string;
  type: string;
  status: string;
  projectId?: string;
  createdAtLabel: string;
}

export interface AgentActivityRowViewModel {
  id: string;
  kind: "operation" | "audit";
  title: string;
  status?: string;
  timestampLabel: string;
}

export interface AgentCapabilityViewModel {
  permissions: string[];
  methods: string[];
}

export interface AgentDaemonViewModel {
  connected: boolean;
  endpoint: string;
  daemonVersion?: string;
  sessionId?: string;
  authenticatedClientId?: string;
  spawned: boolean;
  lastError?: string;
  state?: string;
  connectedClients?: number;
  pendingApprovals?: number;
  activeOperations?: number;
  pluginHostStatus?: string;
  diagnosticsJson?: string;
}

export interface AgentSetupViewModel {
  exampleClientId: string;
  commandSnippet: string;
  notes: string[];
}
