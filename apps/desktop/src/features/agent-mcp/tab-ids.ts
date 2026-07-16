import type { AgentMcpTabId } from "./view-models.js";

export function tabId(id: AgentMcpTabId): string {
  return `agent-mcp-tab-${id}`;
}

export function tabPanelId(id: AgentMcpTabId): string {
  return `agent-mcp-panel-${id}`;
}
