import type { IntegrationTab } from "./view-models.js";

export function tabKey(tab: IntegrationTab): string {
  return tab.kind === "home" ? "home" : tab.connectionId;
}

export function tabPanelId(key: string): string {
  return `integrations-panel-${key}`;
}

export function tabId(key: string): string {
  return `integrations-tab-${key}`;
}
