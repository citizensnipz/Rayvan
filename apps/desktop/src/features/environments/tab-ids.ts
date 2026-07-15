import type { EnvironmentTab } from "./view-models.js";

export function tabKey(tab: EnvironmentTab): string {
  switch (tab.kind) {
    case "overview":
      return "overview";
    case "matrix":
      return "matrix";
    case "resources":
      return "resources";
    case "environment":
      return `environment:${tab.environmentId}`;
    case "configurationKey":
      return `key:${tab.configurationKeyId}`;
  }
}

export function tabPanelId(key: string): string {
  return `environments-panel-${key}`;
}

export function tabId(key: string): string {
  return `environments-tab-${key}`;
}

export function tabLabel(tab: EnvironmentTab): string {
  switch (tab.kind) {
    case "overview":
      return "Overview";
    case "matrix":
      return "Configuration Matrix";
    case "resources":
      return "Resources";
    case "environment":
      return tab.label;
    case "configurationKey":
      return tab.label;
  }
}

export function isClosableTab(tab: EnvironmentTab): boolean {
  return tab.kind === "environment" || tab.kind === "configurationKey";
}
