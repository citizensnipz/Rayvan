import type { FindingsTab } from "./view-models.js";

export function tabKey(tab: FindingsTab): string {
  switch (tab.kind) {
    case "list":
      return "list";
    case "detail":
      return `detail:${tab.findingId}`;
  }
}

export function tabPanelId(key: string): string {
  return `findings-panel-${key}`;
}

export function tabId(key: string): string {
  return `findings-tab-${key}`;
}

export function tabLabel(tab: FindingsTab): string {
  switch (tab.kind) {
    case "list":
      return "Findings";
    case "detail":
      return tab.label;
  }
}

export function isClosableTab(tab: FindingsTab): boolean {
  return tab.kind === "detail";
}
