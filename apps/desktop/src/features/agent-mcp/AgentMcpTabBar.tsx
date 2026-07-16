import type { CSSProperties, KeyboardEvent } from "react";

import { tabId, tabPanelId } from "./tab-ids.js";
import { AGENT_MCP_TABS, type AgentMcpTabId } from "./view-models.js";

const tabListStyle: CSSProperties = {
  display: "flex",
  gap: "0.25rem",
  overflowX: "auto",
  whiteSpace: "nowrap",
  borderBottom: "1px solid var(--color-border)",
  paddingBottom: "0.25rem",
  marginBottom: "1rem",
};

function tabButtonStyle(active: boolean): CSSProperties {
  return {
    display: "inline-flex",
    alignItems: "center",
    gap: "0.4rem",
    padding: "0.4rem 0.75rem",
    borderRadius: "6px 6px 0 0",
    border: "none",
    borderBottom: active
      ? "2px solid var(--color-text)"
      : "2px solid transparent",
    background: active ? "var(--color-surface-muted)" : "transparent",
    color: "var(--color-text)",
    cursor: "pointer",
    flexShrink: 0,
    fontWeight: active ? 600 : 400,
  };
}

interface AgentMcpTabBarProps {
  activeTab: AgentMcpTabId;
  onSelect: (id: AgentMcpTabId) => void;
}

export function AgentMcpTabBar({ activeTab, onSelect }: AgentMcpTabBarProps) {
  function focusTabAt(index: number) {
    const next = AGENT_MCP_TABS[index];
    if (!next) {
      return;
    }
    onSelect(next.id);
    window.requestAnimationFrame(() => {
      document.getElementById(tabId(next.id))?.focus();
    });
  }

  function handleTabKeyDown(
    event: KeyboardEvent<HTMLButtonElement>,
    index: number,
  ) {
    if (event.key === "ArrowRight") {
      event.preventDefault();
      focusTabAt((index + 1) % AGENT_MCP_TABS.length);
      return;
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      focusTabAt((index - 1 + AGENT_MCP_TABS.length) % AGENT_MCP_TABS.length);
      return;
    }
    if (event.key === "Home") {
      event.preventDefault();
      focusTabAt(0);
      return;
    }
    if (event.key === "End") {
      event.preventDefault();
      focusTabAt(AGENT_MCP_TABS.length - 1);
    }
  }

  return (
    <div role="tablist" aria-label="Agent / MCP tabs" style={tabListStyle}>
      {AGENT_MCP_TABS.map((tab, index) => {
        const active = tab.id === activeTab;
        return (
          <button
            key={tab.id}
            type="button"
            id={tabId(tab.id)}
            role="tab"
            aria-selected={active}
            aria-controls={tabPanelId(tab.id)}
            tabIndex={active ? 0 : -1}
            style={tabButtonStyle(active)}
            onClick={() => onSelect(tab.id)}
            onKeyDown={(event) => handleTabKeyDown(event, index)}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
