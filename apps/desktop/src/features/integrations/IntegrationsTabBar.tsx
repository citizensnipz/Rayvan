import type { CSSProperties, KeyboardEvent } from "react";

import { tabId, tabKey, tabPanelId } from "./tab-ids.js";
import type { IntegrationTab } from "./view-models.js";

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
    borderBottom: active ? "2px solid var(--color-text)" : "2px solid transparent",
    background: active ? "var(--color-surface-muted)" : "transparent",
    color: "var(--color-text)",
    cursor: "pointer",
    flexShrink: 0,
    fontWeight: active ? 600 : 400,
  };
}

function tabLabel(tab: IntegrationTab): string {
  return tab.kind === "home" ? "Home" : tab.label;
}

interface IntegrationsTabBarProps {
  tabs: IntegrationTab[];
  activeTabKey: string;
  onSelect: (key: string) => void;
  onClose: (connectionId: string) => void;
}

export function IntegrationsTabBar({
  tabs,
  activeTabKey,
  onSelect,
  onClose,
}: IntegrationsTabBarProps) {
  function focusTabAt(index: number) {
    const next = tabs[index];
    if (!next) {
      return;
    }
    const key = tabKey(next);
    onSelect(key);
    window.requestAnimationFrame(() => {
      document.getElementById(tabId(key))?.focus();
    });
  }

  function handleTabKeyDown(
    event: KeyboardEvent<HTMLButtonElement>,
    index: number,
  ) {
    if (event.key === "ArrowRight") {
      event.preventDefault();
      focusTabAt((index + 1) % tabs.length);
      return;
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      focusTabAt((index - 1 + tabs.length) % tabs.length);
      return;
    }
    if (event.key === "Home") {
      event.preventDefault();
      focusTabAt(0);
      return;
    }
    if (event.key === "End") {
      event.preventDefault();
      focusTabAt(tabs.length - 1);
    }
  }

  return (
    <div role="tablist" aria-label="Integrations tabs" style={tabListStyle}>
      {tabs.map((tab, index) => {
        const key = tabKey(tab);
        const active = key === activeTabKey;
        const label = tabLabel(tab);

        return (
          <div key={key} style={{ display: "inline-flex", alignItems: "center" }}>
            <button
              type="button"
              id={tabId(key)}
              role="tab"
              aria-selected={active}
              aria-controls={tabPanelId(key)}
              tabIndex={active ? 0 : -1}
              style={tabButtonStyle(active)}
              onClick={() => onSelect(key)}
              onKeyDown={(event) => handleTabKeyDown(event, index)}
            >
              {label}
            </button>
            {tab.kind === "detail" ? (
              <button
                type="button"
                aria-label={`Close ${label} tab`}
                style={{
                  border: "none",
                  background: "transparent",
                  color: "var(--color-text-muted)",
                  cursor: "pointer",
                  fontSize: "0.85rem",
                  lineHeight: 1,
                  padding: "0 0.35rem",
                }}
                onClick={() => onClose(tab.connectionId)}
              >
                &times;
              </button>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
