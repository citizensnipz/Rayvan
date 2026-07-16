import type { CSSProperties, KeyboardEvent } from "react";

import { SeverityBadge } from "./severity.js";
import type { FindingListItemViewModel } from "./view-models.js";

const itemStyle: CSSProperties = {
  display: "block",
  width: "100%",
  textAlign: "left",
  padding: "0.75rem 0.85rem",
  borderRadius: "8px",
  border: "1px solid var(--color-border)",
  background: "var(--color-surface)",
  cursor: "pointer",
  color: "var(--color-text)",
};

interface FindingListItemProps {
  item: FindingListItemViewModel;
  selected?: boolean;
  onOpen: (findingId: string) => void;
}

export function FindingListItem({ item, selected, onOpen }: FindingListItemProps) {
  function handleKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onOpen(item.findingId);
    }
  }

  const absoluteTitle = `${item.title} — ${item.severityLabel}, ${item.statusLabel}, last detected ${item.lastDetectedAt}`;

  return (
    <button
      type="button"
      style={{
        ...itemStyle,
        borderColor: selected ? "var(--color-border-strong)" : "var(--color-border)",
        background: selected ? "var(--color-surface-muted)" : "var(--color-surface)",
      }}
      aria-label={`${item.title}. Severity ${item.severityLabel}. Status ${item.statusLabel}. Last detected ${item.lastDetectedAt}`}
      title={absoluteTitle}
      onClick={() => onOpen(item.findingId)}
      onKeyDown={handleKeyDown}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          gap: "0.75rem",
          flexWrap: "wrap",
        }}
      >
        <strong style={{ minWidth: 0 }}>{item.title}</strong>
        <span style={{ flexShrink: 0 }}>
          <SeverityBadge severity={item.severity} label={item.severityLabel} />
        </span>
      </div>
      <div
        style={{
          marginTop: "0.35rem",
          fontSize: "0.85rem",
          color: "var(--color-text-secondary)",
        }}
      >
        <span aria-label={`Status: ${item.statusLabel}`}>{item.statusLabel}</span>
        {" · "}
        {item.categoryLabel}
        {item.environmentLabel ? ` · ${item.environmentLabel}` : ""}
        {item.integrationLabel ? ` · ${item.integrationLabel}` : ""}
      </div>
      <p
        style={{
          margin: "0.35rem 0 0",
          fontSize: "0.85rem",
          color: "var(--color-text-secondary)",
        }}
      >
        {item.summary}
      </p>
      <div
        style={{
          marginTop: "0.35rem",
          fontSize: "0.75rem",
          color: "var(--color-text-muted)",
        }}
        title={item.lastDetectedAt}
      >
        {item.sourceLabel} · Detected {item.lastDetectedLabel}
        {item.remediable ? " · Remediable" : ""}
      </div>
    </button>
  );
}
