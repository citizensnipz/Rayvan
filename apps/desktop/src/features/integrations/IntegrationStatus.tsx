import type { CSSProperties } from "react";

import type { IntegrationStatus } from "./view-models.js";

interface StatusMeta {
  symbol: string;
  color: string;
}

const STATUS_META: Record<IntegrationStatus, StatusMeta> = {
  connected: { symbol: "\u2713", color: "#16a34a" },
  syncing: { symbol: "\u27F3", color: "#2563eb" },
  attention_required: { symbol: "\u26A0", color: "#d97706" },
  expired: { symbol: "\u23F3", color: "#d97706" },
  disconnected: { symbol: "\u25CB", color: "var(--color-text-muted)" },
  error: { symbol: "\u2715", color: "var(--color-danger)" },
};

interface IntegrationStatusIndicatorProps {
  status: IntegrationStatus;
  label: string;
}

/**
 * Renders integration status as text + a distinct symbol, so meaning never
 * depends on color alone.
 */
export function IntegrationStatusIndicator({
  status,
  label,
}: IntegrationStatusIndicatorProps) {
  const meta = STATUS_META[status];
  const style: CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    gap: "0.35rem",
    fontSize: "0.85rem",
    color: "var(--color-text-secondary)",
  };

  return (
    <span style={style} data-integration-status={status}>
      <span aria-hidden="true" style={{ color: meta.color, fontWeight: 700 }}>
        {meta.symbol}
      </span>
      <span>{label}</span>
    </span>
  );
}
