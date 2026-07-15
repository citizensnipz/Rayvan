import type { CSSProperties } from "react";
import { Button } from "@rayvan/ui";

import type { EnvironmentSyncState } from "../../lib/environments/index.js";

const bannerStyle: CSSProperties = {
  marginBottom: "1rem",
  padding: "0.75rem 1rem",
  borderRadius: "8px",
  border: "1px solid var(--color-border-strong)",
  background: "var(--color-surface-muted)",
};

interface SyncEnvironmentsBannerProps {
  syncState: EnvironmentSyncState | null;
  onSync: () => void;
  onCancel: () => void;
  disabled?: boolean;
}

export function SyncEnvironmentsBanner({
  syncState,
  onSync,
  onCancel,
  disabled,
}: SyncEnvironmentsBannerProps) {
  const inProgress = syncState?.inProgress ?? false;
  const last = syncState?.lastResult;
  const failedPlugins =
    last?.plugins.filter((plugin) => plugin.status === "failed").map((plugin) => plugin.pluginId) ??
    [];

  return (
    <div style={bannerStyle} role="status">
      <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap" }}>
        <div>
          <strong>Sync with integrations</strong>
          <p style={{ margin: "0.25rem 0 0", color: "var(--color-text-secondary)", fontSize: "0.9rem" }}>
            Read-only discovery. Rayvan never writes to providers from this action.
          </p>
          {inProgress ? (
            <p style={{ margin: "0.35rem 0 0", fontSize: "0.9rem" }}>
              {syncState?.progressLabel ?? `Syncing (${syncState?.phase})…`}
            </p>
          ) : null}
          {!inProgress && last ? (
            <p style={{ margin: "0.35rem 0 0", fontSize: "0.9rem" }}>
              Last sync: {last.phase}
              {failedPlugins.length > 0
                ? ` · partial failure (${failedPlugins.join(", ")})`
                : null}
              {last.cancelled ? " · cancelled" : null}
            </p>
          ) : null}
        </div>
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "flex-start" }}>
          {inProgress ? (
            <Button onClick={onCancel}>Cancel sync</Button>
          ) : (
            <Button onClick={onSync} disabled={disabled}>
              Sync environments
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
