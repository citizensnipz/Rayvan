import { useState } from "react";
import { Button } from "@rayvan/ui";

import {
  FINDINGS_DISMISS_REASONS,
  FINDINGS_SUPPRESS_PRESETS,
  type FindingsDismissReason,
  type FindingsSuppressPreset,
} from "../../lib/findings/index.js";
import type { FindingDetailViewModel } from "./view-models.js";

interface FindingActionsProps {
  detail: FindingDetailViewModel;
  busy?: boolean;
  onAcknowledge: () => void;
  onDismiss: (reason: FindingsDismissReason) => void;
  onSuppress: (preset: FindingsSuppressPreset) => void;
  onOpenEnvironment?: (environmentId: string) => void;
  onOpenIntegration?: (connectionId: string) => void;
  onResync?: () => void;
}

export function FindingActions({
  detail,
  busy,
  onAcknowledge,
  onDismiss,
  onSuppress,
  onOpenEnvironment,
  onOpenIntegration,
  onResync,
}: FindingActionsProps) {
  const [dismissReason, setDismissReason] =
    useState<FindingsDismissReason>("Expected behaviour");
  const [suppressPreset, setSuppressPreset] =
    useState<FindingsSuppressPreset>("until_next_sync");

  const remediation = detail.remediation;

  return (
    <div style={{ display: "grid", gap: "0.75rem" }}>
      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        {detail.canAcknowledge ? (
          <Button disabled={busy} onClick={onAcknowledge}>
            Acknowledge
          </Button>
        ) : null}
        {detail.finding.environmentId && onOpenEnvironment ? (
          <Button
            disabled={busy}
            onClick={() => onOpenEnvironment(detail.finding.environmentId!)}
          >
            Open environment
          </Button>
        ) : null}
        {detail.finding.connectionId && onOpenIntegration ? (
          <Button
            disabled={busy}
            onClick={() => onOpenIntegration(detail.finding.connectionId!)}
          >
            Open integration
          </Button>
        ) : null}
        {onResync &&
        (remediation?.type === "resync" ||
          remediation?.type === "reauthenticate") ? (
          <Button disabled={busy} onClick={onResync}>
            {remediation.type === "reauthenticate" ? "Reauthenticate" : "Resync"}
          </Button>
        ) : null}
      </div>

      {detail.canDismiss ? (
        <div
          style={{
            display: "flex",
            gap: "0.5rem",
            flexWrap: "wrap",
            alignItems: "end",
          }}
        >
          <label style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
            Dismissal reason
            <select
              aria-label="Dismissal reason"
              value={dismissReason}
              onChange={(event) =>
                setDismissReason(event.target.value as FindingsDismissReason)
              }
              style={{
                padding: "0.4rem 0.55rem",
                borderRadius: "6px",
                border: "1px solid var(--color-border)",
              }}
            >
              {FINDINGS_DISMISS_REASONS.map((reason) => (
                <option key={reason} value={reason}>
                  {reason}
                </option>
              ))}
            </select>
          </label>
          <Button disabled={busy} onClick={() => onDismiss(dismissReason)}>
            Dismiss
          </Button>
        </div>
      ) : null}

      {detail.canSuppress ? (
        <div
          style={{
            display: "flex",
            gap: "0.5rem",
            flexWrap: "wrap",
            alignItems: "end",
          }}
        >
          <label style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
            Suppress until
            <select
              aria-label="Suppress until"
              value={suppressPreset}
              onChange={(event) =>
                setSuppressPreset(event.target.value as FindingsSuppressPreset)
              }
              style={{
                padding: "0.4rem 0.55rem",
                borderRadius: "6px",
                border: "1px solid var(--color-border)",
              }}
            >
              {FINDINGS_SUPPRESS_PRESETS.map((preset) => (
                <option key={preset.id} value={preset.id}>
                  {preset.label}
                </option>
              ))}
            </select>
          </label>
          <Button disabled={busy} onClick={() => onSuppress(suppressPreset)}>
            Suppress
          </Button>
        </div>
      ) : null}

      {remediation?.type === "manual" ? (
        <div
          style={{
            padding: "0.75rem",
            borderRadius: "8px",
            border: "1px dashed var(--color-border-strong)",
            background: "var(--color-surface-muted)",
          }}
        >
          <strong>{remediation.label}</strong>
          <p style={{ margin: "0.35rem 0 0", color: "var(--color-text-secondary)" }}>
            {remediation.instructions}
          </p>
        </div>
      ) : null}

      {remediation?.type === "generate_change_plan" ? (
        <p style={{ margin: 0, color: "var(--color-text-secondary)", fontSize: "0.9rem" }}>
          {remediation.label}: open the environment and review an apply plan. Rayvan does
          not execute infrastructure mutations from Findings.
        </p>
      ) : null}
    </div>
  );
}
