import {
  useEffect,
  useId,
  useRef,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import { Button } from "@rayvan/ui";

import type { ConfigurationCellSelection } from "./view-models.js";

const panelStyle: CSSProperties = {
  position: "fixed",
  top: 0,
  right: 0,
  width: "min(24rem, 100%)",
  height: "100vh",
  background: "var(--color-surface)",
  borderLeft: "1px solid var(--color-border)",
  boxShadow: "var(--shadow-dialog)",
  zIndex: 30,
  padding: "1.25rem",
  overflowY: "auto",
};

interface ConfigurationCellDetailProps {
  selection: ConfigurationCellSelection | null;
  onClose: () => void;
  onOpenKey: (configurationKeyId: string, label: string) => void;
  onOpenEnvironment: (environmentId: string, label: string) => void;
}

export function ConfigurationCellDetail({
  selection,
  onClose,
  onOpenKey,
  onOpenEnvironment,
}: ConfigurationCellDetailProps) {
  const titleId = useId();
  const panelRef = useRef<HTMLElement>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!selection) {
      return;
    }
    previouslyFocusedRef.current =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    window.requestAnimationFrame(() => {
      panelRef.current?.focus();
    });
    return () => {
      previouslyFocusedRef.current?.focus();
    };
  }, [selection]);

  if (!selection) {
    return null;
  }

  const valueLabel = selection.accessLocked
    ? "Locked (value not readable)"
    : selection.safeVisibleValue ??
      (selection.requiredMissing ? "Missing required value" : "No safe visible value");

  function handleKeyDown(event: ReactKeyboardEvent<HTMLElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      onClose();
    }
  }

  return (
    <aside
      ref={panelRef}
      style={panelStyle}
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
      aria-label={`Configuration cell: ${selection.configurationKeyName} in ${selection.environmentName}`}
      tabIndex={-1}
      onKeyDown={handleKeyDown}
    >
      <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem" }}>
        <h2 id={titleId} style={{ margin: 0, fontSize: "1.1rem" }}>
          Cell detail
        </h2>
        <Button onClick={onClose}>Close</Button>
      </div>

      <dl style={{ marginTop: "1rem", display: "grid", gap: "0.65rem" }}>
        <div>
          <dt style={{ color: "var(--color-text-muted)", fontSize: "0.8rem" }}>Key</dt>
          <dd style={{ margin: 0, fontWeight: 600 }}>{selection.configurationKeyName}</dd>
        </div>
        <div>
          <dt style={{ color: "var(--color-text-muted)", fontSize: "0.8rem" }}>Environment</dt>
          <dd style={{ margin: 0, fontWeight: 600 }}>{selection.environmentName}</dd>
        </div>
        <div>
          <dt style={{ color: "var(--color-text-muted)", fontSize: "0.8rem" }}>Status</dt>
          <dd style={{ margin: 0 }} aria-label={`Status: ${selection.statusLabel}`}>
            {selection.statusLabel}
          </dd>
        </div>
        <div>
          <dt style={{ color: "var(--color-text-muted)", fontSize: "0.8rem" }}>Value</dt>
          <dd style={{ margin: 0 }} aria-label={valueLabel}>
            {valueLabel}
          </dd>
        </div>
        <div>
          <dt style={{ color: "var(--color-text-muted)", fontSize: "0.8rem" }}>Occurrences</dt>
          <dd style={{ margin: 0 }}>{selection.occurrenceIds.length}</dd>
        </div>
      </dl>

      <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", marginTop: "1.25rem" }}>
        <Button
          onClick={() =>
            onOpenKey(selection.configurationKeyId, selection.configurationKeyName)
          }
        >
          Open key detail
        </Button>
        <Button
          onClick={() =>
            onOpenEnvironment(selection.environmentId, selection.environmentName)
          }
        >
          Open environment
        </Button>
        <Button disabled title="Not yet implemented">
          Add missing key (preview)
        </Button>
        <Button disabled title="Not yet implemented">
          Copy from Staging (preview)
        </Button>
      </div>
      <p style={{ fontSize: "0.8rem", color: "var(--color-text-muted)", marginTop: "0.75rem" }}>
        Preview actions are stubs and do not write to providers.
      </p>
    </aside>
  );
}
