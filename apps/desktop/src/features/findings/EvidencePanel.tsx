import type { FindingEvidence, SafeFindingValue } from "@rayvan/core";

function renderSafeValue(value: SafeFindingValue) {
  if (value.access === "readable") {
    return <code>{value.value}</code>;
  }
  if (value.access === "masked") {
    return (
      <span aria-label="Masked sensitive value">
        <span aria-hidden="true">{value.maskedValue ?? "••••••••"}</span>
      </span>
    );
  }
  if (value.access === "fingerprint") {
    return (
      <span aria-label="Sensitive value represented by fingerprint only">
        <span aria-hidden="true">fingerprint:{value.fingerprint.slice(0, 12)}…</span>
      </span>
    );
  }
  return (
    <span aria-label={`Sensitive value access: ${value.access}`}>
      <span aria-hidden="true">[{value.access}]</span>
    </span>
  );
}

interface EvidencePanelProps {
  evidence: FindingEvidence[];
}

export function EvidencePanel({ evidence }: EvidencePanelProps) {
  if (evidence.length === 0) {
    return (
      <p style={{ color: "var(--color-text-secondary)" }}>No evidence attached.</p>
    );
  }

  return (
    <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: "0.65rem" }}>
      {evidence.map((item, index) => (
        <li
          key={`${item.type}-${index}`}
          style={{
            padding: "0.65rem 0.75rem",
            border: "1px solid var(--color-border)",
            borderRadius: "8px",
            background: "var(--color-surface)",
            fontSize: "0.9rem",
          }}
        >
          {item.type === "message" ? (
            <p style={{ margin: 0 }}>{item.message}</p>
          ) : null}

          {item.type === "connection_error" ? (
            <div>
              <strong>Connection error</strong>
              <div style={{ color: "var(--color-text-secondary)", marginTop: "0.25rem" }}>
                {item.safeMessage}
                {item.errorCode ? ` (${item.errorCode})` : ""}
              </div>
            </div>
          ) : null}

          {item.type === "resource_state" ? (
            <div>
              <strong>Resource state</strong>
              <div style={{ color: "var(--color-text-secondary)", marginTop: "0.25rem" }}>
                {item.state}
                {item.observedAt ? ` · ${item.observedAt}` : ""}
              </div>
            </div>
          ) : null}

          {item.type === "deployment_state" ? (
            <div>
              <strong>Deployment state</strong>
              <div style={{ color: "var(--color-text-secondary)", marginTop: "0.25rem" }}>
                Status: {item.status}
                {item.observedAt ? ` · ${item.observedAt}` : ""}
              </div>
            </div>
          ) : null}

          {item.type === "configuration_comparison" ? (
            <div>
              <strong>Configuration comparison</strong>
              <div style={{ marginTop: "0.35rem", color: "var(--color-text-secondary)" }}>
                Key: {item.configurationKeyId}
              </div>
              {item.expectedState ? (
                <div style={{ marginTop: "0.35rem" }}>
                  Expected: {renderSafeValue(item.expectedState)}
                </div>
              ) : null}
              <ul style={{ margin: "0.35rem 0 0", paddingLeft: "1.1rem" }}>
                {item.observedStates.length === 0 ? (
                  <li>No observed states</li>
                ) : (
                  item.observedStates.map((observed, observedIndex) => (
                    <li key={observedIndex}>
                      {observed.label ?? observed.pluginId ?? "Observed"}:{" "}
                      {renderSafeValue(observed.value)}
                    </li>
                  ))
                )}
              </ul>
            </div>
          ) : null}
        </li>
      ))}
    </ul>
  );
}
