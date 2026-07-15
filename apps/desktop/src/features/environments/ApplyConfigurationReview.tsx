import type { CSSProperties } from "react";
import type {
  ConfigurationApplyPlan,
  ConfigurationApplyResult,
} from "@rayvan/config-engine";
import { Button } from "@rayvan/ui";

const panelStyle: CSSProperties = {
  marginTop: "0.75rem",
  padding: "1rem",
  border: "1px solid var(--color-border-strong)",
  borderRadius: "8px",
  background: "var(--color-surface-muted)",
  display: "grid",
  gap: "0.75rem",
};

interface ApplyConfigurationReviewProps {
  plan: ConfigurationApplyPlan;
  result: ConfigurationApplyResult | null;
  onApprove: () => void;
  onClose: () => void;
}

export function ApplyConfigurationReview({
  plan,
  result,
  onApprove,
  onClose,
}: ApplyConfigurationReviewProps) {
  const groups = new Map<
    string,
    {
      pluginId: string;
      resourceBindingId: string;
      resourceName?: string;
      operations: ConfigurationApplyPlan["operations"];
    }
  >();

  for (const operation of plan.operations) {
    const key = `${operation.pluginId}:${operation.resourceBindingId}`;
    const existing = groups.get(key);
    if (existing) {
      existing.operations.push(operation);
    } else {
      groups.set(key, {
        pluginId: operation.pluginId,
        resourceBindingId: operation.resourceBindingId,
        resourceName: operation.resourceName,
        operations: [operation],
      });
    }
  }

  return (
    <section
      role="dialog"
      aria-label="Review configuration apply plan"
      style={panelStyle}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          gap: "0.75rem",
          flexWrap: "wrap",
        }}
      >
        <div>
          <h3 style={{ margin: 0 }}>Review apply plan</h3>
          <p style={{ margin: "0.35rem 0 0", color: "var(--color-text-secondary)" }}>
            {plan.summary}. This stub writes applied state only — no provider API calls.
          </p>
        </div>
        <Button onClick={onClose}>Close</Button>
      </div>

      {[...groups.values()].map((group) => (
        <div
          key={`${group.pluginId}:${group.resourceBindingId}`}
          style={{
            padding: "0.75rem",
            borderRadius: "8px",
            border: "1px solid var(--color-border)",
            background: "var(--color-surface)",
          }}
        >
          <strong>
            {group.pluginId}
            {group.resourceName ? ` · ${group.resourceName}` : ""}
          </strong>
          <ul style={{ margin: "0.5rem 0 0", paddingLeft: "1.1rem" }}>
            {group.operations.map((operation) => (
              <li key={operation.id}>
                <code>{operation.configurationKeyName}</code>
                {" → "}
                {operation.sensitive ? (
                  <span aria-label="Redacted secret value">••••••••</span>
                ) : (
                  operation.displayValue
                )}
                {operation.warnings.length > 0 ? (
                  <div
                    style={{
                      fontSize: "0.8rem",
                      color: "var(--color-text-secondary)",
                      marginTop: "0.2rem",
                    }}
                  >
                    {operation.warnings.join(" ")}
                  </div>
                ) : null}
              </li>
            ))}
          </ul>
        </div>
      ))}

      {result ? (
        <div role="status">
          Apply result: <strong>{result.status}</strong>
          <ul style={{ margin: "0.35rem 0 0", paddingLeft: "1.1rem" }}>
            {result.items.map((item) => (
              <li key={item.operationId}>
                {item.configurationKeyId}: {item.status}
                {item.errorMessage ? ` — ${item.errorMessage}` : ""}
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <Button onClick={onApprove}>Approve and apply</Button>
          <Button onClick={onClose}>Cancel</Button>
        </div>
      )}
    </section>
  );
}
