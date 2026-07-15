import type { CSSProperties } from "react";
import { Button } from "@rayvan/ui";

import type { MappingSuggestionViewModel } from "./view-models.js";

const sectionStyle: CSSProperties = {
  marginTop: "1.5rem",
  padding: "1rem",
  borderRadius: "10px",
  border: "1px solid var(--color-border)",
  background: "var(--color-surface)",
};

interface EnvironmentMappingSuggestionsProps {
  suggestions: MappingSuggestionViewModel[];
  onAccept: (suggestionId: string) => void;
  onReject: (suggestionId: string) => void;
}

export function EnvironmentMappingSuggestions({
  suggestions,
  onAccept,
  onReject,
}: EnvironmentMappingSuggestionsProps) {
  if (suggestions.length === 0) {
    return null;
  }

  return (
    <section style={sectionStyle} aria-label="Mapping suggestions">
      <h3 style={{ marginTop: 0, marginBottom: "0.35rem" }}>Mapping suggestions</h3>
      <p style={{ margin: "0 0 0.75rem", color: "var(--color-text-secondary)", fontSize: "0.9rem" }}>
        Suggestions are never applied automatically. Accept to bind a resource to an environment.
      </p>
      <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "grid", gap: "0.75rem" }}>
        {suggestions.map((suggestion) => (
          <li
            key={suggestion.suggestionId}
            style={{
              border: "1px solid var(--color-border)",
              borderRadius: "8px",
              padding: "0.75rem",
              background: "var(--color-surface-muted)",
            }}
          >
            <div style={{ fontWeight: 600 }}>{suggestion.resourceName}</div>
            <div style={{ fontSize: "0.85rem", color: "var(--color-text-secondary)" }}>
              {suggestion.pluginId}
              {suggestion.suggestedEnvironmentName
                ? ` → ${suggestion.suggestedEnvironmentName}`
                : " → choose an environment"}
              {suggestion.confidence !== undefined
                ? ` · confidence ${(suggestion.confidence * 100).toFixed(0)}%`
                : null}
            </div>
            {suggestion.reasons.length > 0 ? (
              <ul style={{ margin: "0.4rem 0 0", paddingLeft: "1.1rem", fontSize: "0.85rem" }}>
                {suggestion.reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            ) : null}
            <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.65rem" }}>
              <Button
                onClick={() => onAccept(suggestion.suggestionId)}
                disabled={!suggestion.suggestedEnvironmentId}
              >
                Accept
              </Button>
              <Button onClick={() => onReject(suggestion.suggestionId)}>Reject</Button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
