import type { ConfigurationKey, ConfigurationOccurrence, Environment } from "@rayvan/core";
import type { ConfigurationMatrixViewModel } from "@rayvan/config-engine";
import { Button } from "@rayvan/ui";

interface ConfigurationKeyDetailProps {
  configurationKey: ConfigurationKey;
  environments: Environment[];
  occurrences: ConfigurationOccurrence[];
  matrix: ConfigurationMatrixViewModel | null;
  onOpenEnvironment: (environmentId: string, label: string) => void;
}

export function ConfigurationKeyDetail({
  configurationKey,
  environments,
  occurrences,
  matrix,
  onOpenEnvironment,
}: ConfigurationKeyDetailProps) {
  const row = matrix?.rows.find(
    (item) => item.configurationKeyId === configurationKey.id,
  );
  const envById = new Map<string, Environment>(
    environments.map((environment) => [environment.id, environment]),
  );
  const keyOccurrences = occurrences.filter(
    (occurrence) => occurrence.configurationKeyId === configurationKey.id,
  );

  return (
    <section>
      <h2 style={{ marginTop: 0 }}>{configurationKey.name}</h2>
      <p style={{ color: "var(--color-text-secondary)" }}>
        {configurationKey.description ?? "No description yet."}
      </p>

      <dl
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(10rem, 1fr))",
          gap: "0.75rem",
          margin: "1rem 0",
        }}
      >
        <div>
          <dt style={{ color: "var(--color-text-muted)", fontSize: "0.8rem" }}>Type</dt>
          <dd style={{ margin: 0 }}>{configurationKey.valueType}</dd>
        </div>
        <div>
          <dt style={{ color: "var(--color-text-muted)", fontSize: "0.8rem" }}>Required</dt>
          <dd style={{ margin: 0 }}>{configurationKey.required ? "Yes" : "No"}</dd>
        </div>
        <div>
          <dt style={{ color: "var(--color-text-muted)", fontSize: "0.8rem" }}>Sensitive</dt>
          <dd style={{ margin: 0 }}>{configurationKey.sensitive ? "Yes" : "No"}</dd>
        </div>
        <div>
          <dt style={{ color: "var(--color-text-muted)", fontSize: "0.8rem" }}>Source</dt>
          <dd style={{ margin: 0 }}>{configurationKey.source}</dd>
        </div>
      </dl>

      <h3>Per-environment status</h3>
      <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: "0.5rem" }}>
        {(row?.cells ?? []).map((cell) => {
          const environment = envById.get(cell.environmentId);
          return (
            <li
              key={cell.environmentId}
              style={{
                display: "flex",
                justifyContent: "space-between",
                gap: "0.75rem",
                padding: "0.65rem 0.75rem",
                border: "1px solid var(--color-border)",
                borderRadius: "8px",
                background: "var(--color-surface)",
              }}
            >
              <div>
                <strong>{environment?.name ?? cell.environmentId}</strong>
                <div style={{ fontSize: "0.85rem", color: "var(--color-text-secondary)" }}>
                  Status: {cell.statusLabel}
                  {cell.accessLocked ? " · Locked" : null}
                  {cell.safeVisibleValue ? ` · ${cell.safeVisibleValue}` : null}
                </div>
              </div>
              <Button
                onClick={() =>
                  onOpenEnvironment(
                    cell.environmentId,
                    environment?.name ?? "Environment",
                  )
                }
              >
                Open
              </Button>
            </li>
          );
        })}
      </ul>

      <h3 style={{ marginTop: "1.5rem" }}>Occurrences ({keyOccurrences.length})</h3>
      <ul style={{ paddingLeft: "1.1rem", color: "var(--color-text-secondary)" }}>
        {keyOccurrences.map((occurrence) => (
          <li key={occurrence.id}>
            {occurrence.pluginId} · {occurrence.valueAccess}
            {occurrence.environmentId
              ? ` · ${envById.get(occurrence.environmentId)?.name ?? occurrence.environmentId}`
              : " · unmapped"}
          </li>
        ))}
      </ul>

      <div style={{ display: "flex", gap: "0.5rem", marginTop: "1.25rem", flexWrap: "wrap" }}>
        <Button disabled title="Not yet implemented">
          Add missing key (preview)
        </Button>
        <Button disabled title="Not yet implemented">
          Copy from Staging (preview)
        </Button>
      </div>
    </section>
  );
}
