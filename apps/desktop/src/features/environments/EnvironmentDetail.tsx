import { useEffect, useState } from "react";
import type {
  AppliedConfigurationState,
  ConfigurationKey,
  ConfigurationOccurrence,
  DesiredConfigurationValue,
  Environment,
  FindingRecord,
} from "@rayvan/core";
import type {
  EnvironmentConfigurationStatusViewModel,
} from "@rayvan/config-engine";
import { Button } from "@rayvan/ui";

import type { EnvironmentsGateway } from "../../lib/environments/index.js";
import { EnvironmentConfigurationEditor } from "./EnvironmentConfigurationEditor.js";
import type {
  EnvironmentCardViewModel,
  ResourceListItemViewModel,
} from "./view-models.js";
import { ENVIRONMENT_KIND_LABELS, ENVIRONMENT_STATUS_LABELS } from "./view-models.js";

type DetailSection = "summary" | "configuration" | "resources" | "findings" | "activity";

interface EnvironmentDetailProps {
  projectId: string;
  environment: Environment;
  card?: EnvironmentCardViewModel;
  resources: ResourceListItemViewModel[];
  findings: FindingRecord[];
  keys: ConfigurationKey[];
  occurrences: ConfigurationOccurrence[];
  gateway: EnvironmentsGateway;
  onSync: (environmentId: string) => void;
  onArchive: (environmentId: string) => void;
  onOpenMatrix: () => void;
  onRefresh: () => Promise<void>;
  onBanner: (message: string) => void;
}

export function EnvironmentDetail({
  projectId,
  environment,
  card,
  resources,
  findings,
  keys,
  occurrences,
  gateway,
  onSync,
  onArchive,
  onOpenMatrix,
  onRefresh,
  onBanner,
}: EnvironmentDetailProps) {
  const [section, setSection] = useState<DetailSection>("summary");
  const [desired, setDesired] = useState<DesiredConfigurationValue[]>([]);
  const [applied, setApplied] = useState<AppliedConfigurationState[]>([]);
  const [configStatus, setConfigStatus] =
    useState<EnvironmentConfigurationStatusViewModel | null>(null);

  const envFindings = findings.filter(
    (finding) =>
      finding.environmentId === environment.id &&
      (finding.status === "open" ||
        finding.status === "acknowledged" ||
        finding.status === "suppressed"),
  );

  useEffect(() => {
    let cancelled = false;
    async function loadConfiguration() {
      const [nextDesired, nextApplied, nextStatus] = await Promise.all([
        gateway.listDesiredValuesByEnvironment(environment.id),
        gateway.listAppliedByEnvironment(environment.id),
        gateway.getEnvironmentConfigurationStatus(projectId, environment.id),
      ]);
      if (cancelled) {
        return;
      }
      setDesired(nextDesired);
      setApplied(nextApplied);
      setConfigStatus(nextStatus);
    }
    void loadConfiguration();
    return () => {
      cancelled = true;
    };
  }, [gateway, projectId, environment.id, occurrences]);

  async function refreshConfiguration() {
    await onRefresh();
    const [nextDesired, nextApplied, nextStatus] = await Promise.all([
      gateway.listDesiredValuesByEnvironment(environment.id),
      gateway.listAppliedByEnvironment(environment.id),
      gateway.getEnvironmentConfigurationStatus(projectId, environment.id),
    ]);
    setDesired(nextDesired);
    setApplied(nextApplied);
    setConfigStatus(nextStatus);
  }

  return (
    <section>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          gap: "1rem",
          flexWrap: "wrap",
          marginBottom: "1rem",
        }}
      >
        <div>
          <h2 style={{ marginTop: 0, marginBottom: "0.35rem" }}>{environment.name}</h2>
          <p style={{ margin: 0, color: "var(--color-text-secondary)" }}>
            {ENVIRONMENT_KIND_LABELS[environment.kind]} ·{" "}
            <span aria-label={`Status: ${ENVIRONMENT_STATUS_LABELS[environment.status]}`}>
              {ENVIRONMENT_STATUS_LABELS[environment.status]}
            </span>
            {card?.configAggregate ? (
              <>
                {" · "}
                <span aria-label={`Configuration: ${card.configAggregate.headlineLabel}`}>
                  {card.configAggregate.headlineLabel}
                </span>
              </>
            ) : null}
          </p>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <Button onClick={() => onSync(environment.id)}>Sync</Button>
          <Button onClick={onOpenMatrix}>Compare with other environments</Button>
          <Button onClick={() => onArchive(environment.id)}>Archive</Button>
        </div>
      </div>

      <div
        role="tablist"
        aria-label={`${environment.name} sections`}
        style={{ display: "flex", gap: "0.35rem", marginBottom: "1rem", flexWrap: "wrap" }}
      >
        {(
          [
            ["summary", "Summary"],
            ["configuration", "Configuration"],
            ["resources", "Resources"],
            ["findings", "Findings"],
            ["activity", "Activity"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={section === id}
            onClick={() => setSection(id)}
            style={{
              padding: "0.4rem 0.75rem",
              borderRadius: "6px",
              border: "1px solid var(--color-border)",
              background:
                section === id ? "var(--color-surface-muted)" : "var(--color-surface)",
              cursor: "pointer",
              fontWeight: section === id ? 600 : 400,
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {section === "summary" ? (
        <div>
          <p style={{ color: "var(--color-text-secondary)" }}>
            {environment.description ?? "No description."}
          </p>
          {card ? (
            <dl
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(9rem, 1fr))",
                gap: "0.75rem",
              }}
            >
              <div>
                <dt>Resources</dt>
                <dd style={{ margin: 0, fontWeight: 600 }}>{card.resourceCount}</dd>
              </div>
              <div>
                <dt>Integrations</dt>
                <dd style={{ margin: 0, fontWeight: 600 }}>{card.integrationCount}</dd>
              </div>
              <div>
                <dt>Findings</dt>
                <dd
                  style={{ margin: 0, fontWeight: 600 }}
                  aria-label={card.findingsLabel}
                >
                  {card.findingsLabel}
                </dd>
              </div>
              <div>
                <dt>Last sync</dt>
                <dd style={{ margin: 0, fontWeight: 600 }}>{card.lastSyncLabel}</dd>
              </div>
              {card.configAggregate ? (
                <>
                  <div>
                    <dt>In sync</dt>
                    <dd style={{ margin: 0, fontWeight: 600 }}>
                      {card.configAggregate.inSyncCount}
                    </dd>
                  </div>
                  <div>
                    <dt>Not applied</dt>
                    <dd style={{ margin: 0, fontWeight: 600 }}>
                      {card.configAggregate.changesNotAppliedCount}
                    </dd>
                  </div>
                  <div>
                    <dt>Missing remotely</dt>
                    <dd style={{ margin: 0, fontWeight: 600 }}>
                      {card.configAggregate.missingRemoteCount}
                    </dd>
                  </div>
                </>
              ) : null}
            </dl>
          ) : null}
        </div>
      ) : null}

      {section === "configuration" ? (
        <EnvironmentConfigurationEditor
          projectId={projectId}
          environment={environment}
          keys={keys}
          occurrences={occurrences}
          desired={desired}
          applied={applied}
          status={configStatus}
          gateway={gateway}
          onOpenMatrix={onOpenMatrix}
          onRefresh={refreshConfiguration}
          onBanner={onBanner}
        />
      ) : null}

      {section === "resources" ? (
        <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: "0.5rem" }}>
          {resources.length === 0 ? (
            <li style={{ color: "var(--color-text-secondary)" }}>
              No resources bound to this environment.
            </li>
          ) : (
            resources.map((resource) => (
              <li
                key={resource.discoveredResourceId}
                style={{
                  padding: "0.65rem 0.75rem",
                  border: "1px solid var(--color-border)",
                  borderRadius: "8px",
                  background: "var(--color-surface)",
                }}
              >
                <strong>{resource.name}</strong>
                <div style={{ fontSize: "0.85rem", color: "var(--color-text-secondary)" }}>
                  {resource.pluginId} · {resource.resourceType}
                  {resource.syncStatusLabel ? ` · ${resource.syncStatusLabel}` : ""}
                </div>
              </li>
            ))
          )}
        </ul>
      ) : null}

      {section === "findings" ? (
        <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: "0.5rem" }}>
          {envFindings.length === 0 ? (
            <li style={{ color: "var(--color-text-secondary)" }}>No findings for this environment.</li>
          ) : (
            envFindings.map((finding) => (
              <li
                key={finding.id}
                style={{
                  padding: "0.65rem 0.75rem",
                  border: "1px solid var(--color-border)",
                  borderRadius: "8px",
                  background: "var(--color-surface)",
                }}
              >
                <strong>{finding.title}</strong>
                <div style={{ fontSize: "0.85rem", color: "var(--color-text-secondary)" }}>
                  <span aria-label={`Severity: ${finding.severity}`}>{finding.severity}</span>
                  {" · "}
                  <span aria-label={`Status: ${finding.status}`}>{finding.status}</span>
                  {" · "}
                  {finding.category}
                </div>
                <p style={{ margin: "0.35rem 0 0" }}>{finding.summary}</p>
                <p style={{ margin: "0.5rem 0 0", fontSize: "0.85rem" }}>
                  Open the Findings section and search for this title to review the full record.
                </p>
              </li>
            ))
          )}
        </ul>
      ) : null}

      {section === "activity" ? (
        <div
          style={{
            padding: "1rem",
            border: "1px dashed var(--color-border-strong)",
            borderRadius: "8px",
            color: "var(--color-text-secondary)",
          }}
        >
          Activity history is a placeholder in this workspace preview.
        </div>
      ) : null}
    </section>
  );
}
