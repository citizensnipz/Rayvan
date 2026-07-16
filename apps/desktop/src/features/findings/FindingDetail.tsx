import type {
  FindingsDismissReason,
  FindingsSuppressPreset,
} from "../../lib/findings/index.js";
import { EvidencePanel } from "./EvidencePanel.js";
import { FindingActions } from "./FindingActions.js";
import { SeverityBadge } from "./severity.js";
import type { FindingDetailViewModel } from "./view-models.js";

interface FindingDetailProps {
  detail: FindingDetailViewModel;
  busy?: boolean;
  onAcknowledge: () => void;
  onDismiss: (reason: FindingsDismissReason) => void;
  onSuppress: (preset: FindingsSuppressPreset) => void;
  onOpenEnvironment?: (environmentId: string) => void;
  onOpenIntegration?: (connectionId: string) => void;
  onResync?: () => void;
}

export function FindingDetail({
  detail,
  busy,
  onAcknowledge,
  onDismiss,
  onSuppress,
  onOpenEnvironment,
  onOpenIntegration,
  onResync,
}: FindingDetailProps) {
  const { finding } = detail;

  return (
    <article aria-label={`Finding detail: ${finding.title}`}>
      <header style={{ marginBottom: "1rem" }}>
        <h2 style={{ marginTop: 0, marginBottom: "0.35rem" }}>{finding.title}</h2>
        <p
          style={{
            margin: 0,
            color: "var(--color-text-secondary)",
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            gap: "0.35rem 0.5rem",
          }}
        >
          <SeverityBadge
            severity={finding.severity}
            label={detail.severityLabel}
            size="md"
          />
          <span aria-hidden="true">·</span>
          <span aria-label={`Status: ${detail.statusLabel}`}>{detail.statusLabel}</span>
          <span aria-hidden="true">·</span>
          <span>{detail.categoryLabel}</span>
          <span aria-hidden="true">·</span>
          <span>{detail.sourceLabel}</span>
        </p>
        <p
          style={{ margin: "0.35rem 0 0", fontSize: "0.85rem", color: "var(--color-text-muted)" }}
        >
          <span title={detail.firstDetectedAbsolute}>
            First detected {detail.firstDetectedLabel}
          </span>
          {" · "}
          <span title={detail.lastDetectedAbsolute}>
            Last detected {detail.lastDetectedLabel}
          </span>
          {" · "}
          Seen {finding.occurrenceCount} time{finding.occurrenceCount === 1 ? "" : "s"}
        </p>
      </header>

      <section style={{ marginBottom: "1.25rem" }}>
        <h3 style={{ marginTop: 0 }}>Summary</h3>
        <p style={{ margin: 0 }}>{finding.summary}</p>
        {finding.description ? (
          <p style={{ margin: "0.5rem 0 0", color: "var(--color-text-secondary)" }}>
            {finding.description}
          </p>
        ) : null}
      </section>

      <section style={{ marginBottom: "1.25rem" }}>
        <h3>Affected objects</h3>
        <ul style={{ margin: 0, paddingLeft: "1.1rem", color: "var(--color-text-secondary)" }}>
          {detail.environmentLabel ? <li>Environment: {detail.environmentLabel}</li> : null}
          {detail.integrationLabel ? <li>Integration: {detail.integrationLabel}</li> : null}
          {finding.configurationKeyId ? (
            <li>Configuration key: {finding.configurationKeyId}</li>
          ) : null}
          {finding.resourceBindingId ? (
            <li>Resource binding: {finding.resourceBindingId}</li>
          ) : null}
          {!detail.environmentLabel &&
          !detail.integrationLabel &&
          !finding.configurationKeyId &&
          !finding.resourceBindingId ? (
            <li>Project-scoped</li>
          ) : null}
        </ul>
      </section>

      <section style={{ marginBottom: "1.25rem" }}>
        <h3>Evidence</h3>
        <EvidencePanel evidence={finding.evidence} />
      </section>

      <section style={{ marginBottom: "1.25rem" }}>
        <h3>Remediation</h3>
        <FindingActions
          detail={detail}
          busy={busy}
          onAcknowledge={onAcknowledge}
          onDismiss={onDismiss}
          onSuppress={onSuppress}
          onOpenEnvironment={onOpenEnvironment}
          onOpenIntegration={onOpenIntegration}
          onResync={onResync}
        />
      </section>

      <section>
        <h3>History</h3>
        {detail.lifecycleEvents.length === 0 ? (
          <p style={{ color: "var(--color-text-secondary)" }}>No lifecycle events.</p>
        ) : (
          <ol style={{ margin: 0, paddingLeft: "1.1rem", display: "grid", gap: "0.35rem" }}>
            {detail.lifecycleEvents.map((event) => (
              <li key={event.id}>
                <span title={event.createdAt}>{new Date(event.createdAt).toLocaleString()}</span>
                {": "}
                {event.type}
                {event.reason ? ` — ${event.reason}` : ""}
                {event.actor.kind === "user" && event.actor.displayName
                  ? ` (${event.actor.displayName})`
                  : ` (${event.actor.kind})`}
              </li>
            ))}
          </ol>
        )}
      </section>
    </article>
  );
}
