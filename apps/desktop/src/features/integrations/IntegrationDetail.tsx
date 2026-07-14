import type { CSSProperties } from "react";
import { Button } from "@rayvan/ui";

import { IntegrationIcon } from "./icons.js";
import { IntegrationStatusIndicator } from "./IntegrationStatus.js";
import { resolveIntegrationTheme } from "./theme.js";
import type {
  IntegrationFieldGroup,
  PluginIntegrationDetailViewModel,
} from "./view-models.js";

const headerStyle: CSSProperties = {
  display: "flex",
  gap: "1rem",
  alignItems: "flex-start",
  marginBottom: "1.5rem",
};

const sectionStyle: CSSProperties = {
  marginBottom: "1.5rem",
  padding: "1rem",
  borderRadius: "10px",
  border: "1px solid var(--color-border)",
  background: "var(--color-surface)",
};

const groupStyle: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
  gap: "0.75rem",
  marginTop: "0.5rem",
};

function FieldGroup({ group }: { group: IntegrationFieldGroup }) {
  return (
    <div>
      <h4 style={{ margin: "0 0 0.35rem", fontSize: "0.85rem", color: "var(--color-text-secondary)" }}>
        {group.title}
      </h4>
      <div style={groupStyle}>
        {group.fields.map((field) => (
          <div key={field.label}>
            <div style={{ fontSize: "0.75rem", color: "var(--color-text-muted)" }}>
              {field.label}
            </div>
            <div style={{ fontWeight: 600 }}>{field.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

interface IntegrationDetailProps {
  detail: PluginIntegrationDetailViewModel;
  onDisconnect: (connectionId: string) => void;
  onReconnect: (connectionId: string) => void;
}

export function IntegrationDetail({
  detail,
  onDisconnect,
  onReconnect,
}: IntegrationDetailProps) {
  const theme = resolveIntegrationTheme(detail.theme);

  return (
    <section aria-label={`${detail.connectionName} details`}>
      <div style={headerStyle}>
        <IntegrationIcon icon={detail.icon} theme={theme} size={48} />
        <div>
          <h2 style={{ margin: "0 0 0.15rem" }}>{detail.connectionName}</h2>
          <p style={{ margin: "0 0 0.35rem", color: "var(--color-text-secondary)" }}>
            {detail.pluginName} &middot; {detail.publisher} &middot; v{detail.version}
          </p>
          <IntegrationStatusIndicator status={detail.status} label={detail.statusLabel} />
        </div>
      </div>

      <div style={sectionStyle}>
        <h3 style={{ marginTop: 0 }}>Overview</h3>
        {detail.overview.groups.map((group) => (
          <FieldGroup key={group.title} group={group} />
        ))}
      </div>

      <div style={sectionStyle}>
        <h3 style={{ marginTop: 0 }}>Resources</h3>
        {detail.resources.isEmpty ? (
          <p style={{ margin: 0, color: "var(--color-text-secondary)" }}>
            No resources discovered yet.
          </p>
        ) : (
          detail.resources.groups.map((group) => <FieldGroup key={group.title} group={group} />)
        )}
      </div>

      <div style={sectionStyle}>
        <h3 style={{ marginTop: 0 }}>Configuration</h3>
        <div style={groupStyle}>
          {detail.configuration.grants.map((grant) => (
            <div key={grant.permission}>
              <div style={{ fontSize: "0.75rem", color: "var(--color-text-muted)" }}>
                {grant.permission}
              </div>
              <div style={{ fontWeight: 600 }}>{grant.granted ? "Granted" : "Revoked"}</div>
            </div>
          ))}
        </div>
        <div style={{ display: "flex", gap: "0.5rem", marginTop: "1rem" }}>
          <Button onClick={() => onReconnect(detail.connectionId)}>Reconnect</Button>
          <Button onClick={() => onDisconnect(detail.connectionId)}>Disconnect</Button>
        </div>
      </div>

      <div style={sectionStyle}>
        <h3 style={{ marginTop: 0 }}>Activity</h3>
        {detail.activity.isEmpty ? (
          <p style={{ margin: 0, color: "var(--color-text-secondary)" }}>
            No recent activity to show.
          </p>
        ) : (
          <ul style={{ margin: 0, paddingLeft: "1.25rem" }}>
            {detail.activity.items.map((item) => (
              <li key={item.label}>{item.label}</li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
