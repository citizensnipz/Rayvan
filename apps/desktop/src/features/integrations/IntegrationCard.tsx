import type { CSSProperties } from "react";
import { Button } from "@rayvan/ui";

import { IntegrationIcon } from "./icons.js";
import { IntegrationStatusIndicator } from "./IntegrationStatus.js";
import { resolveIntegrationTheme } from "./theme.js";
import type {
  IntegrationCardActionId,
  PluginIntegrationCardViewModel,
} from "./view-models.js";

const cardStyle: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: "0.75rem",
  padding: "1rem",
  borderRadius: "10px",
  border: "1px solid var(--color-border)",
  background: "var(--color-surface)",
  boxShadow: "inset 0 0 0 1px transparent",
};

const headerStyle: CSSProperties = {
  display: "flex",
  alignItems: "flex-start",
  gap: "0.75rem",
};

const titleGroupStyle: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: "0.15rem",
  minWidth: 0,
};

const fieldsListStyle: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: "0.2rem",
  margin: 0,
  fontSize: "0.85rem",
  color: "var(--color-text-secondary)",
};

const footerStyle: CSSProperties = {
  display: "flex",
  gap: "0.5rem",
  marginTop: "auto",
  paddingTop: "0.5rem",
  flexWrap: "wrap",
};

interface IntegrationCardProps {
  card: PluginIntegrationCardViewModel;
  onOpen: (connectionId: string) => void;
  onAction: (connectionId: string, actionId: IntegrationCardActionId) => void;
}

export function IntegrationCard({ card, onOpen, onAction }: IntegrationCardProps) {
  const theme = resolveIntegrationTheme(card.theme);
  const primaryActions = card.actions.filter((action) => action.kind === "primary");
  const secondaryActions = card.actions.filter((action) => action.kind === "secondary");

  return (
    <article
      aria-label={`${card.pluginName}: ${card.connectionName}`}
      style={{
        ...cardStyle,
        borderTop: `3px solid ${theme.accentColor}`,
      }}
    >
      <div style={headerStyle}>
        <IntegrationIcon icon={card.icon} theme={theme} />
        <div style={titleGroupStyle}>
          <strong
            title={card.connectionName}
            style={{
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {card.connectionName}
          </strong>
          <span
            style={{ fontSize: "0.8rem", color: "var(--color-text-muted)" }}
            title={`${card.pluginName} · ${card.publisher}`}
          >
            {card.pluginName} &middot; {card.publisher}
          </span>
          <IntegrationStatusIndicator status={card.status} label={card.statusLabel} />
        </div>
      </div>

      {card.fields.length > 0 ? (
        <dl style={fieldsListStyle}>
          {card.fields.map((field, index) => (
            <div
              key={`${field.label}-${index}`}
              style={{ display: "flex", justifyContent: "space-between", gap: "0.5rem" }}
            >
              <dt>{field.label}</dt>
              <dd
                style={{
                  margin: 0,
                  fontWeight: 600,
                  color: "var(--color-text)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  maxWidth: "60%",
                }}
                title={field.value}
              >
                {field.value}
              </dd>
            </div>
          ))}
        </dl>
      ) : null}

      <div style={footerStyle}>
        {primaryActions.map((action) => (
          <Button
            key={action.id}
            onClick={() => onAction(card.connectionId, action.id)}
          >
            {action.label}
          </Button>
        ))}
        {secondaryActions.map((action) => (
          <Button
            key={action.id}
            onClick={() => onAction(card.connectionId, action.id)}
          >
            {action.label}
          </Button>
        ))}
        {primaryActions.length === 0 ? (
          <Button onClick={() => onOpen(card.connectionId)}>Open</Button>
        ) : null}
      </div>
    </article>
  );
}
