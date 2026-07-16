import { useState, type CSSProperties } from "react";
import { Button } from "@rayvan/ui";

import type {
  EnvironmentCardActionId,
  EnvironmentCardViewModel,
} from "./view-models.js";

const cardStyle: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: "0.75rem",
  padding: "1rem",
  borderRadius: "10px",
  border: "1px solid var(--color-border)",
  background: "var(--color-surface)",
};

const headerStyle: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  gap: "0.75rem",
  alignItems: "flex-start",
};

const metaStyle: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: "0.35rem 0.75rem",
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
  alignItems: "center",
};

function colorAccent(color?: string): string {
  switch (color) {
    case "blue":
      return "#2563eb";
    case "amber":
      return "#d97706";
    case "rose":
      return "#e11d48";
    case "violet":
      return "#7c3aed";
    case "cyan":
      return "#0891b2";
    case "green":
      return "#16a34a";
    default:
      return "var(--color-border-strong)";
  }
}

interface EnvironmentCardProps {
  card: EnvironmentCardViewModel;
  onAction: (environmentId: string, actionId: EnvironmentCardActionId) => void;
}

export function EnvironmentCard({ card, onAction }: EnvironmentCardProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const primary = card.actions.filter((action) => action.kind === "primary");
  const secondary = card.actions.filter((action) => action.kind === "secondary");
  const overflow = card.actions.filter((action) => action.kind === "overflow");

  return (
    <article
      aria-label={`${card.name}: ${card.statusLabel}`}
      style={{
        ...cardStyle,
        borderTop: `3px solid ${colorAccent(card.color)}`,
      }}
    >
      <div style={headerStyle}>
        <div style={{ minWidth: 0 }}>
          <strong
            title={card.name}
            style={{
              display: "block",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {card.name}
          </strong>
          <div style={{ fontSize: "0.8rem", color: "var(--color-text-muted)" }}>
            {card.kindLabel}
          </div>
          <span
            aria-label={`Status: ${card.statusLabel}`}
            style={{
              display: "inline-block",
              marginTop: "0.35rem",
              fontSize: "0.8rem",
              fontWeight: 600,
              color: "var(--color-text)",
            }}
          >
            {card.statusLabel}
          </span>
        </div>
      </div>

      <dl style={metaStyle}>
        <div>
          <dt>Resources</dt>
          <dd style={{ margin: 0, fontWeight: 600, color: "var(--color-text)" }}>
            {card.resourceCount}
          </dd>
        </div>
        <div>
          <dt>Integrations</dt>
          <dd style={{ margin: 0, fontWeight: 600, color: "var(--color-text)" }}>
            {card.integrationCount}
          </dd>
        </div>
        <div>
          <dt>Config keys</dt>
          <dd style={{ margin: 0, fontWeight: 600, color: "var(--color-text)" }}>
            {card.configurationKeyCount}
          </dd>
        </div>
        <div>
          <dt>Findings</dt>
          <dd
            style={{ margin: 0, fontWeight: 600, color: "var(--color-text)" }}
            aria-label={card.findingsLabel}
          >
            {card.findingsLabel}
          </dd>
        </div>
      </dl>

      <div style={{ fontSize: "0.8rem", color: "var(--color-text-secondary)" }}>
        Health: {card.health.healthy} healthy · {card.health.missing} missing ·{" "}
        {card.health.mismatched} mismatched
      </div>
      {card.configAggregate ? (
        <div style={{ fontSize: "0.8rem", color: "var(--color-text-secondary)" }}>
          Config: {card.configAggregate.headlineLabel} ·{" "}
          {card.configAggregate.inSyncCount} in sync ·{" "}
          {card.configAggregate.changesNotAppliedCount} not applied ·{" "}
          {card.configAggregate.missingRemoteCount} missing remotely
        </div>
      ) : null}
      <div style={{ fontSize: "0.8rem", color: "var(--color-text-muted)" }}>
        Last sync: {card.lastSyncLabel}
      </div>

      <div style={footerStyle}>
        {primary.map((action) => (
          <Button
            key={action.id}
            onClick={() => onAction(card.environmentId, action.id)}
          >
            {action.label}
          </Button>
        ))}
        {secondary.map((action) => (
          <Button
            key={action.id}
            onClick={() => onAction(card.environmentId, action.id)}
          >
            {action.label}
          </Button>
        ))}
        {overflow.length > 0 ? (
          <div style={{ position: "relative", marginLeft: "auto" }}>
            <Button
              aria-haspopup="menu"
              aria-expanded={menuOpen}
              onClick={() => setMenuOpen((open) => !open)}
            >
              More
            </Button>
            {menuOpen ? (
              <div
                role="menu"
                style={{
                  position: "absolute",
                  right: 0,
                  top: "110%",
                  minWidth: "8rem",
                  background: "var(--color-surface)",
                  border: "1px solid var(--color-border)",
                  borderRadius: "8px",
                  boxShadow: "var(--shadow-dialog)",
                  zIndex: 2,
                  padding: "0.25rem",
                }}
              >
                {overflow.map((action) => (
                  <button
                    key={action.id}
                    type="button"
                    role="menuitem"
                    style={{
                      display: "block",
                      width: "100%",
                      textAlign: "left",
                      border: "none",
                      background: "transparent",
                      padding: "0.45rem 0.6rem",
                      cursor: "pointer",
                      color: "var(--color-text)",
                    }}
                    onClick={() => {
                      setMenuOpen(false);
                      onAction(card.environmentId, action.id);
                    }}
                  >
                    {action.label}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    </article>
  );
}

export function EnvironmentCardGrid({
  cards,
  onAction,
}: {
  cards: EnvironmentCardViewModel[];
  onAction: (environmentId: string, actionId: EnvironmentCardActionId) => void;
}) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(16rem, 1fr))",
        gap: "1rem",
      }}
    >
      {cards.map((card) => (
        <EnvironmentCard key={card.environmentId} card={card} onAction={onAction} />
      ))}
    </div>
  );
}
