import type { CSSProperties } from "react";
import { Button } from "@rayvan/ui";

import { IntegrationCardGrid } from "./IntegrationCardGrid.js";
import { IntegrationEmptyState } from "./IntegrationEmptyState.js";
import type {
  IntegrationCardActionId,
  PluginIntegrationCardViewModel,
} from "./view-models.js";

const headerStyle: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "flex-start",
  gap: "1rem",
  marginBottom: "1.25rem",
};

interface IntegrationsHomeProps {
  cards: PluginIntegrationCardViewModel[];
  canAddIntegration: boolean;
  onOpen: (connectionId: string) => void;
  onAction: (connectionId: string, actionId: IntegrationCardActionId) => void;
  onAddIntegration: () => void;
}

export function IntegrationsHome({
  cards,
  canAddIntegration,
  onOpen,
  onAction,
  onAddIntegration,
}: IntegrationsHomeProps) {
  return (
    <section>
      <div style={headerStyle}>
        <div>
          <h2 style={{ marginTop: 0, marginBottom: "0.35rem" }}>Integrations</h2>
          <p style={{ margin: 0, color: "var(--color-text-secondary)" }}>
            Connect Rayvan to the services used by this project.
          </p>
        </div>
        <Button onClick={onAddIntegration} disabled={!canAddIntegration}>
          + Add integration
        </Button>
      </div>

      {cards.length === 0 ? (
        <IntegrationEmptyState
          onAddIntegration={canAddIntegration ? onAddIntegration : undefined}
        />
      ) : (
        <IntegrationCardGrid cards={cards} onOpen={onOpen} onAction={onAction} />
      )}
    </section>
  );
}
