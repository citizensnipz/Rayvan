import type { CSSProperties } from "react";

import { IntegrationCard } from "./IntegrationCard.js";
import type {
  IntegrationCardActionId,
  PluginIntegrationCardViewModel,
} from "./view-models.js";

const gridStyle: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
  gap: "1rem",
};

interface IntegrationCardGridProps {
  cards: PluginIntegrationCardViewModel[];
  onOpen: (connectionId: string) => void;
  onAction: (connectionId: string, actionId: IntegrationCardActionId) => void;
}

export function IntegrationCardGrid({ cards, onOpen, onAction }: IntegrationCardGridProps) {
  return (
    <div style={gridStyle} data-testid="integration-card-grid">
      {cards.map((card) => (
        <IntegrationCard
          key={card.connectionId}
          card={card}
          onOpen={onOpen}
          onAction={onAction}
        />
      ))}
    </div>
  );
}
