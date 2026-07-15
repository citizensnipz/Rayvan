import type { CSSProperties } from "react";
import { Button } from "@rayvan/ui";

import { EnvironmentCardGrid } from "./EnvironmentCard.js";
import { EnvironmentEmptyState } from "./EnvironmentEmptyState.js";
import { EnvironmentMappingSuggestions } from "./EnvironmentMappingSuggestions.js";
import { SyncEnvironmentsBanner } from "./SyncEnvironmentsBanner.js";
import type { EnvironmentSyncState } from "../../lib/environments/index.js";
import type {
  EnvironmentCardActionId,
  EnvironmentCardViewModel,
  EnvironmentComparisonSummary,
  MappingSuggestionViewModel,
} from "./view-models.js";

const headerStyle: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "flex-start",
  gap: "1rem",
  marginBottom: "1.25rem",
};

const summaryStyle: CSSProperties = {
  margin: "0 0 1.25rem",
  padding: "0.85rem 1rem",
  borderRadius: "8px",
  border: "1px solid var(--color-border)",
  background: "var(--color-surface)",
};

interface EnvironmentsOverviewProps {
  cards: EnvironmentCardViewModel[];
  comparison: EnvironmentComparisonSummary;
  suggestions: MappingSuggestionViewModel[];
  syncState: EnvironmentSyncState | null;
  showLocalOnlyBanner: boolean;
  canMutate: boolean;
  onCreateEnvironment: () => void;
  onOpenMatrix: () => void;
  onCardAction: (environmentId: string, actionId: EnvironmentCardActionId) => void;
  onAcceptSuggestion: (suggestionId: string) => void;
  onRejectSuggestion: (suggestionId: string) => void;
  onSync: () => void;
  onCancelSync: () => void;
}

export function EnvironmentsOverview({
  cards,
  comparison,
  suggestions,
  syncState,
  showLocalOnlyBanner,
  canMutate,
  onCreateEnvironment,
  onOpenMatrix,
  onCardAction,
  onAcceptSuggestion,
  onRejectSuggestion,
  onSync,
  onCancelSync,
}: EnvironmentsOverviewProps) {
  return (
    <section>
      <div style={headerStyle}>
        <div>
          <h2 style={{ marginTop: 0, marginBottom: "0.35rem" }}>Environments</h2>
          <p style={{ margin: 0, color: "var(--color-text-secondary)" }}>
            Compare configuration and map discovered resources across environments.
          </p>
        </div>
        <Button onClick={onCreateEnvironment} disabled={!canMutate}>
          + Create environment
        </Button>
      </div>

      <SyncEnvironmentsBanner
        syncState={syncState}
        onSync={onSync}
        onCancel={onCancelSync}
        disabled={!canMutate || cards.length === 0}
      />

      {showLocalOnlyBanner ? (
        <div
          role="status"
          style={{
            ...summaryStyle,
            borderStyle: "dashed",
            background: "var(--color-surface-muted)",
          }}
        >
          Environments are local-only until you connect integrations and sync. Sync is read-only
          and never writes to providers.
        </div>
      ) : null}

      {cards.length > 0 ? (
        <div style={summaryStyle}>
          <strong>Comparison summary</strong>
          <p style={{ margin: "0.35rem 0 0.75rem", color: "var(--color-text-secondary)" }}>
            {comparison.environmentCount} environments · {comparison.keyCount} keys ·{" "}
            {comparison.healthyCellCount} healthy · {comparison.missingCellCount} missing ·{" "}
            {comparison.mismatchedCellCount} mismatched
          </p>
          <Button onClick={onOpenMatrix}>Open Configuration Matrix</Button>
        </div>
      ) : null}

      {cards.length === 0 ? (
        <EnvironmentEmptyState onCreateEnvironment={canMutate ? onCreateEnvironment : undefined} />
      ) : (
        <EnvironmentCardGrid cards={cards} onAction={onCardAction} />
      )}

      <EnvironmentMappingSuggestions
        suggestions={suggestions}
        onAccept={onAcceptSuggestion}
        onReject={onRejectSuggestion}
      />
    </section>
  );
}
