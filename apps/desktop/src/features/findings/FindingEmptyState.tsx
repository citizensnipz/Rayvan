import { Button, EmptyState } from "@rayvan/ui";

import type { FindingEmptyVariant } from "./view-models.js";

interface FindingEmptyStateProps {
  variant: FindingEmptyVariant;
  onScan?: () => void;
  onClearFilters?: () => void;
}

/**
 * How to open development fixtures:
 * Select a project, open Findings — the dev gateway auto-seeds on ensureProjectSeeded.
 * Use “Scan project” to run evaluateProject against the seeded context.
 */
export function FindingEmptyState({
  variant,
  onScan,
  onClearFilters,
}: FindingEmptyStateProps) {
  if (variant === "no-project") {
    return (
      <EmptyState
        title="Select a project"
        description="Choose a project to inspect configuration, drift, and integration findings."
      />
    );
  }

  if (variant === "scanning") {
    return (
      <EmptyState
        title="Scanning for findings"
        description="Rayvan is evaluating this project. Findings appear when the scan completes."
      />
    );
  }

  if (variant === "never-evaluated") {
    return (
      <div>
        <EmptyState
          title="No findings yet"
          description="This project has not been scanned. Run a project scan to evaluate configuration, environments, and integrations. In development, fixtures also seed automatically when you open Findings."
        />
        {onScan ? (
          <div style={{ marginTop: "1rem" }}>
            <Button onClick={onScan}>Scan project</Button>
          </div>
        ) : null}
      </div>
    );
  }

  if (variant === "partial-failure") {
    return (
      <div>
        <EmptyState
          title="Scan finished with partial failures"
          description="Some evaluators failed. Existing findings are still shown; retry the scan after checking integration health."
        />
        {onScan ? (
          <div style={{ marginTop: "1rem" }}>
            <Button onClick={onScan}>Retry scan</Button>
          </div>
        ) : null}
      </div>
    );
  }

  if (variant === "no-matches") {
    return (
      <div>
        <EmptyState
          title="No findings match these filters"
          description="Try clearing search or status filters to see more findings."
        />
        {onClearFilters ? (
          <div style={{ marginTop: "1rem" }}>
            <Button onClick={onClearFilters}>Clear filters</Button>
          </div>
        ) : null}
      </div>
    );
  }

  return (
    <div>
      <EmptyState
        title="No open findings"
        description="There are no open or acknowledged findings right now. Adjust filters to include resolved or dismissed history, or run a new scan."
      />
      {onScan ? (
        <div style={{ marginTop: "1rem" }}>
          <Button onClick={onScan}>Scan project</Button>
        </div>
      ) : null}
    </div>
  );
}
