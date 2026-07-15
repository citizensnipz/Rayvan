import { Button, EmptyState } from "@rayvan/ui";

interface IntegrationEmptyStateProps {
  onAddIntegration?: () => void;
}

export function IntegrationEmptyState({ onAddIntegration }: IntegrationEmptyStateProps) {
  return (
    <div>
      <EmptyState
        title="No integrations configured"
        description="Connect Rayvan to the services used by this project."
      />
      {onAddIntegration ? (
        <div style={{ marginTop: "1rem" }}>
          <Button onClick={onAddIntegration}>Add integration</Button>
        </div>
      ) : null}
    </div>
  );
}
